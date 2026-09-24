"""创建时间：2026-09-23；功能：直接调用 Python ONNX Runtime 或 TensorRT 推理；目的：与 C++ 入口按相同条件测速。"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np


def summarize(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "mean_ms": statistics.mean(samples),
        "median_ms": statistics.median(samples),
        "p95_ms": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
    }


def benchmark_onnx(path: Path, data: np.ndarray, threads: int, warmup: int, iterations: int) -> dict:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    samples = []
    output_sum = 0.0
    for index in range(warmup + iterations):
        start = time.perf_counter_ns()
        result = session.run([output_name], {input_name: data})[0]
        end = time.perf_counter_ns()
        if index >= warmup:
            samples.append((end - start) / 1_000_000)
        if index == warmup:
            output_sum = float(result.astype(np.float64).sum())
    return {"backend": "onnx_cpu", **summarize(samples), "output_sum": output_sum,
            "onnxruntime_version": ort.__version__}


def read_engine(path: Path) -> bytes:
    contents = path.read_bytes()
    header_size = int.from_bytes(contents[:4], "little")
    if 0 < header_size < len(contents) - 4 and contents[4:5] == b"{":
        json.loads(contents[4:4 + header_size])
        return contents[4 + header_size:]
    return contents


def benchmark_trt(path: Path, data: np.ndarray, warmup: int, iterations: int) -> dict:
    import tensorrt as trt
    import torch

    torch.cuda.set_device(0)
    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(read_engine(path))
    if engine is None:
        raise RuntimeError("TensorRT engine 反序列化失败")
    context = engine.create_execution_context()
    names = [engine.get_tensor_name(index) for index in range(engine.num_io_tensors)]
    input_name = next(name for name in names if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT)
    output_name = next(name for name in names if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT)
    input_tensor = torch.from_numpy(data).to("cuda")
    output_shape = tuple(engine.get_tensor_shape(output_name))
    if any(dimension <= 0 for dimension in output_shape):
        raise RuntimeError("本测试仅支持静态 batch=1 engine")
    output_tensor = torch.empty(output_shape, dtype=torch.float32, device="cuda")
    context.set_tensor_address(input_name, input_tensor.data_ptr())
    context.set_tensor_address(output_name, output_tensor.data_ptr())
    stream = torch.cuda.Stream()
    samples = []
    for index in range(warmup + iterations):
        start = time.perf_counter_ns()
        if not context.execute_async_v3(stream_handle=stream.cuda_stream):
            raise RuntimeError("TensorRT 推理失败")
        stream.synchronize()
        end = time.perf_counter_ns()
        if index >= warmup:
            samples.append((end - start) / 1_000_000)
    return {"backend": "tensorrt", **summarize(samples),
            "output_sum": float(output_tensor.double().sum().item()),
            "tensorrt_version": trt.__version__}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("backend", choices=["onnx", "trt"])
    parser.add_argument("model", type=Path)
    parser.add_argument("input", type=Path)
    parser.add_argument("threads", type=int)
    parser.add_argument("warmup", type=int)
    parser.add_argument("iterations", type=int)
    args = parser.parse_args()
    if args.threads <= 0 or args.warmup < 0 or args.iterations <= 0:
        parser.error("线程数、预热次数或测量次数无效")
    data = np.fromfile(args.input, dtype=np.float32)
    if data.size != 3 * 640 * 640:
        parser.error("输入文件必须是 1x3x640x640 FP32")
    data = data.reshape(1, 3, 640, 640)
    if args.backend == "onnx":
        result = benchmark_onnx(args.model, data, args.threads, args.warmup, args.iterations)
    else:
        result = benchmark_trt(args.model, data, args.warmup, args.iterations)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
