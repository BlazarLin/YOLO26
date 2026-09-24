"""创建时间：2026-09-24；功能：直接执行 YOLO26 CPU ONNX 与 TensorRT 检测；目的：保存逐图结果并测量预热后的完整检测耗时。"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SIZE = 640
CONFIDENCE = 0.25


def prepare(image: np.ndarray) -> tuple[np.ndarray, float, int, int]:
    height, width = image.shape[:2]
    ratio = min(SIZE / height, SIZE / width)
    resized_width = int(np.floor(width * ratio + 0.5))
    resized_height = int(np.floor(height * ratio + 0.5))
    left = int(np.floor((SIZE - resized_width) / 2 - 0.1 + 0.5))
    top = int(np.floor((SIZE - resized_height) / 2 - 0.1 + 0.5))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(resized, top, SIZE - resized_height - top,
                                left, SIZE - resized_width - left,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
    pixels = np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32)
    pixels *= np.float32(1 / 255.0)
    return pixels[None], ratio, left, top


def decode(output: np.ndarray, image: np.ndarray, ratio: float,
           left: int, top: int) -> np.ndarray:
    if output.shape != (1, 300, 6):
        raise ValueError(f"只支持 YOLO26 640 FP32 免 NMS 输出 (1,300,6)，实际 {output.shape}")
    boxes = output[0, output[0, :, 4] >= CONFIDENCE].copy()
    if boxes.size:
        boxes[:, [0, 2]] = np.clip((boxes[:, [0, 2]] - left) / ratio, 0, image.shape[1])
        boxes[:, [1, 3]] = np.clip((boxes[:, [1, 3]] - top) / ratio, 0, image.shape[0])
    return boxes


def read_engine(path: Path) -> bytes:
    contents = path.read_bytes()
    if len(contents) < 5:
        raise ValueError("TensorRT engine 文件过短")
    header_size = int.from_bytes(contents[:4], "little")
    if 0 < header_size < len(contents) - 4 and contents[4:5] == b"{":
        json.loads(contents[4:4 + header_size])
        return contents[4 + header_size:]
    return contents


class OnnxBackend:
    def __init__(self, path: Path, threads: int) -> None:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(path), sess_options=options,
                                            providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def run(self, pixels: np.ndarray) -> np.ndarray:
        return self.session.run([self.output_name], {self.input_name: pixels})[0]


class TrtBackend:
    def __init__(self, path: Path) -> None:
        import tensorrt as trt
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("TensorRT 推理需要 CUDA GPU")
        self.torch = torch
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(read_engine(path))
        if self.engine is None:
            raise RuntimeError("TensorRT engine 反序列化失败")
        self.context = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(index) for index in range(self.engine.num_io_tensors)]
        input_name = next(name for name in names
                          if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT)
        output_name = next(name for name in names
                           if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT)
        if tuple(self.engine.get_tensor_shape(input_name)) != (1, 3, SIZE, SIZE):
            raise ValueError("TensorRT 输入不是静态 1x3x640x640")
        output_shape = tuple(self.engine.get_tensor_shape(output_name))
        if output_shape != (1, 300, 6):
            raise ValueError(f"TensorRT 输出不是免 NMS 格式：{output_shape}")
        self.host_input = torch.empty((1, 3, SIZE, SIZE), dtype=torch.float32, pin_memory=True)
        self.host_output = torch.empty(output_shape, dtype=torch.float32, pin_memory=True)
        self.device_input = torch.empty_like(self.host_input, device="cuda")
        self.device_output = torch.empty(output_shape, dtype=torch.float32, device="cuda")
        self.stream = torch.cuda.Stream()
        if not self.context.set_tensor_address(input_name, self.device_input.data_ptr()):
            raise RuntimeError("TensorRT 输入地址设置失败")
        if not self.context.set_tensor_address(output_name, self.device_output.data_ptr()):
            raise RuntimeError("TensorRT 输出地址设置失败")

    def run(self, pixels: np.ndarray) -> np.ndarray:
        np.copyto(self.host_input.numpy(), pixels)
        with self.torch.cuda.stream(self.stream):
            self.device_input.copy_(self.host_input, non_blocking=True)
            if not self.context.execute_async_v3(stream_handle=self.stream.cuda_stream):
                raise RuntimeError("TensorRT enqueueV3 失败")
            self.host_output.copy_(self.device_output, non_blocking=True)
        self.stream.synchronize()
        return self.host_output.numpy()


def render(image: np.ndarray, boxes: np.ndarray, destination: Path) -> None:
    annotated = image.copy()
    for x1, y1, x2, y2, confidence, class_id in boxes:
        first = (round(float(x1)), round(float(y1)))
        second = (round(float(x2)), round(float(y2)))
        cv2.rectangle(annotated, first, second, (0, 0, 255), 2)
        cv2.putText(annotated, f"class {int(class_id)} {confidence:.2f}",
                    (first[0], max(20, first[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 0, 255), 2)
    if len(boxes) == 0:
        cv2.putText(annotated, "0 detections", (20, 35), cv2.FONT_HERSHEY_SIMPLEX,
                    0.85, (0, 0, 255), 2)
    valid, encoded = cv2.imencode(".png", annotated)
    if not valid:
        raise RuntimeError(f"结果图编码失败：{destination}")
    encoded.tofile(str(destination))


def summarize(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {"mean_ms": round(statistics.mean(samples), 3),
            "median_ms": round(statistics.median(samples), 3),
            "p95_ms": round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))], 3)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO26 免 NMS CPU ONNX / TensorRT 检测")
    parser.add_argument("backend", choices=("onnx", "trt"))
    parser.add_argument("model", type=Path)
    parser.add_argument("source", type=Path, help="图片文件或图片目录")
    parser.add_argument("--output-dir", type=Path, help="结果目录，默认 x64/output/时间戳")
    parser.add_argument("--threads", type=int, default=4, help="CPU ONNX 线程数")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=1, help="每张图的测量次数")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.threads <= 0 or args.warmup < 0 or args.iterations <= 0:
        raise ValueError("线程数、预热次数或测量次数无效")
    if not args.model.is_file():
        raise FileNotFoundError(args.model)
    if args.source.is_file():
        paths = [args.source]
    elif args.source.is_dir():
        paths = sorted(path for path in args.source.iterdir()
                       if path.suffix.lower() in {".bmp", ".jpg", ".jpeg", ".png"})
    else:
        raise FileNotFoundError(args.source)
    if not paths:
        raise RuntimeError("没有可推理的图片")
    output_dir = (args.output_dir or ROOT / "x64" / "output" /
                  datetime.now().strftime("%Y%m%d_%H%M%S")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    backend = OnnxBackend(args.model.resolve(), args.threads) if args.backend == "onnx" \
        else TrtBackend(args.model.resolve())
    first = cv2.imdecode(np.fromfile(str(paths[0]), dtype=np.uint8), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"图片读取失败：{paths[0]}")
    for _ in range(args.warmup):
        pixels, ratio, left, top = prepare(first)
        decode(backend.run(pixels), first, ratio, left, top)
    results = []
    for path in paths:
        image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"图片读取失败：{path}")
        samples = []
        boxes = np.empty((0, 6), dtype=np.float32)
        for _ in range(args.iterations):
            start = time.perf_counter_ns()
            pixels, ratio, left, top = prepare(image)
            boxes = decode(backend.run(pixels), image, ratio, left, top)
            samples.append((time.perf_counter_ns() - start) / 1_000_000)
        destination = output_dir / f"{path.stem}_{path.suffix[1:].lower()}.png"
        render(image, boxes, destination)
        item = {"source": str(path.resolve()), "result": str(destination),
                "count": len(boxes), **summarize(samples)}
        results.append(item)
        print(f"{path.name}: {item['count']} 框，平均 {item['mean_ms']} ms", flush=True)
    report = {"backend": args.backend, "model": str(args.model.resolve()),
              "warmup": args.warmup, "iterations_per_image": args.iterations,
              "cpu_threads": args.threads if args.backend == "onnx" else None,
              "timing_scope": "预处理 + 运行时 + 后处理；不含读图、绘图、写盘与模型加载",
              "results": results}
    (output_dir / "inference.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结果目录：{output_dir}")


if __name__ == "__main__":
    main()
