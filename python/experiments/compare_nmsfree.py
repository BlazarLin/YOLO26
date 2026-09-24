"""创建时间：2026-09-23。功能：比较默认与免 NMS ONNX 路径。目的：核对官网测速条件。"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import onnxruntime
from ultralytics import YOLO

from experiment import benchmark_one, load_images


def main() -> None:
    parser = argparse.ArgumentParser(description="核对 YOLO26 免 NMS 路径的 CPU ONNX 耗时")
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=128)
    args = parser.parse_args()

    source_run = args.source_run.resolve()
    output_dir = (args.output_dir or Path(__file__).resolve().parents[2] / "x64" / "output" /
                  datetime.now().strftime("%Y%m%d_%H%M%S")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_pt = source_run / "runs" / "yolo26n" / "weights" / "best.pt"
    e2e_pt = output_dir / "yolo26n_e2e.pt"
    e2e_onnx = e2e_pt.with_suffix(".onnx")
    if not e2e_pt.is_file():
        shutil.copy2(source_pt, e2e_pt)
    if not e2e_onnx.is_file():
        YOLO(str(e2e_pt)).export(format="onnx", imgsz=640, batch=1,
                                dynamic=False, half=False, opset=17,
                                simplify=True, device="cpu", nms=False)

    paths = {
        "yolov8n_default": source_run / "runs" / "yolov8n" / "weights" / "best.onnx",
        "yolo26n_default": source_run / "runs" / "yolo26n" / "weights" / "best.onnx",
        "yolo26n_nmsfree": e2e_onnx,
    }
    shapes = {}
    for name, path in paths.items():
        session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        shapes[name] = session.get_outputs()[0].shape
    if shapes["yolo26n_nmsfree"] != [1, 300, 6]:
        raise RuntimeError(f"免 NMS 导出形状不符：{shapes['yolo26n_nmsfree']}")

    images, image_paths = load_images(source_run / "datasets" / "coco128.yaml", 128)
    orders = (
        ("yolov8n_default", "yolo26n_default", "yolo26n_nmsfree"),
        ("yolo26n_nmsfree", "yolo26n_default", "yolov8n_default"),
    )
    rounds = []
    for order in orders:
        measurements = {}
        for name in order:
            print(f"测速：{name}", flush=True)
            measurements[name] = benchmark_one(str(paths[name]), "onnx_cpu", images,
                                               args.warmup, args.iterations, 640)
        rounds.append(measurements)
    report = {
        "created_at": datetime.now().astimezone().isoformat(),
        "source_run": str(source_run), "output_shapes": shapes,
        "warmup": args.warmup, "iterations": args.iterations,
        "images": image_paths, "rounds": rounds,
    }
    output = output_dir / "compare_nmsfree.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for index, measurements in enumerate(rounds, start=1):
        for name, item in measurements.items():
            print(f"第 {index} 轮 {name}: 完整检测 {item['wall_detect']['mean_ms']} ms; "
                  f"模型执行 {item['model_inference']['mean_ms']} ms", flush=True)
    print(f"结果文件：{output}", flush=True)


if __name__ == "__main__":
    main()
