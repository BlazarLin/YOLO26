"""创建时间：2026-09-24；功能：导出 YOLO26 FP32 TensorRT 与 CPU ONNX；目的：固定免 NMS 分支并避免 engine 导出覆盖最终 ONNX。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出 YOLO26 免 NMS CPU ONNX 与 TensorRT engine")
    parser.add_argument("--weights", type=Path, required=True, help="训练得到的 best.pt")
    parser.add_argument("--output-dir", type=Path, help="生成目录，默认 x64/output/时间戳")
    parser.add_argument("--name", default="yolo26n_e2e", help="导出文件名，不含扩展名")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--onnx-only", action="store_true", help="无 NVIDIA GPU 时只导出 ONNX")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.weights.resolve()
    if not source.is_file() or args.imgsz != 640 or not re.fullmatch(r"[A-Za-z0-9_-]+", args.name):
        raise ValueError("权重路径、图像尺寸或模型名无效")
    output_dir = (args.output_dir or ROOT / "x64" / "output" /
                  datetime.now().strftime("%Y%m%d_%H%M%S")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    weight = output_dir / f"{args.name}.pt"
    if source != weight:
        if weight.is_file() and digest(weight) != digest(source):
            raise FileExistsError(f"目标权重与源权重不同：{weight}")
        if not weight.is_file():
            shutil.copy2(source, weight)

    import onnxruntime as ort
    import torch
    from ultralytics import YOLO, __version__ as ultralytics_version

    engine = weight.with_suffix(".engine")
    onnx = weight.with_suffix(".onnx")
    if not args.onnx_only:
        if not torch.cuda.is_available():
            raise RuntimeError("TensorRT 导出需要 CUDA GPU；仅导出 ONNX 请使用 --onnx-only")
        if not engine.is_file():
            print("导出 TensorRT engine...", flush=True)
            YOLO(str(weight)).export(format="engine", imgsz=args.imgsz, batch=1,
                                     dynamic=False, half=False, opset=17,
                                     simplify=True, device=0, nms=False)
    print("导出独立 CPU ONNX...", flush=True)
    YOLO(str(weight)).export(format="onnx", imgsz=args.imgsz, batch=1,
                             dynamic=False, half=False, opset=17,
                             simplify=True, device="cpu", nms=False)
    if not onnx.is_file() or (not args.onnx_only and not engine.is_file()):
        raise RuntimeError("导出文件缺失")
    session = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
    shape = session.get_outputs()[0].shape
    if shape != [1, 300, 6]:
        raise RuntimeError(f"ONNX 输出不是免 NMS 格式：(1,300,6)，实际 {shape}")
    report = {
        "created_at": datetime.now().astimezone().isoformat(),
        "source_weight": str(source), "weight": str(weight),
        "onnx_cpu": str(onnx), "tensorrt": str(engine) if engine.is_file() else None,
        "output_shape": shape, "imgsz": args.imgsz, "batch": 1, "precision": "FP32",
        "opset": 17, "nms": False, "ultralytics": ultralytics_version,
        "onnxruntime_python": ort.__version__,
    }
    (output_dir / "export_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
