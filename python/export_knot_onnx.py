"""创建时间：2026-09-28；功能：导出打结检测的动态 ONNX；目的：保持 V6 one-to-many + C++ NMS 的验证口径。"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

import onnx
import onnxruntime as ort
from ultralytics import YOLO, __version__ as ultralytics_version


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="导出 one-to-many 打结 ONNX")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", default="v6_knot")
    args = parser.parse_args()
    source = args.weights.resolve()
    output = args.output_dir.resolve()
    if not source.is_file() or not args.name.replace("_", "").isalnum():
        raise ValueError("权重路径或模型名无效")
    output.mkdir(parents=True, exist_ok=True)
    weight = output / f"{args.name}.pt"
    if weight.exists():
        if digest(weight) != digest(source):
            raise FileExistsError(f"目标权重与源权重不同：{weight}")
    else:
        shutil.copy2(source, weight)

    exported = Path(YOLO(str(weight)).export(
        format="onnx", imgsz=640, batch=1, dynamic=True, opset=17,
        simplify=True, device="cpu", nms=None))
    if not exported.is_file():
        raise RuntimeError("ONNX 导出文件缺失")
    session = ort.InferenceSession(str(exported), providers=["CPUExecutionProvider"])
    input_shape = session.get_inputs()[0].shape
    output_shape = session.get_outputs()[0].shape
    if input_shape != ["batch", 3, "height", "width"] or output_shape != ["batch", 5, "anchors"]:
        raise RuntimeError(f"ONNX 形状不是一类原始检测分支：{input_shape} -> {output_shape}")
    model = onnx.load(str(exported), load_external_data=False)
    report = {
        "created_at": datetime.now().astimezone().isoformat(),
        "source_weight": str(source), "source_sha256": digest(source),
        "copied_weight": str(weight), "onnx": str(exported),
        "onnx_sha256": digest(exported),
        "onnx_ir": model.ir_version,
        "opset": [(item.domain, item.version) for item in model.opset_import],
        "input_shape": input_shape, "output_shape": output_shape,
        "dynamic": True, "nms": None, "precision": "FP32",
        "ultralytics": ultralytics_version,
        "onnxruntime_python_validation": ort.__version__,
    }
    (output / "export_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
