"""创建时间：2026-09-28；功能：用 PyTorch YOLO26 对任意图片推理并保存叠图；目的：在 C++ 部署前复核口罩打结结果。"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO26 PyTorch 口罩打结推理")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True, help="单图或图片目录")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--nms-free", action="store_true", help="使用 one-to-one 免 NMS 分支")
    parser.add_argument("--device", default="0")
    parser.add_argument("--normalize-mean", type=float,
                        help="可选：每张图按平均灰度平移到指定值；当前实验建议 72")
    parser.add_argument("--blur-kernel", type=int, default=0,
                        help="可选：奇数核的整图高斯模糊；0 表示关闭")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (not 0 < args.confidence < 1 or args.imgsz <= 0 or
            (args.normalize_mean is not None and not 0 <= args.normalize_mean <= 255) or
            (args.blur_kernel and (args.blur_kernel < 3 or args.blur_kernel % 2 == 0))):
        raise ValueError("置信度、图像尺寸或模糊核无效")
    source = args.source.resolve()
    if source.is_file():
        files = [source]
    elif source.is_dir():
        files = sorted(file for file in source.iterdir()
                       if file.is_file() and file.suffix.lower() in EXTENSIONS)
    else:
        raise FileNotFoundError(source)
    if not files:
        raise ValueError("输入路径没有可处理图片")
    output = (args.output_dir or ROOT / "x64" / "output" /
              datetime.now().strftime("%Y%m%d_%H%M%S") / "predict_pt").resolve()
    output.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.weights.resolve()))
    records = []
    for index, path in enumerate(files, 1):
        image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"无法解码：{path}")
        if args.normalize_mean is not None:
            gray_mean = float(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).mean())
            image = np.clip(image.astype(np.float32) + args.normalize_mean - gray_mean,
                            0, 255).astype(np.uint8)
        if args.blur_kernel:
            image = cv2.GaussianBlur(image, (args.blur_kernel, args.blur_kernel), 0)
        result = model.predict(image, imgsz=args.imgsz, conf=args.confidence, iou=0.5,
                               max_det=100, verbose=False, device=args.device,
                               nms=not args.nms_free)[0]
        boxes = []
        marked = image.copy()
        for xyxy, confidence in zip(result.boxes.xyxy.cpu().numpy(),
                                    result.boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = map(int, xyxy)
            cv2.rectangle(marked, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(marked, f"knot {confidence:.3f}", (x1, max(18, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
            boxes.append({"xyxy": [float(value) for value in xyxy],
                          "confidence": float(confidence)})
        target = output / f"{path.stem}_pred.jpg"
        cv2.imencode(".jpg", marked)[1].tofile(str(target))
        records.append({"source": str(path), "result": str(target), "boxes": boxes})
        print(f"[推理] {index}/{len(files)} {path.name}：{len(boxes)} 框", flush=True)
    (output / "predictions.json").write_text(
        json.dumps({"weights": str(args.weights.resolve()), "imgsz": args.imgsz,
                    "confidence": args.confidence, "nms_free": args.nms_free,
                    "normalize_mean": args.normalize_mean, "blur_kernel": args.blur_kernel,
                    "items": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"推理结果：{output}", flush=True)


if __name__ == "__main__":
    main()
