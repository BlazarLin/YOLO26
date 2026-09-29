"""创建时间：2026-09-28；功能：测量 Python YOLO26 单图推理耗时；目的：比较模型尺寸与免 NMS 分支的延迟代价。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Python 打结模型预热后推理计时")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--second-weights", type=Path,
                        help="额外运行第二个模型，测量双模型并集所需的两次推理代价")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--nms-free", action="store_true")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--normalize-mean", type=float)
    parser.add_argument("--blur-kernel", type=int, default=0)
    parser.add_argument("--tile-fraction", type=float,
                        help="启用两块重叠横向裁图，并把预测框映回原图合并")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.warmup < 1 or args.iterations < 10:
        raise ValueError("预热或计时次数过少")
    if (not 0 < args.confidence < 1 or
            (args.blur_kernel and (args.blur_kernel < 3 or args.blur_kernel % 2 == 0))):
        raise ValueError("置信度或模糊核无效")
    if args.tile_fraction is not None and not 0.5 < args.tile_fraction < 1.0:
        raise ValueError("裁图比例无效")
    image = cv2.imdecode(np.fromfile(str(args.image), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"图片无法读取：{args.image}")
    model = YOLO(str(args.weights))
    second_model = YOLO(str(args.second_weights)) if args.second_weights else None
    if second_model is not None and args.tile_fraction is not None:
        raise ValueError("双模型与双裁图计时需分别进行")
    def collect(result) -> list[dict]:
        return [{"box": [float(value) for value in box],
                 "confidence": float(confidence)}
                for box, confidence in zip(result.boxes.xyxy.cpu().numpy(),
                                           result.boxes.conf.cpu().numpy())]
    def infer() -> None:
        prepared = image
        if args.normalize_mean is not None:
            gray_mean = float(cv2.cvtColor(prepared, cv2.COLOR_BGR2GRAY).mean())
            prepared = np.clip(prepared.astype(np.float32) + args.normalize_mean - gray_mean,
                               0, 255).astype(np.uint8)
        if args.blur_kernel:
            prepared = cv2.GaussianBlur(prepared, (args.blur_kernel, args.blur_kernel), 0)
        if args.tile_fraction is None:
            first = model.predict(prepared, imgsz=args.imgsz, conf=args.confidence, iou=0.5,
                                  max_det=100, verbose=False, device=0,
                                  nms=not args.nms_free)[0]
            predictions = collect(first)
            if second_model is not None:
                from evaluate_knot_tiles import merge_predictions

                second = second_model.predict(prepared, imgsz=args.imgsz,
                                              conf=args.confidence, iou=0.5, max_det=100,
                                              verbose=False, device=0,
                                              nms=not args.nms_free)[0]
                predictions.extend(collect(second))
                merge_predictions(predictions)
        else:
            from evaluate_knot_tiles import merge_predictions

            width = prepared.shape[1]
            tile_width = round(width * args.tile_fraction)
            predictions = []
            for start in (0, width - tile_width):
                result = model.predict(prepared[:, start:start + tile_width],
                                       imgsz=args.imgsz, conf=args.confidence,
                                       iou=0.5, max_det=100, verbose=False,
                                       device=0, nms=not args.nms_free)[0]
                for box, confidence in zip(result.boxes.xyxy.cpu().numpy(),
                                           result.boxes.conf.cpu().numpy()):
                    x1, y1, x2, y2 = box
                    predictions.append({"box": [float(x1 + start), float(y1),
                                                float(x2 + start), float(y2)],
                                        "confidence": float(confidence)})
            merge_predictions(predictions)
    for _ in range(args.warmup):
        infer()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    times = []
    for _ in range(args.iterations):
        start = time.perf_counter()
        infer()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    result = {"name": args.name, "weights": str(args.weights.resolve()),
              "second_weights": str(args.second_weights.resolve()) if args.second_weights else None,
              "image": str(args.image.resolve()), "imgsz": args.imgsz,
              "nms_free": args.nms_free, "warmup": args.warmup,
              "confidence": args.confidence, "normalize_mean": args.normalize_mean,
              "blur_kernel": args.blur_kernel, "tile_fraction": args.tile_fraction,
              "iterations": args.iterations, "device": torch.cuda.get_device_name(0),
              "scope": "预加载的单张图；含可选归一/模糊、每个 model.predict 的预处理/网络/后处理、预测框 CPU 转存及可选双模型或双裁图框合并，不含读盘与模型加载",
              "mean_ms": float(np.mean(times)), "median_ms": float(np.median(times)),
              "p90_ms": float(np.percentile(times, 90)),
              "min_ms": float(np.min(times)), "max_ms": float(np.max(times))}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / f"{args.name}.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
