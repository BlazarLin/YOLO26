"""创建时间：2026-09-28；功能：分别扰动打结区与背景亮度；目的：诊断模型分数依赖局部形态还是整图采集条件。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from evaluate_knot import iou, load_image


def make_mask(shape: tuple[int, int], box: list[float], margin: int = 20) -> np.ndarray:
    height, width = shape
    x1, y1, x2, y2 = box
    mask = np.zeros((height, width), dtype=np.float32)
    xa, ya = max(0, int(x1) - margin), max(0, int(y1) - margin)
    xb, yb = min(width, int(x2) + margin), min(height, int(y2) + margin)
    mask[ya:yb, xa:xb] = 1.0
    return cv2.GaussianBlur(mask, (0, 0), sigmaX=10)


def score(model: YOLO, image: np.ndarray, truth: list[list[float]], imgsz: int) -> tuple[float, float]:
    result = model.predict(image, imgsz=imgsz, conf=0.001, iou=0.5,
                           max_det=100, verbose=False, device=0, nms=True)[0]
    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    matched = max((float(confidence) for box, confidence in zip(boxes, scores)
                   if any(iou(np.asarray(gt), box) >= 0.5 for gt in truth)), default=0.0)
    any_score = max((float(confidence) for confidence in scores), default=0.0)
    return matched, any_score


def main() -> None:
    parser = argparse.ArgumentParser(description="打结区和背景亮度扰动诊断")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shift", type=float, default=11.0)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    source = args.run_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((source / "dataset_manifest.json").read_text(encoding="utf-8"))
    model = YOLO(str(args.weights))
    rows = []
    variants = ("original", "global", "target", "background")
    for index, item in enumerate(manifest["items"], 1):
        if item["split"] != "val" or item["class"] != "NG":
            continue
        image = load_image(source / "dataset" / "images" / "val" / item["file"])
        mask = np.maximum.reduce([make_mask(image.shape[:2], box) for box in item["boxes"]])
        weights = (np.zeros_like(mask), np.ones_like(mask), mask, 1.0 - mask)
        measurements = {}
        for name, weight in zip(variants, weights):
            changed = np.clip(image.astype(np.float32) +
                              args.shift * weight[:, :, None], 0, 255).astype(np.uint8)
            matched, any_score = score(model, changed, item["boxes"], args.imgsz)
            measurements[name] = {"matched_score": matched, "any_score": any_score,
                                  "strict_hit_025": matched >= 0.25}
        rows.append({"file": item["file"], "truth": item["boxes"],
                     "measurements": measurements})
        if len(rows) % 10 == 0:
            print(f"[局部光照] {len(rows)}/29", flush=True)
    summary = {"shift": args.shift, "count": len(rows)}
    for name in variants:
        values = np.asarray([row["measurements"][name]["matched_score"] for row in rows])
        base = np.asarray([row["measurements"]["original"]["matched_score"] for row in rows])
        summary[name] = {"strict_hits_025": int((values >= 0.25).sum()),
                         "median_score": float(np.median(values)),
                         "median_score_change": float(np.median(values - base)),
                         "score_drop_over_005": int(((base - values) > 0.05).sum())}
    (output / "local_brightness.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
