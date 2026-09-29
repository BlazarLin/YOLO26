"""创建时间：2026-09-28；功能：遮挡打结框并对比预测分数；目的：验证模型是否依赖标注区域的图像证据。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="打结区域遮挡诊断")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--nms-free", action="store_true")
    return parser.parse_args()


def iou(a: list[float], b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return intersection / max(1e-9, area_a + area_b - intersection)


def score(model: YOLO, image: np.ndarray, box: list[float], imgsz: int, nms_free: bool) -> float:
    result = model.predict(image, imgsz=imgsz, conf=0.001, iou=0.5, max_det=100,
                           verbose=False, device=0, nms=not nms_free)[0]
    return max((float(confidence) for predicted, confidence in
                zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy())
                if iou(box, predicted) >= 0.5), default=0.0)


def blur_region(image: np.ndarray, box: list[float]) -> np.ndarray:
    output = image.copy()
    x1, y1, x2, y2 = map(round, box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
    blurred = cv2.GaussianBlur(image, (51, 51), sigmaX=0)
    output[y1:y2, x1:x2] = blurred[y1:y2, x1:x2]
    return output


def main() -> None:
    args = parse_args()
    run = args.run_dir.resolve()
    manifest = json.loads((run / "dataset_manifest.json").read_text(encoding="utf-8"))
    items = [item for item in manifest["items"] if item["split"] == "val" and item["class"] == "NG"]
    model = YOLO(str(args.weights))
    output = run / "occlusion" / args.name
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, item in enumerate(items, 1):
        path = run / "dataset" / "images" / "val" / item["file"]
        image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"无法读取 {path}")
        target = item["boxes"][0]
        width = target[2] - target[0]
        center_x = (target[0] + target[2]) / 2
        control_x = image.shape[1] - center_x
        control = [max(0, control_x - width / 2), target[1],
                   min(image.shape[1], control_x + width / 2), target[3]]
        variants = [image, blur_region(image, target), blur_region(image, control)]
        scores = [score(model, variant, target, args.imgsz, args.nms_free)
                  for variant in variants]
        if scores[0] < 0.25 or abs(scores[0] - scores[1]) < 0.05:
            panels = []
            for variant, label, value in zip(variants, ["original", "target_blur", "control_blur"], scores):
                panel = cv2.resize(variant, (640, round(variant.shape[0] * 640 / variant.shape[1])))
                cv2.putText(panel, f"{label} score={value:.3f}", (8, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 255), 2)
                panels.append(panel)
            montage = np.concatenate(panels, axis=0)
            cv2.imencode(".jpg", montage)[1].tofile(
                str(output / f"{Path(item['file']).stem}.jpg"))
        rows.append({"file": item["file"], "original_score": scores[0],
                     "target_blur_score": scores[1], "control_blur_score": scores[2],
                     "target_drop": scores[0] - scores[1],
                     "control_drop": scores[0] - scores[2]})
        if index % 10 == 0:
            print(f"[遮挡] {index}/{len(items)}", flush=True)
    (output / "occlusion_scores.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"遮挡诊断：{output}", flush=True)


if __name__ == "__main__":
    main()
