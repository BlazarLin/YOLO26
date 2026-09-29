"""创建时间：2026-09-28；功能：评估 C++ ONNX 全量预测；目的：逐图核对 NG 定位、OK 误报和 Python 结果。"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估 C++ ONNX 打结预测")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python-reference", type=Path, action="append")
    return parser.parse_args()


def iou(first: list[float], second: list[float]) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    return intersection / max(1e-9, first_area + second_area - intersection)


def draw_error(image_path: Path, target: Path, truth: list[list[float]],
               predictions: list[dict]) -> None:
    image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图像：{image_path}")
    for box in truth:
        x1, y1, x2, y2 = map(round, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(image, "GT", (x1, max(20, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2)
    for pred in predictions:
        x1, y1, x2, y2 = map(round, pred["box"])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 220), 2)
        cv2.putText(image, f"{pred['confidence']:.3f}",
                    (x1, min(image.shape[0] - 5, y2 + 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 220), 2)
    target.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(".jpg", image)
    if not success:
        raise OSError(f"无法编码复核图：{target}")
    encoded.tofile(str(target))


def read_predictions(path: Path) -> tuple[dict[tuple[str, str], list[dict]], dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    timings = {}
    with path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if None in row.values():
                raise ValueError(f"CSV 列数错误：{row}")
            key = (row["split"], row["file"])
            timings[key] = {name: float(row[name]) for name in
                            ("read_ms", "preprocess_ms", "inference_ms", "postprocess_ms")}
            grouped[key]
            if row["box_index"]:
                grouped[key].append({
                    "box": [float(row[name]) for name in ("x1", "y1", "x2", "y2")],
                    "confidence": float(row["confidence"]),
                })
    return dict(grouped), timings


def summarize(rows: list[dict]) -> dict:
    ng = [row for row in rows if row["category"] == "NG"]
    ok = [row for row in rows if row["category"] == "OK"]
    return {"images": len(rows), "ng_total": len(ng),
            "ng_detected_iou50": sum(row["status"] == "检出" for row in ng),
            "ng_missed": sum(row["status"] == "漏检" for row in ng),
            "ok_total": len(ok),
            "ok_false_alarms": sum(row["status"] == "误报" for row in ok)}


def parity(rows: list[dict], predictions: dict, reference_path: Path) -> dict:
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    differences = []
    status_differences = []
    statuses = {(row["split"], row["file"]): row["status"] for row in rows}
    max_box_difference = 0.0
    max_score_difference = 0.0
    compared_boxes = 0
    for item in reference:
        key = (item["split"], item["file"])
        cpp = sorted(predictions[key], key=lambda pred: -pred["confidence"])
        python = sorted((pred for pred in item["predictions"]
                         if pred["confidence"] >= 0.25),
                        key=lambda pred: -pred["confidence"])
        truth = item["truth"]
        python_hit = bool(truth) and all(
            any(iou(gt, pred["box"]) >= 0.5 for pred in python) for gt in truth)
        python_status = (("检出" if python_hit else "漏检") if truth else
                         ("误报" if python else "正确OK"))
        if statuses[key] != python_status:
            status_differences.append({"file": item["file"], "cpp": statuses[key],
                                       "python": python_status})
        if len(cpp) != len(python):
            differences.append({"file": item["file"], "cpp": len(cpp),
                                "python": len(python)})
            continue
        for left, right in zip(cpp, python):
            max_box_difference = max(max_box_difference,
                                     max(abs(a - b) for a, b in
                                         zip(left["box"], right["box"])))
            max_score_difference = max(max_score_difference,
                                       abs(left["confidence"] - right["confidence"]))
            compared_boxes += 1
    return {"reference": str(reference_path), "images": len(reference),
            "box_count_disagreements": differences,
            "status_disagreements": status_differences,
            "compared_boxes": compared_boxes,
            "max_coordinate_difference_px": max_box_difference,
            "max_confidence_difference": max_score_difference}


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    predictions, timings = read_predictions(args.predictions)
    expected = {(item["split"], item["file"]) for item in manifest["items"]}
    if len(expected) != len(manifest["items"]) or set(predictions) != expected:
        raise ValueError(f"预测与清单不匹配：缺失 {expected - set(predictions)}；多出 {set(predictions) - expected}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in manifest["items"]:
        key = (item["split"], item["file"])
        boxes = predictions[key]
        truth = item["boxes"]
        best_iou = max((iou(gt, pred["box"]) for gt in truth for pred in boxes),
                       default=0.0)
        hit = bool(truth) and all(any(iou(gt, pred["box"]) >= 0.5 for pred in boxes)
                                  for gt in truth)
        status = ("检出" if hit else "漏检") if truth else ("误报" if boxes else "正确OK")
        row = {"split": item["split"], "file": item["file"],
               "category": item["class"], "status": status,
               "detections": len(boxes), "best_iou": round(best_iou, 6),
               "max_confidence": round(max((pred["confidence"] for pred in boxes),
                                        default=0.0), 6), **timings[key]}
        rows.append(row)
        if status in ("漏检", "误报"):
            draw_error(args.image_root / item["split"] / item["file"],
                       args.output_dir / "errors" / item["split"] /
                       (Path(item["file"]).stem + ".jpg"), truth, boxes)

    fields = ("split", "file", "category", "status", "detections", "best_iou",
              "max_confidence", "read_ms", "preprocess_ms", "inference_ms", "postprocess_ms")
    with (args.output_dir / "per_image.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    by_split = {name: summarize([row for row in rows if row["split"] == name])
                for name in ("train", "val")}
    by_ok_origin = {name: summarize([row for row in rows
                                     if row["category"] == "OK" and
                                     row["file"].lower().endswith(suffix)])
                    for name, suffix in (("old_bmp", ".bmp"), ("new_png", ".png"))}
    times = sorted(row["inference_ms"] for row in rows)
    summary = {"model": "V6 YOLO26n ONNX raw one-to-many + C++ NMS",
               "confidence": 0.25, "nms_iou": 0.5, "strict_match_iou": 0.5,
               "all": summarize(rows), "by_split": by_split,
               "ok_by_origin": by_ok_origin,
               "cpu_onnx_inference_ms": {"median": times[len(times) // 2],
                                         "p90": times[int(len(times) * 0.9)]},
               "missed_files": [row["file"] for row in rows if row["status"] == "漏检"],
               "false_alarm_files": [row["file"] for row in rows if row["status"] == "误报"]}
    if args.python_reference:
        summary["python_pt_parity"] = [parity(rows, predictions, path)
                                        for path in args.python_reference]
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
