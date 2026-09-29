"""创建时间：2026-09-28；功能：两块重叠裁图推理并映回原图；目的：在固定原图验证集上检验放大小目标的收益与误报。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from evaluate_knot import draw, iou, load_image


def merge_predictions(predictions: list[dict], overlap: float = 0.5) -> list[dict]:
    ordered = sorted(predictions, key=lambda row: row["confidence"], reverse=True)
    kept: list[dict] = []
    for candidate in ordered:
        if all(iou(np.asarray(candidate["box"]), np.asarray(previous["box"])) <= overlap
               for previous in kept):
            kept.append(candidate)
        if len(kept) >= 100:
            break
    return kept


def main() -> None:
    parser = argparse.ArgumentParser(description="固定原图拆分上的重叠裁图推理")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--tile-fraction", type=float, default=0.6)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    args = parser.parse_args()
    if not 0.5 < args.tile_fraction < 1.0:
        raise ValueError("裁图宽度比例必须在 0.5 与 1 之间")
    source = args.run_dir.resolve()
    output = args.output_dir.resolve() / "evaluations" / args.name
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((source / "dataset_manifest.json").read_text(encoding="utf-8"))
    items = [row for row in manifest["items"] if row["split"] == args.split]
    model = YOLO(str(args.weights))
    rows = []
    for index, item in enumerate(items, 1):
        image = load_image(source / "dataset" / "images" / args.split / item["file"])
        width = image.shape[1]
        tile_width = round(width * args.tile_fraction)
        predictions = []
        for side, start in zip(("L", "R"), (0, width - tile_width)):
            tile = image[:, start:start + tile_width]
            result = model.predict(tile, imgsz=args.imgsz, conf=0.001, iou=0.5,
                                   max_det=100, verbose=False, device=0, nms=True)[0]
            for box, confidence in zip(result.boxes.xyxy.cpu().numpy(),
                                       result.boxes.conf.cpu().numpy()):
                x1, y1, x2, y2 = box
                predictions.append({"box": [float(x1 + start), float(y1),
                                            float(x2 + start), float(y2)],
                                    "confidence": float(confidence), "tile": side})
        predictions = merge_predictions(predictions)
        truth = item["boxes"]
        match_scores = [max((pred["confidence"] for pred in predictions
                             if iou(np.asarray(box), np.asarray(pred["box"])) >= 0.5),
                            default=0.0) for box in truth]
        best_iou = max((iou(np.asarray(box), np.asarray(pred["box"]))
                        for box in truth for pred in predictions), default=0.0)
        row = {"file": item["file"], "category": item["class"], "split": args.split,
               "tile_fraction": args.tile_fraction, "truth": truth,
               "match_scores_iou50": match_scores, "best_iou_any_conf": best_iou,
               "max_confidence": max((pred["confidence"] for pred in predictions), default=0.0),
               "predictions": predictions}
        rows.append(row)
        hit = bool(truth) and all(score >= args.threshold for score in match_scores)
        false_alarm = not truth and row["max_confidence"] >= args.threshold
        if (truth and not hit) or false_alarm:
            picture = draw(image, truth, predictions, args.threshold, item["class"])
            cv2.imencode(".jpg", picture)[1].tofile(str(output / f"{Path(item['file']).stem}.jpg"))
        if index % 10 == 0:
            print(f"[裁图评估] {index}/{len(items)}", flush=True)
    summary = []
    ng = [row for row in rows if row["category"] == "NG"]
    ok = [row for row in rows if row["category"] == "OK"]
    for threshold in (0.01, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60):
        detected = sum(all(score >= threshold for score in row["match_scores_iou50"])
                       for row in ng)
        false_alarms = sum(row["max_confidence"] >= threshold for row in ok)
        unmatched_per_image = [sum(
            pred["confidence"] >= threshold and
            all(iou(np.asarray(box), np.asarray(pred["box"])) < 0.5 for box in row["truth"])
            for pred in row["predictions"]) for row in ng]
        summary.append({"threshold": threshold, "ng_detected": detected,
                        "ng_total": len(ng), "ng_recall": detected / len(ng),
                        "ok_false_alarms": false_alarms, "ok_total": len(ok),
                        "ok_false_alarm_rate": false_alarms / len(ok),
                        "ng_unmatched_boxes": sum(unmatched_per_image),
                        "ng_images_with_unmatched": sum(count > 0 for count in unmatched_per_image)})
    (output / "predictions.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                                                 encoding="utf-8")
    (output / "thresholds.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                              encoding="utf-8")
    print(json.dumps(next(row for row in summary if row["threshold"] == 0.25),
                     ensure_ascii=False), flush=True)
    print(f"逐图结果：{output}", flush=True)


if __name__ == "__main__":
    main()
