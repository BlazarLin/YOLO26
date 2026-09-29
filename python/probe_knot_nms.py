"""创建时间：2026-09-28；功能：对比 NMS 抑制前后的打结候选框；目的：区分后处理抑制和检测头定位失准。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from evaluate_knot import iou, load_image


def main() -> None:
    parser = argparse.ArgumentParser(description="关键漏检图 NMS 诊断")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--file", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    manifest = json.loads((run / "dataset_manifest.json").read_text(encoding="utf-8"))
    item = next(row for row in manifest["items"] if row["file"] == args.file)
    image = load_image(run / "dataset" / "images" / item["split"] / item["file"])
    model = YOLO(str(args.weights))
    results = {}
    for label, nms_iou in (("normal_nms", 0.5), ("nearly_disabled_nms", 0.99)):
        prediction = model.predict(image, imgsz=640, conf=0.001, iou=nms_iou,
                                   max_det=1000, verbose=False, device=0, nms=True)[0]
        candidates = []
        for box, score in zip(prediction.boxes.xyxy.cpu().numpy(),
                              prediction.boxes.conf.cpu().numpy()):
            overlap = max(iou(np.asarray(gt), box) for gt in item["boxes"])
            candidates.append({"confidence": float(score), "iou_gt": overlap,
                               "box": [float(value) for value in box]})
        results[label] = {"count": len(candidates),
                          "best_score_iou50": max((row["confidence"] for row in candidates
                                                   if row["iou_gt"] >= .5), default=0.0),
                          "best_iou_score025": max((row["iou_gt"] for row in candidates
                                                    if row["confidence"] >= .25), default=0.0),
                          "top_iou": sorted(candidates, key=lambda row: row["iou_gt"],
                                            reverse=True)[:10],
                          "top_score": sorted(candidates, key=lambda row: row["confidence"],
                                              reverse=True)[:10]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"file": item["file"], "truth": item["boxes"],
                                       "results": results}, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print({name: {key: value for key, value in data.items()
                  if key not in {"top_iou", "top_score"}}
           for name, data in results.items()}, flush=True)


if __name__ == "__main__":
    main()
