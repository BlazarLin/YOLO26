"""创建时间：2026-09-28；功能：合并两份逐图推理结果；目的：评估双模型互补检出、误报及额外推理代价。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_knot import iou
from evaluate_knot_tiles import merge_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="双模型检测结果取并集")
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=.25)
    parser.add_argument("--merge-iou", type=float, default=.5)
    args = parser.parse_args()
    if not 0 < args.threshold < 1 or not 0 < args.merge_iou < 1:
        raise ValueError("置信度阈值或跨模型 NMS IoU 无效")
    first = json.loads(args.first.read_text(encoding="utf-8"))
    second = json.loads(args.second.read_text(encoding="utf-8"))
    lookup = {row["file"]: row for row in second}
    if len(first) != len(second) or len(lookup) != len(second):
        raise ValueError("两份预测图数量不同或文件名重复")
    counts = {"NG": 0, "OK": 0, "ng_strict_first": 0, "ng_strict_second": 0,
              "ng_strict_union": 0, "ok_fp_first": 0, "ok_fp_second": 0,
              "ok_fp_union": 0, "ng_strict_merged": 0, "ok_fp_merged": 0,
              "merged_boxes": 0, "images_multiple_boxes": 0,
              "second_model_calls_cascade": 0,
              "ng_strict_cascade": 0, "ok_fp_cascade": 0}
    disagreements = []
    merge_regressions = []
    for left in first:
        right = lookup.get(left["file"])
        if right is None or left["category"] != right["category"] or left["truth"] != right["truth"]:
            raise ValueError(f"两份结果不对应：{left['file']}")
        defaults = {"rotate_180": False, "flip_vertical": False,
                    "brightness_shift": 0.0, "brightness_gain": 1.0,
                    "blur_kernel": 0, "noise_std": 0.0}
        for key in ("split", "rotate_180", "flip_vertical", "brightness_shift",
                    "brightness_gain", "normalize_mean", "blur_kernel", "noise_std"):
            if left.get(key, defaults.get(key)) != right.get(key, defaults.get(key)):
                raise ValueError(f"输入条件不同：{left['file']} 的 {key}")
        category = left["category"]
        counts[category] += 1
        merged = merge_predictions([
            prediction for prediction in left["predictions"] + right["predictions"]
            if prediction["confidence"] >= args.threshold], overlap=args.merge_iou)
        counts["merged_boxes"] += len(merged)
        counts["images_multiple_boxes"] += len(merged) > 1
        trigger = left["max_confidence"] >= args.threshold
        counts["second_model_calls_cascade"] += trigger
        if category == "NG":
            hit_left = all(value >= args.threshold for value in left["match_scores_iou50"])
            hit_right = all(value >= args.threshold for value in right["match_scores_iou50"])
            union = all(max(a, b) >= args.threshold for a, b in zip(
                left["match_scores_iou50"], right["match_scores_iou50"]))
            counts["ng_strict_first"] += hit_left
            counts["ng_strict_second"] += hit_right
            counts["ng_strict_union"] += union
            merged_hit = all(
                any(iou(np.asarray(truth), np.asarray(prediction["box"])) >= .5
                    for prediction in merged) for truth in left["truth"])
            counts["ng_strict_merged"] += merged_hit
            if union and not merged_hit:
                merge_regressions.append(left["file"])
            counts["ng_strict_cascade"] += trigger and union
        else:
            hit_left = left["max_confidence"] >= args.threshold
            hit_right = right["max_confidence"] >= args.threshold
            union = hit_left or hit_right
            counts["ok_fp_first"] += hit_left
            counts["ok_fp_second"] += hit_right
            counts["ok_fp_union"] += union
            counts["ok_fp_merged"] += bool(merged)
            counts["ok_fp_cascade"] += trigger
        if hit_left != hit_right or (category == "NG" and not union):
            disagreements.append({"file": left["file"], "class": category,
                                  "first": hit_left, "second": hit_right, "union": union,
                                  "first_match_score": left["match_scores_iou50"],
                                  "second_match_score": right["match_scores_iou50"]})
    result = {"first": str(args.first.resolve()), "second": str(args.second.resolve()),
              "threshold": args.threshold, "merge_iou": args.merge_iou,
              "counts": counts, "disagreements": disagreements,
              "merge_regressions": merge_regressions,
              "caveat": "并集需运行两次模型；条件级联仅在第一模型有分数达阈值的任意框时运行第二模型，"
                        "只负责已拦截 NG 的框细化。本验证集参与方案选择，调用比例不代表真实产线。"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"counts": counts, "disagreements": len(disagreements)},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
