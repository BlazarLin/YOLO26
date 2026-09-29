"""创建时间：2026-09-28；功能：仅对预测框贴边图合并补边二次推理结果；目的：检验边界上下文修复能否避免全量补边退化。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="贴边目标条件二次推理复核")
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--padded", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=.25)
    parser.add_argument("--edge-pixels", type=float, default=20)
    args = parser.parse_args()
    manifest = json.loads((args.run_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    dimensions = {row["file"]: (row["width"], row["height"]) for row in manifest["items"]}
    raw = json.loads(args.raw.read_text(encoding="utf-8"))
    padded = {row["file"]: row for row in json.loads(args.padded.read_text(encoding="utf-8"))}
    if not 0 < args.confidence < 1 or args.edge_pixels < 0:
        raise ValueError("置信度或边界距离无效")
    if len(raw) != len(padded):
        raise ValueError("两份预测数量不一致")
    summary = {"ng_total": 0, "ng_raw_strict": 0, "ng_retry_strict": 0,
               "ok_total": 0, "ok_raw_fp": 0, "ok_retry_fp": 0,
               "retry_images": 0, "retry_ng": 0, "retry_ok": 0}
    rows = []
    for left in raw:
        right = padded.get(left["file"])
        if right is None or left["truth"] != right["truth"] or left["category"] != right["category"]:
            raise ValueError(f"预测不对应：{left['file']}")
        defaults = {"rotate_180": False, "flip_vertical": False,
                    "brightness_shift": 0.0, "brightness_gain": 1.0,
                    "blur_kernel": 0, "noise_std": 0.0}
        for key in ("split", "rotate_180", "flip_vertical", "brightness_shift",
                    "brightness_gain", "normalize_mean", "blur_kernel", "noise_std"):
            if left.get(key, defaults.get(key)) != right.get(key, defaults.get(key)):
                raise ValueError(f"两份输入条件不同：{left['file']} 的 {key}")
        if not right.get("border_pad_ratio", 0) > 0:
            raise ValueError("第二份预测不是补边推理")
        width, height = dimensions[left["file"]]
        edge_boxes = []
        for pred in left["predictions"]:
            if pred["confidence"] < args.confidence:
                continue
            x1, y1, x2, y2 = pred["box"]
            distance = min(x1, y1, width - x2, height - y2)
            if distance < args.edge_pixels:
                edge_boxes.append({"score": pred["confidence"], "distance": distance})
        retry = bool(edge_boxes)
        summary["retry_images"] += retry
        category = left["category"]
        summary[f"retry_{category.lower()}"] += retry
        if category == "NG":
            summary["ng_total"] += 1
            hit_raw = all(value >= args.confidence for value in left["match_scores_iou50"])
            hit_retry = all(max(a, b if retry else 0) >= args.confidence for a, b in zip(
                left["match_scores_iou50"], right["match_scores_iou50"]))
            summary["ng_raw_strict"] += hit_raw
            summary["ng_retry_strict"] += hit_retry
        else:
            summary["ok_total"] += 1
            hit_raw = left["max_confidence"] >= args.confidence
            hit_retry = hit_raw or (retry and right["max_confidence"] >= args.confidence)
            summary["ok_raw_fp"] += hit_raw
            summary["ok_retry_fp"] += hit_retry
        rows.append({"file": left["file"], "class": category, "retry": retry,
                     "edge_boxes": edge_boxes, "raw_hit_or_fp": hit_raw,
                     "retry_hit_or_fp": hit_retry,
                     "raw_match_scores": left["match_scores_iou50"],
                     "padded_match_scores": right["match_scores_iou50"]})
    result = {"first": str(args.raw.resolve()), "padded": str(args.padded.resolve()),
              "confidence": args.confidence, "edge_pixels": args.edge_pixels,
              "summary": summary, "rows": rows,
              "caveat": "触发条件由主验证漏检图启发；重试不会恢复第一遍完全无框的 NG。"
                        "本验证集参与方案选择，仍需独立新批次复核。"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
