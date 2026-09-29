"""创建时间：2026-09-28；功能：从已固定的训练/验证清单生成重叠横向裁图；目的：检验小目标有效像素不足的假设。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import yaml

from evaluate_knot import load_image


def main() -> None:
    parser = argparse.ArgumentParser(description="生成口罩打结横向重叠裁图")
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="包含 dataset_manifest.json 的固定原图拆分目录")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tile-fraction", type=float, default=0.6)
    args = parser.parse_args()
    source = args.run_dir.resolve()
    output = args.output_dir.resolve()
    if not 0.5 < args.tile_fraction < 1.0:
        raise ValueError("裁图宽度比例必须在 0.5 与 1 之间")
    if output == source or source in output.parents:
        raise ValueError("裁图目录应位于原始拆分目录之外")
    manifest = json.loads((source / "dataset_manifest.json").read_text(encoding="utf-8"))
    items = manifest["items"]
    if len({row["file"] for row in items}) != len(items):
        raise ValueError("原图清单出现重复文件名")
    image_root = output / "dataset" / "images"
    label_root = output / "dataset" / "labels"
    rows: list[dict] = []
    counts: Counter[str] = Counter()
    for row in items:
        split = row["split"]
        image = load_image(source / "dataset" / "images" / split / row["file"])
        height, width = image.shape[:2]
        if (width, height) != (row["width"], row["height"]):
            raise ValueError(f"图片尺寸发生变化：{row['file']}")
        tile_width = round(width * args.tile_fraction)
        starts = (0, width - tile_width)
        if any(box[2] - box[0] > 2 * tile_width - width for box in row["boxes"]):
            raise ValueError(f"框宽度超过裁图重叠宽度：{row['file']}")
        saved_positive = False
        for side, start in zip(("L", "R"), starts):
            intersects = [box for box in row["boxes"] if box[0] < start + tile_width and box[2] > start]
            # 避免把被切断的打结当成空背景或残缺正样本来训练。
            if any(box[0] < start or box[2] > start + tile_width for box in intersects):
                counts[f"{split}_skipped_partial"] += 1
                continue
            tile_name = f"{Path(row['file']).stem}_{side}.bmp"
            image_dir = image_root / split
            label_dir = label_root / split
            image_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            tile = image[:, start:start + tile_width]
            encoded = cv2.imencode(".bmp", tile)[1]
            encoded.tofile(str(image_dir / tile_name))
            labels = []
            for box in intersects:
                x1, y1, x2, y2 = box
                labels.append(f"0 {(x1+x2-2*start)/(2*tile_width):.9f} "
                              f"{(y1+y2)/(2*height):.9f} "
                              f"{(x2-x1)/tile_width:.9f} {(y2-y1)/height:.9f}")
            (label_dir / f"{Path(tile_name).stem}.txt").write_text(
                "\n".join(labels) + ("\n" if labels else ""), encoding="ascii")
            rows.append({"original": row["file"], "tile": tile_name, "split": split,
                         "side": side, "x_offset": start, "boxes": intersects})
            counts[f"{split}_{'NG' if intersects else 'OK'}_tiles"] += 1
            saved_positive |= bool(intersects)
        if row["boxes"] and not saved_positive:
            raise ValueError(f"没有保留完整正样本：{row['file']}")
    if {r["original"] for r in rows if r["split"] == "train"} & {
            r["original"] for r in rows if r["split"] == "val"}:
        raise ValueError("同一原图跨训练/验证")
    dataset = output / "dataset"
    config = {"path": str(dataset), "train": "images/train", "val": "images/val",
              "nc": 1, "names": {0: "打结"}}
    output.mkdir(parents=True, exist_ok=True)
    (output / "data.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (output / "tile_manifest.json").write_text(
        json.dumps({"source_manifest": str(source / "dataset_manifest.json"),
                    "tile_fraction": args.tile_fraction, "counts": dict(counts), "items": rows},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"裁图生成完成：{dict(counts)}；输出：{output}", flush=True)


if __name__ == "__main__":
    main()
