"""创建时间：2026-09-28；功能：合并训练原图与同拆分的裁图；目的：分别检验局部放大和同批次正常耳带硬负例。"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="生成原图与局部裁图混合训练集")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--tiles-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include-background-tiles", action="store_true",
                        help="加入同批次无打结的正常耳带裁图")
    parser.add_argument("--background-from-ng-only", action="store_true",
                        help="背景裁图只取自原 NG 图，保证负例与正例同拍摄批次")
    parser.add_argument("--skip-positive-tiles", action="store_true",
                        help="只加入正常区域裁图，检验误报压制与正例裁图的作用")
    args = parser.parse_args()
    source = args.run_dir.resolve()
    tiles = args.tiles_dir.resolve()
    output = args.output_dir.resolve()
    if output in (source, tiles) or source in output.parents or tiles in output.parents:
        raise ValueError("输出不能覆盖输入数据")
    if args.skip_positive_tiles and not args.include_background_tiles:
        raise ValueError("只保留正常区域裁图时需要启用 --include-background-tiles")
    original = json.loads((source / "dataset_manifest.json").read_text(encoding="utf-8"))
    tiled = json.loads((tiles / "tile_manifest.json").read_text(encoding="utf-8"))
    original_split = {row["file"]: row["split"] for row in original["items"]}
    original_class = {row["file"]: row["class"] for row in original["items"]}
    if any(original_split[row["original"]] != row["split"] for row in tiled["items"]):
        raise ValueError("裁图与原图拆分不一致")
    data = output / "dataset"
    counts: Counter[str] = Counter()
    for row in original["items"]:
        split = row["split"]
        image_target = data / "images" / split / row["file"]
        label_target = data / "labels" / split / f"{Path(row['file']).stem}.txt"
        image_target.parent.mkdir(parents=True, exist_ok=True)
        label_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / "dataset" / "images" / split / row["file"], image_target)
        shutil.copy2(source / "dataset" / "labels" / split /
                     f"{Path(row['file']).stem}.txt", label_target)
        counts[f"{split}_original_{row['class']}"] += 1
    for row in tiled["items"]:
        if row["split"] != "train":
            continue
        if row["boxes"] and args.skip_positive_tiles:
            continue
        if not row["boxes"] and not args.include_background_tiles:
            continue
        if (not row["boxes"] and args.background_from_ng_only and
                original_class[row["original"]] != "NG"):
            continue
        image_target = data / "images" / "train" / row["tile"]
        label_target = data / "labels" / "train" / f"{Path(row['tile']).stem}.txt"
        if image_target.exists() or label_target.exists():
            raise ValueError(f"文件名重复：{row['tile']}")
        shutil.copy2(tiles / "dataset" / "images" / "train" / row["tile"], image_target)
        shutil.copy2(tiles / "dataset" / "labels" / "train" /
                     f"{Path(row['tile']).stem}.txt", label_target)
        counts[f"train_tile_{'NG' if row['boxes'] else 'background'}"] += 1
    config = {"path": str(data), "train": "images/train", "val": "images/val",
              "nc": 1, "names": {0: "打结"}}
    output.mkdir(parents=True, exist_ok=True)
    (output / "data.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (output / "mixed_manifest.json").write_text(json.dumps({
        "original_manifest": str(source / "dataset_manifest.json"),
        "tile_manifest": str(tiles / "tile_manifest.json"),
        "include_background_tiles": args.include_background_tiles,
        "background_from_ng_only": args.background_from_ng_only,
        "skip_positive_tiles": args.skip_positive_tiles,
        "counts": dict(counts)}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"混合训练集：{dict(counts)}；输出：{output}", flush=True)


if __name__ == "__main__":
    main()
