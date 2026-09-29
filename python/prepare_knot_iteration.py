"""创建时间：2026-09-28；功能：按 NG/OK 比例拆分 LabelMe 打结数据；目的：固定可复现的训练与验证样本。"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按类别比例随机拆分口罩打结数据")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--val-ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--baseline-manifest", type=Path,
                        help="旧数据清单；保持旧图训练/验证归属，只按比例拆分新增图片")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = (args.output_dir or ROOT / "x64" / "output" /
              datetime.now().strftime("%Y%m%d_%H%M%S")).resolve()
    if not source.is_dir() or not 0 < args.val_ratio < 1:
        raise ValueError("源目录或验证比例无效")
    if output == source or source in output.parents:
        raise ValueError("输出目录不能位于原始数据内")
    annotations = sorted(source.glob("*.json"))
    image_files = [file for file in source.iterdir()
                   if file.is_file() and file.suffix.lower() in {".bmp", ".png"}]
    images = {file.stem: file for file in image_files}
    if (not annotations or len(images) != len(image_files) or
            {file.stem for file in annotations} != set(images)):
        raise ValueError("BMP/PNG 与 JSON 未一一对应，或图片文件名重复")

    by_class: dict[str, list[dict]] = {"NG": [], "OK": []}
    hashes: dict[str, str] = {}
    for annotation in annotations:
        document = json.loads(annotation.read_text(encoding="utf-8"))
        image_path = images[annotation.stem]
        if Path(document["imagePath"]).name != image_path.name:
            raise ValueError(f"图片路径不匹配：{annotation.name}")
        image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"无法解码：{image_path.name}")
        height, width = image.shape[:2]
        if (width, height) != (document["imageWidth"], document["imageHeight"]):
            raise ValueError(f"图片尺寸与标注不符：{annotation.name}")
        labels = []
        boxes = []
        for shape in document["shapes"]:
            if shape["label"] != "打结" or shape["shape_type"] not in {"rectangle", "polygon"}:
                raise ValueError(f"不支持的标注：{annotation.name}")
            points = np.asarray(shape["points"], dtype=np.float64)
            if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
                raise ValueError(f"标注坐标无效：{annotation.name}")
            x1, y1 = points.min(axis=0)
            x2, y2 = points.max(axis=0)
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise ValueError(f"标注越界：{annotation.name}")
            labels.append(f"0 {(x1+x2)/(2*width):.9f} {(y1+y2)/(2*height):.9f} "
                          f"{(x2-x1)/width:.9f} {(y2-y1)/height:.9f}")
            boxes.append([float(x1), float(y1), float(x2), float(y2)])
        sha = hashlib.sha256(image_path.read_bytes()).hexdigest()
        if sha in hashes:
            raise ValueError(f"重复图片：{image_path.name} 与 {hashes[sha]}")
        hashes[sha] = image_path.name
        category = "NG" if labels else "OK"
        by_class[category].append({"image": image_path, "labels": labels, "boxes": boxes,
                                   "width": width, "height": height, "sha256": sha})

    dataset = output / "dataset"
    for split in ("train", "val"):
        (dataset / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset / "labels" / split).mkdir(parents=True, exist_ok=True)
    randomizer = random.Random(args.seed)
    baseline_split: dict[str, str] = {}
    if args.baseline_manifest:
        old = json.loads(args.baseline_manifest.read_text(encoding="utf-8"))
        baseline_split = {row["file"]: row["split"] for row in old["items"]}
        current = {row["image"].name for group in by_class.values() for row in group}
        if not baseline_split or not set(baseline_split) <= current:
            raise ValueError("旧数据清单与当前样本不一致")
        if any(split not in {"train", "val"} for split in baseline_split.values()):
            raise ValueError("旧数据清单的训练/验证标记无效")
    manifest = []
    counts = Counter()
    for category, items in by_class.items():
        randomizer.shuffle(items)
        n_val = round(len(items) * args.val_ratio)
        fixed_val = sum(baseline_split.get(row["image"].name) == "val" for row in items)
        fixed_train = sum(baseline_split.get(row["image"].name) == "train" for row in items)
        n_val = max(n_val, fixed_val)
        if len(items) - fixed_train < n_val:
            raise ValueError(f"{category} 的旧训练集与新验证比例冲突")
        if n_val == 0 or n_val == len(items):
            raise ValueError(f"{category} 样本太少，无法拆分")
        new_val_needed = n_val - fixed_val
        for item in items:
            old_split = baseline_split.get(item["image"].name)
            if old_split:
                split = old_split
            elif new_val_needed > 0:
                split = "val"
                new_val_needed -= 1
            else:
                split = "train"
            image = item["image"]
            shutil.copy2(image, dataset / "images" / split / image.name)
            # 空标签文件明确表明该图片为 OK 背景图。
            (dataset / "labels" / split / f"{image.stem}.txt").write_text(
                "\n".join(item["labels"]) + ("\n" if item["labels"] else ""), encoding="ascii")
            manifest.append({"file": image.name, "class": category, "split": split,
                             "width": item["width"], "height": item["height"],
                             "boxes": item["boxes"], "sha256": item["sha256"]})
            counts[f"{split}_{category}"] += 1
    config = {"path": str(dataset), "train": "images/train", "val": "images/val",
              "nc": 1, "names": {0: "打结"}}
    output.mkdir(parents=True, exist_ok=True)
    (output / "data.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (output / "dataset_manifest.json").write_text(
        json.dumps({"source": str(source), "seed": args.seed, "val_ratio": args.val_ratio,
                    "baseline_manifest": str(args.baseline_manifest) if args.baseline_manifest else None,
                    "counts": dict(counts), "items": sorted(manifest, key=lambda row: row["file"])},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"拆分完成：{dict(counts)}；清单：{output / 'dataset_manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
