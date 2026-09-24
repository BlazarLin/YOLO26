"""创建时间：2026-09-24；功能：将按拍摄时间分组的 LabelMe 数据转换为 YOLO；目的：避免相邻帧同时进入训练与验证集。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]


def capture_time(path: Path) -> float:
    match = re.search(r"_(\d{2})h(\d{2})m(\d{1,2})s(\d+)ms_", path.stem)
    if not match:
        raise ValueError(f"文件名缺少拍摄时间：{path.name}")
    hour, minute, second, millisecond = map(int, match.groups())
    return hour * 3600 + minute * 60 + second + millisecond / 1000


def grouped_files(paths: list[Path], gap_seconds: float) -> list[list[Path]]:
    groups: list[list[Path]] = []
    for path in sorted(paths, key=capture_time):
        if not groups or capture_time(path) - capture_time(groups[-1][-1]) > gap_seconds:
            groups.append([])
        groups[-1].append(path)
    return groups


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LabelMe 打结数据转换为 YOLO 检测集")
    parser.add_argument("--source", type=Path, required=True, help="包含同名 BMP 和 JSON 的目录")
    parser.add_argument("--output-dir", type=Path, help="生成目录，默认 x64/output/时间戳")
    parser.add_argument("--class-name", default="打结")
    parser.add_argument("--val-groups", default="0,15,22", help="按时间排序后的验证组编号")
    parser.add_argument("--gap-seconds", type=float, default=3.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    if not source.is_dir() or args.gap_seconds <= 0:
        raise ValueError("源目录不存在或分组间隔无效")
    output_dir = (args.output_dir or ROOT / "x64" / "output" /
                  datetime.now().strftime("%Y%m%d_%H%M%S")).resolve()
    if output_dir == source or source in output_dir.parents:
        raise ValueError("输出目录不能位于原始数据目录内")
    dataset = output_dir / "dataset"
    annotations = sorted(source.glob("*.json"))
    images = {path.stem: path for path in source.glob("*.bmp")}
    if not annotations or {path.stem for path in annotations} != set(images):
        raise RuntimeError("BMP 与 JSON 未一一对应")
    groups = grouped_files(annotations, args.gap_seconds)
    val_groups = {int(item.strip()) for item in args.val_groups.split(",") if item.strip()}
    if not val_groups or min(val_groups) < 0 or max(val_groups) >= len(groups):
        raise ValueError(f"验证组编号越界；共 {len(groups)} 组")
    for split in ("train", "val"):
        (dataset / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset / "labels" / split).mkdir(parents=True, exist_ok=True)
    manifest = []
    counts = Counter()
    hashes: dict[str, str] = {}
    for group_index, group in enumerate(groups):
        split = "val" if group_index in val_groups else "train"
        for annotation in group:
            document = json.loads(annotation.read_text(encoding="utf-8"))
            image_path = images[annotation.stem]
            if Path(document["imagePath"]).name != image_path.name:
                raise ValueError(f"图片路径不匹配：{annotation.name}")
            image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError(f"图片无法解码：{image_path.name}")
            height, width = image.shape[:2]
            if (width, height) != (document["imageWidth"], document["imageHeight"]):
                raise ValueError(f"图片尺寸与标注不一致：{annotation.name}")
            labels = []
            for shape in document["shapes"]:
                if shape["label"] != args.class_name or shape["shape_type"] not in {"rectangle", "polygon"}:
                    raise ValueError(f"不支持的标注：{annotation.name}")
                points = np.asarray(shape["points"], dtype=np.float64)
                if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
                    raise ValueError(f"标注坐标无效：{annotation.name}")
                x1, y1 = points.min(axis=0)
                x2, y2 = points.max(axis=0)
                if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                    raise ValueError(f"标注越界：{annotation.name}")
                x, y = (x1 + x2) / (2 * width), (y1 + y2) / (2 * height)
                bw, bh = (x2 - x1) / width, (y2 - y1) / height
                labels.append(f"0 {x:.9f} {y:.9f} {bw:.9f} {bh:.9f}")
            if not labels:
                raise ValueError(f"图片无目标：{annotation.name}")
            shutil.copy2(image_path, dataset / "images" / split / image_path.name)
            (dataset / "labels" / split / f"{annotation.stem}.txt").write_text(
                "\n".join(labels) + "\n", encoding="ascii")
            sha = hashlib.sha256(image_path.read_bytes()).hexdigest()
            if sha in hashes:
                raise ValueError(f"重复图片：{image_path.name} 与 {hashes[sha]}")
            hashes[sha] = image_path.name
            manifest.append({"file": image_path.name, "split": split, "group": group_index,
                             "width": width, "height": height, "objects": len(labels), "sha256": sha})
            counts[split] += 1
    config = {"path": str(dataset), "train": "images/train", "val": "images/val",
              "nc": 1, "names": {0: args.class_name}}
    (output_dir / "data.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps({"source": str(source), "split_counts": dict(counts),
                    "validation_groups": sorted(val_groups), "items": manifest},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"转换完成：训练 {counts['train']} 张，验证 {counts['val']} 张；YAML：{output_dir / 'data.yaml'}")


if __name__ == "__main__":
    main()
