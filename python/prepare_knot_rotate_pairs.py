"""创建时间：2026-09-28；功能：仅对训练图生成精确 180 度旋转副本；目的：覆盖产线两种已确认的产品朝向。"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import cv2
import yaml

from evaluate_knot import load_image


def rotated_label(line: str) -> str:
    parts = line.split()
    if len(parts) != 5:
        raise ValueError(f"YOLO 标注格式无效：{line}")
    category, x, y, width, height = parts
    return f"{category} {1 - float(x):.8f} {1 - float(y):.8f} {float(width):.8f} {float(height):.8f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="生成产线双朝向训练副本")
    parser.add_argument("--source-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source_yaml = args.source_data.resolve()
    config = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))
    source = Path(config["path"]).resolve()
    output = args.output_dir.resolve()
    if output == source or source in output.parents:
        raise ValueError("输出不能覆盖或嵌套在输入数据集内")
    destination = output / "dataset"
    counts: Counter[str] = Counter()
    for split in ("train", "val"):
        image_dir = source / "images" / split
        label_dir = source / "labels" / split
        output_images = destination / "images" / split
        output_labels = destination / "labels" / split
        output_images.mkdir(parents=True, exist_ok=True)
        output_labels.mkdir(parents=True, exist_ok=True)
        for image_path in sorted(image_dir.iterdir()):
            if image_path.suffix.lower() not in {".bmp", ".png", ".jpg", ".jpeg"}:
                continue
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                raise FileNotFoundError(label_path)
            shutil.copy2(image_path, output_images / image_path.name)
            shutil.copy2(label_path, output_labels / label_path.name)
            counts[f"{split}_original"] += 1
            if split == "val":
                continue
            image = load_image(image_path)
            rotated = cv2.rotate(image, cv2.ROTATE_180)
            name = f"{image_path.stem}_rot180.png"
            encoded, buffer = cv2.imencode(".png", rotated)
            if not encoded:
                raise OSError(f"无法保存旋转图：{name}")
            buffer.tofile(str(output_images / name))
            lines = label_path.read_text(encoding="utf-8").splitlines()
            (output_labels / f"{Path(name).stem}.txt").write_text(
                "\n".join(rotated_label(line) for line in lines if line.strip()) +
                ("\n" if lines else ""), encoding="utf-8")
            counts["train_rot180"] += 1
    output.mkdir(parents=True, exist_ok=True)
    new_config = {"path": str(destination), "train": "images/train", "val": "images/val",
                  "nc": config["nc"], "names": config["names"]}
    (output / "data.yaml").write_text(
        yaml.safe_dump(new_config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (output / "rotation_manifest.json").write_text(
        json.dumps({"source": str(source_yaml), "rotation": 180, "counts": dict(counts),
                    "validation_unchanged": True}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(dict(counts), flush=True)


if __name__ == "__main__":
    main()
