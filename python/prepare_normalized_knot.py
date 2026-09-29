"""创建时间：2026-09-28；功能：生成按整图平均灰度归一的口罩训练副本；目的：检验跨批次亮度混杂的影响。"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成平均灰度归一的 YOLO 数据副本")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mean", type=float, default=72.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = args.run_dir.resolve()
    output = args.output_dir.resolve()
    source_dataset = run / "dataset"
    if (not 0 <= args.mean <= 255 or output == run or output == source_dataset or
            source_dataset in output.parents):
        raise ValueError("输出目录或目标平均灰度无效")
    manifest = json.loads((run / "dataset_manifest.json").read_text(encoding="utf-8"))
    dataset = output / "dataset"
    converted = []
    for item in manifest["items"]:
        split, file_name = item["split"], item["file"]
        source = run / "dataset" / "images" / split / file_name
        image = cv2.imdecode(np.fromfile(str(source), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"无法读取 {source}")
        gray_mean = float(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).mean())
        shift = args.mean - gray_mean
        image = np.clip(image.astype(np.float32) + shift, 0, 255).astype(np.uint8)
        image_target = dataset / "images" / split / file_name
        image_target.parent.mkdir(parents=True, exist_ok=True)
        cv2.imencode(".bmp", image)[1].tofile(str(image_target))
        label_source = run / "dataset" / "labels" / split / f"{Path(file_name).stem}.txt"
        label_target = dataset / "labels" / split / label_source.name
        label_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(label_source, label_target)
        converted.append({**item, "source_sha256": item["sha256"],
                          "sha256": hashlib.sha256(image_target.read_bytes()).hexdigest(),
                          "original_mean_gray": gray_mean, "mean_shift": shift})
    config = {"path": str(dataset), "train": "images/train", "val": "images/val",
              "nc": 1, "names": {0: "打结"}}
    output.mkdir(parents=True, exist_ok=True)
    (output / "data.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (output / "dataset_manifest.json").write_text(
        json.dumps({**manifest, "normalization_target_mean": args.mean, "items": converted},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"归一化完成：{len(converted)} 张；输出 {output}", flush=True)


if __name__ == "__main__":
    main()
