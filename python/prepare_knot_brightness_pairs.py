"""创建时间：2026-09-28；功能：为训练图生成双向亮度副本；目的：让 OK/NG 的平均亮度重叠，检验采集亮度捷径。"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

from evaluate_knot import load_image


def main() -> None:
    parser = argparse.ArgumentParser(description="生成双向亮度扰动训练副本")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shift", type=int, default=12)
    args = parser.parse_args()
    source = args.run_dir.resolve()
    output = args.output_dir.resolve()
    if not 1 <= args.shift <= 30 or output == source or source in output.parents:
        raise ValueError("亮度范围或输出路径无效")
    manifest = json.loads((source / "dataset_manifest.json").read_text(encoding="utf-8"))
    data = output / "dataset"
    counts: Counter[str] = Counter()
    means: dict[str, list[float]] = {"NG": [], "OK": []}
    generated = []
    for row in manifest["items"]:
        split = row["split"]
        input_image = source / "dataset" / "images" / split / row["file"]
        input_label = source / "dataset" / "labels" / split / f"{Path(row['file']).stem}.txt"
        image_dir = data / "images" / split
        label_dir = data / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(input_image, image_dir / row["file"])
        shutil.copy2(input_label, label_dir / input_label.name)
        counts[f"{split}_original_{row['class']}"] += 1
        if split != "train":
            continue
        image = load_image(input_image)
        for shift in (-args.shift, args.shift):
            variant = np.clip(image.astype(np.int16) + shift, 0, 255).astype(np.uint8)
            name = f"{Path(row['file']).stem}_{'minus' if shift < 0 else 'plus'}{abs(shift)}.png"
            cv2.imencode(".png", variant)[1].tofile(str(image_dir / name))
            shutil.copy2(input_label, label_dir / f"{Path(name).stem}.txt")
            mean = float(cv2.cvtColor(variant, cv2.COLOR_BGR2GRAY).mean())
            means[row["class"]].append(mean)
            generated.append({"original": row["file"], "variant": name,
                              "class": row["class"], "shift": shift, "mean_gray": mean})
            counts[f"train_shifted_{row['class']}"] += 1
    config = {"path": str(data), "train": "images/train", "val": "images/val",
              "nc": 1, "names": {0: "打结"}}
    output.mkdir(parents=True, exist_ok=True)
    (output / "data.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    report = {"source_manifest": str(source / "dataset_manifest.json"),
              "shift": args.shift, "counts": dict(counts),
              "augmented_mean_ranges": {key: [min(value), max(value)]
                                        for key, value in means.items()},
              "generated": generated}
    (output / "brightness_pair_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print({key: value for key, value in report.items() if key != "generated"}, flush=True)


if __name__ == "__main__":
    main()
