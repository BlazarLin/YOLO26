"""创建时间：2026-09-28；功能：绘制骨干与定位参数实验对照；目的：同时展示 NG 检出和 OK 误报，避免单看 mAP。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_metrics(folder: Path) -> dict:
    rows = json.loads((folder / "thresholds.json").read_text(encoding="utf-8"))
    return next(row for row in rows if row["threshold"] == 0.25)


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制骨干网络实验的业务指标")
    parser.add_argument("--previous-root", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    old = args.previous_root / "evaluations"
    new = args.experiment_root / "evaluations"
    configurations = [
        ("V6 baseline", old / "v6_expanded_val", old / "v6_plus11_expanded"),
        ("n full", new / "n_full_raw", new / "n_full_plus11"),
        ("n freeze11", new / "n_freeze11_raw", new / "n_freeze11_plus11"),
        ("n L1 x2", new / "n_dfl3_raw", new / "n_dfl3_plus11"),
        ("n COCO", new / "n_pretrained_raw", new / "n_pretrained_plus11"),
        ("s COCO", new / "s_pretrained_raw", new / "s_pretrained_plus11"),
    ]
    names = []
    raw = []
    bright = []
    false_positives = []
    for label, raw_path, bright_path in configurations:
        if not (raw_path / "thresholds.json").is_file() or not (bright_path / "thresholds.json").is_file():
            continue
        first, second = read_metrics(raw_path), read_metrics(bright_path)
        names.append(label)
        raw.append(first["ng_detected"])
        bright.append(second["ng_detected"])
        false_positives.append(first["ok_false_alarms"])
    positions = np.arange(len(names))
    figure, axis = plt.subplots(figsize=(12, max(4.7, 0.68 * len(names) + 1.5)))
    width = 0.34
    left = axis.barh(positions - width / 2, raw, width, color="#167ba0", label="Original NG / 29")
    right = axis.barh(positions + width / 2, bright, width, color="#df8c3e", label="Brightness +11 NG / 29")
    for bars in (left, right):
        for bar in bars:
            axis.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height() / 2,
                      f"{int(bar.get_width())}/29", va="center", fontsize=9)
    axis.axvline(27, color="#aa3344", linestyle="--", linewidth=1,
                 label="At least 27/29 (observed 90% target)")
    axis.set_yticks(positions, names)
    axis.invert_yaxis()
    axis.set_xlim(0, 32)
    axis.set_xlabel("Strict detections at confidence 0.25 and IoU >= 0.5")
    axis.set_title("YOLO26 backbone and localization experiments on the same validation split\n"
                   "Blue: original NG   Orange: brightness +11 NG   Red line: at least 27/29\n"
                   f"Original-image OK false positives: {max(false_positives)}/11 for all models",
                   pad=14)
    axis.grid(axis="x", alpha=0.16)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    plt.close(figure)
    print(f"对照图：{args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
