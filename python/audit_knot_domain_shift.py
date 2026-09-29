"""创建时间：2026-09-28；功能：量化批次亮度和印字位置差异；目的：揭示验证集与类别共变的图像捷径。"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="打结数据采集批次差异审计")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = args.run_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((source / "dataset_manifest.json").read_text(encoding="utf-8"))
    rows = []
    for item in manifest["items"]:
        path = source / "dataset" / "images" / item["split"] / item["file"]
        image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"无法读取：{path}")
        height, width = image.shape
        # 两块固定 ROI 分别覆盖印字可能出现的上下位置；仅作批次诊断，不参与推理。
        upper = image[int(.12 * height):int(.34 * height),
                      int(.68 * width):int(.91 * width)]
        lower = image[int(.66 * height):int(.86 * height),
                      int(.68 * width):int(.91 * width)]
        rows.append({"file": item["file"], "class": item["class"],
                     "split": item["split"], "date": item["file"].split("_")[1],
                     "mean_gray": float(image.mean()),
                     "upper_minus_lower_bright_fraction":
                     float((upper > 180).mean() - (lower > 180).mean())})
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[f"{row['class']} {row['date']}"].append(row)
    style = {"NG 260313": ("#b43129", "o"),
             "OK 260310": ("#2b73b7", "^"),
             "OK 260928": ("#28a58a", "s")}
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axis = plt.subplots(figsize=(11, 6.5), dpi=160)
    for name, members in groups.items():
        color, marker = style.get(name, ("#777777", "x"))
        axis.scatter([row["mean_gray"] for row in members],
                     [row["upper_minus_lower_bright_fraction"] for row in members],
                     c=color, marker=marker, alpha=.75, s=40,
                     label=f"{name} (n={len(members)})")
    axis.set_xlabel("整图平均灰度")
    axis.set_ylabel("上方减下方的高亮像素比例")
    axis.set_title("采集批次特征与 OK / NG 类别同时变化")
    axis.grid(alpha=.2)
    axis.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output / "domain_shift_scatter.png")
    plt.close(fig)
    summary = {name: {"count": len(members),
                      "mean_gray_range": [min(row["mean_gray"] for row in members),
                                          max(row["mean_gray"] for row in members)],
                      "position_score_range": [
                          min(row["upper_minus_lower_bright_fraction"] for row in members),
                          max(row["upper_minus_lower_bright_fraction"] for row in members)]}
               for name, members in groups.items()}
    (output / "domain_shift.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
