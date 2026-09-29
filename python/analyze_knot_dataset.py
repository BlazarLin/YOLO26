"""创建时间：2026-09-28；功能：绘制口罩打结数据的亮度和标注分布；目的：识别批次混杂与难例位置。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="口罩打结数据分布分析")
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = args.run_dir.resolve()
    items = json.loads((run / "dataset_manifest.json").read_text(encoding="utf-8"))["items"]
    records = []
    for item in items:
        path = run / "dataset" / "images" / item["split"] / item["file"]
        image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"无法读取 {path}")
        record = {"file": item["file"], "class": item["class"], "split": item["split"],
                  "mean_gray": float(image.mean()), "std_gray": float(image.std())}
        if item["boxes"]:
            box = item["boxes"][0]
            record.update({"center_x": (box[0] + box[2]) / (2 * item["width"]),
                           "center_y": (box[1] + box[3]) / (2 * item["height"]),
                           "box_width": (box[2] - box[0]) / item["width"],
                           "box_height": (box[3] - box[1]) / item["height"]})
        records.append(record)

    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    ng = [row for row in records if row["class"] == "NG"]
    ok = [row for row in records if row["class"] == "OK"]
    axes[0, 0].hist([row["mean_gray"] for row in ng], bins=25, color="#d04e2b", alpha=.75,
                    label=f"NG {len(ng)} 张")
    axes[0, 0].hist([row["mean_gray"] for row in ok], bins=10, color="#2385b5", alpha=.75,
                    label=f"OK {len(ok)} 张")
    axes[0, 0].set_title("整图平均灰度：NG 与 OK 完全分离")
    axes[0, 0].set_xlabel("平均灰度（0～255）")
    axes[0, 0].set_ylabel("图片数")
    axes[0, 0].legend()
    axes[0, 1].boxplot([[row["std_gray"] for row in ng], [row["std_gray"] for row in ok]],
                       tick_labels=["NG", "OK"])
    axes[0, 1].set_title("整图灰度标准差：对比度也有批次差异")
    axes[0, 1].set_ylabel("灰度标准差")
    for split, color in [("train", "#2779ad"), ("val", "#e56b2e")]:
        part = [row for row in ng if row["split"] == split]
        axes[1, 0].scatter([row["center_x"] for row in part],
                           [row["center_y"] for row in part], s=24, alpha=.7,
                           color=color, label=f"{split} {len(part)} 张")
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_xlim(0, 1)
    axes[1, 0].set_ylim(1, 0)
    axes[1, 0].set_title("NG 标注中心位置：训练/验证均覆盖上下耳带")
    axes[1, 0].set_xlabel("归一化 X")
    axes[1, 0].set_ylabel("归一化 Y")
    axes[1, 0].legend()
    axes[1, 1].scatter([row["box_width"] for row in ng],
                       [row["box_height"] for row in ng], s=25, alpha=.7,
                       color="#6f5a9b")
    axes[1, 1].set_title("NG 标注框宽高：形态跨度较大")
    axes[1, 1].set_xlabel("框宽 / 图宽")
    axes[1, 1].set_ylabel("框高 / 图高")
    target = run / "dataset_profile.png"
    fig.savefig(target, dpi=160)
    plt.close(fig)
    (run / "dataset_profile.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"数据分布图：{target}")


if __name__ == "__main__":
    main()
