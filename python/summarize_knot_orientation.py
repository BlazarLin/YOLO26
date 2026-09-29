"""创建时间：2026-09-28；功能：绘制三份比例拆分的上下翻转增强结果；目的：判断亮度与朝向训练收益能否复现。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总三份双朝向训练复核")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root.resolve()
    methods = ("base", "brightness", "flipud")
    conditions = ("raw", "plus11", "flipud")
    seeds = (42, 123, 777)
    data = {}
    for seed in seeds:
        data[str(seed)] = {}
        for method in methods:
            data[str(seed)][method] = {}
            for condition in conditions:
                path = root / "evaluations" / f"seed{seed}_{method}_{condition}" / "thresholds.json"
                items = json.loads(path.read_text(encoding="utf-8"))
                row = next(item for item in items if item["threshold"] == .25)
                data[str(seed)][method][condition] = {
                    "ng_detected": row["ng_detected"], "ng_total": row["ng_total"],
                    "ok_false_alarms": row["ok_false_alarms"], "ok_total": row["ok_total"]}
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), dpi=160, sharey=True)
    labels = {"base": "原模型", "brightness": "亮度微调", "flipud": "亮度+上下翻转"}
    colors = {"base": "#2974ad", "brightness": "#24a17b", "flipud": "#b67937"}
    for axis, condition, title in zip(axes, conditions,
                                      ("原图", "统一加亮 11", "上下翻转")):
        for offset, method in enumerate(methods):
            heights = [data[str(seed)][method][condition]["ng_detected"] for seed in seeds]
            x = np.arange(3) + (offset - 1) * .25
            axis.bar(x, heights, width=.22, color=colors[method], label=labels[method])
            for x_value, value in zip(x, heights):
                axis.text(x_value, value + .1, str(value), ha="center", fontsize=9)
        axis.set_xticks(np.arange(3), [str(seed) for seed in seeds])
        axis.set_xlabel("随机比例拆分种子")
        axis.set_title(title)
        axis.set_ylim(23, 35.5)
        axis.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("NG 严格检出 / 34 张")
    axes[-1].legend(loc="lower right")
    fig.suptitle("三份比例拆分：双朝向训练需同时看原图、亮度和方向；种子 123 原图新增 1/13 OK 误报")
    fig.tight_layout()
    target = root / "diagnostics" / "orientation_replicate_comparison.png"
    fig.savefig(target)
    plt.close(fig)
    (root / "diagnostics" / "orientation_replicate_comparison.json").write_text(
        json.dumps({"note": "各份验证图互有重叠，翻转输入是软件变换，全部 OK 误报数在当前阈值下为零",
                    "data": data}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"双朝向复核图：{target}", flush=True)


if __name__ == "__main__":
    main()
