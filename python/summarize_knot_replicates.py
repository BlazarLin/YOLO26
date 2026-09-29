"""创建时间：2026-09-28；功能：汇总三份比例拆分的原图和加亮对照；目的：直观看出稳健性收益与插值反例。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="生成打结重复拆分结果图")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root.resolve()
    methods = ("base", "brightness", "alpha05")
    seeds = (42, 123, 777)
    result = {}
    for seed in seeds:
        result[str(seed)] = {}
        for method in methods:
            result[str(seed)][method] = {}
            for condition in ("raw", "plus11"):
                path = (root / "evaluations" /
                        f"seed{seed}_{method}_{condition}" / "thresholds.json")
                summary = json.loads(path.read_text(encoding="utf-8"))
                row = next(item for item in summary if item["threshold"] == .25)
                result[str(seed)][method][condition] = {
                    "ng_detected": row["ng_detected"], "ng_total": row["ng_total"],
                    "ok_false_alarms": row["ok_false_alarms"], "ok_total": row["ok_total"]}
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=160, sharey=True)
    colors = ("#2974ad", "#24a17b", "#b67937")
    for axis, condition, title in zip(axes, ("raw", "plus11"),
                                      ("原图 NG 严格检出", "统一加亮 11 后 NG 严格检出")):
        positions = np.arange(len(seeds))
        for index, (method, color) in enumerate(zip(methods, colors)):
            values = [result[str(seed)][method][condition]["ng_detected"]
                      for seed in seeds]
            x = positions + (index - 1) * .24
            axis.bar(x, values, width=.21, color=color,
                     label={"base": "原模型", "brightness": "亮度微调",
                            "alpha05": "50% 权重插值"}[method])
            for xx, value in zip(x, values):
                axis.text(xx, value + .15, str(value), ha="center", va="bottom", fontsize=10)
        axis.set_xticks(positions, [str(seed) for seed in seeds])
        axis.set_xlabel("随机拆分种子")
        axis.set_title(title)
        axis.set_ylim(24, 35.5)
        axis.set_yticks(range(24, 35, 2))
        axis.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("严格检出数 / 34 张 NG")
    axes[1].legend(loc="lower left")
    fig.suptitle("三份训练/验证比例拆分：权重插值在种子 777 出现反例")
    fig.tight_layout()
    target = root / "diagnostics" / "replicate_comparison.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target)
    plt.close(fig)
    (root / "diagnostics" / "replicate_comparison.json").write_text(
        json.dumps({"note": "各拆分验证图重叠，不能合计为独立样本",
                    "results": result}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"重复拆分结果图：{target}", flush=True)


if __name__ == "__main__":
    main()
