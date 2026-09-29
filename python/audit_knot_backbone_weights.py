"""创建时间：2026-09-28；功能：逐层比较微调前后参数变化；目的：核实骨干冻结与定位参数实验实际修改了哪些层。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="审计 YOLO26 微调前后的逐层参数变化")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = YOLO(str(args.base)).model.model
    candidate = YOLO(str(args.candidate)).model.model
    if len(base) != len(candidate):
        raise ValueError("网络层数不同，不能逐层比较参数")
    rows = []
    for nLayer, (before, after) in enumerate(zip(base, candidate)):
        reference = dict(before.named_parameters())
        updated = dict(after.named_parameters())
        if reference.keys() != updated.keys():
            raise ValueError(f"第 {nLayer} 层参数名不一致")
        baseline_energy = 0.0
        delta_energy = 0.0
        for name, old in reference.items():
            new = updated[name]
            if old.shape != new.shape:
                raise ValueError(f"第 {nLayer} 层参数维度不同：{name}")
            old64 = old.detach().cpu().to(torch.float64)
            new64 = new.detach().cpu().to(torch.float64)
            baseline_energy += float(torch.sum(old64 * old64))
            delta_energy += float(torch.sum((new64 - old64) ** 2))
        stage = "backbone" if nLayer <= 10 else "neck" if nLayer <= 22 else "head"
        rows.append({"layer": nLayer, "stage": stage,
                     "module": type(before).__name__, "parameter_tensors": len(reference),
                     "relative_l2_change": (delta_energy / max(baseline_energy, 1e-30)) ** 0.5,
                     "delta_energy": delta_energy, "base_energy": baseline_energy})
    stages = {}
    for stage in ("backbone", "neck", "head"):
        group = [row for row in rows if row["stage"] == stage]
        stages[stage] = {
            "relative_l2_change": (sum(row["delta_energy"] for row in group) /
                                   max(sum(row["base_energy"] for row in group), 1e-30)) ** 0.5,
            "unchanged_layers": sum(row["delta_energy"] == 0 for row in group),
            "layer_count": len(group),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"base": str(args.base.resolve()),
                                       "candidate": str(args.candidate.resolve()),
                                       "stages": stages, "layers": rows},
                                      ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stages, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
