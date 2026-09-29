"""创建时间：2026-09-28；功能：同架构打结模型权重线性插值；目的：检验原图精度与亮度稳健性能否在单模型内兼顾。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from ultralytics import YOLO


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="同初始化 YOLO26 权重插值")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--alpha", type=float, required=True,
                        help="0 为原模型，1 为候选模型")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.alpha < 1:
        raise ValueError("插值比例必须在 0 与 1 之间")
    base_path = args.base.resolve()
    candidate_path = args.candidate.resolve()
    output = args.output.resolve()
    if output in (base_path, candidate_path):
        raise ValueError("不能覆盖源权重")
    base = YOLO(str(base_path))
    candidate = YOLO(str(candidate_path))
    first = base.model.state_dict()
    second = candidate.model.state_dict()
    if first.keys() != second.keys() or any(first[k].shape != second[k].shape for k in first):
        raise ValueError("模型结构不一致，不能插值")
    blended = {}
    with torch.no_grad():
        for key, tensor in first.items():
            other = second[key]
            if tensor.is_floating_point():
                blended[key] = torch.lerp(tensor.float(), other.float(),
                                          args.alpha).to(dtype=tensor.dtype)
            else:
                blended[key] = tensor.clone()
        base.model.load_state_dict(blended, strict=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    base.save(output)
    result = {"base": str(base_path), "base_sha256": sha256(base_path),
              "candidate": str(candidate_path), "candidate_sha256": sha256(candidate_path),
              "alpha": args.alpha, "output": str(output), "output_sha256": sha256(output)}
    output.with_suffix(".json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
