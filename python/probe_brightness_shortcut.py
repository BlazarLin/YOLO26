"""创建时间：2026-09-28；功能：验证平均灰度能否区分类别；目的：量化采集日期与标签绑定的捷径风险。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="训练集定阈值的亮度捷径诊断")
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    manifest = json.loads((run / "dataset_manifest.json").read_text(encoding="utf-8"))
    rows = []
    for item in manifest["items"]:
        path = run / "dataset" / "images" / item["split"] / item["file"]
        image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"无法读取 {path}")
        rows.append({"file": item["file"], "split": item["split"],
                     "class": item["class"], "mean_gray": float(image.mean())})
    train_ok = [row["mean_gray"] for row in rows
                if row["split"] == "train" and row["class"] == "OK"]
    train_ng = [row["mean_gray"] for row in rows
                if row["split"] == "train" and row["class"] == "NG"]
    if not train_ok or not train_ng:
        raise ValueError("训练集需要同时包含 OK 和 NG")
    # 仅用训练集确定一个固定阈值，验证集不参与选择。
    boundary = (max(train_ok) + min(train_ng)) / 2
    val = [row for row in rows if row["split"] == "val"]
    correct = sum(("NG" if row["mean_gray"] > boundary else "OK") == row["class"]
                  for row in val)
    result = {"train_ok_range": [min(train_ok), max(train_ok)],
              "train_ng_range": [min(train_ng), max(train_ng)],
              "threshold_from_train_only": boundary,
              "val_correct": correct, "val_total": len(val),
              "val_ok_correct": sum(row["class"] == "OK" and
                                    row["mean_gray"] <= boundary for row in val),
              "val_ng_correct": sum(row["class"] == "NG" and
                                    row["mean_gray"] > boundary for row in val),
              "note": "仅用于量化数据捷径，不能作为打结识别方法"}
    target = run / "brightness_shortcut_probe.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
