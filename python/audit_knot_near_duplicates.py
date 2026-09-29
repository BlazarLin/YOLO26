"""创建时间：2026-09-28；功能：审计训练与验证的打结局部相似度；目的：排查随机比例拆分中的近重复泄漏。"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np

from evaluate_knot import load_image


def crop_target(image: np.ndarray, box: list[float]) -> np.ndarray:
    x1, y1, x2, y2 = box
    side = max(x2 - x1, y2 - y1) * 1.6
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    left, top = round(cx - side / 2), round(cy - side / 2)
    right, bottom = round(cx + side / 2), round(cy + side / 2)
    canvas = np.full((bottom - top, right - left, 3), 100, np.uint8)
    src_x1, src_y1 = max(0, left), max(0, top)
    src_x2, src_y2 = min(image.shape[1], right), min(image.shape[0], bottom)
    canvas[src_y1 - top:src_y2 - top, src_x1 - left:src_x2 - left] = (
        image[src_y1:src_y2, src_x1:src_x2])
    return cv2.resize(canvas, (128, 128), interpolation=cv2.INTER_AREA)


def hash_patch(patch: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(small))[:8, :8]
    values = dct.flatten()[1:]
    return values > np.median(values)


def timestamp_ms(name: str) -> int:
    match = re.search(r"_(\d+)h(\d+)m(\d+)s(\d+)ms", name)
    if match is None:
        raise ValueError(f"文件名没有采集时间：{name}")
    hour, minute, second, millisecond = map(int, match.groups())
    return ((hour * 60 + minute) * 60 + second) * 1000 + millisecond


def main() -> None:
    parser = argparse.ArgumentParser(description="打结局部近重复审计")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    output = args.output_dir.resolve()
    manifest = json.loads((run / "dataset_manifest.json").read_text(encoding="utf-8"))
    groups: dict[str, list[dict]] = {"train": [], "val": []}
    for item in manifest["items"]:
        if item["class"] != "NG":
            continue
        image = load_image(run / "dataset" / "images" / item["split"] / item["file"])
        patch = crop_target(image, item["boxes"][0])
        groups[item["split"]].append({"file": item["file"], "patch": patch,
                                      "hash": hash_patch(patch)})
    results = []
    for valid in groups["val"]:
        distances = [(int(np.count_nonzero(valid["hash"] != train["hash"])), train)
                     for train in groups["train"]]
        distance, nearest = min(distances, key=lambda pair: pair[0])
        gray_val = cv2.cvtColor(valid["patch"], cv2.COLOR_BGR2GRAY)
        gray_train = cv2.cvtColor(nearest["patch"], cv2.COLOR_BGR2GRAY)
        correlation = float(cv2.matchTemplate(gray_val, gray_train,
                                               cv2.TM_CCOEFF_NORMED)[0, 0])
        results.append({"validation": valid["file"], "nearest_training": nearest["file"],
                        "phash_distance": distance, "pixel_correlation": correlation})
    results.sort(key=lambda row: (row["phash_distance"], -row["pixel_correlation"]))
    timing = []
    for valid in groups["val"]:
        gap, nearest = min(((abs(timestamp_ms(valid["file"]) - timestamp_ms(train["file"])),
                             train) for train in groups["train"]), key=lambda pair: pair[0])
        correlation = float(cv2.matchTemplate(
            cv2.cvtColor(valid["patch"], cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(nearest["patch"], cv2.COLOR_BGR2GRAY),
            cv2.TM_CCOEFF_NORMED)[0, 0])
        timing.append({"validation": valid["file"], "nearest_time_training": nearest["file"],
                       "gap_ms": gap, "pixel_correlation": correlation})
    timing.sort(key=lambda row: row["gap_ms"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "near_duplicate_pairs.json").write_text(
        json.dumps({"method": "人工框外扩到最大边 1.6 倍并归一到 128 像素；63 位 DCT 感知哈希",
                    "caveat": "相似度仅提示人工审阅，不能自动判定同一实物",
                    "pairs": results, "temporal_neighbors": timing,
                    "temporal_count_under_1s": sum(row["gap_ms"] < 1000 for row in timing),
                    "temporal_count_under_5s": sum(row["gap_ms"] < 5000 for row in timing)},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    panels = []
    by_name = {item["file"]: item for group in groups.values() for item in group}
    for row in results[:12]:
        left = cv2.resize(by_name[row["validation"]]["patch"], (320, 320))
        right = cv2.resize(by_name[row["nearest_training"]]["patch"], (320, 320))
        panel = np.concatenate((left, right), axis=1)
        cv2.putText(panel, f"val / train; hash={row['phash_distance']}; r={row['pixel_correlation']:.2f}",
                    (8, 25), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 250, 250), 2)
        panels.append(panel)
    montage = np.concatenate([np.concatenate(panels[i:i + 3], axis=1)
                              for i in range(0, len(panels), 3)], axis=0)
    cv2.imencode(".jpg", montage)[1].tofile(str(output / "near_duplicate_review.jpg"))
    print(f"局部相似度审计：{output}; 最小距离 {results[0]['phash_distance']}", flush=True)


if __name__ == "__main__":
    main()
