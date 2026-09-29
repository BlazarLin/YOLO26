"""创建时间：2026-09-28；功能：统计打结框尺度与贴边情况并生成局部审阅图；目的：查找定位学习和评估规则的不一致。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from evaluate_knot import load_image


def main() -> None:
    parser = argparse.ArgumentParser(description="打结框尺寸与边界审计")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = args.run_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((source / "dataset_manifest.json").read_text(encoding="utf-8"))
    records = []
    for row in manifest["items"]:
        for box in row["boxes"]:
            x1, y1, x2, y2 = box
            records.append({"file": row["file"], "split": row["split"],
                            "width": row["width"], "height": row["height"],
                            "box": box, "box_width": x2 - x1, "box_height": y2 - y1,
                            "area": (x2 - x1) * (y2 - y1),
                            "edge_distance": min(x1, y1, row["width"] - x2,
                                                 row["height"] - y2)})
    records.sort(key=lambda row: row["file"])
    by_area = sorted(records, key=lambda row: row["area"])
    by_edge = sorted(records, key=lambda row: row["edge_distance"])
    wanted = by_area[:4] + by_area[-4:] + by_edge[:4]
    # 漏检图即使不属于面积或边缘极值，也单列方便人工比较框的含义。
    for suffix in ("53s193ms", "53s569ms", "56s219ms", "50s204ms"):
        match = next((row for row in records if suffix in row["file"]), None)
        if match and match not in wanted:
            wanted.append(match)
    tile_width, tile_height = 360, 280
    columns = 4
    sheet = np.full((tile_height * ((len(wanted) + columns - 1) // columns),
                     tile_width * columns, 3), 245, dtype=np.uint8)
    for index, row in enumerate(wanted):
        image = load_image(source / "dataset" / "images" / row["split"] / row["file"])
        x1, y1, x2, y2 = row["box"]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        crop_w = max(250, 2.8 * (x2 - x1))
        crop_h = max(190, 2.5 * (y2 - y1))
        left = max(0, min(image.shape[1] - int(crop_w), int(cx - crop_w / 2)))
        top = max(0, min(image.shape[0] - int(crop_h), int(cy - crop_h / 2)))
        right = min(image.shape[1], left + int(crop_w))
        bottom = min(image.shape[0], top + int(crop_h))
        crop = image[top:bottom, left:right].copy()
        cv2.rectangle(crop, (round(x1 - left), round(y1 - top)),
                      (round(x2 - left), round(y2 - top)), (0, 220, 0), 2)
        preview = cv2.resize(crop, (tile_width - 8, tile_height - 44))
        col, line = index % columns, index // columns
        ox, oy = col * tile_width + 4, line * tile_height + 4
        sheet[oy:oy + preview.shape[0], ox:ox + preview.shape[1]] = preview
        label = f"{row['file'][3:31]} {row['split']}"
        metrics = f"box {row['box_width']:.0f}x{row['box_height']:.0f} edge {row['edge_distance']:.0f}px"
        cv2.putText(sheet, label, (ox, oy + tile_height - 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, (30, 30, 30), 1)
        cv2.putText(sheet, metrics, (ox, oy + tile_height - 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (30, 30, 30), 1)
    cv2.imencode(".jpg", sheet)[1].tofile(str(output / "annotation_audit.jpg"))
    widths = np.asarray([row["box_width"] for row in records])
    heights = np.asarray([row["box_height"] for row in records])
    edges = np.asarray([row["edge_distance"] for row in records])
    report = {"positive_boxes": len(records),
              "width_q10_q50_q90": np.quantile(widths, [0.1, 0.5, 0.9]).tolist(),
              "height_q10_q50_q90": np.quantile(heights, [0.1, 0.5, 0.9]).tolist(),
              "edge_under_20px": int((edges < 20).sum()),
              "edge_under_40px": int((edges < 40).sum()),
              "rows": records}
    (output / "annotation_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print({key: value for key, value in report.items() if key != "rows"}, flush=True)


if __name__ == "__main__":
    main()
