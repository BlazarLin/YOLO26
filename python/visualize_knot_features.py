"""创建时间：2026-09-28；功能：导出口罩打结模型关键层激活图；目的：检查缺陷区域是否保留可分辨的特征。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox


LAYERS = {1: "early_conv", 4: "backbone_p3", 6: "backbone_p4", 16: "neck_p3"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="打结特征图可视化")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--files", nargs="+", required=True, help="数据清单中的 BMP 文件名")
    parser.add_argument("--imgsz", type=int, default=640)
    return parser.parse_args()


def prepare(image: np.ndarray, size: int, stride: int) -> tuple[np.ndarray, int, int, int, int]:
    height, width = image.shape[:2]
    ratio = min(size / width, size / height)
    new_width, new_height = round(width * ratio), round(height * ratio)
    # 与单张 model.predict 的 rect=True、auto=True 最小矩形补边保持一致。
    canvas = LetterBox(new_shape=(size, size), auto=True, stride=stride)(image=image)
    left = round((canvas.shape[1] - new_width) / 2 - 0.1)
    top = round((canvas.shape[0] - new_height) / 2 - 0.1)
    return canvas, left, top, new_width, new_height


def project(map_data: np.ndarray, shape: tuple[int, int], left: int, top: int,
            new_width: int, new_height: int, canvas_shape: tuple[int, int]) -> np.ndarray:
    resized = cv2.resize(map_data, (canvas_shape[1], canvas_shape[0]),
                         interpolation=cv2.INTER_LINEAR)
    cropped = resized[top:top + new_height, left:left + new_width]
    return cv2.resize(cropped, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)


def colorize(image: np.ndarray, activity: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(activity, [5, 99])
    normalized = np.clip((activity - lo) / max(1e-6, hi - lo), 0, 1)
    color = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    return cv2.addWeighted(image, 0.55, color, 0.45, 0)


def main() -> None:
    args = parse_args()
    manifest = json.loads((args.run_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    lookup = {item["file"]: item for item in manifest["items"]}
    model = YOLO(str(args.weights)).model.eval().to("cuda" if torch.cuda.is_available() else "cpu")
    captured = {}
    handles = []
    for index in LAYERS:
        def make_hook(layer_index: int):
            def hook(_module, _inputs, output):
                captured[layer_index] = output.detach().float().cpu().numpy()[0]
            return hook
        handles.append(model.model[index].register_forward_hook(make_hook(index)))
    output = args.run_dir / "features" / args.name
    output.mkdir(parents=True, exist_ok=True)
    metrics = []
    try:
        for file_name in args.files:
            item = lookup[file_name]
            path = args.run_dir / "dataset" / "images" / item["split"] / file_name
            image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"无法读取图片：{path}")
            canvas, left, top, new_width, new_height = prepare(
                image, args.imgsz, int(model.stride.max()))
            tensor = torch.from_numpy(canvas[:, :, ::-1].copy()).permute(2, 0, 1)
            tensor = tensor.unsqueeze(0).float().div_(255).to(next(model.parameters()).device)
            captured.clear()
            with torch.inference_mode():
                model(tensor)
            panels = []
            original = image.copy()
            for box in item["boxes"]:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(original, (x1, y1), (x2, y2), (0, 255, 0), 3)
            panels.append(original)
            layer_metrics = {}
            for index, layer_name in LAYERS.items():
                feature = captured[index]
                energy = np.mean(np.abs(feature), axis=0)
                projected = project(energy, image.shape[:2], left, top,
                                    new_width, new_height, canvas.shape[:2])
                overlay = colorize(image, projected)
                ratios = []
                for box in item["boxes"]:
                    x1, y1, x2, y2 = map(int, box)
                    roi = projected[max(0, y1):min(image.shape[0], y2),
                                    max(0, x1):min(image.shape[1], x2)]
                    ratios.append(float(np.mean(roi) / max(1e-8, np.mean(projected))))
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 3)
                layer_metrics[layer_name] = {"feature_shape": list(feature.shape),
                                             "gt_to_global_energy_ratio": ratios}
                panels.append(overlay)
                # 同一层最高能量的四个通道单独保存，便于检查平均图掩盖的局部响应。
                channel_strength = np.mean(np.abs(feature), axis=(1, 2))
                channels = np.argsort(channel_strength)[-4:][::-1]
                channel_panels = []
                for channel in channels:
                    channel_map = project(np.abs(feature[channel]), image.shape[:2],
                                          left, top, new_width, new_height, canvas.shape[:2])
                    channel_panels.append(colorize(image, channel_map))
                channel_row = np.concatenate(channel_panels, axis=1)
                cv2.imencode(".jpg", channel_row)[1].tofile(
                    str(output / f"{Path(file_name).stem}_{layer_name}_channels.jpg"))
            panel_height = 320
            scaled = [cv2.resize(panel, (round(panel.shape[1] * panel_height / panel.shape[0]),
                                         panel_height)) for panel in panels]
            for panel, label in zip(scaled, ["original_GT", *LAYERS.values()]):
                cv2.putText(panel, label, (8, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 2)
            blank = np.zeros_like(scaled[0])
            montage = np.concatenate((np.concatenate(scaled[:3], axis=1),
                                      np.concatenate([*scaled[3:], blank], axis=1)), axis=0)
            cv2.imencode(".jpg", montage)[1].tofile(str(output / f"{Path(file_name).stem}_summary.jpg"))
            metrics.append({"file": file_name, "class": item["class"], "layers": layer_metrics})
            print(f"[特征图] {file_name}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    (output / "feature_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"特征图输出：{output}", flush=True)


if __name__ == "__main__":
    main()
