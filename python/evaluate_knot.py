"""创建时间：2026-09-28；功能：逐图评估打结模型；目的：同时统计 NG 检出与 OK 误报并保存可复核图片。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="口罩打结逐图评估")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path,
                        help="评估产物的根目录；默认使用数据拆分目录")
    parser.add_argument("--name", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--nms-free", action="store_true", help="切换到 YOLO26 one-to-one 免 NMS 检测头")
    parser.add_argument("--clahe", action="store_true", help="推理前对灰度亮度做 CLAHE 对照实验")
    parser.add_argument("--brightness-shift", type=float, default=0.0,
                        help="亮度扰动对照：每个像素统一加此值，随后截断到 0～255")
    parser.add_argument("--brightness-gain", type=float, default=1.0,
                        help="亮度增益扰动对照：每个像素乘此值")
    parser.add_argument("--normalize-mean", type=float,
                        help="把每张图的平均灰度平移到指定值，用于跨光照对照")
    parser.add_argument("--blur-kernel", type=int, default=0,
                        help="奇数核的轻度高斯模糊对照，0 表示关闭")
    parser.add_argument("--noise-std", type=float, default=0.0,
                        help="固定随机种子的高斯噪声标准差对照")
    parser.add_argument("--rotate-180", action="store_true",
                        help="整图旋转 180 度并同步变换标注框，用于朝向扰动诊断")
    parser.add_argument("--flip-vertical", action="store_true",
                        help="整图上下翻转并同步变换标注框，用于真实朝向差异诊断")
    parser.add_argument("--border-pad-ratio", type=float, default=0.0,
                        help="推理时在四周加边框并映回原图，用于贴边目标诊断")
    parser.add_argument("--border-mode", choices=["reflect", "constant"], default="reflect")
    return parser.parse_args()


def iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1, y1 = np.maximum(box_a[:2], box_b[:2])
    x2, y2 = np.minimum(box_a[2:], box_b[2:])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    return float(intersection / max(1e-9, area_a + area_b - intersection))


def load_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"图片无法读取：{path}")
    return image


def draw(image: np.ndarray, truth: list[list[float]], predictions: list[dict],
         threshold: float, category: str) -> np.ndarray:
    out = image.copy()
    for box in truth:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(out, "GT", (x1, max(15, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 200, 0), 2)
    for pred in predictions:
        if pred["confidence"] < threshold:
            continue
        x1, y1, x2, y2 = map(int, pred["box"])
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 220), 2)
        cv2.putText(out, f"{pred['confidence']:.2f}", (x1, min(out.shape[0] - 5, y2 + 17)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 220), 2)
    cv2.putText(out, f"{category} | green=GT red=pred | threshold={threshold:.2f}",
                (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
    return out


def main() -> None:
    args = parse_args()
    manifest = json.loads((args.run_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    items = [item for item in manifest["items"] if item["split"] == args.split]
    if not items:
        raise ValueError("拆分中没有样本")
    if args.blur_kernel and (args.blur_kernel < 3 or args.blur_kernel % 2 == 0):
        raise ValueError("模糊核必须是大于等于 3 的奇数")
    if args.noise_std < 0:
        raise ValueError("噪声标准差不能为负")
    if not 0 <= args.border_pad_ratio <= .2:
        raise ValueError("边框比例必须在 0～0.2 之间")
    output = (args.output_dir or args.run_dir) / "evaluations" / args.name
    output.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.weights))
    rows = []
    for index, item in enumerate(items, 1):
        image_path = args.run_dir / "dataset" / "images" / args.split / item["file"]
        image = load_image(image_path)
        truth = [box.copy() for box in item["boxes"]]
        if args.rotate_180 and args.flip_vertical:
            raise ValueError("两种朝向变换不能同时启用")
        if args.rotate_180:
            height, width = image.shape[:2]
            image = cv2.rotate(image, cv2.ROTATE_180)
            truth = [[width - x2, height - y2, width - x1, height - y1]
                     for x1, y1, x2, y2 in truth]
        if args.flip_vertical:
            height = image.shape[0]
            image = cv2.flip(image, 0)
            truth = [[x1, height - y2, x2, height - y1]
                     for x1, y1, x2, y2 in truth]
        if args.clahe:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
            image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if args.brightness_shift or args.brightness_gain != 1.0:
            image = np.clip(image.astype(np.float32) * args.brightness_gain +
                            args.brightness_shift,
                            0, 255).astype(np.uint8)
        if args.noise_std:
            noise = np.random.default_rng(42 + index).normal(0, args.noise_std, image.shape[:2])
            image = np.clip(image.astype(np.float32) + noise[:, :, None],
                            0, 255).astype(np.uint8)
        if args.normalize_mean is not None:
            shift = args.normalize_mean - float(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).mean())
            image = np.clip(image.astype(np.float32) + shift, 0, 255).astype(np.uint8)
        if args.blur_kernel:
            image = cv2.GaussianBlur(image, (args.blur_kernel, args.blur_kernel), 0)
        height, width = image.shape[:2]
        pad_x, pad_y = round(width * args.border_pad_ratio), round(height * args.border_pad_ratio)
        if pad_x or pad_y:
            mode = cv2.BORDER_REFLECT_101 if args.border_mode == "reflect" else cv2.BORDER_CONSTANT
            prepared = cv2.copyMakeBorder(image, pad_y, pad_y, pad_x, pad_x, mode,
                                          value=(100, 100, 100))
        else:
            prepared = image
        result = model.predict(prepared, imgsz=args.imgsz, conf=0.001, iou=0.5,
                               max_det=100, verbose=False, device=0, nms=not args.nms_free)[0]
        predictions = []
        for box, confidence in zip(result.boxes.xyxy.cpu().numpy(),
                                   result.boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = box
            mapped = [float(np.clip(x1 - pad_x, 0, width)),
                      float(np.clip(y1 - pad_y, 0, height)),
                      float(np.clip(x2 - pad_x, 0, width)),
                      float(np.clip(y2 - pad_y, 0, height))]
            if mapped[0] >= mapped[2] or mapped[1] >= mapped[3]:
                continue
            predictions.append({"box": mapped,
                                "confidence": float(confidence)})
        match_scores = [max((pred["confidence"] for pred in predictions
                             if iou(np.asarray(gt), np.asarray(pred["box"])) >= 0.5),
                            default=0.0) for gt in truth]
        best_iou = max((iou(np.asarray(gt), np.asarray(pred["box"]))
                        for gt in truth for pred in predictions), default=0.0)
        row = {"file": item["file"], "category": item["class"], "split": args.split,
               "nms_free": args.nms_free,
               "rotate_180": args.rotate_180,
               "flip_vertical": args.flip_vertical,
               "border_pad_ratio": args.border_pad_ratio,
               "border_mode": args.border_mode,
               "brightness_shift": args.brightness_shift,
               "brightness_gain": args.brightness_gain,
               "normalize_mean": args.normalize_mean,
               "blur_kernel": args.blur_kernel,
               "noise_std": args.noise_std,
               "truth": truth, "match_scores_iou50": match_scores,
               "best_iou_any_conf": best_iou,
               "max_confidence": max((pred["confidence"] for pred in predictions), default=0.0),
               "predictions": predictions}
        rows.append(row)
        hit = all(score >= args.threshold for score in match_scores) if truth else False
        false_alarm = not truth and any(pred["confidence"] >= args.threshold for pred in predictions)
        if (truth and not hit) or false_alarm:
            picture = draw(image, truth, predictions, args.threshold, item["class"])
            cv2.imencode(".jpg", picture)[1].tofile(str(output / f"{Path(item['file']).stem}.jpg"))
        if index % 10 == 0:
            print(f"[评估] {index}/{len(items)}", flush=True)

    thresholds = [0.01, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60]
    summary = []
    for threshold in thresholds:
        ng = [row for row in rows if row["category"] == "NG"]
        ok = [row for row in rows if row["category"] == "OK"]
        detected = sum(all(score >= threshold for score in row["match_scores_iou50"])
                       for row in ng)
        false_alarms = sum(row["max_confidence"] >= threshold for row in ok)
        unmatched_per_image = [sum(
            pred["confidence"] >= threshold and
            all(iou(np.asarray(gt), np.asarray(pred["box"])) < 0.5 for gt in row["truth"])
            for pred in row["predictions"]) for row in ng]
        summary.append({"threshold": threshold, "ng_detected": detected, "ng_total": len(ng),
                        "ng_recall": detected / len(ng), "ok_false_alarms": false_alarms,
                        "ok_total": len(ok), "ok_false_alarm_rate": false_alarms / len(ok),
                        "ng_unmatched_boxes": sum(unmatched_per_image),
                        "ng_images_with_unmatched": sum(count > 0 for count in unmatched_per_image)})
    (output / "predictions.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "thresholds.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"逐图结果：{output}", flush=True)


if __name__ == "__main__":
    main()
