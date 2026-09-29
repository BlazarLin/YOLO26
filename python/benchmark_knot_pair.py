"""创建时间：2026-09-28；功能：交错计时两种 YOLO26 骨干；目的：降低单独启动进程造成的温度和调度漂移。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="交错比较两种骨干的预热后单图耗时")
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    if args.warmup < 1 or args.iterations < 10:
        raise ValueError("预热或计时次数过少")
    image = cv2.imdecode(np.fromfile(str(args.image), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图片：{args.image}")
    models = [YOLO(str(args.first)), YOLO(str(args.second))]

    def infer(nIndex: int) -> float:
        started = time.perf_counter()
        result = models[nIndex].predict(image, imgsz=640, conf=0.25, iou=0.5,
                                        max_det=100, verbose=False, device=0, nms=True)[0]
        result.boxes.xyxy.cpu().numpy()
        result.boxes.conf.cpu().numpy()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return (time.perf_counter() - started) * 1000

    for _ in range(args.warmup):
        infer(0)
        infer(1)
    timings: list[list[float]] = [[], []]
    for nIteration in range(args.iterations):
        for nIndex in ((0, 1) if nIteration % 2 == 0 else (1, 0)):
            timings[nIndex].append(infer(nIndex))
    summaries = []
    for path, values in zip((args.first, args.second), timings):
        summaries.append({"weights": str(path.resolve()),
                          "median_ms": float(np.median(values)),
                          "p90_ms": float(np.percentile(values, 90)),
                          "mean_ms": float(np.mean(values))})
    result = {"image": str(args.image.resolve()), "device": torch.cuda.get_device_name(0),
              "imgsz": 640, "confidence": 0.25, "warmup_each": args.warmup,
              "iterations_each": args.iterations,
              "scope": "预加载单图；含 Python 预处理、网络、后处理和框转存 CPU；不含读盘与权重加载",
              "models": summaries,
              "second_to_first_median_ratio": summaries[1]["median_ms"] / summaries[0]["median_ms"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
