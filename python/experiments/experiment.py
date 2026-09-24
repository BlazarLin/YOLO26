"""创建时间：2026-09-23。功能：训练、导出并测量两代 YOLO nano 检测模型。目的：复用相同数据和计时口径验证部署链。"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import statistics
import time
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import onnxruntime
import tensorrt
import torch
import yaml
from ultralytics import YOLO, __version__ as ultralytics_version


ROOT = Path(__file__).resolve().parents[2]
MODEL_NAMES = ("yolov8n", "yolo26n")
COCO128_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"


def prepare_dataset(run_dir: Path, data: str) -> Path:
    if data != "coco128":
        custom = Path(data).resolve()
        if not custom.is_file():
            raise FileNotFoundError(f"数据集 YAML 不存在：{custom}")
        return custom

    dataset_dir = run_dir / "datasets" / "coco128"
    yaml_path = run_dir / "datasets" / "coco128.yaml"
    images_dir = dataset_dir / "images" / "train2017"
    if not images_dir.is_dir() or len(list(images_dir.glob("*.jpg"))) != 128:
        zip_path = run_dir / "datasets" / "coco128.zip"
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        if not zip_path.is_file():
            urllib.request.urlretrieve(COCO128_URL, zip_path)
        with zipfile.ZipFile(zip_path) as archive:
            for member in archive.infolist():
                target = (zip_path.parent / member.filename).resolve()
                if not target.is_relative_to(zip_path.parent.resolve()):
                    raise ValueError(f"数据包路径越界：{member.filename}")
            archive.extractall(zip_path.parent)
    if len(list(images_dir.glob("*.jpg"))) != 128:
        raise RuntimeError("COCO128 解压后不是 128 张训练图")

    from ultralytics import __file__ as ultralytics_file

    source_yaml = Path(ultralytics_file).parent / "cfg" / "datasets" / "coco128.yaml"
    with source_yaml.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    config["path"] = str(dataset_dir)
    config.pop("download", None)
    with yaml_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, allow_unicode=True, sort_keys=False)
    return yaml_path


def train_models(run_dir: Path, data_yaml: Path, epochs: int, batch: int, imgsz: int) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("训练需要 CUDA PyTorch；当前 torch.cuda.is_available() 为 False")
    trained = {}
    for name in MODEL_NAMES:
        weight_path = run_dir / "runs" / name / "weights" / "best.pt"
        if not weight_path.is_file():
            print(f"开始训练：{name}", flush=True)
            YOLO(f"{name}.pt").train(
                data=str(data_yaml), epochs=epochs, imgsz=imgsz, batch=batch,
                device=0, workers=2, seed=42, deterministic=True,
                project=str(run_dir / "runs"), name=name, exist_ok=True,
                plots=False, verbose=False,
            )
        if not weight_path.is_file():
            raise RuntimeError(f"训练未产生权重：{weight_path}")
        trained[name] = str(weight_path)
    return trained


def export_models(run_dir: Path, trained: dict, imgsz: int) -> dict:
    exported = {}
    for name, weight in trained.items():
        onnx_path = Path(weight).with_suffix(".onnx")
        engine_path = Path(weight).with_suffix(".engine")
        engine_built = False
        if not engine_path.is_file():
            print(f"导出 TensorRT：{name}", flush=True)
            YOLO(weight).export(format="engine", imgsz=imgsz, batch=1, dynamic=False,
                                half=False, opset=17, simplify=True, device=0)
            engine_built = True
        if not onnx_path.is_file() or engine_built:
            print(f"导出 CPU ONNX：{name}", flush=True)
            YOLO(weight).export(format="onnx", imgsz=imgsz, batch=1, dynamic=False,
                                half=False, opset=17, simplify=True, device="cpu")
        if not onnx_path.is_file() or not engine_path.is_file():
            raise RuntimeError(f"导出文件缺失：{name}")
        exported[name] = {"onnx_cpu": str(onnx_path), "tensorrt": str(engine_path)}
        if name == "yolo26n":
            e2e_weight = Path(weight).with_name("best_nmsfree.pt")
            e2e_onnx = e2e_weight.with_suffix(".onnx")
            e2e_engine = e2e_weight.with_suffix(".engine")
            if not e2e_weight.is_file():
                shutil.copy2(weight, e2e_weight)
            e2e_engine_built = False
            if not e2e_engine.is_file():
                print("导出 YOLO26 免 NMS TensorRT", flush=True)
                YOLO(str(e2e_weight)).export(format="engine", imgsz=imgsz, batch=1,
                                             dynamic=False, half=False, opset=17,
                                             simplify=True, device=0, nms=False)
                e2e_engine_built = True
            if not e2e_onnx.is_file() or e2e_engine_built:
                print("导出 YOLO26 免 NMS CPU ONNX", flush=True)
                YOLO(str(e2e_weight)).export(format="onnx", imgsz=imgsz, batch=1,
                                             dynamic=False, half=False, opset=17,
                                             simplify=True, device="cpu", nms=False)
            if not e2e_onnx.is_file() or not e2e_engine.is_file():
                raise RuntimeError("YOLO26 免 NMS 导出文件缺失")
            exported[name]["onnx_cpu_nmsfree"] = str(e2e_onnx)
            exported[name]["tensorrt_nmsfree"] = str(e2e_engine)
    return exported


def load_images(data_yaml: Path, count: int) -> tuple[list[np.ndarray], list[str]]:
    with data_yaml.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    root = Path(config.get("path") or data_yaml.parent)
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    image_dir = root / config["val"]
    paths = sorted(path for path in image_dir.iterdir()
                   if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})[:count]
    if not paths:
        raise RuntimeError(f"验证集无图像：{image_dir}")
    images = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"图像读取失败：{path}")
        images.append(image)
    return images, [str(path) for path in paths]


def summarize(values: list[float]) -> dict:
    ordered = sorted(values)
    return {
        "mean_ms": round(statistics.mean(values), 3),
        "median_ms": round(statistics.median(values), 3),
        "p95_ms": round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))], 3),
        "min_ms": round(ordered[0], 3),
        "max_ms": round(ordered[-1], 3),
    }


def benchmark_one(path: str, backend: str, images: list[np.ndarray],
                  warmup: int, iterations: int, imgsz: int) -> dict:
    device = "cpu" if backend.startswith("onnx_cpu") else 0
    predictor = YOLO(path)

    def detect(image: np.ndarray):
        return predictor.predict(source=image, imgsz=imgsz, device=device,
                                 conf=0.25, iou=0.7, verbose=False, rect=False)[0]

    for index in range(warmup):
        detect(images[index % len(images)])
    if backend.startswith("tensorrt"):
        torch.cuda.synchronize()
    wall_ms, preprocess_ms, inference_ms, postprocess_ms, pipeline_ms, counts = [], [], [], [], [], []
    for index in range(iterations):
        image = images[index % len(images)]
        if backend.startswith("tensorrt"):
            torch.cuda.synchronize()
        start = time.perf_counter_ns()
        result = detect(image)
        if backend.startswith("tensorrt"):
            torch.cuda.synchronize()
        wall_ms.append((time.perf_counter_ns() - start) / 1e6)
        preprocess_ms.append(float(result.speed["preprocess"]))
        inference_ms.append(float(result.speed["inference"]))
        postprocess_ms.append(float(result.speed["postprocess"]))
        pipeline_ms.append(sum(float(result.speed[key]) for key in
                               ("preprocess", "inference", "postprocess")))
        counts.append(len(result.boxes))
    return {
        "backend": backend, "model_path": path,
        "wall_detect": summarize(wall_ms),
        "pipeline": summarize(pipeline_ms),
        "preprocess": summarize(preprocess_ms),
        "model_inference": summarize(inference_ms),
        "postprocess": summarize(postprocess_ms),
        "detections_per_image_mean": round(statistics.mean(counts), 3),
    }


def benchmark_models(exported: dict, data_yaml: Path, warmup: int,
                     iterations: int, imgsz: int, image_count: int) -> dict:
    images, paths = load_images(data_yaml, image_count)
    results = {}
    for name in MODEL_NAMES:
        results[name] = {}
        for backend in exported[name]:
            print(f"测速：{name} / {backend}", flush=True)
            results[name][backend] = benchmark_one(
                exported[name][backend], backend, images, warmup, iterations, imgsz)
    return {"images": paths, "results": results}


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLOv8-n 和 YOLO26-n 训练及推理对比")
    parser.add_argument("phase", choices=("all", "prepare", "train", "export", "benchmark"))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--data", default="coco128", help="coco128 或自定义数据集 YAML 绝对路径")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=128)
    parser.add_argument("--image-count", type=int, default=128)
    parser.add_argument("--output-name", default="benchmark.json")
    args = parser.parse_args()
    if Path(args.output_name).name != args.output_name or not args.output_name.endswith(".json"):
        parser.error("--output-name 必须是当前输出目录中的 JSON 文件名")
    data = (str(Path(args.data).resolve()) if args.data != "coco128" else args.data)
    run_dir = (args.run_dir or ROOT / "x64" / "output" /
               datetime.now().strftime("%Y%m%d_%H%M%S")).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(run_dir)
    data_yaml = prepare_dataset(run_dir, data)
    if args.phase == "prepare":
        print(data_yaml)
        return

    trained = {name: str(run_dir / "runs" / name / "weights" / "best.pt")
               for name in MODEL_NAMES}
    if args.phase in ("all", "train"):
        trained = train_models(run_dir, data_yaml, args.epochs, args.batch, args.imgsz)
    if args.phase == "train":
        return
    for path in trained.values():
        if not Path(path).is_file():
            raise FileNotFoundError(f"缺少训练权重：{path}")

    exported = {name: {"onnx_cpu": str(Path(path).with_suffix(".onnx")),
                       "tensorrt": str(Path(path).with_suffix(".engine"))}
                for name, path in trained.items()}
    e2e_weight = Path(trained["yolo26n"]).with_name("best_nmsfree.pt")
    if e2e_weight.with_suffix(".onnx").is_file() and e2e_weight.with_suffix(".engine").is_file():
        exported["yolo26n"]["onnx_cpu_nmsfree"] = str(e2e_weight.with_suffix(".onnx"))
        exported["yolo26n"]["tensorrt_nmsfree"] = str(e2e_weight.with_suffix(".engine"))
    if args.phase in ("all", "export"):
        exported = export_models(run_dir, trained, args.imgsz)
    if args.phase == "export":
        return
    for files in exported.values():
        for path in files.values():
            if not Path(path).is_file():
                raise FileNotFoundError(f"缺少导出模型：{path}")

    bench = benchmark_models(exported, data_yaml, args.warmup,
                             args.iterations, args.imgsz, args.image_count)
    training = {}
    for name in MODEL_NAMES:
        args_yaml = run_dir / "runs" / name / "args.yaml"
        with args_yaml.open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        training[name] = {key: config.get(key) for key in
                          ("model", "data", "epochs", "batch", "imgsz", "seed", "optimizer")}
    report = {
        "created_at": datetime.now().astimezone().isoformat(),
        "dataset_yaml": str(data_yaml), "training": training,
        "imgsz": args.imgsz, "warmup": args.warmup,
        "iterations": args.iterations, "precision": "FP32",
        "python": platform.python_version(), "torch": torch.__version__,
        "onnxruntime": onnxruntime.__version__, "tensorrt": tensorrt.__version__,
        "ultralytics": ultralytics_version,
        "cpu": platform.processor(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        **bench,
    }
    output = run_dir / args.output_name
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["results"], ensure_ascii=False, indent=2), flush=True)
    print(f"结果文件：{output}", flush=True)


if __name__ == "__main__":
    main()
