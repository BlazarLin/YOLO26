"""创建时间：2026-09-24；功能：训练 YOLO26 并记录进度；目的：让自定义数据训练可复现、可由网页看板监控。"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 YOLO26 检测模型")
    parser.add_argument("--data", type=Path, required=True, help="YOLO 数据集 YAML")
    parser.add_argument("--weights", default="yolo26n.pt", help="预训练权重路径或 Ultralytics 模型名")
    parser.add_argument("--output-dir", type=Path, help="任务输出目录，默认 x64/output/时间戳")
    parser.add_argument("--name", default="yolo26n_custom")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", default="0")
    return parser.parse_args()


class TrainingMonitor:
    def __init__(self, status_path: Path, epochs: int) -> None:
        self.status_path = status_path
        self.epochs = epochs
        self.started = time.monotonic()
        self.last_write = 0.0
        self.completed_epochs = 0
        self.current_batch = 0
        self.epoch_durations: list[float] = []
        self.state: dict = {
            "state": "准备中", "phase": "准备数据与模型", "pid": os.getpid(),
            "started_at": datetime.now().astimezone().isoformat(),
            "epoch": 0, "completed_epochs": 0, "total_epochs": epochs,
            "batch": 0, "batches_per_epoch": 0, "progress_percent": 0.0,
            "elapsed_seconds": 0, "eta_seconds": None, "metrics": {}, "losses": {},
            "gpu_reserved_mb": 0, "best_weight": None,
        }
        self.write(force=True)

    def write(self, force: bool = False) -> None:
        import torch

        now = time.monotonic()
        if not force and now - self.last_write < 1.5:
            return
        self.last_write = now
        batches = self.state["batches_per_epoch"]
        progress = self.completed_epochs + self.current_batch / batches if batches else self.completed_epochs
        self.state["progress_percent"] = round(min(100.0, progress / self.epochs * 100), 2)
        self.state["elapsed_seconds"] = round(now - self.started)
        if self.state["state"] == "已完成":
            self.state["progress_percent"] = 100.0
            self.state["eta_seconds"] = 0
        elif self.epoch_durations:
            seconds_per_epoch = sum(self.epoch_durations[-5:]) / len(self.epoch_durations[-5:])
            self.state["eta_seconds"] = round(max(0.0, self.epochs - progress) * seconds_per_epoch)
        elif self.current_batch >= 2 and batches:
            batch_seconds = (now - self.started) / self.current_batch
            self.state["eta_seconds"] = round(max(0.0, self.epochs * batches - self.current_batch) * batch_seconds)
        self.state["updated_at"] = datetime.now().astimezone().isoformat()
        if torch.cuda.is_available():
            self.state["gpu_reserved_mb"] = round(torch.cuda.memory_reserved(0) / 1048576)
        temporary = self.status_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.state, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, self.status_path)

    def on_train_start(self, trainer) -> None:
        self.state["state"] = "训练中"
        self.state["phase"] = "训练"
        self.state["batches_per_epoch"] = len(trainer.train_loader)
        self.write(force=True)

    def on_train_epoch_start(self, trainer) -> None:
        self.current_batch = 0
        self.state["epoch"] = trainer.epoch + 1
        self.state["batch"] = 0
        self.state["phase"] = "训练"
        self.write(force=True)

    def on_train_batch_end(self, trainer) -> None:
        self.current_batch += 1
        self.state["batch"] = self.current_batch
        self.write()

    def on_train_epoch_end(self, trainer) -> None:
        self.state["phase"] = "验证与保存"
        self.write(force=True)

    def on_fit_epoch_end(self, trainer) -> None:
        if self.state["state"] == "已完成":
            return
        results_csv = Path(trainer.save_dir) / "results.csv"
        recorded_epochs = (max(0, len(results_csv.read_text(encoding="utf-8").splitlines()) - 1)
                           if results_csv.is_file() else trainer.epoch + 1)
        self.completed_epochs = min(self.epochs, trainer.epoch + 1, recorded_epochs)
        self.current_batch = 0
        self.state["completed_epochs"] = self.completed_epochs
        self.state["batch"] = 0
        self.state["metrics"] = {name: round(float(value), 6)
                                 for name, value in trainer.metrics.items()
                                 if isinstance(value, (int, float)) or hasattr(value, "item")}
        epoch_time = getattr(trainer, "epoch_time", None)
        if epoch_time is not None:
            self.epoch_durations.append(float(epoch_time))
        self.write(force=True)
        print(f"[进度] {self.completed_epochs}/{self.epochs} epoch，"
              f"预计剩余 {self.state['eta_seconds']} 秒，指标 {self.state['metrics']}", flush=True)

    def on_train_end(self, trainer) -> None:
        self.state["state"] = "已完成"
        self.state["phase"] = "完成"
        self.state["early_stopped"] = self.completed_epochs < self.epochs
        self.state["best_weight"] = str(Path(trainer.save_dir) / "weights" / "best.pt")
        self.state["last_weight"] = str(Path(trainer.save_dir) / "weights" / "last.pt")
        self.write(force=True)


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch <= 0 or args.imgsz <= 0 or args.workers < 0:
        raise ValueError("训练参数无效")
    data = args.data.resolve()
    if not data.is_file():
        raise FileNotFoundError(data)
    output_dir = (args.output_dir or ROOT / "x64" / "output" /
                  datetime.now().strftime("%Y%m%d_%H%M%S")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    monitor = TrainingMonitor(output_dir / "training_status.json", args.epochs)
    try:
        import torch
        from ultralytics import YOLO

        if args.device != "cpu" and not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用")
        model = YOLO(args.weights)
        for event in ("on_train_start", "on_train_epoch_start", "on_train_batch_end",
                      "on_train_epoch_end", "on_fit_epoch_end", "on_train_end"):
            model.add_callback(event, getattr(monitor, event))
        model.train(data=str(data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                    device=args.device, workers=args.workers, seed=42, deterministic=True,
                    project=str(output_dir / "runs"), name=args.name, exist_ok=True,
                    patience=args.patience, plots=True, verbose=False,
                    mosaic=0.2, close_mosaic=10, fliplr=0.0, flipud=0.0,
                    hsv_h=0.0, hsv_s=0.0, hsv_v=0.05,
                    degrees=3.0, translate=0.05, scale=0.2)
        print(f"训练输出：{output_dir}", flush=True)
    except Exception as error:
        monitor.state["state"] = "失败"
        monitor.state["phase"] = "异常终止"
        monitor.state["error"] = str(error)
        monitor.state["traceback"] = traceback.format_exc()
        monitor.write(force=True)
        raise


if __name__ == "__main__":
    main()
