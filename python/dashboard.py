"""创建时间：2026-09-24；功能：提供 YOLO26 训练状态网页；目的：在本机浏览器查看进度、剩余时间与验证曲线。"""

from __future__ import annotations

import argparse
import csv
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


PAGE = Path(__file__).with_name("dashboard.html")


class DashboardHandler(BaseHTTPRequestHandler):
    status_path: Path
    results_path: Path

    def do_GET(self) -> None:
        if self.path in {"/", "/dashboard.html"}:
            self.respond(PAGE.read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/status":
            payload = (self.status_path.read_bytes() if self.status_path.is_file()
                       else '{"state":"等待训练启动"}'.encode("utf-8"))
            self.respond(payload, "application/json; charset=utf-8")
        elif self.path == "/api/history":
            rows = []
            if self.results_path.is_file():
                with self.results_path.open("r", encoding="utf-8-sig", newline="") as stream:
                    rows = [{key.strip(): value.strip() for key, value in row.items()}
                            for row in csv.DictReader(stream) if row]
            self.respond(json.dumps(rows[-100:], ensure_ascii=False).encode("utf-8"),
                         "application/json; charset=utf-8")
        else:
            self.send_error(404)

    def respond(self, payload: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format_string: str, *args) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO26 本机训练看板")
    parser.add_argument("--run-dir", type=Path, required=True, help="训练脚本的输出目录")
    parser.add_argument("--name", default="yolo26n_custom", help="训练任务名")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        raise ValueError("端口无效")
    DashboardHandler.status_path = args.run_dir.resolve() / "training_status.json"
    DashboardHandler.results_path = args.run_dir.resolve() / "runs" / args.name / "results.csv"
    server = ThreadingHTTPServer(("127.0.0.1", args.port), DashboardHandler)
    print(f"训练看板：http://127.0.0.1:{args.port}/", flush=True)
    server.serve_forever(poll_interval=0.5)


if __name__ == "__main__":
    main()
