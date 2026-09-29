"""创建时间：2026-09-28；功能：聚合多轮 YOLO26 训练进度与 ETA；目的：让连续优化实验只需一个本机看板入口。"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


PAGE = """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>口罩打结训练监控</title><style>body{font:16px/1.5 'Microsoft YaHei',sans-serif;background:#f4f7fa;color:#203040;margin:30px}
h1{font-size:25px}.hero{background:white;padding:18px;border-left:5px solid #1680cb;margin:15px 0}
table{width:100%;border-collapse:collapse;background:white}th,td{padding:9px;border-bottom:1px solid #dce4eb;text-align:left}
th{background:#e9f1f8}.running{color:#14694b;font-weight:bold}.failed{color:#b1342c;font-weight:bold}
.bar{width:140px;height:12px;border-radius:6px;background:#dce7ee}.fill{height:100%;border-radius:6px;background:#18a679}</style></head><body>
<h1>口罩打结检测 · 连续训练监控</h1><div class="hero" id="headline">连接中…</div>
<table><thead><tr><th>轮次</th><th>状态</th><th>进度</th><th>预计剩余</th><th>mAP50</th><th>GPU 预留</th></tr></thead><tbody id="rows"></tbody></table>
<p>每 3 秒刷新；ETA 由当前轮次的训练脚本估算，前几轮波动较大，早停时可能提前结束。<a href="http://127.0.0.1:__REPORT_PORT__/%E5%8F%A3%E7%BD%A9%E6%89%93%E7%BB%93%E6%A0%B9%E5%9B%A0%E4%BC%98%E5%8C%96%E5%8F%AF%E8%A7%86%E5%8C%96.html">查看逐图可视化</a></p>
<script>async function update(){try{const r=await fetch('/api/status',{cache:'no-store'});const d=await r.json();
document.getElementById('headline').textContent=d.active ? `当前：${d.active.name} · ${d.active.epoch}/${d.active.total_epochs} epoch · 预计剩余 ${d.active.eta_seconds ?? '计算中'} 秒` : `当前没有运行中的训练；已记录 ${d.items.length} 轮。`;
document.getElementById('rows').innerHTML=d.items.map(x=>`<tr><td>${x.name}</td><td class="${x.state==='训练中'?'running':x.state==='失败'?'failed':''}">${x.state}</td>
<td><div class="bar"><div class="fill" style="width:${x.progress_percent||0}%"></div></div>${x.completed_epochs}/${x.total_epochs} epoch</td>
<td>${x.state==='训练中'?(x.eta_seconds??'计算中')+' 秒':'—'}</td><td>${x.map50??'—'}</td><td>${x.gpu_reserved_mb??0} MB</td></tr>`).join('');
}catch(e){document.getElementById('headline').textContent='状态读取失败：'+e;}}update();setInterval(update,3000);</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    run_dir: Path
    report_port: int

    def do_GET(self) -> None:
        if self.path == "/":
            payload = PAGE.replace("__REPORT_PORT__", str(self.report_port)).encode("utf-8")
            content_type = "text/html; charset=utf-8"
        elif self.path == "/api/status":
            items = []
            for path in sorted((self.run_dir / "experiments").glob("*/training_status.json")):
                try:
                    state = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                items.append({"name": path.parent.name, "state": state.get("state", "未知"),
                              "epoch": state.get("epoch", 0),
                              "completed_epochs": state.get("completed_epochs", 0),
                              "total_epochs": state.get("total_epochs", 0),
                              "progress_percent": state.get("progress_percent", 0),
                              "eta_seconds": state.get("eta_seconds"),
                              "gpu_reserved_mb": state.get("gpu_reserved_mb"),
                              "map50": state.get("metrics", {}).get("metrics/mAP50(B)"),
                              "updated_at": state.get("updated_at", "")})
            active = next((item for item in reversed(items) if item["state"] == "训练中"), None)
            payload = json.dumps({"active": active, "items": items}, ensure_ascii=False).encode("utf-8")
            content_type = "application/json; charset=utf-8"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, _format: str, *_args) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="多轮训练进度看板")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--report-port", type=int, default=8772)
    args = parser.parse_args()
    if not args.run_dir.is_dir() or not 1 <= args.port <= 65535 or not 1 <= args.report_port <= 65535:
        raise ValueError("任务目录或端口无效")
    Handler.run_dir = args.run_dir.resolve()
    Handler.report_port = args.report_port
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"迭代看板：http://127.0.0.1:{args.port}/", flush=True)
    server.serve_forever(poll_interval=0.5)


if __name__ == "__main__":
    main()
