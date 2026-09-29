"""创建时间：2026-09-28；功能：汇总打结实验和逐图预测为本机 HTML；目的：让 NG 漏检、OK 误报与特征图容易复核。"""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成口罩打结实验可视化报告")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True, help="evaluations 下已有的模型名")
    parser.add_argument("--selected", required=True, help="逐图展示的模型名")
    parser.add_argument("--threshold", type=float, required=True)
    return parser.parse_args()


def svg_plot(data: dict[str, list[dict]], field: str, title: str) -> str:
    width, height = 650, max(260, 245 + 16 * math.ceil(len(data) / 3))
    colors = ["#2563eb", "#e05a2a", "#159570", "#a34db6", "#b58112",
              "#495361", "#c22778", "#825331"]
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
             f'<text x="20" y="25" font-size="18" font-weight="bold">{html.escape(title)}</text>']
    for y, label in [(60, "100%"), (130, "50%"), (200, "0%")]:
        parts.append(f'<line x1="65" y1="{y}" x2="610" y2="{y}" stroke="#dbe3ea"/>')
        parts.append(f'<text x="10" y="{y + 4}" font-size="12">{label}</text>')
    for x, label in [(65, "0"), (174, "0.1"), (283, "0.2"), (392, "0.3"),
                     (501, "0.4"), (610, "0.5")]:
        parts.append(f'<text x="{x - 8}" y="220" font-size="12">{label}</text>')
    for index, (name, rows) in enumerate(data.items()):
        color = colors[index % len(colors)]
        points = []
        for row in rows:
            threshold = row["threshold"]
            if threshold > 0.5:
                continue
            x = 65 + threshold / 0.5 * 545
            y = 200 - row[field] * 140
            points.append(f"{x:.1f},{y:.1f}")
        parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" '
                     f'stroke-width="2.5"/>')
        parts.append(f'<text x="{65 + (index % 3) * 185}" y="{239 + (index // 3) * 16}" '
                     f'fill="{color}" font-size="12">● {html.escape(name)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def svg_scores(rows: dict[str, dict], threshold: float) -> str:
    ng = sorted((row["match_scores_iou50"][0], row["file"])
                for row in rows.values() if row["category"] == "NG")
    ok = sorted((row["max_confidence"], row["file"])
                for row in rows.values() if row["category"] == "OK")
    def y_position(value: float) -> float:
        return 225 - (math.log10(max(0.001, value)) + 3) / 3 * 165
    parts = ['<svg viewBox="0 0 1000 280" role="img" aria-label="验证图匹配分数分布">',
             '<text x="18" y="25" font-size="18" font-weight="bold">验证图逐张分数：左为 NG 匹配框，右为 OK 最高框</text>']
    for value in [1.0, 0.1, 0.01, 0.001]:
        y = y_position(value)
        parts.append(f'<line x1="60" y1="{y:.1f}" x2="970" y2="{y:.1f}" '
                     f'stroke="#dbe3ea"/><text x="8" y="{y+4:.1f}" font-size="12">{value:g}</text>')
    y = y_position(threshold)
    parts.append(f'<line x1="60" y1="{y:.1f}" x2="970" y2="{y:.1f}" '
                 f'stroke="#c83734" stroke-dasharray="6 4" stroke-width="2"/>')
    parts.append(f'<text x="840" y="{max(45, y-7):.1f}" fill="#c83734" '
                 f'font-size="13">阈值 {threshold:.3f}</text>')
    for index, (score, file_name) in enumerate(ng):
        x = 72 + index * 24
        color = "#198754" if score >= threshold else "#d04734"
        parts.append(f'<circle cx="{x}" cy="{y_position(score):.1f}" r="5" fill="{color}">'
                     f'<title>{html.escape(file_name)}：匹配分数 {score:.4f}</title></circle>')
    for index, (score, file_name) in enumerate(ok):
        x = 825 + index * 29
        parts.append(f'<circle cx="{x}" cy="{y_position(score):.1f}" r="5" fill="#2778ac">'
                     f'<title>{html.escape(file_name)}：最高框分数 {score:.4f}</title></circle>')
    parts.append('<text x="350" y="258" font-size="14">29 张 NG</text>'
                 '<text x="845" y="258" font-size="14">5 张 OK</text></svg>')
    return "".join(parts)


def image_with_boxes(image: np.ndarray, item: dict, row: dict, threshold: float) -> np.ndarray:
    output = image.copy()
    for box in item["boxes"]:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(output, (x1, y1), (x2, y2), (25, 200, 25), 3)
    visible_index = 0
    for prediction in row["predictions"]:
        if prediction["confidence"] < threshold:
            continue
        x1, y1, x2, y2 = map(int, prediction["box"])
        cv2.rectangle(output, (x1, y1), (x2, y2), (30, 35, 230), 2)
        label_y = (max(18, y1 - 5) if visible_index % 2 == 0
                   else min(output.shape[0] - 8, y2 - 5))
        cv2.putText(output, f"{prediction['confidence']:.3f}",
                    (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 35, 230), 2)
        visible_index += 1
    return output


def main() -> None:
    args = parse_args()
    run = args.run_dir.resolve()
    manifest = json.loads((run / "dataset_manifest.json").read_text(encoding="utf-8"))
    validation = {item["file"]: item for item in manifest["items"] if item["split"] == "val"}
    summaries = {}
    predictions = {}
    for name in args.models:
        folder = run / "evaluations" / name
        summaries[name] = json.loads((folder / "thresholds.json").read_text(encoding="utf-8"))
        predictions[name] = {row["file"]: row for row in
                             json.loads((folder / "predictions.json").read_text(encoding="utf-8"))}
        if set(predictions[name]) != set(validation):
            raise ValueError(f"{name} 评估图片与当前验证集不同")
    if args.selected not in predictions:
        raise ValueError("所选模型未纳入报告")

    image_dir = run / "report_images"
    image_dir.mkdir(exist_ok=True)
    cards = []
    selected = predictions[args.selected]
    for file_name, item in sorted(validation.items(), key=lambda pair: (pair[1]["class"] != "NG", pair[0])):
        row = selected[file_name]
        image_path = run / "dataset" / "images" / "val" / file_name
        image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"无法读取 {image_path}")
        picture = image_with_boxes(image, item, row, args.threshold)
        target = image_dir / f"{Path(file_name).stem}.jpg"
        cv2.imencode(".jpg", picture)[1].tofile(str(target))
        if item["class"] == "NG":
            score = min(row["match_scores_iou50"])
            good = score >= args.threshold
            status = "NG 已检出" if good else "NG 漏检"
            score_text = f"标注匹配分数 {score:.3f}；最佳 IoU {row['best_iou_any_conf']:.3f}"
        else:
            good = row["max_confidence"] < args.threshold
            status = "OK 正确" if good else "OK 误报"
            score_text = f"最高预测分数 {row['max_confidence']:.3f}"
        cards.append(f'<article class="card {"good" if good else "bad"}">'
                     f'<div class="cap"><b>{html.escape(status)}</b><span>{html.escape(file_name)}</span></div>'
                     f'<img loading="lazy" src="report_images/{html.escape(target.name)}">'
                     f'<p>{html.escape(score_text)}；绿色=标注，红色=预测</p></article>')

    table_rows = []
    for name in args.models:
        cells = []
        for threshold in [0.01, 0.03, 0.05, 0.25]:
            result = next((row for row in summaries[name]
                           if abs(row["threshold"] - threshold) < 1e-8), None)
            cells.append(f'{result["ng_detected"]}/{result["ng_total"]}；'
                         f'OK {result["ok_false_alarms"]}/{result["ok_total"]}' if result else "—")
        table_rows.append(f'<tr><td>{html.escape(name)}</td>'
                          + "".join(f'<td>{cell}</td>' for cell in cells) + '</tr>')

    selected_ng = sum(row["category"] == "NG" and
                      min(row["match_scores_iou50"]) >= args.threshold for row in selected.values())
    selected_ok_fp = sum(row["category"] == "OK" and
                         row["max_confidence"] >= args.threshold for row in selected.values())

    default_models = {name: rows for name, rows in summaries.items() if not name.endswith("_e2e")}
    e2e_models = {name: rows for name, rows in summaries.items() if name.endswith("_e2e")}
    plots = "".join(svg_plot(group, "ng_recall", title)
                    for group, title in [(default_models, "默认分支 NG 检出率"),
                                         (e2e_models, "免 NMS 分支 NG 检出率")] if group)
    plots += "".join(svg_plot(group, "ok_false_alarm_rate", title)
                     for group, title in [(default_models, "默认分支 OK 误报率"),
                                          (e2e_models, "免 NMS 分支 OK 误报率")] if group)
    feature_cards = []
    comparison_cards = []
    last_miss_card = ""
    for file_name in ["C1_260313_15h57m53s193ms_dajie.bmp",
                      "C1_260313_15h57m53s569ms_dajie.bmp",
                      "C1_260313_15h57m56s219ms_dajie.bmp",
                      "C1_260313_16h18m50s204ms_dajie.bmp"]:
        stem = Path(file_name).stem
        if "baseline_ngonly" in predictions:
            item = validation[file_name]
            source_path = run / "dataset" / "images" / "val" / file_name
            source_image = cv2.imdecode(np.fromfile(str(source_path), dtype=np.uint8),
                                       cv2.IMREAD_COLOR)
            if source_image is None:
                raise ValueError(f"无法读取 {source_path}")
            old_row = predictions["baseline_ngonly"][file_name]
            new_row = selected[file_name]
            old_marked = image_with_boxes(source_image, item, old_row, args.threshold)
            new_marked = image_with_boxes(source_image, item, new_row, args.threshold)
            comparison = np.concatenate([old_marked, new_marked], axis=1)
            comparison_path = image_dir / f"{stem}_old_new.jpg"
            cv2.imencode(".jpg", comparison)[1].tofile(str(comparison_path))
            x1, y1, x2, y2 = map(int, item["boxes"][0])
            margin = 75
            left, top = max(0, x1 - margin), max(0, y1 - margin)
            right = min(source_image.shape[1], x2 + margin)
            bottom = min(source_image.shape[0], y2 + margin)
            zoom_old = old_marked[top:bottom, left:right]
            zoom_new = new_marked[top:bottom, left:right]
            zoom_pair = np.concatenate([zoom_old, zoom_new], axis=1)
            zoom_pair = cv2.resize(zoom_pair, None, fx=2, fy=2,
                                   interpolation=cv2.INTER_NEAREST)
            zoom_path = image_dir / f"{stem}_old_new_zoom.jpg"
            cv2.imencode(".jpg", zoom_pair)[1].tofile(str(zoom_path))
            old_score = min(old_row["match_scores_iou50"])
            new_score = min(new_row["match_scores_iou50"])
            comparison_cards.append(
                f'<article class="card"><div class="cap"><b>{html.escape(file_name)}</b>'
                f'<span>首版 {old_score:.3f} → {html.escape(args.selected)} {new_score:.3f}</span>'
                f'</div><img loading="lazy" src="report_images/{html.escape(comparison_path.name)}">'
                f'<img loading="lazy" src="report_images/{html.escape(zoom_path.name)}" '
                f'alt="局部放大对比">'
                f'<p>左：首版；右：当前模型。绿色是标注框，红色是达到阈值的预测框。</p></article>')
            if (args.selected == "v6_box10" and "56s219ms" in file_name and
                    "v14_box15" in predictions):
                v14_row = predictions["v14_box15"][file_name]
                v14_marked = image_with_boxes(source_image, item, v14_row, 0.07)
                v14_zoom = v14_marked[top:bottom, left:right]
                v6_v14_pair = np.concatenate([zoom_new, v14_zoom], axis=1)
                v6_v14_pair = cv2.resize(v6_v14_pair, None, fx=2, fy=2,
                                         interpolation=cv2.INTER_NEAREST)
                v6_v14_path = image_dir / f"{stem}_v6_v14_zoom.jpg"
                cv2.imencode(".jpg", v6_v14_pair)[1].tofile(str(v6_v14_path))
                last_miss_card = (
                    f'<img src="report_images/{html.escape(v6_v14_path.name)}" '
                    f'alt="贴边漏检图 V6 与 V14 局部对比" style="max-width:1100px;width:100%">')
        old_path = run / "features" / "baseline_ngonly" / f"{stem}_summary.jpg"
        new_path = run / "features" / args.selected / f"{stem}_summary.jpg"
        if old_path.is_file() and new_path.is_file():
            third_path = run / "features" / "v14_box15" / f"{stem}_summary.jpg"
            third_card = (f'<div><b>v14_box15</b><img loading="lazy" '
                          f'src="features/v14_box15/{html.escape(third_path.name)}"></div>'
                          if third_path.is_file() else "")
            feature_cards.append(f'<details><summary>{html.escape(file_name)}：旧版与当前模型特征</summary>'
                                 f'<div class="feature"><div><b>首版</b><img loading="lazy" '
                                 f'src="features/baseline_ngonly/{html.escape(old_path.name)}"></div>'
                                 f'<div><b>{html.escape(args.selected)}</b><img loading="lazy" '
                                 f'src="features/{html.escape(args.selected)}/{html.escape(new_path.name)}">'
                                 f'</div>{third_card}</div></details>')
    light_rows = []
    for name, label in [("v6_box10", "原图"),
                        ("v6_box10_bright_plus11", "全部加亮 11"),
                        ("v6_box10_bright_minus11", "全部变暗 11"),
                        ("v6_box10_norm72_infer", "平均灰度归一到 72"),
                        ("v6_box10_plus11_norm72", "加亮 11 后归一"),
                        ("v6_box10_minus11_norm72", "变暗 11 后归一"),
                        ("v6_box10_gain085_norm72", "增益 0.85 后归一")]:
        path = run / "evaluations" / name / "thresholds.json"
        if path.is_file():
            row = next(value for value in json.loads(path.read_text(encoding="utf-8"))
                       if value["threshold"] == 0.25)
            light_rows.append(f'<tr><td>{label}</td><td>{row["ng_detected"]}/29</td>'
                              f'<td>{row["ok_false_alarms"]}/5</td></tr>')
    v14_rows = []
    for name, label in [("v14_box15", "原图"),
                        ("v14_box15_plus11", "统一加亮 11"),
                        ("v14_box15_noise5", "固定噪声标准差 5"),
                        ("v14_box15_gain085_norm72", "增益 0.85 后归一到 72")]:
        path = run / "evaluations" / name / "thresholds.json"
        if path.is_file():
            values = json.loads(path.read_text(encoding="utf-8"))
            low = next(row for row in values if row["threshold"] == 0.07)
            safe = next(row for row in values if row["threshold"] == 0.10)
            v14_rows.append(f'<tr><td>{label}</td>'
                            f'<td>{low["ng_detected"]}/29；OK {low["ok_false_alarms"]}/5</td>'
                            f'<td>{safe["ng_detected"]}/29；OK {safe["ok_false_alarms"]}/5</td></tr>')
    repeat_rows = []
    for seed in [42, 123, 777]:
        path = (run / "splits" / f"seed{seed}" / "evaluations" /
                f"v8_random_seed{seed}" / "thresholds.json")
        if path.is_file():
            values = json.loads(path.read_text(encoding="utf-8"))
            low = next(row for row in values if row["threshold"] == 0.05)
            standard = next(row for row in values if row["threshold"] == 0.25)
            next_name = {42: "v18_seed42_box15", 123: "v16_seed123_box15",
                         777: "v17_seed777_box15"}[seed]
            next_path = run / "splits" / f"seed{seed}" / "evaluations" / next_name / "thresholds.json"
            next_values = (json.loads(next_path.read_text(encoding="utf-8"))
                           if next_path.is_file() else [])
            next_low = next((row for row in next_values if row["threshold"] == 0.05), None)
            next_standard = next((row for row in next_values if row["threshold"] == 0.25), None)
            next_low_ng = f'{next_low["ng_detected"]}/34' if next_low else '—'
            next_low_ok = f'{next_low["ok_false_alarms"]}/5' if next_low else '—'
            next_standard_ng = f'{next_standard["ng_detected"]}/34' if next_standard else '—'
            repeat_rows.append(f'<tr><td>{seed}</td><td>{low["ng_detected"]}/34</td>'
                               f'<td>{low["ok_false_alarms"]}/5</td>'
                               f'<td>{standard["ng_detected"]}/34</td>'
                               f'<td>{next_low_ng}</td><td>{next_low_ok}</td>'
                               f'<td>{next_standard_ng}</td></tr>')
    shortcut_path = run / "brightness_shortcut_probe.json"
    shortcut_text = "NG 与 OK 的平均灰度分布存在采集差异，跨批次仍需补图验证。"
    if shortcut_path.is_file():
        shortcut = json.loads(shortcut_path.read_text(encoding="utf-8"))
        shortcut_text = (
            f'仅用训练集选出的平均灰度阈值 {shortcut["threshold_from_train_only"]:.3f}，'
            f'就能在验证集分对 {shortcut["val_ng_correct"]}/29 NG 和 '
            f'{shortcut["val_ok_correct"]}/5 OK；该规则没有识别打结，只是在利用采集条件。'
            '因此 0/5 OK 误报不能推断为跨批次的产线水平。')
    benchmark_rows = []
    for name in ["b0_default_raw", "v6_default_raw", "v6_ok_raw", "v6_edge_ng_raw",
                 "v14_default_raw", "v6_e2e_raw", "v6_default_norm72_blur5",
                 "v5s_default_raw"]:
        path = run / "benchmarks" / f"{name}.json"
        if path.is_file():
            value = json.loads(path.read_text(encoding="utf-8"))
            benchmark_rows.append(f'<tr><td>{html.escape(name)}</td>'
                                  f'<td>{value["median_ms"]:.2f} ms</td>'
                                  f'<td>{value["p90_ms"]:.2f} ms</td></tr>')
    last_miss_section = (
        '<h2>最后一张贴边图：V6 与 V14</h2><p>左侧 V6 阈值 0.25 只保留耳带下段局部框；'
        '右侧 V14 阈值 0.07 同时出现局部框和覆盖完整标注的框。'
        '右侧的低阈值在噪声试验中不稳定。</p>' + last_miss_card
        if last_miss_card else "")
    doc = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>口罩打结检测实验报告</title>
<style>body{{font:16px/1.6 "Microsoft YaHei",sans-serif;background:#f4f7fa;color:#172331;margin:0}}
main{{max-width:1500px;margin:auto;padding:28px}}h1{{font-size:28px}}h2{{margin-top:38px}}.lead{{background:#fff;padding:18px;border-left:5px solid #2563eb}}
.charts{{display:grid;grid-template-columns:repeat(auto-fit,minmax(580px,1fr));gap:16px}}svg{{background:#fff;border-radius:8px;width:100%;height:auto}}
.kpis{{display:flex;gap:14px;flex-wrap:wrap;margin:18px 0}}.kpi{{background:#fff;border-radius:8px;padding:16px 24px;min-width:185px}}
.kpi b{{font-size:27px;color:#155c9a}}.kpi small{{display:block;color:#506274}}
.features details{{background:#fff;border:1px solid #d8e0e8;border-radius:8px;margin:9px 0;padding:10px}}
.features summary{{cursor:pointer;font-weight:bold}}.feature{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px}}
.feature img{{display:block;width:100%;height:auto;margin-top:5px}}
table{{width:100%;border-collapse:collapse;background:#fff}}th,td{{border-bottom:1px solid #d8e0e8;padding:7px 10px;text-align:left}}th{{background:#e9f0f7;position:sticky;top:0}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(440px,1fr));gap:14px}}.card{{background:#fff;border-radius:8px;overflow:hidden;border:2px solid #b8d9ca}}
.card.bad{{border-color:#ed7d69}}.card img{{width:100%;display:block}}.card p{{margin:4px 12px 12px;font-size:14px}}.cap{{display:flex;justify-content:space-between;gap:12px;padding:9px 12px;font-size:14px}}
.cap span{{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}.bad b{{color:#ad2e22}}.good b{{color:#176b47}}</style></head><body><main>
<h1>口罩下成像打结检测：训练与验证可视化</h1>
<div class="lead">源数据 134 NG + 21 OK；合并后按比例拆分，未按日期分组。固定训练集 105 NG + 16 OK，验证集 29 NG + 5 OK。只显示验证集，训练图片未进入本报告的指标。<br>
逐图显示模型：<b>{html.escape(args.selected)}</b>，阈值 <b>{args.threshold:.3f}</b>。NG 需框与标注 IoU ≥ 0.5 才算检出；OK 有任意达阈值框就算误报。两天采集条件与类别绑定，OK 验证仅 5 张，产线泛化仍需补图。</div>
<div class="kpis"><div class="kpi"><b>{selected_ng}/29</b><small>NG 检出（{selected_ng/29:.1%}）</small></div>
<div class="kpi"><b>{selected_ok_fp}/5</b><small>OK 误报</small></div>
<div class="kpi"><b>{29-selected_ng}</b><small>NG 漏检图</small></div></div>
<h2>数据审计</h2><p>{html.escape(shortcut_text)}</p>
<img src="dataset_profile.png" alt="数据亮度和标注分布" style="width:100%;background:white;border-radius:8px">
<h2>光照扰动复测（V6，阈值 0.25）</h2><p>同一批验证图片的模拟亮度变化；归一可以抵消均匀加减亮度，较大对比度变化仍会漏检。</p>
<table><thead><tr><th>输入处理</th><th>NG 检出</th><th>OK 误报</th></tr></thead><tbody>{''.join(light_rows)}</tbody></table>
<h2>V14 的高检出阈值与扰动</h2><p>0.07 在原图达到 29/29，但固定噪声后出现 OK 误报；0.10 的原图为 28/29。扰动是软件模拟，不能代替新采集批次。</p>
<table><thead><tr><th>输入处理</th><th>阈值 0.07</th><th>阈值 0.10</th></tr></thead><tbody>{''.join(v14_rows)}</tbody></table>
<h2>逐张分数分布</h2><p>纵轴采用对数刻度；红色 NG 点低于阈值，蓝色是 OK 图片上的最高预测分数。鼠标悬停可看文件名。</p>
{svg_scores(selected, args.threshold)}
<h2>阈值与检出 / 误报</h2><div class="charts">{plots}</div>
<h2>各轮对比</h2><p>每格依次为 NG 检出数、OK 误报数；两条推理分支分别比较。</p>
<table><thead><tr><th>模型 / 分支</th><th>阈值 0.01</th><th>阈值 0.03</th><th>阈值 0.05</th><th>阈值 0.25</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table>
<h2>V6 训练曲线</h2><p>同一 34 张验证图反复用于各 epoch 的指标；曲线可看训练过程，不是独立测试成绩。较小验证集会使 recall 和 mAP 波动。</p>
<img src="experiments/v6_box10/runs/yolo26n_knot/results.png" alt="V6 训练和验证曲线" style="width:100%;background:white;border-radius:8px">
<h2>预热后 Python 单图耗时</h2><p>RTX 3060，640 输入；加载后预热 30 次、计时 100 次并同步 GPU。含 model.predict 预处理、网络、后处理；归一与模糊也计入。无读盘、画框、写盘及模型加载。</p>
<table><thead><tr><th>模型与输入</th><th>中位数</th><th>P90</th></tr></thead><tbody>{''.join(benchmark_rows)}</tbody></table>
<h2>按比例随机划分的重复复核</h2><p>每个种子都从通用权重重新训练。随机划分可能把外观相近的图片放在两侧，不能与上方固定首版验证图的结果直接比较。</p>
<table><thead><tr><th>种子</th><th>框权重 10：0.05 NG</th><th>框权重 10：0.05 OK 误报</th><th>框权重 10：0.25 NG</th><th>框权重 15：0.05 NG</th><th>框权重 15：0.05 OK 误报</th><th>框权重 15：0.25 NG</th></tr></thead><tbody>{''.join(repeat_rows)}</tbody></table>
<h2>首版四张漏检：左右对比</h2><p>并排比较同一张图在首版和当前模型下的框，分数是与标注 IoU ≥ 0.5 的最高匹配分数。高分局部框若 IoU 不足，仍按严格漏检计。</p>
<div class="grid">{''.join(comparison_cards)}</div>
{last_miss_section}
<h2>关键层特征图</h2><p>绿色框是打结标注，热色代表相对更强的卷积激活。早期层主要响应边缘，深层响应形态；亮区不等于已经正确识别打结，需要与下方逐图预测同时看。</p>
<div class="features">{''.join(feature_cards)}</div>
<h2>固定验证集逐图结果</h2><div class="grid">{''.join(cards)}</div>
<p>完整实验策略、漏检归因和复核限制见仓库 docs/260928口罩打结检测迭代记录.md。</p>
</main></body></html>'''
    result_path = run / "口罩打结检测可视化.html"
    result_path.write_text(doc, encoding="utf-8")
    print(f"可视化报告：{result_path}")


if __name__ == "__main__":
    main()
