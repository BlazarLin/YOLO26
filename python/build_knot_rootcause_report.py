"""创建时间：2026-09-28；功能：汇总打结根因实验与逐图叠框为本机 HTML；目的：让训练取舍和验证局限可直接复核。"""

from __future__ import annotations

import argparse
import html
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from build_knot_report import image_with_boxes


def load_result(root: Path, name: str) -> tuple[dict, dict]:
    folder = root / "evaluations" / name
    thresholds = json.loads((folder / "thresholds.json").read_text(encoding="utf-8"))
    rows = json.loads((folder / "predictions.json").read_text(encoding="utf-8"))
    stats = next(row for row in thresholds if row["threshold"] == 0.25)
    return stats, {row["file"]: row for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser(description="生成打结根因可视化报告")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--selected", default="alpha05_raw")
    parser.add_argument("--backbone-root", type=Path,
                        help="补充骨干与定位参数对照的实验目录")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    root = args.output_root.resolve()
    manifest = json.loads((run / "dataset_manifest.json").read_text(encoding="utf-8"))
    validation = {row["file"]: row for row in manifest["items"] if row["split"] == "val"}
    scenarios = [
        ("原图", "v6_expanded_val", "alpha05_raw", "v23_raw", "v29_raw"),
        ("统一加亮 11", "v6_plus11_expanded", "alpha05_plus11", "v23_plus11", "v29_plus11"),
        ("统一减亮 11", "v6_minus11_expanded", "alpha05_minus11", "", ""),
        ("增益 ×0.85", "v6_gain085_expanded", "alpha05_gain085", "v23_gain085", ""),
        ("噪声标准差 5", "v6_noise5_expanded", "alpha05_noise5", "v23_noise5", ""),
        ("旋转 180 度", "v6_rotate180", "alpha05_rotate180", "v23_rotate180", ""),
        ("上下翻转（对应现有双朝向）", "v6_flipud", "alpha05_flipud", "v23_flipud", "v29_flipud"),
        ("双裁图视野压力测试（旧验证图）", "v6_tile_infer", "alpha05_tile_stress", "", ""),
        ("双裁图视野压力测试（扩充后验证图）", "v6_tile_expanded_stress",
         "alpha05_tile_expanded_stress", "v23_tile_expanded_stress", "v29_tile_expanded_stress"),
    ]
    scenario_rows = []
    for label, base, blend, trained, oriented in scenarios:
        cells = []
        for name in (base, blend, trained, oriented):
            if not name:
                cells.append("—")
                continue
            stats, _ = load_result(root, name)
            cells.append(f"{stats['ng_detected']}/{stats['ng_total']}；"
                         f"OK {stats['ok_false_alarms']}/{stats['ok_total']}")
        scenario_rows.append(f"<tr><td>{html.escape(label)}</td>" +
                             "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
    ensemble_rows = []
    for label, name in (("原图", "main_raw"), ("统一加亮 11", "main_plus11"),
                        ("上下翻转", "main_flipud")):
        counts = json.loads((root / "ensembles" / f"{name}.json").read_text(
            encoding="utf-8"))["counts"]
        ensemble_rows.append(f"<tr><td>{label}</td>"
                             f"<td>{counts['ng_strict_first']}/29</td>"
                             f"<td>{counts['ng_strict_second']}/29</td>"
                             f"<td>{counts['ng_strict_union']}/29</td>"
                             f"<td>{counts['ng_strict_merged']}/29</td>"
                             f"<td>{counts['ng_strict_cascade']}/29；"
                             f"第二模型调用 {counts['second_model_calls_cascade']}/40</td></tr>")
    relaxed_merge = json.loads((root / "ensembles" / "main_raw_nms08.json").read_text(
        encoding="utf-8"))["counts"]
    benchmark_rows = []
    for label, name in (("V6 单模型", "v6_full"), ("V23 单模型", "v23_full"),
                        ("V29 单模型", "v29_full"), ("V6 + V23 双模型", "v6_v23_dual")):
        timing = json.loads((root / "benchmarks" / f"{name}.json").read_text(
            encoding="utf-8"))
        benchmark_rows.append(f"<tr><td>{label}</td><td>{timing['median_ms']:.2f} ms</td>"
                              f"<td>{timing['p90_ms']:.2f} ms</td></tr>")
    backbone_panel = ""
    if args.backbone_root is not None:
        backbone_root = args.backbone_root.resolve()
        chart = backbone_root / "diagnostics" / "backbone_business_comparison.png"
        if not chart.is_file():
            raise FileNotFoundError(chart)
        shutil.copy2(chart, root / "diagnostics" / chart.name)
        backbone_rows = []
        for label, raw_name, bright_name in (
            ("V6 原模型", "", ""),
            ("n 全网络续训", "n_full_raw", "n_full_plus11"),
            ("n 冻结前 11 层", "n_freeze11_raw", "n_freeze11_plus11"),
            ("n 框距离 L1 权重加倍", "n_dfl3_raw", "n_dfl3_plus11"),
            ("n 官方预训练微调", "n_pretrained_raw", "n_pretrained_plus11"),
            ("s 官方预训练微调", "s_pretrained_raw", "s_pretrained_plus11"),
        ):
            if not raw_name:
                raw, _ = load_result(root, "v6_expanded_val")
                bright, _ = load_result(root, "v6_plus11_expanded")
            else:
                raw, bright = (next(row for row in json.loads(
                    (backbone_root / "evaluations" / name / "thresholds.json").read_text(
                        encoding="utf-8")) if row["threshold"] == 0.25)
                               for name in (raw_name, bright_name))
            backbone_rows.append(f"<tr><td>{label}</td><td>{raw['ng_detected']}/29</td>"
                                 f"<td>{bright['ng_detected']}/29</td>"
                                 f"<td>{raw['ok_false_alarms']}/11</td></tr>")
        paired = json.loads((backbone_root / "benchmarks" / "n_vs_s_interleaved.json").read_text(
            encoding="utf-8"))
        backbone_panel = f'''<section class="panel"><h2>补充实验：骨干、框距离损失和输入大小</h2>
<img class="wide" src="diagnostics/backbone_business_comparison.png" alt="骨干和定位参数的同验证集逐图对照">
<table><thead><tr><th>方案</th><th>原图 NG 严格检出</th><th>加亮 11 NG 严格检出</th><th>原图 OK 误报</th></tr></thead>
<tbody>{''.join(backbone_rows)}</tbody></table>
<p>同一比例拆分、0.25 阈值、IoU≥0.5。全网络续训骨干参数相对 L2 变化 0.1456；冻结前 11 层后这 11 层变化均为 0，
但仍漏掉原贴边困难图。框距离 L1 权重翻倍进一步退化。仅改变 V6 推理输入，800 为 28/29 NG 且 1/11 OK 误报，
960 为 26/29 NG。n/s 同起点交错预热计时分别为 {paired['models'][0]['median_ms']:.2f} ms 和
{paired['models'][1]['median_ms']:.2f} ms；s 的原图 NG 更低。当前不替换 V6。</p>
<div class="grid casegrid"><div><b>冻结骨干的贴边困难图特征</b>
<img src="expanded/features/n_freeze11_backbone_audit/C1_260313_15h57m56s219ms_dajie_summary.jpg" alt="冻结骨干的关键层特征"></div>
<div><b>YOLO26s 的同图特征</b>
<img src="expanded/features/s_backbone_audit/C1_260313_15h57m56s219ms_dajie_summary.jpg" alt="YOLO26s 关键层特征"></div></div>
<p>热图按每张图各层的百分位独立着色，不能跨模型比绝对能量；耳带区域均有响应，
而 s 的完整匹配框仍仅 0.001 分。特征存在不等于定位正确。</p></section>'''
    hard_rows = []
    for label, full_name, tile_name in (
        ("V6 原模型", "v6_expanded_val", "v6_tile_expanded_stress"),
        ("V23 亮度副本", "v23_raw", "v23_tile_expanded_stress"),
        ("V29 双朝向", "v29_raw", "v29_tile_expanded_stress"),
        ("V34 正例裁图 + 正常裁图", "v34_raw", "v34_tile_expanded_stress"),
        ("V35 仅正常裁图", "v35_raw", "v35_tile_expanded_stress"),
    ):
        if not (root / "evaluations" / full_name / "thresholds.json").exists():
            continue
        full, _ = load_result(root, full_name)
        tile, _ = load_result(root, tile_name)
        hard_rows.append(f"<tr><td>{label}</td><td>{full['ng_detected']}/29；"
                         f"OK {full['ok_false_alarms']}/11</td>"
                         f"<td>{tile['ng_detected']}/29；"
                         f"OK {tile['ok_false_alarms']}/11</td></tr>")
    selected_stats, selected = load_result(root, args.selected)
    _, v6_raw = load_result(root, "v6_expanded_val")
    _, v6_plus = load_result(root, "v6_plus11_expanded")
    _, v23_raw = load_result(root, "v23_raw")
    _, v23_plus = load_result(root, "v23_plus11")
    _, v29_raw = load_result(root, "v29_raw")
    _, v29_plus = load_result(root, "v29_plus11")
    _, alpha_plus = load_result(root, "alpha05_plus11")
    if set(selected) != set(validation):
        raise ValueError("模型评估文件与验证清单不一致")
    image_dir = root / "report_images"
    image_dir.mkdir(exist_ok=True)
    cards = []
    image_cache = {}
    for file_name, item in sorted(validation.items(),
                                  key=lambda pair: (pair[1]["class"] != "NG", pair[0])):
        row = selected[file_name]
        path = run / "dataset" / "images" / "val" / file_name
        image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"无法读取：{path}")
        image_cache[file_name] = image
        overlay = image_with_boxes(image, item, row, 0.25)
        picture = image_dir / f"{Path(file_name).stem}.jpg"
        cv2.imencode(".jpg", overlay)[1].tofile(str(picture))
        is_ng = item["class"] == "NG"
        strict = row["match_scores_iou50"][0] >= 0.25 if is_ng else False
        bad = not strict if is_ng else row["max_confidence"] >= 0.25
        state = ("NG 定位漏检" if bad else "NG 严格检出") if is_ng else (
            "OK 误报" if bad else "OK 正确")
        measurement = (f"IoU≥0.5 匹配分数 {row['match_scores_iou50'][0]:.3f}；"
                       f"任意框最高分 {row['max_confidence']:.3f}；"
                       f"最佳 IoU {row['best_iou_any_conf']:.3f}" if is_ng
                       else f"最高预测分数 {row['max_confidence']:.3f}")
        cards.append(f'<article class="card {"bad" if bad else "good"}" '
                     f'data-bad="{1 if bad else 0}"><header><b>{html.escape(state)}</b>'
                     f'<span>{html.escape(file_name)}</span></header>'
                     f'<img loading="lazy" src="report_images/{html.escape(picture.name)}">'
                     f'<p>{html.escape(measurement)}；绿色=标注，红色=预测</p></article>')
    cases = []
    case_dir = root / "case_images"
    case_dir.mkdir(exist_ok=True)
    for file_name, item in sorted(validation.items()):
        if item["class"] != "NG":
            continue
        group = (("V6 原图", v6_raw), ("V6 +11", v6_plus),
                 ("V23 原图", v23_raw), ("V23 +11", v23_plus),
                 ("V29 原图", v29_raw), ("V29 +11", v29_plus),
                 ("插值原图", selected), ("插值 +11", alpha_plus))
        if all(rows[file_name]["match_scores_iou50"][0] >= .25 for _, rows in group):
            continue
        cells = []
        for label, rows in group:
            value = rows[file_name]["match_scores_iou50"][0]
            maximum = rows[file_name]["max_confidence"]
            overlap = rows[file_name]["best_iou_any_conf"]
            state = "检出" if value >= .25 else (
                "高分框未匹配" if maximum >= .25 else "匹配框分数低")
            cells.append(f"<tr><td>{label}</td><td>{value:.3f}</td>"
                         f"<td>{maximum:.3f}</td><td>{overlap:.3f}</td><td>{state}</td></tr>")
        pictures = []
        for label, code, rows in (("V6 原图", "v6", v6_raw),
                                  ("V23 原图", "v23", v23_raw),
                                  ("V29 原图", "v29", v29_raw),
                                  ("插值原图", "alpha", selected)):
            overlay = image_with_boxes(image_cache[file_name], item, rows[file_name], .25)
            picture = case_dir / f"{Path(file_name).stem}_{code}.jpg"
            cv2.imencode(".jpg", overlay)[1].tofile(str(picture))
            pictures.append(f'<div><b>{label}</b><img loading="lazy" '
                            f'src="case_images/{html.escape(picture.name)}"></div>')
        cases.append(f'<article class="panel"><h3>{html.escape(file_name)}</h3>'
                     f'<div class="grid casegrid">{"".join(pictures)}</div>'
                     '<table><thead><tr><th>模型/输入</th><th>IoU≥0.5 框分数</th>'
                     '<th>任意框最高分</th><th>任意框最佳 IoU</th><th>现象</th></tr></thead>'
                     f'<tbody>{"".join(cells)}</tbody></table></article>')
    probe = json.loads((root / "diagnostics" / "local_brightness.json").read_text(encoding="utf-8"))
    local_rows = []
    for name, label in (("original", "不扰动"), ("global", "整图 +11"),
                        ("target", "只加亮打结框及周边"),
                        ("background", "只加亮框外背景")):
        row = probe["summary"][name]
        local_rows.append(f"<tr><td>{label}</td><td>{row['strict_hits_025']}/29</td>"
                          f"<td>{row['median_score_change']:+.3f}</td>"
                          f"<td>{row['score_drop_over_005']}/29</td></tr>")
    audit = json.loads((root / "diagnostics" / "annotation_audit.json").read_text(encoding="utf-8"))
    edge_file = "C1_260313_15h57m56s219ms_dajie.bmp"
    edge_rows = []
    for label, name in (("无补边", "v6_expanded_val"),
                        ("镜像补边 5%", "v6_pad05_reflect"),
                        ("灰度 100 补边 5%", "v6_pad05_constant"),
                        ("灰度 100 补边 2%", "v6_pad02_constant")):
        stats, predictions = load_result(root, name)
        score = predictions[edge_file]["match_scores_iou50"][0]
        edge_rows.append(f"<tr><td>{label}</td><td>{stats['ng_detected']}/29</td>"
                         f"<td>{stats['ok_false_alarms']}/11</td><td>{score:.3f}</td></tr>")
        if name == "v6_pad02_constant":
            overlay = image_with_boxes(image_cache[edge_file], validation[edge_file],
                                       predictions[edge_file], .25)
            cv2.imencode(".jpg", overlay)[1].tofile(str(case_dir / "edge_pad_constant.jpg"))
    edge_retry = json.loads((root / "edge_retry" / "main_raw.json").read_text(
        encoding="utf-8"))["summary"]
    ng_count = sum(row["class"] == "NG" for row in validation.values())
    ok_count = len(validation) - ng_count
    document = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>口罩打结根因与优化复核</title>
<style>body{{margin:0;background:#edf2f5;color:#1a2b34;font:16px/1.6 'Microsoft YaHei',sans-serif}}
main{{max-width:1370px;margin:30px auto;padding:0 22px}}h1{{font-size:28px}}h2{{font-size:22px;margin-top:32px}}
.panel{{background:white;padding:20px 24px;border-radius:12px;margin:18px 0;box-shadow:0 2px 12px #10253411}}
.alert{{border-left:5px solid #d57d1a;background:#fff7e8}}.goodline{{border-left:5px solid #258f68}}
table{{width:100%;border-collapse:collapse}}td,th{{padding:10px;border-bottom:1px solid #dfe8ec;text-align:left}}
th{{background:#e8f2f7}}.wide{{width:100%;max-height:630px;object-fit:contain;background:#fff}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(400px,1fr));gap:15px}}
.card{{background:white;border-radius:9px;overflow:hidden;border:2px solid #dce8e3}}
.card.bad{{border-color:#d65549}}.card header{{padding:10px 12px;display:flex;justify-content:space-between;gap:10px}}
.card header span{{font-size:12px;overflow-wrap:anywhere}}.card img{{width:100%}}.card p{{padding:0 12px 8px;margin:0;font-size:13px}}
.casegrid{{grid-template-columns:repeat(auto-fit,minmax(290px,1fr));gap:10px}}
.casegrid img{{width:100%}}
button{{font:inherit;padding:6px 12px;margin-right:8px;border-radius:6px;border:1px solid #93aab6;background:#fff;cursor:pointer}}
.muted{{color:#60717d}}a{{color:#146da3}}</style></head><body><main>
<h1>口罩下成像打结：根因与优化复核</h1>
<p class="muted">数据：134 张 NG + 51 张 OK；固定验证：{ng_count} 张 NG + {ok_count} 张 OK；模型均按置信度 0.25、IoU≥0.5 统计。</p>
<section class="panel goodline"><b>稳定参照为 V6；50% 权重插值是探索候选</b>。
原图严格检出 {selected_stats['ng_detected']}/{selected_stats['ng_total']} NG，
OK 误报 {selected_stats['ok_false_alarms']}/{selected_stats['ok_total']}；不增加双模型推理耗时。
但双裁图视野压力测试出现 2/5 OK 误报，且这组验证图片已参与方案选择，不能直接部署。</section>
<section class="panel alert"><b>生产验收限制：</b>全部 NG 拍于 3 月 13 日；21 张旧 OK 拍于 3 月 10 日，30 张新 OK 拍于 9 月 28 日。
两类整图平均灰度完全分开；新 OK 的印字位置也与 NG 不同。用户确认机位和曝光相同、产线两种朝向都会出现；
仍需要同班次、各朝向、相近材料下同时采集 OK 与 NG，
再做独立验收。当前 0/{ok_count} OK 误报无法保证量产误报率。</section>
<section class="panel"><h2>固定阈值下的模型与扰动对照</h2>
<table><thead><tr><th>输入</th><th>原 V6</th><th>50% 插值单模型</th><th>亮度副本训练 V23</th><th>亮度和上下翻转训练 V29</th></tr></thead>
<tbody>{''.join(scenario_rows)}</tbody></table>
<p class="muted">每格为 NG 严格检出数；OK 误报数。旧验证图的双裁图行用 29 NG + 5 OK，其余用扩充后的 29 NG + 11 OK。
对照现有图片，印字右下与右上更接近上下翻转；180° 旋转会把右侧印字移到左侧，不能代表该真实朝向。所有软件扰动都不能代替真实工况。</p></section>
{backbone_panel}
<section class="panel"><h2>两模型互补与运行代价</h2>
<table><thead><tr><th>输入</th><th>V6</th><th>亮度微调</th><th>直接并集</th><th>跨模型 NMS 后</th><th>V6 有框才调用第二模型（合并前）</th></tr></thead>
<tbody>{''.join(ensemble_rows)}</tbody></table>
<p>此表 OK 误报均为 0/11，但当前 OK/NG 存在采集批次线索。主验证原图直接并集的 29/29 在跨模型 NMS 后退回 28/29：
高分局部框压掉了低分完整框。每张图运行两次模型仍需测实际耗时；条件级联也会漏掉 V6 完全无框的图。
放宽跨模型 NMS 到 0.8 后原图重新达到 {relaxed_merge['ng_strict_merged']}/29，但输出
{relaxed_merge['merged_boxes']} 个框，其中 {relaxed_merge['images_multiple_boxes']} 张图出现多框；
这是看过漏检后的探索设置。种子 123 无收益，独立新批次前不推荐部署。</p></section>
<section class="panel"><h2>预热后 Python 单图耗时</h2>
<table><thead><tr><th>方式</th><th>中位数</th><th>P90</th></tr></thead>
<tbody>{''.join(benchmark_rows)}</tbody></table>
<p>RTX 3060、同一张 1050×542 NG、640 输入，预热 30 次、计时 100 次。
包含 Python 预处理、网络、后处理、框转存 CPU，双模型含跨模型 NMS；不含读盘和权重加载。
双模型约为单模型两倍，却未在常用 NMS 下提升主验证原图的严格检出。</p></section>
<section class="panel"><h2>三份按比例拆分的复核</h2>
<img class="wide" src="diagnostics/replicate_comparison.png" alt="三份比例拆分的原图与加亮检出数">
<p>亮度微调在种子 42 的加亮图上有收益，在种子 123 基本持平；50% 权重插值在种子 777 反而退化。
三份验证图互有重叠，不能累加为 102 张独立 NG。上图三种方法的 OK 误报均为 0/13；
新增的双朝向训练另出现过 OK 误报，且 OK/NG 采集差异仍在。</p></section>
<section class="panel"><h2>双朝向训练的三份拆分复核</h2>
<img class="wide" src="diagnostics/orientation_replicate_comparison.png" alt="三份拆分上原图、加亮和上下翻转检出数">
<p>两份拆分的原图和加亮图各增一张 NG 检出，但两份的翻转图各少一张；
种子 123 原图新增 1/13 OK 误报。不能只挑其中一栏决定训练策略。</p></section>
<section class="panel"><h2>双朝向训练引入的 OK 边角误报</h2>
<img class="wide" src="evaluations/seed123_flipud_raw/C1_260310_16h11m32s254ms.jpg" alt="种子 123 的 OK 左上角误报">
<p>种子 123 的双朝向训练模型在这张旧 OK 左上空白边角产生 0.317 分框，
原亮度微调模型在同一验证集无 OK 误报。即使 NG 多检出一张，也要把误报同时计入。</p></section>
<section class="panel"><h2>裁图后正常耳带交叉处被误判</h2>
<div class="grid casegrid"><div><b>V6，新增 PNG OK，0.475 分误报</b>
<img src="evaluations/v6_tile_expanded_stress/C1_260928_10h07m49s872ms.jpg" alt="V6 裁图后正常耳带误报"></div>
<div><b>V29，同一张 OK，0.495 分误报</b>
<img src="evaluations/v29_tile_expanded_stress/C1_260928_10h07m49s872ms.jpg" alt="V29 裁图后正常耳带误报"></div></div>
<p>整图输入时这张 OK 没有 ≥0.25 的预测框。60% 宽度双裁图改变视野和目标尺度后，
两模型都在正常耳带交叉处输出高分框。扩充验证集上 V6、V23、V29、插值模型分别有 3/11、6/11、2/11、4/11 OK 误报；
所以裁图只用作压力测试，不是当前可采用的推理方式。</p></section>
<section class="panel"><h2>正常区域硬负例：误报与漏检的交换</h2>
<table><thead><tr><th>训练方案</th><th>整图 NG 严格检出 / OK 误报</th><th>双裁图 NG 严格检出 / OK 误报</th></tr></thead>
<tbody>{''.join(hard_rows)}</tbody></table>
<img class="wide" src="evaluations/v34_raw/C1_260313_15h57m54s705ms_dajie.jpg" alt="V34 将打结只定位到局部而漏掉人工框的连接段">
<p>V34 加入训练图生成的正例和正常裁图后，裁图 OK 误报消失，但整图 NG 严格检出下降；
红色预测框收缩到拐弯局部，未覆盖绿色人工框中的横向连接段。V35 只加入正常裁图，同样消除了裁图 OK 误报，
但整图原图仅 27/29 NG，软件加亮 11 后仅 18/29，较 V6 的 22/29 更差。
图像级有框与缺陷提取位置要分别验收。</p></section>
<section class="panel"><h2>为什么目前的 OK 指标偏乐观</h2>
<img class="wide" src="diagnostics/domain_shift_scatter.png" alt="采集批次灰度与印字位置分布">
<p>横轴是整图平均灰度；纵轴是上方减下方的高亮像素比例。NG 与两批 OK 在灰度上不相交；
新增 PNG 的印字偏上，NG 偏下。用训练集确定一个灰度阈值就能在 40 张验证图上得到 40/40，
因此同批次 OK/NG 才能检验模型是否依靠打结形态。</p></section>
<section class="panel"><h2>真实朝向的图像对照</h2>
<div class="grid casegrid"><div><b>新增 OK 原图：印字在右上</b>
<img src="expanded/dataset/images/val/C1_260928_10h06m09s897ms.png" alt="新增 OK 的印字在右上"></div>
<div><b>旧 NG 上下翻转：印字仍在右上</b>
<img src="diagnostics/flipud_label_check.jpg" alt="旧 NG 上下翻转后的绿框标注"></div>
<div><b>旧 NG 旋转 180°：印字移到左上</b>
<img src="diagnostics/rotation_label_check.jpg" alt="旧 NG 旋转 180 度后的绿框标注"></div></div>
<p>用户确认两种朝向均在产线出现。现有图片更接近上下翻转，因此旋转 180° 训练轮次作为失败实验保留；
上下翻转同样只是软件变换，真实反向 NG 仍需补齐。</p></section>
<section class="panel"><h2>模型真正敏感的图像区域</h2>
<table><thead><tr><th>V6 输入</th><th>NG 严格检出</th><th>匹配分数中位变化</th><th>分数下降超过 0.05</th></tr></thead>
<tbody>{''.join(local_rows)}</tbody></table>
<p>只加亮打结附近比只加亮背景影响大；这提示局部亮度和对比度敏感。局部处理边界经过平滑，
仍属于人为干预诊断，不能证明网络学到了正确的打结语义。</p></section>
<section class="panel"><h2>关键卷积层：漏检图的特征流</h2>
<div class="grid casegrid"><div><b>V6：56s219ms，贴边局部框</b>
<img src="expanded/features/v6_rootcause/C1_260313_15h57m56s219ms_dajie_summary.jpg" alt="V6 贴边漏检图关键层特征"></div>
<div><b>V29：同一张图，原图定位恢复</b>
<img src="expanded/features/v29_rootcause/C1_260313_15h57m56s219ms_dajie_summary.jpg" alt="V29 同一图关键层特征"></div>
<div><b>V6：53s193ms，亮度微调后定位退化</b>
<img src="expanded/features/v6_rootcause/C1_260313_15h57m53s193ms_dajie_summary.jpg" alt="V6 第二张困难图特征"></div>
<div><b>V29：同一张图，仍只有局部高分框</b>
<img src="expanded/features/v29_rootcause/C1_260313_15h57m53s193ms_dajie_summary.jpg" alt="V29 第二张困难图特征"></div></div>
<p>展示 early Conv、backbone P3/P4 和 neck P3 的通道平均响应。缺陷框内与整图能量比在不同层约为 0.70～1.29；
两模型在缺陷附近均有响应，同时正常耳带、边缘和印字也会激活。
热图说明特征流没有完全丢失该区域，不能据此断言识别语义正确，更不能从通道平均图解释单个预测框。</p></section>
<section class="panel"><h2>贴边图：补边修复单图但无法跨拆分复现</h2>
<table><thead><tr><th>V6 输入</th><th>NG 严格检出</th><th>OK 误报</th><th>原漏检图匹配分数</th></tr></thead>
<tbody>{''.join(edge_rows)}</tbody></table>
<img class="wide" src="case_images/edge_pad_constant.jpg" alt="固定灰度补边后贴边图的绿框和红框">
<p>2% 固定灰度补边把原漏检图升到 0.691，但另一图改为漏检。
只对第一遍高分框距边界不足 20 像素的图片补边重跑，主验证原图为
{edge_retry['ng_retry_strict']}/29，触发 {edge_retry['retry_images']}/40；
在三份比例拆分中均没有增加检出。边界上下文有影响，此规则不作部署推荐。</p></section>
<section class="panel"><h2>标注范围与贴边样本</h2>
<img class="wide" src="diagnostics/annotation_audit.jpg" alt="不同打结框和贴边示例">
<p>{audit['positive_boxes']} 个正例框中 {audit['edge_under_20px']} 个距离图像边界不足 20 像素。
框宽中位数 {audit['width_q10_q50_q90'][1]:.0f} 像素，高中位数 {audit['height_q10_q50_q90'][1]:.0f} 像素。
绿色框有时只覆盖打结核心，有时包含较长垂带；需要人工统一提取范围。</p></section>
<section class="panel"><h2>训练与验证的近似画面审计</h2>
<img class="wide" src="diagnostics/near_duplicate_review.jpg" alt="训练与验证缺陷局部并排对比">
<p>每组左边是验证 NG、右边是感知哈希最相近的训练 NG。结构相似不等于同一实物；
按采集文件名，验证 NG 距最近训练 NG 均超过 9 秒，未发现 5 秒内跨集合的直接相邻帧。</p></section>
<section class="panel"><h2>逐张反查漏检：{len(cases)} 张困难 NG</h2>
<p>收录 V6、V23、V29、插值模型在原图或 +11 条件下任一次严格漏检的全部图片。
绿色为人工框、红色为达到 0.25 的预测框。匹配框分数低与高分框未匹配是不同问题；
具体框形和人工框范围需一起看，表格状态只作快速提示。</p></section>
{''.join(cases)}
<section class="panel"><h2>逐图验证：{html.escape(args.selected)}</h2>
<button onclick="show('all')">全部 {len(cards)} 张</button><button onclick="show('bad')">只看漏检或误报</button>
<div class="grid" id="cards">{''.join(cards)}</div></section>
<script>function show(mode){{document.querySelectorAll('.card').forEach(x=>{{x.style.display=(mode==='all'||x.dataset.bad==='1')?'':'none'}})}}
</script></main></body></html>"""
    target = root / "口罩打结根因优化可视化.html"
    target.write_text(document, encoding="utf-8")
    print(f"可视化报告：{target}", flush=True)


if __name__ == "__main__":
    main()
