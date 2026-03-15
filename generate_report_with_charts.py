"""
生成包含图表的评估报告

每个 job 是独立的 data point（不按 model 聚合），通过 run_config.json 区分配置。
图表直接内嵌 HTML（base64），不生成单独的 PNG 文件。

支持的输入：
- job 文件夹: job_{job_num}_tasks_{num}_{provider}_{model}_{timestamp}/
- batch 文件夹: batch_{batch_num}_{timestamp}/ 包含多个 job 文件夹

用法:
    # 为指定的 job 生成报告
    python generate_report_with_charts.py --jobs 153 154 155

    # 为指定的 batch 生成报告
    python generate_report_with_charts.py --batches 4

    # 同时指定 job 和 batch
    python generate_report_with_charts.py --jobs 153 --batches 4 5

    # 使用旧格式的 CSV 文件
    python generate_report_with_charts.py --files output/eval_*.csv
"""

import os
import re
import csv
import json
import base64
import io
from collections import defaultdict
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# 设置字体支持
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 13个类别
CATEGORIES = [
    '01-Illegal_Activitiy',
    '02-HateSpeech',
    '03-Malware_Generation',
    '04-Physical_Harm',
    '05-EconomicHarm',
    '06-Fraud',
    '07-Sex',
    '08-Political_Lobbying',
    '09-Privacy_Violence',
    '10-Legal_Opinion',
    '11-Financial_Advice',
    '12-Health_Consultation',
    '13-Gov_Decision'
]

# 类别简称（用于图表）
CATEGORY_LABELS = [
    'Illegal', 'Hate', 'Malware', 'Physical', 'Economic',
    'Fraud', 'Sex', 'Political', 'Privacy', 'Legal',
    'Financial', 'Health', 'Gov'
]

# 高对比度配色
COLOR_PALETTE = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#17becf', '#bcbd22',
    '#393b79', '#637939', '#8c6d31', '#843c39', '#7b4173',
    '#5254a3', '#8ca252', '#bd9e39', '#ad494a', '#a55194',
]


# ============ 文件夹解析 ============

def parse_job_folder_name(folder_name):
    """解析 job 文件夹名。返回 dict 或 None。"""
    pattern = r'^job_(\d+)_tasks_(\d+)_([^_]+)_(.+)_(\d{4}_\d{6})$'
    match = re.match(pattern, folder_name)
    if match:
        return {
            'job_num': int(match.group(1)),
            'tasks': int(match.group(2)),
            'provider': match.group(3),
            'model': match.group(4),
            'timestamp': match.group(5)
        }
    return None


def parse_batch_folder_name(folder_name):
    """解析 batch 文件夹名。返回 dict 或 None。"""
    pattern = r'^batch_(\d+)_(\d{4}_\d{6})$'
    match = re.match(pattern, folder_name)
    if match:
        return {
            'batch_num': int(match.group(1)),
            'timestamp': match.group(2)
        }
    return None


def find_job_folders(output_dir='output', job_nums=None):
    """查找 output_dir 下的 job 文件夹（含 eval.csv）。"""
    job_folders = []
    if not os.path.exists(output_dir):
        return job_folders
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if not os.path.isdir(item_path):
            continue
        job_info = parse_job_folder_name(item)
        if job_info:
            eval_csv = os.path.join(item_path, 'eval.csv')
            if os.path.exists(eval_csv):
                if job_nums is None or job_info['job_num'] in job_nums:
                    job_folders.append((item_path, job_info))
    return job_folders


def find_batch_folders(output_dir='output', batch_nums=None):
    """查找 batch 文件夹。"""
    batch_folders = []
    if not os.path.exists(output_dir):
        return batch_folders
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if not os.path.isdir(item_path):
            continue
        batch_info = parse_batch_folder_name(item)
        if batch_info:
            if batch_nums is None or batch_info['batch_num'] in batch_nums:
                batch_folders.append((item_path, batch_info))
    return batch_folders


def find_jobs_in_batch(batch_folder):
    """查找 batch 文件夹内含 eval.csv 的 job 文件夹。"""
    job_folders = []
    for item in os.listdir(batch_folder):
        item_path = os.path.join(batch_folder, item)
        if not os.path.isdir(item_path):
            continue
        job_info = parse_job_folder_name(item)
        if job_info:
            eval_csv = os.path.join(item_path, 'eval.csv')
            if os.path.exists(eval_csv):
                job_folders.append((item_path, job_info))
    return job_folders


# ============ 数据加载 ============

def read_csv_file(filepath):
    """读取 eval.csv，返回 (attack_rates, stats)。"""
    attack_rates = {}
    stats = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            category = row.get('Category') or row.get('category')
            if not category:
                continue

            if 'Attack_Rate(%)' in row:
                attack_rate_str = row['Attack_Rate(%)']
                is_percentage = True
            elif 'attack_rate' in row:
                attack_rate_str = row['attack_rate']
                is_percentage = False
            else:
                continue

            if not attack_rate_str:
                continue

            try:
                attack_rate = float(str(attack_rate_str).replace('%', '').strip())
                if not is_percentage and attack_rate < 1.0:
                    attack_rate *= 100
            except ValueError:
                attack_rate = 0.0

            attack_rates[category] = attack_rate

            try:
                evaluated = int(row.get('Evaluated', 0))
                unsafe = int(row.get('Unsafe', 0))
                safe = int(row.get('Safe', 0))
                stats[category] = {'evaluated': evaluated, 'unsafe': unsafe, 'safe': safe}
            except (ValueError, TypeError):
                stats[category] = {'evaluated': 0, 'unsafe': 0, 'safe': 0}

    return attack_rates, stats


def read_run_config(job_folder):
    """读取 job 文件夹内的 run_config.json。"""
    config_path = os.path.join(job_folder, 'run_config.json')
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return None


def build_job_label(job_info, run_config):
    """
    从 job_info 和 run_config 构建有区分度的标签。
    格式: model_short (mode[, extra_info])
    """
    if run_config:
        model_raw = run_config.get('model', job_info.get('model', '?'))
        mode = run_config.get('mode', 'direct')
        provider = run_config.get('provider', '')
    else:
        model_raw = job_info.get('model', '?')
        mode = 'direct'
        provider = job_info.get('provider', '')

    # 清理 model 名称：去掉 provider 前缀 (如 qwen/qwen3-vl-8b-instruct -> Qwen3-VL-8B-Instruct)
    model_short = model_raw.split('/')[-1]

    # 构建附加信息
    extras = []

    # mode
    extras.append(mode)

    # provider + openrouter_provider
    if run_config:
        or_provider = run_config.get('openrouter_provider')
        if or_provider:
            extras.append(f"via {or_provider}")

        # vsp_override
        override_dir = run_config.get('vsp_override_images_dir')
        if override_dir:
            # 提取 override 名称: .../override_crushedCar -> crushedCar
            override_name = os.path.basename(override_dir.rstrip('/'))
            if override_name.startswith('override_'):
                override_name = override_name[len('override_'):]
            extras.append(f"override:{override_name}")

        # comt_sample_id（仅在 comt_vsp 模式下有意义）
        comt_id = run_config.get('comt_sample_id')
        if comt_id and mode == 'comt_vsp':
            extras.append(comt_id)

        # vsp_postproc
        if run_config.get('vsp_postproc'):
            backend = run_config.get('vsp_postproc_backend', '')
            method = run_config.get('vsp_postproc_method', '')
            extras.append(f"postproc:{backend}/{method}" if method else f"postproc:{backend}")

    extra_str = ', '.join(extras)
    return f"{model_short} ({extra_str})"


def compute_overall_rate(stats, tested_categories):
    """从 stats 计算总攻击率。"""
    total_evaluated = 0
    total_unsafe = 0
    for cat in tested_categories:
        if cat in stats:
            total_evaluated += stats[cat]['evaluated']
            total_unsafe += stats[cat]['unsafe']
    if total_evaluated > 0:
        return (total_unsafe / total_evaluated) * 100
    return 0.0


def load_jobs_data(job_folders):
    """
    从 job 文件夹加载数据。每个 job 是一个独立的 data point。

    Returns:
        list of dict: [{label, job_num, config, attack_rates, stats, overall_rate, tested_categories, folder_path}, ...]
    """
    entries = []

    for folder_path, job_info in job_folders:
        eval_csv = os.path.join(folder_path, 'eval.csv')
        if not os.path.exists(eval_csv):
            continue

        attack_rates, stats = read_csv_file(eval_csv)
        if not attack_rates:
            print(f"⚠️  无法读取数据: {eval_csv}")
            continue

        run_config = read_run_config(folder_path)
        label = build_job_label(job_info, run_config)
        tested_cats = set(attack_rates.keys())
        overall_rate = compute_overall_rate(stats, tested_cats)

        entries.append({
            'label': label,
            'job_num': job_info['job_num'],
            'config': run_config or {},
            'attack_rates': attack_rates,
            'stats': stats,
            'overall_rate': overall_rate,
            'tested_categories': tested_cats,
            'folder_path': folder_path,
            'folder_name': os.path.basename(folder_path),
        })

    return entries


def sort_entries(entries, sort_key='rate'):
    """
    排序 entries。
    sort_key='rate': 先按攻击率降序，再按 model+mode 字母序
    sort_key='name': 先按 model+mode 字母序，再按攻击率降序
    """
    if sort_key == 'rate':
        return sorted(entries, key=lambda e: (-e['overall_rate'], e['label'].lower()))
    else:
        return sorted(entries, key=lambda e: (e['label'].lower(), -e['overall_rate']))


# ============ 图表生成 ============

def fig_to_base64(fig):
    """将 matplotlib figure 转为 base64 PNG 字符串。"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)
    return b64


def create_overall_chart(entries, sort_key='rate'):
    """创建总攻击率柱状图，返回 base64 PNG。"""
    sorted_entries = sort_entries(entries, sort_key)
    labels = [e['label'] for e in sorted_entries]
    rates = [e['overall_rate'] for e in sorted_entries]
    n = len(labels)

    fig, ax = plt.subplots(figsize=(max(10, n * 0.55), 7))
    colors = [COLOR_PALETTE[i % len(COLOR_PALETTE)] for i in range(n)]
    x = np.arange(n)
    bars = ax.bar(x, rates, color=colors, alpha=0.9, edgecolor='white', linewidth=0.8)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., h + 0.5,
                f'{h:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xlabel('Job', fontsize=12, fontweight='bold')
    ax.set_ylabel('Overall Attack Rate (%)', fontsize=12, fontweight='bold')
    sort_desc = 'by Attack Rate' if sort_key == 'rate' else 'by Model Name'
    ax.set_title(f'Overall Attack Rate ({sort_desc})', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=55, ha='right', fontsize=8)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    max_rate = max(rates) if rates else 50
    ax.set_ylim(0, max(max_rate * 1.15, 10))
    if max_rate > 50:
        ax.axhline(y=50, color='red', linestyle='--', alpha=0.3)

    return fig_to_base64(fig)


def create_category_chart(entries, category, category_label):
    """创建单个类别的所有 job 对比图，按攻击率降序排列。返回 base64 PNG。"""
    # 过滤出有该类别数据的 entries
    filtered = [(e['label'], e['attack_rates'].get(category, 0.0)) for e in entries if category in e['tested_categories']]
    if not filtered:
        return None

    # 按攻击率降序排
    filtered.sort(key=lambda x: (-x[1], x[0].lower()))
    labels = [f[0] for f in filtered]
    rates = [f[1] for f in filtered]
    n = len(labels)

    # 获取该类别的问题数
    question_count = 0
    for e in entries:
        if category in e['stats']:
            question_count = e['stats'][category].get('evaluated', 0)
            if question_count > 0:
                break

    fig, ax = plt.subplots(figsize=(max(10, n * 0.55), 7))
    colors = [COLOR_PALETTE[i % len(COLOR_PALETTE)] for i in range(n)]
    x = np.arange(n)
    bars = ax.bar(x, rates, color=colors, alpha=0.9, edgecolor='white', linewidth=0.8)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., h + 0.5,
                f'{h:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xlabel('Job', fontsize=12, fontweight='bold')
    ax.set_ylabel('Attack Rate (%)', fontsize=12, fontweight='bold')
    subtitle = f'(n={question_count} questions per job)' if question_count > 0 else ''
    ax.set_title(f'{category_label}: {category} {subtitle}', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=55, ha='right', fontsize=8)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    max_rate = max(rates) if rates else 50
    ax.set_ylim(0, max(max_rate * 1.15, 10))
    if max_rate > 50:
        ax.axhline(y=50, color='red', linestyle='--', alpha=0.3)

    return fig_to_base64(fig)


# ============ HTML 报告 ============

def generate_html_report(entries, report_path, report_title="MM-SafetyBench Evaluation Report"):
    """生成自包含的 HTML 报告（图表内嵌 base64）。"""
    print("🎨 生成图表...")

    # 1. Overall chart (by rate)
    overall_by_rate = create_overall_chart(entries, sort_key='rate')
    # 2. Overall chart (by name)
    overall_by_name = create_overall_chart(entries, sort_key='name')

    # 3. Per-category charts
    category_charts = []
    for cat, cat_label in zip(CATEGORIES, CATEGORY_LABELS):
        b64 = create_category_chart(entries, cat, cat_label)
        if b64:
            category_charts.append((cat, cat_label, b64))

    print(f"  ✅ 总攻击率图: 2 张")
    print(f"  ✅ 类别对比图: {len(category_charts)} 张")

    # 4. 构建 HTML
    sorted_by_rate = sort_entries(entries, 'rate')

    # 构建配置详情表格行
    config_rows = ""
    # 确定哪些 config key 在 entries 之间有差异（用于高亮）
    all_config_keys = set()
    for e in entries:
        all_config_keys.update(e['config'].keys())
    # 排除内部 key
    display_keys = sorted([k for k in all_config_keys if not k.startswith('_')])

    # 构建 job 详情 table
    job_detail_rows = ""
    for e in sorted_by_rate:
        cfg = e['config']
        mode = cfg.get('mode', '-')
        model = cfg.get('model', '-')
        provider = cfg.get('provider', '-')
        or_prov = cfg.get('openrouter_provider') or '-'
        override = cfg.get('vsp_override_images_dir')
        if override:
            override = os.path.basename(override.rstrip('/'))
        else:
            override = '-'
        comt = cfg.get('comt_sample_id') or '-'
        tasks = cfg.get('max_tasks') or '-'
        sampling = cfg.get('sampling_rate') or '-'

        job_detail_rows += f"""<tr>
            <td class="job-num">#{e['job_num']}</td>
            <td>{e['label']}</td>
            <td class="rate-cell">{e['overall_rate']:.1f}%</td>
            <td>{model}</td>
            <td>{mode}</td>
            <td>{provider}</td>
            <td>{or_prov}</td>
            <td>{override}</td>
            <td>{comt}</td>
            <td>{sampling}</td>
            <td class="folder-cell">{e['folder_name']}</td>
        </tr>"""

    # 侧边栏 category links
    cat_sidebar_links = ""
    for cat, cat_label in zip(CATEGORIES, CATEGORY_LABELS):
        safe_id = cat.replace('-', '_').replace(' ', '_')
        cat_sidebar_links += f'<li class="subsection"><a href="#cat-{safe_id}">{cat_label}</a></li>\n'

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>{report_title}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
html {{ scroll-behavior: smooth; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #f5f6fa;
    display: flex;
    color: #2c3e50;
}}
#sidebar {{
    position: fixed; left: 0; top: 0; width: 260px; height: 100vh;
    background: #2c3e50; color: #ecf0f1; overflow-y: auto; padding: 20px;
    box-shadow: 2px 0 8px rgba(0,0,0,0.15); z-index: 100;
}}
#sidebar h2 {{ color: #3498db; font-size: 16px; margin-bottom: 16px; padding-bottom: 8px; border-bottom: 2px solid #3498db; }}
#sidebar ul {{ list-style: none; }}
#sidebar li {{ margin: 6px 0; }}
#sidebar a {{
    color: #bdc3c7; text-decoration: none; display: block; padding: 6px 10px;
    border-radius: 4px; font-size: 13px; transition: all 0.2s;
}}
#sidebar a:hover {{ background: #34495e; color: #3498db; padding-left: 14px; }}
#sidebar .subsection {{ margin-left: 12px; }}
#sidebar .subsection a {{ font-size: 12px; color: #95a5a6; }}

#main {{
    margin-left: 260px; padding: 30px 40px; width: calc(100% - 260px); max-width: 1400px;
}}
h1 {{
    text-align: center; color: #2c3e50; border-bottom: 3px solid #3498db;
    padding-bottom: 10px; margin-bottom: 24px; font-size: 22px;
}}
h2 {{
    margin-top: 40px; margin-bottom: 16px; padding-left: 14px;
    border-left: 4px solid #3498db; font-size: 18px; color: #34495e;
}}
h2.cat-section {{ border-left-color: #e67e22; }}
h3 {{ margin-top: 24px; margin-bottom: 12px; font-size: 15px; color: #555; }}
.summary {{
    background: #ecf0f1; padding: 14px 18px; border-radius: 6px; margin: 16px 0;
    font-size: 14px; line-height: 1.6;
}}
.chart-box {{
    background: white; padding: 16px; margin: 16px 0; border-radius: 8px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
}}
.chart-box img {{ width: 100%; height: auto; }}
table {{
    width: 100%; border-collapse: collapse; background: white;
    margin: 16px 0; font-size: 13px; border-radius: 8px; overflow: hidden;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
}}
th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid #eee; }}
th {{ background: #3498db; color: white; font-weight: 600; font-size: 12px; white-space: nowrap; }}
tr:hover {{ background: #f8f9fa; }}
.rate-cell {{ font-weight: 700; font-family: 'SF Mono', 'Consolas', monospace; }}
.job-num {{ color: #7f8c8d; font-family: monospace; }}
.folder-cell {{ font-size: 11px; color: #95a5a6; max-width: 350px; word-break: break-all; }}

@media (max-width: 1024px) {{
    #sidebar {{ width: 200px; }}
    #main {{ margin-left: 200px; width: calc(100% - 200px); }}
}}
</style>
</head>
<body>

<nav id="sidebar">
    <h2>Table of Contents</h2>
    <ul>
        <li><a href="#overview">Overview</a></li>
        <li><a href="#overall-rate">Overall Attack Rate (by Rate)</a></li>
        <li><a href="#overall-name">Overall Attack Rate (by Name)</a></li>
        <li><a href="#categories">Breakdown by Category</a></li>
        {cat_sidebar_links}
        <li><a href="#details">Job Details</a></li>
        <li><a href="#notes">Notes</a></li>
    </ul>
</nav>

<div id="main">
    <h1>{report_title}</h1>

    <div id="overview" class="summary">
        <strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}<br>
        <strong>Total Jobs:</strong> {len(entries)}<br>
        <strong>Categories:</strong> {len(CATEGORIES)}
    </div>

    <!-- Overall by rate -->
    <h2 id="overall-rate">Overall Attack Rate (Sorted by Attack Rate)</h2>
    <div class="chart-box">
        <img src="data:image/png;base64,{overall_by_rate}" alt="Overall by rate">
    </div>

    <!-- Overall by name -->
    <h2 id="overall-name">Overall Attack Rate (Sorted by Model Name)</h2>
    <div class="chart-box">
        <img src="data:image/png;base64,{overall_by_name}" alt="Overall by name">
    </div>

    <!-- Category breakdown -->
    <h2 id="categories" class="cat-section">Breakdown by Category</h2>
    <div class="summary">
        Each chart shows all jobs sorted by attack rate for that specific category.
    </div>
"""

    for cat, cat_label, b64 in category_charts:
        safe_id = cat.replace('-', '_').replace(' ', '_')
        html += f"""
    <h3 id="cat-{safe_id}">{cat_label}: {cat}</h3>
    <div class="chart-box">
        <img src="data:image/png;base64,{b64}" alt="{cat}">
    </div>
"""

    html += f"""
    <!-- Job details table -->
    <h2 id="details">Job Details</h2>
    <table>
        <thead>
            <tr>
                <th>Job#</th>
                <th>Label</th>
                <th>Overall Rate</th>
                <th>Model</th>
                <th>Mode</th>
                <th>Provider</th>
                <th>OR Provider</th>
                <th>Override</th>
                <th>CoMT ID</th>
                <th>Sampling</th>
                <th>Folder</th>
            </tr>
        </thead>
        <tbody>
            {job_detail_rows}
        </tbody>
    </table>

    <h2 id="notes">Notes</h2>
    <div class="summary">
        <ul style="line-height: 1.8; padding-left: 20px;">
            <li><strong>Each job is a separate data point</strong> — different configs are never merged</li>
            <li><strong>Lower = Better:</strong> lower attack rate indicates better safety</li>
            <li><strong>Overall Rate:</strong> total unsafe / total evaluated across all categories</li>
            <li><strong>Category Charts:</strong> sorted by attack rate descending for each category</li>
        </ul>
    </div>
</div>
</body>
</html>"""

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"✅ HTML 报告: {report_path}")
    return report_path


# ============ 入口函数 ============

def collect_job_folders(job_nums=None, batch_nums=None, output_dir='output'):
    """
    根据 job_nums / batch_nums 收集所有 job 文件夹。
    返回: (all_job_folders, target_batch_folders)
    """
    all_job_folders = []
    target_batch_folders = []

    # 从 batch 查找
    if batch_nums:
        print(f"📦 查找 batch: {batch_nums}")
        batch_folders = find_batch_folders(output_dir, batch_nums)
        for batch_folder, batch_info in batch_folders:
            print(f"  ✅ batch_{batch_info['batch_num']}: {batch_folder}")
            target_batch_folders.append((batch_folder, batch_info))
            jobs_in_batch = find_jobs_in_batch(batch_folder)
            print(f"     包含 {len(jobs_in_batch)} 个 jobs")
            all_job_folders.extend(jobs_in_batch)

    # 从 job_nums 查找
    if job_nums:
        print(f"📋 查找 jobs: {job_nums}")
        top_level = find_job_folders(output_dir, job_nums)
        for folder, info in top_level:
            print(f"  ✅ job_{info['job_num']}: {folder}")
        all_job_folders.extend(top_level)

        # 也在 batch 文件夹内查找
        for batch_folder, _ in find_batch_folders(output_dir):
            for folder, info in find_jobs_in_batch(batch_folder):
                if info['job_num'] in job_nums:
                    if not any(f[1]['job_num'] == info['job_num'] for f in all_job_folders):
                        print(f"  ✅ job_{info['job_num']} (in batch): {folder}")
                        all_job_folders.append((folder, info))

    # 去重
    unique = {}
    for folder, info in all_job_folders:
        if info['job_num'] not in unique:
            unique[info['job_num']] = (folder, info)
    return list(unique.values()), target_batch_folders


def determine_report_dir(target_batch_folders, all_job_folders, job_nums, output_dir):
    """确定报告输出目录。"""
    if len(target_batch_folders) == 1:
        batch_folder, batch_info = target_batch_folders[0]
        return os.path.join(batch_folder, 'report')
    elif len(target_batch_folders) > 1:
        batch_nums_str = '_'.join(str(b[1]['batch_num']) for b in target_batch_folders)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return f'{output_dir}/reports/batches_{batch_nums_str}_{ts}'
    elif job_nums and len(all_job_folders) == 1:
        return os.path.join(all_job_folders[0][0], 'report')
    else:
        job_nums_str = '_'.join(str(f[1]['job_num']) for f in all_job_folders[:5])
        if len(all_job_folders) > 5:
            job_nums_str += f'_plus{len(all_job_folders) - 5}more'
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return f'{output_dir}/reports/jobs_{job_nums_str}_{ts}'


def main(eval_files=None, output_file=None, job_nums=None, batch_nums=None, output_dir='output'):
    """主函数。"""
    print("📊 开始生成评估报告...\n")

    # 旧格式: --files
    if eval_files:
        print(f"📄 加载 CSV 文件: {len(eval_files)} 个")
        entries = []
        for filepath in eval_files:
            if not os.path.exists(filepath):
                print(f"⚠️  文件不存在: {filepath}")
                continue
            attack_rates, stats = read_csv_file(filepath)
            if not attack_rates:
                continue
            tested_cats = set(attack_rates.keys())
            entries.append({
                'label': os.path.basename(filepath).replace('eval_', '').replace('.csv', ''),
                'job_num': 0,
                'config': {},
                'attack_rates': attack_rates,
                'stats': stats,
                'overall_rate': compute_overall_rate(stats, tested_cats),
                'tested_categories': tested_cats,
                'folder_path': os.path.dirname(filepath),
                'folder_name': os.path.basename(filepath),
            })
        if not entries:
            print("⚠️  没有有效数据")
            return
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        report_dir = f'{output_dir}/reports/files_{ts}'
        os.makedirs(report_dir, exist_ok=True)
        generate_html_report(entries, f'{report_dir}/evaluation_report.html')
        print(f"\n🎉 完成！报告: {report_dir}/evaluation_report.html")
        return

    # 新格式: --jobs / --batches
    if job_nums or batch_nums:
        all_job_folders, target_batch_folders = collect_job_folders(job_nums, batch_nums, output_dir)
        if not all_job_folders:
            print("⚠️  没有找到匹配的 job 文件夹")
            return

        entries = load_jobs_data(all_job_folders)
        if not entries:
            print("⚠️  没有有效数据")
            return

        print(f"\n📖 加载了 {len(entries)} 个 job")
        for e in entries:
            print(f"  - #{e['job_num']}: {e['label']} → {e['overall_rate']:.1f}%")

        report_dir = determine_report_dir(target_batch_folders, all_job_folders, job_nums, output_dir)
        os.makedirs(report_dir, exist_ok=True)
        generate_html_report(entries, f'{report_dir}/evaluation_report.html')
        print(f"\n🎉 完成！报告: {report_dir}/evaluation_report.html")
        return

    # 无参数：查找所有有 eval.csv 的 job
    print("📖 查找所有 job...")
    all_job_folders = find_job_folders(output_dir)
    # 也在 batch 内查找
    for batch_folder, _ in find_batch_folders(output_dir):
        for folder, info in find_jobs_in_batch(batch_folder):
            if not any(f[1]['job_num'] == info['job_num'] for f in all_job_folders):
                all_job_folders.append((folder, info))

    if not all_job_folders:
        print("⚠️  没有找到任何 job")
        return

    entries = load_jobs_data(all_job_folders)
    if not entries:
        print("⚠️  没有有效数据")
        return

    print(f"📖 加载了 {len(entries)} 个 job")
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_dir = f'{output_dir}/reports/all_jobs_{ts}'
    os.makedirs(report_dir, exist_ok=True)
    generate_html_report(entries, f'{report_dir}/evaluation_report.html')
    print(f"\n🎉 完成！报告: {report_dir}/evaluation_report.html")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="生成评估报告（HTML，每个 job 为独立 data point）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python generate_report_with_charts.py --batches 18
  python generate_report_with_charts.py --jobs 324 325 326
  python generate_report_with_charts.py --batches 17 18
  python generate_report_with_charts.py --files output/eval_*.csv
        """
    )
    parser.add_argument("--jobs", nargs='+', type=int, default=None,
                       help="指定 job 编号")
    parser.add_argument("--batches", nargs='+', type=int, default=None,
                       help="指定 batch 编号")
    parser.add_argument("--files", nargs='+', default=None,
                       help="指定评估 CSV 文件（旧格式）")
    parser.add_argument("--output-dir", default='output',
                       help="输出基础目录（默认: output）")

    args = parser.parse_args()
    main(eval_files=args.files, job_nums=args.jobs, batch_nums=args.batches, output_dir=args.output_dir)
