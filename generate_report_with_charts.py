"""
生成包含图表的评估报告

对每个模型系列，生成柱状图显示不同类别的攻击率

支持新的输出格式：
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
from collections import defaultdict
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# 设置中文字体支持
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
    'Illegal',
    'Hate',
    'Malware',
    'Physical',
    'Economic',
    'Fraud',
    'Sex',
    'Political',
    'Privacy',
    'Legal',
    'Financial',
    'Health',
    'Gov'
]

# ============ 新的文件夹解析函数 ============

def parse_job_folder_name(folder_name):
    """
    从 job 文件夹名称解析信息
    格式: job_{job_num}_tasks_{num}_{provider}_{model}_{timestamp}
    例如: job_154_tasks_202_Openrouter_qwen_qwen3-vl-235b-a22b-instruct_0123_222923
    
    Returns:
        dict: {job_num, tasks, provider, model, timestamp} 或 None 如果解析失败
    """
    # 匹配模式: job_数字_tasks_数字_提供者_模型_时间戳
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
    """
    从 batch 文件夹名称解析信息
    格式: batch_{batch_num}_{timestamp}
    例如: batch_4_0123_222923
    
    Returns:
        dict: {batch_num, timestamp} 或 None 如果解析失败
    """
    pattern = r'^batch_(\d+)_(\d{4}_\d{6})$'
    match = re.match(pattern, folder_name)
    
    if match:
        return {
            'batch_num': int(match.group(1)),
            'timestamp': match.group(2)
        }
    return None


def find_job_folders(output_dir='output', job_nums=None):
    """
    查找 job 文件夹
    
    Args:
        output_dir: 输出目录
        job_nums: 指定的 job 编号列表，None 表示查找所有
        
    Returns:
        list: [(folder_path, job_info), ...]
    """
    job_folders = []
    
    if not os.path.exists(output_dir):
        return job_folders
    
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if not os.path.isdir(item_path):
            continue
        
        # 解析 job 文件夹
        job_info = parse_job_folder_name(item)
        if job_info:
            # 检查是否有 eval.csv
            eval_csv = os.path.join(item_path, 'eval.csv')
            if os.path.exists(eval_csv):
                if job_nums is None or job_info['job_num'] in job_nums:
                    job_folders.append((item_path, job_info))
    
    return job_folders


def find_batch_folders(output_dir='output', batch_nums=None):
    """
    查找 batch 文件夹
    
    Args:
        output_dir: 输出目录
        batch_nums: 指定的 batch 编号列表，None 表示查找所有
        
    Returns:
        list: [(folder_path, batch_info), ...]
    """
    batch_folders = []
    
    if not os.path.exists(output_dir):
        return batch_folders
    
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if not os.path.isdir(item_path):
            continue
        
        # 解析 batch 文件夹
        batch_info = parse_batch_folder_name(item)
        if batch_info:
            if batch_nums is None or batch_info['batch_num'] in batch_nums:
                batch_folders.append((item_path, batch_info))
    
    return batch_folders


def find_jobs_in_batch(batch_folder):
    """
    查找 batch 文件夹内的所有 job 文件夹
    
    Args:
        batch_folder: batch 文件夹路径
        
    Returns:
        list: [(folder_path, job_info), ...]
    """
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


def get_postproc_info_from_job(job_folder):
    """
    从 job 文件夹中读取后处理信息
    
    Args:
        job_folder: job 文件夹路径
        
    Returns:
        dict: {fallback_backend, fallback_method} 或 None
    """
    prebaked_file = os.path.join(job_folder, 'prebaked_report_data.json')
    
    if not os.path.exists(prebaked_file):
        return None
    
    try:
        with open(prebaked_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not data or not isinstance(data, list) or len(data) == 0:
            return None
        
        # 取第一条记录的后处理信息（假设整个 job 使用相同的后处理配置）
        first_record = data[0]
        return {
            'fallback_backend': first_record.get('fallback_backend'),
            'fallback_method': first_record.get('fallback_method'),
            'comt_sample_id': first_record.get('comt_sample_id')
        }
    except Exception as e:
        print(f"⚠️  读取后处理信息失败 {prebaked_file}: {e}")
        return None


def format_postproc_suffix(postproc_info):
    """
    格式化后处理后缀
    
    Args:
        postproc_info: {fallback_backend, fallback_method} 或 None
        
    Returns:
        str: 后处理后缀，如 "(visual_mask)" 或 "(sd-good)"
    """
    if not postproc_info:
        return ""
    
    backend = postproc_info.get('fallback_backend')
    method = postproc_info.get('fallback_method')
    
    if not backend or not method:
        return ""
    
    # 格式化后缀
    if backend == 'ask':
        # ask backend 直接使用 method 名称
        return f" ({method})"
    elif backend == 'sd':
        # sd backend 使用 sd-method 格式
        return f" (sd-{method})"
    else:
        return f" ({backend}-{method})"


def parse_model_info_from_job(job_info, job_folder=None):
    """
    从 job_info 中提取模型显示名称和品牌
    
    Args:
        job_info: parse_job_folder_name 返回的字典
        job_folder: job 文件夹路径（用于读取后处理信息）
        
    Returns:
        (brand, model_display_name)
    """
    provider = job_info['provider'].lower()
    model = job_info['model'].lower()
    
    # 检查是否是 CoMT/VSP 或普通请求
    is_comt_vsp = 'comtvsp' in provider or provider == 'comt_vsp'
    is_vsp = 'vsp' in provider and not is_comt_vsp
    
    # 获取后处理信息
    postproc_info = None
    if job_folder and is_comt_vsp:
        postproc_info = get_postproc_info_from_job(job_folder)
    
    # 构建 VSP 后缀
    postproc_suffix = format_postproc_suffix(postproc_info)
    vsp_suffix = ' + CoMT/VSP' + postproc_suffix if is_comt_vsp else (' + VSP' if is_vsp else '')
    
    # 根据模型名称确定品牌和显示名称
    if 'gemini' in model:
        brand = 'Gemini'
        if '2.5-flash' in model or '2-5-flash' in model:
            model_display_name = 'Gemini-2.5-Flash' + vsp_suffix
        elif '2.0-flash' in model or '2-0-flash' in model:
            model_display_name = 'Gemini-2.0-Flash' + vsp_suffix
        else:
            model_display_name = 'Gemini' + vsp_suffix
    
    elif 'gpt-5' in model or 'gpt5' in model:
        brand = 'OpenAI'
        model_display_name = 'GPT-5' + vsp_suffix
    
    elif 'gpt-4' in model or 'gpt4' in model:
        brand = 'OpenAI'
        if 'vision' in model or 'turbo' in model:
            model_display_name = 'GPT-4-Vision' + vsp_suffix
        else:
            model_display_name = 'GPT-4' + vsp_suffix
    
    elif 'qwen' in model:
        # 检查是否是 Thinking 模式
        is_thinking = 'thinking' in model
        
        if is_thinking:
            brand = 'Qwen (Thinking)'
        else:
            brand = 'Qwen'
        
        # 区分不同的 Qwen 模型
        if 'qwen3-vl-235b' in model:
            base_name = 'Qwen3-VL-235B-Thinking' if is_thinking else 'Qwen3-VL-235B-Instruct'
        elif 'qwen3-vl-30b' in model:
            base_name = 'Qwen3-VL-30B-Thinking' if is_thinking else 'Qwen3-VL-30B-Instruct'
        elif 'qwen3-vl-8b' in model:
            base_name = 'Qwen3-VL-8B-Thinking' if is_thinking else 'Qwen3-VL-8B-Instruct'
        elif 'qwen2.5-vl' in model or 'qwen2-5-vl' in model:
            if '72b' in model:
                base_name = 'Qwen2.5-VL-72B'
            elif '7b' in model:
                base_name = 'Qwen2.5-VL-7B'
            else:
                base_name = 'Qwen2.5-VL'
        else:
            base_name = 'Qwen-VL (Unknown)'
        
        model_display_name = base_name + vsp_suffix
    
    elif 'internvl' in model:
        brand = 'InternVL'
        if '78b' in model:
            model_display_name = 'InternVL3-78B' + vsp_suffix
        elif '38b' in model:
            model_display_name = 'InternVL3-38B' + vsp_suffix
        else:
            model_display_name = 'InternVL' + vsp_suffix
    
    elif 'mistral' in model or 'ministral' in model:
        brand = 'Mistral'
        if 'ministral-14b' in model:
            model_display_name = 'Ministral-14B' + vsp_suffix
        elif 'ministral-8b' in model:
            model_display_name = 'Ministral-8B' + vsp_suffix
        else:
            model_display_name = 'Mistral' + vsp_suffix
    
    elif 'claude' in model:
        brand = 'Anthropic'
        if 'claude-3' in model:
            if 'opus' in model:
                model_display_name = 'Claude-3-Opus' + vsp_suffix
            elif 'sonnet' in model:
                model_display_name = 'Claude-3-Sonnet' + vsp_suffix
            elif 'haiku' in model:
                model_display_name = 'Claude-3-Haiku' + vsp_suffix
            else:
                model_display_name = 'Claude-3' + vsp_suffix
        else:
            model_display_name = 'Claude' + vsp_suffix
    
    elif 'llama' in model:
        brand = 'Meta'
        if '3.2' in model or '3-2' in model:
            if '90b' in model:
                model_display_name = 'Llama-3.2-90B' + vsp_suffix
            elif '11b' in model:
                model_display_name = 'Llama-3.2-11B' + vsp_suffix
            else:
                model_display_name = 'Llama-3.2' + vsp_suffix
        else:
            model_display_name = 'Llama' + vsp_suffix
    
    else:
        brand = 'Other'
        # 清理模型名称作为显示名
        clean_model = job_info['model'].replace('_', '-').replace('/', '-')
        model_display_name = clean_model + vsp_suffix
    
    return brand, model_display_name


def load_data_from_jobs(job_folders):
    """
    从 job 文件夹列表加载评估数据
    
    Args:
        job_folders: [(folder_path, job_info), ...]
        
    Returns:
        all_data: {brand: [{model_display_name, timestamp, data, stats, filename, job_info}, ...]}
    """
    all_data = defaultdict(list)
    
    for folder_path, job_info in job_folders:
        eval_csv = os.path.join(folder_path, 'eval.csv')
        
        if not os.path.exists(eval_csv):
            print(f"⚠️  eval.csv 不存在: {folder_path}")
            continue
        
        # 读取 CSV 数据
        attack_rates, stats = read_csv_file(eval_csv)
        
        if not attack_rates:
            print(f"⚠️  无法读取数据: {eval_csv}")
            continue
        
        # 从 job_info 中提取模型信息（传入 folder_path 以读取后处理信息）
        brand, model_display_name = parse_model_info_from_job(job_info, job_folder=folder_path)
        
        all_data[brand].append({
            'model_display_name': model_display_name,
            'timestamp': job_info['timestamp'],
            'data': attack_rates,
            'stats': stats,
            'filename': os.path.basename(folder_path),
            'job_info': job_info,
            'folder_path': folder_path
        })
    
    return all_data


def read_csv_file(filepath):
    """
    读取 CSV 文件并返回数据
    
    Returns:
        tuple: (attack_rates, stats)
        - attack_rates: {category: attack_rate}
        - stats: {category: {'evaluated': int, 'unsafe': int, 'safe': int}}
    """
    attack_rates = {}
    stats = {}
    
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 处理不同的列名格式
            category = row.get('Category') or row.get('category')
            
            if not category:
                continue
            
            # 提取攻击率
            # 两种格式：
            # 1. Attack_Rate(%) - 已经是百分比（6.19 = 6.19%）
            # 2. attack_rate - 是小数（0.0206 = 2.06%，需要×100）
            if 'Attack_Rate(%)' in row:
                attack_rate_str = row['Attack_Rate(%)']
                is_percentage = True
            elif 'attack_rate' in row:
                attack_rate_str = row['attack_rate']
                is_percentage = False  # 小数格式，需要×100
            else:
                continue
            
            if not attack_rate_str:
                continue
            
            try:
                attack_rate = float(str(attack_rate_str).replace('%', '').strip())
                # 如果是小数格式，转换为百分比
                if not is_percentage and attack_rate < 1.0:
                    attack_rate *= 100
            except ValueError:
                print(f"⚠️  无法解析攻击率: {filepath}, {category}, {attack_rate_str}")
                attack_rate = 0.0
            
            attack_rates[category] = attack_rate
            
            # 提取统计数据（Evaluated, Unsafe, Safe）
            try:
                evaluated = int(row.get('Evaluated', 0))
                unsafe = int(row.get('Unsafe', 0))
                safe = int(row.get('Safe', 0))
                
                stats[category] = {
                    'evaluated': evaluated,
                    'unsafe': unsafe,
                    'safe': safe
                }
            except (ValueError, TypeError):
                # 如果无法解析统计数据，使用默认值
                stats[category] = {
                    'evaluated': 0,
                    'unsafe': 0,
                    'safe': 0
                }
    
    return attack_rates, stats

def parse_filename(filename):
    """
    从文件名提取模型信息和时间
    例如: eval_qwen_qwen3-vl-235b-a22b-thinking_2025-11-16_08-06-28_tasks_1680.csv
    返回: (brand, model_display_name, timestamp)
    """
    # 移除 eval_ 前缀和 .csv 后缀
    name = filename.replace('eval_', '').replace('.csv', '')
    
    # 提取时间戳（如果存在）
    parts = name.split('_')
    timestamp = None
    if 'tasks' in name:
        # 找到包含日期的部分
        for i, part in enumerate(parts):
            if '-' in part and len(part) == 10:  # 日期格式 YYYY-MM-DD
                timestamp = f"{parts[i]}_{parts[i+1]}"
                break
    
    # 检查是否使用了 CoMT/VSP（comt_vsp）或普通 VSP
    is_comt_vsp = 'comt_vsp' in name
    is_vsp = 'vsp' in name and not is_comt_vsp
    vsp_suffix = ' + CoMT/VSP' if is_comt_vsp else (' + VSP' if is_vsp else '')
    
    # 识别品牌和生成显示名称
    if 'google_gemini' in name or 'gemini' in name:
        brand = 'Gemini'
        if name.startswith('mini_'):
            model_display_name = 'Gemini-2.5-Flash (Mini-Eval)'
        else:
            model_display_name = 'Gemini-2.5-Flash' + vsp_suffix
    
    elif 'gpt-5' in name or 'gpt5' in name:
        brand = 'OpenAI'
        model_display_name = 'GPT-5' + vsp_suffix
    
    elif 'qwen' in name:
        # 检查是否是 Thinking 模式
        is_thinking = 'thinking' in name
        
        # 根据 Thinking 分组
        if is_thinking:
            brand = 'Qwen (Thinking)'
        else:
            brand = 'Qwen'
        
        # 区分不同的Qwen模型
        if 'qwen3-vl-235b' in name or 'qwen_qwen3-vl-235b' in name:
            if is_thinking:
                base_name = 'Qwen3-VL-235B-Thinking'
            else:
                base_name = 'Qwen3-VL-235B-Instruct'
        elif 'qwen3-vl-30b' in name or 'qwen_qwen3-vl-30b' in name:
            if is_thinking:
                base_name = 'Qwen3-VL-30B-Thinking'
            else:
                base_name = 'Qwen3-VL-30B-Instruct'
        elif 'qwen3-vl-8b' in name or 'qwen_qwen3-vl-8b' in name:
            if is_thinking:
                base_name = 'Qwen3-VL-8B-Thinking'
            else:
                base_name = 'Qwen3-VL-8B-Instruct'
        else:
            base_name = 'Qwen3-VL (Unknown)'
        
        model_display_name = base_name + vsp_suffix
    
    elif 'internvl' in name:
        brand = 'InternVL'
        model_display_name = 'InternVL3-78B' + vsp_suffix
    
    elif 'mistralai' in name or 'ministral' in name:
        brand = 'Mistral'
        if 'ministral-14b' in name or 'ministral-8b' in name:
            # 提取具体的模型大小
            if 'ministral-14b' in name:
                model_display_name = 'Ministral-14B' + vsp_suffix
            else:
                model_display_name = 'Ministral-8B' + vsp_suffix
        elif 'ministral' in name:
            model_display_name = 'Ministral' + vsp_suffix
        else:
            model_display_name = 'Mistral (Unknown)' + vsp_suffix
    
    elif 'comt_vsp' in name or 'vsp' in name:
        brand = 'VSP'
        model_display_name = 'VSP (Unknown Model)'
    
    else:
        brand = 'Other'
        model_display_name = 'Unknown Model'
    
    return brand, model_display_name, timestamp

def load_all_data():
    """加载所有 1680 任务的评估数据，按品牌分组"""
    output_dir = 'output'
    all_data = defaultdict(list)  # {brand: [(model_display_name, timestamp, data, stats), ...]}
    
    for filename in os.listdir(output_dir):
        if filename.startswith('eval_') and filename.endswith('.csv') and 'tasks_1680' in filename:
            filepath = os.path.join(output_dir, filename)
            
            # 解析文件名
            brand, model_display_name, timestamp = parse_filename(filename)
            
            # 读取数据（现在返回 attack_rates 和 stats）
            attack_rates, stats = read_csv_file(filepath)
            
            all_data[brand].append({
                'model_display_name': model_display_name,
                'timestamp': timestamp,
                'data': attack_rates,
                'stats': stats,
                'filename': filename
            })
    
    # 同时检查没有 tasks_1680 标记但是有 14 行（13个类别+表头）或 15 行（+空行）的文件
    for filename in os.listdir(output_dir):
        if filename.startswith('eval_') and filename.endswith('.csv') and 'tasks_1680' not in filename:
            filepath = os.path.join(output_dir, filename)
            
            # 检查行数
            with open(filepath, 'r') as f:
                line_count = sum(1 for _ in f)
            
            if line_count == 14 or line_count == 15:  # 13 类别 + 1 表头 (+ 可能的空行)
                brand, model_display_name, timestamp = parse_filename(filename)
                attack_rates, stats = read_csv_file(filepath)
                
                all_data[brand].append({
                    'model_display_name': model_display_name,
                    'timestamp': timestamp,
                    'data': attack_rates,
                    'stats': stats,
                    'filename': filename
                })
    
    return all_data

def average_multiple_runs(models_data):
    """
    对同一模型的多次运行取平均值
    
    Args:
        models_data: [{model_display_name, timestamp, data, stats}, ...]
    
    Returns:
        averaged_data: {display_name: {category: avg_attack_rate}}
        tested_categories: {display_name: set of actually tested categories}
        averaged_stats: {display_name: {category: {'evaluated': int, 'unsafe': int, 'safe': int}}}
    """
    # 按完整模型名分组
    model_groups = defaultdict(list)
    
    for item in models_data:
        # 直接使用model_display_name作为分组键
        model_display_name = item['model_display_name']
        model_groups[model_display_name].append(item)
    
    # 计算平均值（如果同一个模型有多次运行，取平均）
    averaged_data = {}
    tested_categories = {}  # 记录每个模型实际测试了哪些类别
    averaged_stats = {}  # 统计数据（加总）
    
    for model_name, items in model_groups.items():
        averaged_data[model_name] = {}
        tested_categories[model_name] = set()
        averaged_stats[model_name] = {}
        
        for category in CATEGORIES:
            rates = []
            stats_list = []
            
            for item in items:
                if category in item['data']:
                    rates.append(item['data'][category])
                    if 'stats' in item and category in item['stats']:
                        stats_list.append(item['stats'][category])
            
            if rates:
                averaged_data[model_name][category] = np.mean(rates)
                tested_categories[model_name].add(category)  # 记录实际测试的类别
                
                # 统计数据取平均（对于多次运行）
                if stats_list:
                    averaged_stats[model_name][category] = {
                        'evaluated': int(np.mean([s['evaluated'] for s in stats_list])),
                        'unsafe': int(np.mean([s['unsafe'] for s in stats_list])),
                        'safe': int(np.mean([s['safe'] for s in stats_list]))
                    }
                else:
                    averaged_stats[model_name][category] = {
                        'evaluated': 0,
                        'unsafe': 0,
                        'safe': 0
                    }
            else:
                averaged_data[model_name][category] = 0.0
                averaged_stats[model_name][category] = {
                    'evaluated': 0,
                    'unsafe': 0,
                    'safe': 0
                }
    
    return averaged_data, tested_categories, averaged_stats

def create_bar_chart(brand, averaged_data, output_file, tested_categories=None):
    """
    创建柱状图
    
    Args:
        brand: 品牌名称（如 Qwen, Gemini）
        averaged_data: {model_name: {category: attack_rate}}
        output_file: 输出文件路径
        tested_categories: {model_name: set of tested categories}，如果为 None 则显示所有类别
    """
    # 确定要显示的类别（取所有模型测试过的类别的并集）
    if tested_categories:
        all_tested = set()
        for model_cats in tested_categories.values():
            all_tested.update(model_cats)
        # 按 CATEGORIES 的顺序过滤
        display_categories = [cat for cat in CATEGORIES if cat in all_tested]
        display_labels = [CATEGORY_LABELS[CATEGORIES.index(cat)] for cat in display_categories]
    else:
        display_categories = CATEGORIES
        display_labels = CATEGORY_LABELS
    
    # 如果没有任何类别，返回
    if not display_categories:
        print(f"⚠️  没有可显示的类别，跳过图表: {output_file}")
        return
    
    # 根据类别数量调整图表宽度
    fig_width = max(8, len(display_categories) * 1.2)
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    
    # 准备数据 - 按模型名排序
    variants = sorted(list(averaged_data.keys()))
    x = np.arange(len(display_categories))
    width = 0.8 / len(variants) if len(variants) > 0 else 0.8
    
    # 使用高对比度的颜色方案
    # 定义专业的配色：蓝、红、绿、橙、紫、青、粉
    color_palette = [
        '#1f77b4',  # 深蓝
        '#ff7f0e',  # 橙色
        '#2ca02c',  # 绿色
        '#d62728',  # 红色
        '#9467bd',  # 紫色
        '#8c564b',  # 棕色
        '#e377c2',  # 粉色
        '#7f7f7f',  # 灰色
        '#17becf',  # 青色
        '#bcbd22',  # 黄绿
    ]
    # 根据需要循环使用颜色
    colors = [color_palette[i % len(color_palette)] for i in range(len(variants))]
    
    for i, (variant, color) in enumerate(zip(variants, colors)):
        data = averaged_data[variant]
        # 只取实际显示的类别的数据
        attack_rates = [data.get(cat, 0.0) for cat in display_categories]
        
        offset = (i - len(variants)/2 + 0.5) * width
        bars = ax.bar(x + offset, attack_rates, width, 
                     label=variant, color=color, alpha=0.9, edgecolor='white', linewidth=0.5)
        
        # 在所有柱子上显示数值（包括0）
        for bar in bars:
            height = bar.get_height()
            # 根据高度调整字体大小和位置
            if height > 5:
                fontsize = 7
                y_offset = 0.5
            elif height > 0:
                fontsize = 6
                y_offset = 1
            else:
                # 0 值也要标注
                fontsize = 6
                y_offset = 1
            
            ax.text(bar.get_x() + bar.get_width()/2., height + y_offset,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=fontsize, rotation=0)
    
    # 设置标签和标题
    ax.set_xlabel('Category', fontsize=12, fontweight='bold')
    ax.set_ylabel('Attack Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'{brand} Models - Attack Rate by Category', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, rotation=45, ha='right')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 105)
    
    # 添加水平参考线
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.3, linewidth=1)
    ax.text(len(display_categories)-0.5, 51, '50%', color='red', fontsize=9)
    
    # 保存图表
    plt.subplots_adjust(bottom=0.15, top=0.92, left=0.08, right=0.98)
    try:
        plt.savefig(output_file, dpi=100)  # 进一步降低 dpi
    except Exception as e:
        print(f"⚠️  生成图表失败 {output_file}: {e}")
    finally:
        plt.close()
    
    print(f"✅ 生成图表: {output_file}")

def calculate_overall_attack_rates(averaged_stats, tested_categories):
    """
    计算每个模型的总攻击率
    
    Args:
        averaged_stats: {model_name: {category: {'evaluated': int, 'unsafe': int, 'safe': int}}}
        tested_categories: {model_name: set of tested categories}
    
    Returns:
        {model_name: overall_attack_rate}
    """
    overall_rates = {}
    
    for model_name, stats in averaged_stats.items():
        tested_cats = tested_categories.get(model_name, set())
        
        total_evaluated = 0
        total_unsafe = 0
        
        for category in tested_cats:
            if category in stats:
                total_evaluated += stats[category]['evaluated']
                total_unsafe += stats[category]['unsafe']
        
        if total_evaluated > 0:
            overall_rates[model_name] = (total_unsafe / total_evaluated) * 100
        else:
            overall_rates[model_name] = 0.0
    
    return overall_rates

def create_overall_attack_rate_chart(brand, overall_rates, output_file):
    """
    创建总攻击率对比图
    
    Args:
        brand: 品牌名称
        overall_rates: {model_name: overall_attack_rate}
        output_file: 输出文件路径
    """
    if not overall_rates:
        print(f"⚠️  没有总攻击率数据，跳过图表: {output_file}")
        return
    
    # 准备数据 - 按模型名排序
    models = sorted(list(overall_rates.keys()))
    rates = [overall_rates[model] for model in models]
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(max(8, len(models) * 0.8), 6))
    
    # 使用高对比度的颜色
    color_palette = [
        '#1f77b4',  # 深蓝
        '#ff7f0e',  # 橙色
        '#2ca02c',  # 绿色
        '#d62728',  # 红色
        '#9467bd',  # 紫色
        '#8c564b',  # 棕色
        '#e377c2',  # 粉色
        '#7f7f7f',  # 灰色
        '#17becf',  # 青色
        '#bcbd22',  # 黄绿
    ]
    colors = [color_palette[i % len(color_palette)] for i in range(len(models))]
    
    # 绘制柱状图
    x = np.arange(len(models))
    bars = ax.bar(x, rates, color=colors, alpha=0.9, edgecolor='white', linewidth=0.5)
    
    # 在柱子上显示数值
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
               f'{height:.1f}%',
               ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 设置标签和标题
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Overall Attack Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'{brand} Models - Overall Attack Rate Comparison', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, max(rates) * 1.15 if rates else 100)
    
    # 添加水平参考线
    if max(rates) > 50:
        ax.axhline(y=50, color='red', linestyle='--', alpha=0.3, linewidth=1)
        ax.text(len(models)-0.5, 51, '50%', color='red', fontsize=9)
    
    # 保存图表
    plt.subplots_adjust(bottom=0.25, top=0.92, left=0.08, right=0.98)
    try:
        plt.savefig(output_file, dpi=100)
    except Exception as e:
        print(f"⚠️  生成图表失败 {output_file}: {e}")
    finally:
        plt.close()
    
    print(f"✅ 生成总攻击率图表: {output_file}")

def create_global_overall_chart(all_models_overall_rates, all_models_stats, output_file, sort_by='rate'):
    """
    创建全局总攻击率对比图 - 显示所有模型的总攻击率
    
    Args:
        all_models_overall_rates: {model_name: overall_attack_rate}
        all_models_stats: {model_name: stats_dict}
        output_file: 输出文件路径
        sort_by: 排序方式 - 'rate'（按攻击率降序）或 'name'（按模型名字）
    """
    if not all_models_overall_rates:
        print(f"⚠️  没有数据，跳过全局总攻击率图表")
        return
    
    # 准备数据 - 根据 sort_by 参数排序
    if sort_by == 'name':
        models = sorted(all_models_overall_rates.keys())  # 按名字字母顺序
    else:
        models = sorted(all_models_overall_rates.keys(), key=lambda x: all_models_overall_rates[x], reverse=True)  # 按攻击率降序
    
    rates = [all_models_overall_rates[model] for model in models]
    
    # 计算总问题数（从第一个模型的stats中获取）
    total_questions = 0
    if models and models[0] in all_models_stats:
        for category, cat_stats in all_models_stats[models[0]].items():
            total_questions += cat_stats.get('evaluated', 0)
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(max(12, len(models) * 0.6), 8))
    
    # 使用高对比度的颜色
    color_palette = [
        '#1f77b4',  # 深蓝
        '#ff7f0e',  # 橙色
        '#2ca02c',  # 绿色
        '#d62728',  # 红色
        '#9467bd',  # 紫色
        '#8c564b',  # 棕色
        '#e377c2',  # 粉色
        '#7f7f7f',  # 灰色
        '#17becf',  # 青色
        '#bcbd22',  # 黄绿
    ]
    colors = [color_palette[i % len(color_palette)] for i in range(len(models))]
    
    # 绘制柱状图
    x = np.arange(len(models))
    bars = ax.bar(x, rates, color=colors, alpha=0.9, edgecolor='white', linewidth=1)
    
    # 在柱子上显示数值
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
               f'{height:.1f}%',
               ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 设置标签和标题
    ax.set_xlabel('Model', fontsize=14, fontweight='bold')
    ax.set_ylabel('Overall Attack Rate (%)', fontsize=14, fontweight='bold')
    sort_desc = '(Sorted by Attack Rate)' if sort_by == 'rate' else '(Sorted by Model Name)'
    title = f'All Models - Overall Attack Rate Comparison\n{sort_desc} | Total Questions: {total_questions}'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, max(rates) * 1.15 if rates else 100)
    
    # 添加水平参考线
    if max(rates) > 50:
        ax.axhline(y=50, color='red', linestyle='--', alpha=0.3, linewidth=1)
        ax.text(len(models)-0.5, 51, '50%', color='red', fontsize=9)
    
    # 保存图表
    plt.subplots_adjust(bottom=0.25, top=0.92, left=0.08, right=0.98)
    try:
        plt.savefig(output_file, dpi=120)
    except Exception as e:
        print(f"⚠️  生成图表失败 {output_file}: {e}")
    finally:
        plt.close()
    
    print(f"✅ 生成全局总攻击率图表: {output_file}")

def create_category_comparison_chart(category, all_models_data, all_models_stats, output_file):
    """
    创建单个类别的所有模型对比图
    
    Args:
        category: 类别名称（如 '01-Illegal_Activitiy'）
        all_models_data: {model_name: {category: attack_rate}}
        all_models_stats: {model_name: {category: stats}}
        output_file: 输出文件路径
    """
    # 提取该类别的数据
    models_with_data = {}
    category_question_count = 0
    
    for model_name, data in all_models_data.items():
        if category in data:
            models_with_data[model_name] = data[category]
            # 获取问题数（从stats中）
            if model_name in all_models_stats and category in all_models_stats[model_name]:
                category_question_count = all_models_stats[model_name][category].get('evaluated', 0)
    
    if not models_with_data:
        print(f"⚠️  类别 {category} 没有数据，跳过")
        return
    
    # 按攻击率降序排序
    models = sorted(models_with_data.keys(), key=lambda x: models_with_data[x], reverse=True)
    rates = [models_with_data[model] for model in models]
    
    # 获取类别简称
    category_idx = CATEGORIES.index(category) if category in CATEGORIES else -1
    category_label = CATEGORY_LABELS[category_idx] if category_idx >= 0 else category
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(max(12, len(models) * 0.6), 8))
    
    # 使用高对比度的颜色
    color_palette = [
        '#1f77b4',  # 深蓝
        '#ff7f0e',  # 橙色
        '#2ca02c',  # 绿色
        '#d62728',  # 红色
        '#9467bd',  # 紫色
        '#8c564b',  # 棕色
        '#e377c2',  # 粉色
        '#7f7f7f',  # 灰色
        '#17becf',  # 青色
        '#bcbd22',  # 黄绿
    ]
    colors = [color_palette[i % len(color_palette)] for i in range(len(models))]
    
    # 绘制柱状图
    x = np.arange(len(models))
    bars = ax.bar(x, rates, color=colors, alpha=0.9, edgecolor='white', linewidth=1)
    
    # 在柱子上显示数值
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
               f'{height:.1f}%',
               ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 设置标签和标题
    ax.set_xlabel('Model', fontsize=14, fontweight='bold')
    ax.set_ylabel('Attack Rate (%)', fontsize=14, fontweight='bold')
    title = f'Category: {category_label} ({category})\n(Questions: {category_question_count})'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, max(rates) * 1.15 if rates else 100)
    
    # 添加水平参考线
    if max(rates) > 50:
        ax.axhline(y=50, color='red', linestyle='--', alpha=0.3, linewidth=1)
        ax.text(len(models)-0.5, 51, '50%', color='red', fontsize=9)
    
    # 保存图表
    plt.subplots_adjust(bottom=0.25, top=0.92, left=0.08, right=0.98)
    try:
        plt.savefig(output_file, dpi=120)
    except Exception as e:
        print(f"⚠️  生成图表失败 {output_file}: {e}")
    finally:
        plt.close()
    
    print(f"✅ 生成类别对比图表: {output_file}")

def generate_html_report(all_data, output_file='output/evaluation_report.html'):
    """生成包含所有图表的 HTML 报告"""
    
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>MM-SafetyBench Evaluation Report</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f5f5f5;
            display: flex;
        }
        /* 侧边栏样式 */
        #sidebar {
            position: fixed;
            left: 0;
            top: 0;
            width: 280px;
            height: 100vh;
            background: #2c3e50;
            color: white;
            overflow-y: auto;
            padding: 20px;
            box-shadow: 2px 0 5px rgba(0,0,0,0.1);
            z-index: 1000;
        }
        #sidebar h2 {
            color: #3498db;
            font-size: 18px;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #3498db;
        }
        #sidebar ul {
            list-style: none;
        }
        #sidebar ul li {
            margin: 10px 0;
        }
        #sidebar ul li a {
            color: #ecf0f1;
            text-decoration: none;
            display: block;
            padding: 8px 12px;
            border-radius: 4px;
            transition: all 0.3s;
            font-size: 14px;
        }
        #sidebar ul li a:hover {
            background: #34495e;
            color: #3498db;
            padding-left: 16px;
        }
        #sidebar ul li.subsection {
            margin-left: 15px;
        }
        #sidebar ul li.subsection a {
            font-size: 13px;
            color: #bdc3c7;
        }
        /* 主内容区域 */
        #main-content {
            margin-left: 280px;
            padding: 40px;
            width: calc(100% - 280px);
            max-width: 1400px;
        }
        h1 {
            color: #2c3e50;
            text-align: center;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 30px;
        }
        h2 {
            color: #34495e;
            margin-top: 60px;
            margin-bottom: 20px;
            padding-top: 20px;
            border-left: 5px solid #3498db;
            padding-left: 15px;
        }
        h2.global-section {
            border-left-color: #e74c3c;
            color: #e74c3c;
        }
        h3 {
            color: #34495e;
            margin-top: 30px;
            margin-bottom: 15px;
        }
        .chart-container {
            background: white;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .chart-container img {
            width: 100%;
            height: auto;
        }
        .summary {
            background: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            background: white;
            margin: 20px 0;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #3498db;
            color: white;
            font-weight: bold;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
        .timestamp {
            color: #7f8c8d;
            font-size: 12px;
        }
        /* 平滑滚动 */
        html {
            scroll-behavior: smooth;
        }
        /* 响应式设计 */
        @media (max-width: 1024px) {
            #sidebar {
                width: 220px;
            }
            #main-content {
                margin-left: 220px;
                width: calc(100% - 220px);
            }
        }
    </style>
</head>
<body>
    <!-- 侧边栏目录 -->
    <nav id="sidebar">
        <h2>📑 Table of Contents</h2>
        <ul>
            <li><a href="#overview">Overview</a></li>
            <li><a href="#global-comparison">🌍 Global Comparison</a></li>
            <li class="subsection"><a href="#global-overall-rate">Overall Attack Rate (By Rate)</a></li>
            <li class="subsection"><a href="#global-overall-name">Overall Attack Rate (By Name)</a></li>
            <li class="subsection"><a href="#global-categories">By Category (13 Charts)</a></li>
            <li><a href="#brand-comparison">📊 Brand Comparison</a></li>"""
    
    # 添加品牌到目录
    for i, brand in enumerate(sorted(all_data.keys()), 1):
        safe_brand = brand.replace(" ", "_").replace("+", "")
        html_content += f"""
            <li class="subsection"><a href="#brand-{safe_brand}">{brand}</a></li>"""
    
    html_content += """
            <li><a href="#detailed-info">📋 Detailed Information</a></li>
            <li><a href="#notes">📝 Notes</a></li>
        </ul>
    </nav>
    
    <!-- 主内容区域 -->
    <div id="main-content">
        <h1>📊 MM-SafetyBench Evaluation Report</h1>
        <div id="overview" class="summary">
            <p><strong>Generated:</strong> """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
            <p><strong>Total Brands:</strong> """ + str(len(all_data)) + """</p>
            <p><strong>Total Models:</strong> """ + str(sum(len(models) for models in all_data.values())) + """</p>
        </div>
        
        <!-- 全局对比部分 - 放在最前面 -->
        <h2 id="global-comparison" class="global-section">🌍 Global Comparison - All Models</h2>
        
        <h3 id="global-overall-rate">Global Overall Attack Rate (Sorted by Attack Rate)</h3>
        <div class="chart-container">
            <img src="chart_global_overall_attack_rate.png" alt="Global Overall Attack Rate (By Rate)">
            <p style="color: #7f8c8d; font-size: 14px; margin-top: 10px;">
                This chart shows the overall attack rate across all categories for each model, sorted from highest to lowest attack rate.
            </p>
        </div>
        
        <h3 id="global-overall-name">Global Overall Attack Rate (Sorted by Model Name)</h3>
        <div class="chart-container">
            <img src="chart_global_overall_attack_rate_by_name.png" alt="Global Overall Attack Rate (By Name)">
            <p style="color: #7f8c8d; font-size: 14px; margin-top: 10px;">
                This chart shows the overall attack rate for each model, sorted alphabetically by model name for easy lookup.
            </p>
        </div>
        
        <h3 id="global-categories">Attack Rate by Category - All Models</h3>
        <div class="summary">
            <p>The following 13 charts show model performance in each specific category:</p>
        </div>
"""
    
    # 添加类别对比图表
    for category in CATEGORIES:
        category_idx = CATEGORIES.index(category)
        category_label = CATEGORY_LABELS[category_idx]
        chart_filename = f'chart_category_{category_label}_{category}.png'
        
        html_content += f"""
        <div class="chart-container">
            <h4 style="color: #34495e;">{category_label}: {category}</h4>
            <img src="{chart_filename}" alt="{category} Comparison">
        </div>
"""
    
    html_content += """
        <!-- 品牌对比部分 -->
        <h2 id="brand-comparison">📊 Brand Comparison</h2>
"""
    
    # 为每个品牌生成图表和统计
    for i, (brand, models_data) in enumerate(sorted(all_data.items()), 1):
        safe_brand = brand.replace(" ", "_").replace("+", "")
        
        # 计算平均值（用于HTML显示统计信息）
        averaged_data, tested_categories, averaged_stats = average_multiple_runs(models_data)
        overall_rates = calculate_overall_attack_rates(averaged_stats, tested_categories)
        
        # 图表文件名（图表已由main函数生成）
        chart_file = f'chart_{i}_{safe_brand}.png'
        overall_chart_file = f'chart_{i}_{safe_brand}_overall.png'
        
        # 添加到 HTML（添加锚点）
        html_content += f"""
        <h3 id="brand-{safe_brand}">{i}. {brand}</h3>
        
        <h4>Overall Attack Rate Comparison</h4>
        <div class="chart-container">
            <img src="{overall_chart_file}" alt="{brand} Overall Attack Rate">
        </div>
        
        <h4>Attack Rate by Category</h4>
        <div class="chart-container">
            <img src="{chart_file}" alt="{brand} Category Chart">
        </div>
    
    <div class="stats-grid">
"""
        
        # 添加统计卡片
        for model_name, data in averaged_data.items():
            # 只计算实际测试的类别的平均值
            tested_cats = tested_categories.get(model_name, set())
            if tested_cats:
                tested_values = [data[cat] for cat in tested_cats]
                avg_attack_rate = np.mean(tested_values)
                num_categories = len(tested_cats)
            else:
                avg_attack_rate = 0.0
                num_categories = 0
            
            # 获取总攻击率
            overall_rate = overall_rates.get(model_name, 0.0)
            
            html_content += f"""
        <div class="stat-card">
            <div class="stat-label">{model_name}</div>
            <div class="stat-value">{overall_rate:.1f}%</div>
            <div class="timestamp">Overall Attack Rate | Avg by Category: {avg_attack_rate:.1f}% ({num_categories} categories)</div>
        </div>
"""
        
        html_content += """
        </div>
"""
    
    # 添加详细信息表格
    html_content += """
        <h2 id="detailed-info">📋 Detailed Information</h2>
        <table>
            <thead>
                <tr>
                    <th>Brand</th>
                    <th>Model</th>
                    <th>Runs</th>
                    <th>CSV Files</th>
                </tr>
            </thead>
            <tbody>
"""
    
    # 收集所有品牌的详细信息
    for brand, models_data in sorted(all_data.items()):
        model_info = defaultdict(list)
        for item in models_data:
            model_name = item['model_display_name']
            model_info[model_name].append({
                'timestamp': item['timestamp'],
                'filename': item['filename']
            })
        
        for model_name, info_list in sorted(model_info.items()):
            filenames = [info['filename'] for info in info_list]
            html_content += f"""
                <tr>
                    <td><strong>{brand}</strong></td>
                    <td>{model_name}</td>
                    <td>{len(filenames)}</td>
                    <td class="timestamp" style="max-width: 500px; word-break: break-all; font-size: 11px;">{', '.join(filenames)}</td>
                </tr>
"""
    
    html_content += """
            </tbody>
        </table>
    
        <h2 id="notes">📝 Notes</h2>
        <div class="summary">
            <ul style="line-height: 1.8;">
                <li><strong>Global Comparison:</strong> Shows all models together for easy cross-brand comparison</li>
                <li><strong>Brand Comparison:</strong> Models grouped by brand (e.g., all Qwen models in one chart)</li>
                <li><strong>Attack Rates:</strong> Averaged across multiple runs of the same model</li>
                <li><strong>Category Charts:</strong> Help identify model strengths and weaknesses in specific scenarios</li>
                <li><strong>Lower = Better:</strong> Lower attack rate indicates better safety performance</li>
                <li><strong>Sorting:</strong> Charts sorted by attack rate help identify best/worst performers, while name sorting helps find specific models</li>
            </ul>
        </div>
    </div>
</body>
</html>
"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ 生成 HTML 报告: {output_file}")

def load_specific_data(eval_files: list):
    """
    加载指定的评估文件数据，按品牌分组
    
    Args:
        eval_files: 评估文件路径列表
        
    Returns:
        {brand: [(model_display_name, timestamp, data, stats), ...]}
    """
    all_data = defaultdict(list)
    
    for filepath in eval_files:
        if not os.path.exists(filepath):
            print(f"⚠️  文件不存在: {filepath}")
            continue
        
        filename = os.path.basename(filepath)
        
        # 解析文件名
        brand, model_display_name, timestamp = parse_filename(filename)
        
        # 读取数据（现在返回 attack_rates 和 stats）
        attack_rates, stats = read_csv_file(filepath)
        
        if attack_rates:
            all_data[brand].append({
                'model_display_name': model_display_name,
                'timestamp': timestamp,
                'data': attack_rates,
                'stats': stats,
                'filename': filename
            })
    
    return all_data


def generate_report_to_folder(all_data, report_dir, report_title="MM-SafetyBench Evaluation Report"):
    """
    生成报告到指定目录
    
    Args:
        all_data: {brand: [{model_display_name, timestamp, data, stats, ...}, ...]}
        report_dir: 报告输出目录
        report_title: 报告标题
    """
    os.makedirs(report_dir, exist_ok=True)
    print(f"\n📁 报告目录: {report_dir}")
    
    print("\n🎨 生成图表和报告...")
    
    # ============ 生成全局图表 ============
    print("\n📊 生成全局对比图表...")
    
    # 1. 收集所有模型的数据
    all_models_data = {}  # {model_name: {category: attack_rate}}
    all_models_stats = {}  # {model_name: {category: stats}}
    all_models_overall_rates = {}  # {model_name: overall_attack_rate}
    
    for brand, models_data in all_data.items():
        # 对每个品牌的数据取平均
        averaged_data, tested_categories, averaged_stats = average_multiple_runs(models_data)
        
        # 计算每个模型的总攻击率
        overall_rates = calculate_overall_attack_rates(averaged_stats, tested_categories)
        
        # 合并到全局数据中
        for model_name, data in averaged_data.items():
            all_models_data[model_name] = data
            all_models_stats[model_name] = averaged_stats[model_name]
            all_models_overall_rates[model_name] = overall_rates.get(model_name, 0.0)
    
    # 2. 生成全局总攻击率对比图
    # 2.1 按攻击率排序
    create_global_overall_chart(
        all_models_overall_rates,
        all_models_stats,
        f'{report_dir}/chart_global_overall_attack_rate.png',
        sort_by='rate'
    )
    
    # 2.2 按模型名字排序
    create_global_overall_chart(
        all_models_overall_rates,
        all_models_stats,
        f'{report_dir}/chart_global_overall_attack_rate_by_name.png',
        sort_by='name'
    )
    
    # 3. 为每个类别生成对比图
    print("\n📊 生成各类别对比图表...")
    for category in CATEGORIES:
        category_label = CATEGORY_LABELS[CATEGORIES.index(category)]
        chart_file = f'{report_dir}/chart_category_{category_label}_{category}.png'
        create_category_comparison_chart(
            category,
            all_models_data,
            all_models_stats,
            chart_file
        )
    
    # 4. 生成品牌分组图表
    for i, (brand, models_data) in enumerate(sorted(all_data.items()), 1):
        safe_brand = brand.replace(" ", "_").replace("+", "")
        averaged_data, tested_categories, averaged_stats = average_multiple_runs(models_data)
        
        chart_file = f'{report_dir}/chart_{i}_{safe_brand}.png'
        create_bar_chart(brand, averaged_data, chart_file, tested_categories)
        
        overall_rates = calculate_overall_attack_rates(averaged_stats, tested_categories)
        overall_chart_file = f'{report_dir}/chart_{i}_{safe_brand}_overall.png'
        create_overall_attack_rate_chart(brand, overall_rates, overall_chart_file)
    
    # 5. 生成 HTML 报告
    report_output = f'{report_dir}/evaluation_report.html'
    generate_html_report(all_data, output_file=report_output)
    
    return report_output


def main(eval_files: list = None, output_file: str = None, 
         job_nums: list = None, batch_nums: list = None,
         output_dir: str = 'output'):
    """
    主函数
    
    Args:
        eval_files: 指定的评估文件列表（旧格式，CSV 文件路径）
        output_file: 输出报告文件路径（仅用于旧格式）
        job_nums: 指定的 job 编号列表
        batch_nums: 指定的 batch 编号列表
        output_dir: 输出基础目录，默认 'output'
    """
    print("📊 开始生成评估报告...\n")
    
    # 收集所有 job 文件夹
    all_job_folders = []
    target_batch_folders = []  # 用于保存目标 batch 文件夹（用于输出报告）
    
    # 1. 如果指定了 batch_nums，从 batch 文件夹中查找 jobs
    if batch_nums:
        print(f"📦 查找 batch: {batch_nums}")
        batch_folders = find_batch_folders(output_dir, batch_nums)
        
        for batch_folder, batch_info in batch_folders:
            print(f"  ✅ 找到 batch_{batch_info['batch_num']}: {batch_folder}")
            target_batch_folders.append((batch_folder, batch_info))
            
            # 查找 batch 内的所有 job
            jobs_in_batch = find_jobs_in_batch(batch_folder)
            print(f"     包含 {len(jobs_in_batch)} 个 jobs")
            all_job_folders.extend(jobs_in_batch)
    
    # 2. 如果指定了 job_nums，直接查找 job 文件夹
    if job_nums:
        print(f"📋 查找 jobs: {job_nums}")
        
        # 先从顶层 output 目录查找
        top_level_jobs = find_job_folders(output_dir, job_nums)
        for folder, info in top_level_jobs:
            print(f"  ✅ 找到 job_{info['job_num']}: {folder}")
        all_job_folders.extend(top_level_jobs)
        
        # 再从所有 batch 文件夹中查找（可能 job 在某个 batch 内）
        all_batches = find_batch_folders(output_dir)
        for batch_folder, _ in all_batches:
            jobs_in_batch = find_jobs_in_batch(batch_folder)
            for folder, info in jobs_in_batch:
                if info['job_num'] in job_nums:
                    # 检查是否已经添加过
                    if not any(f[1]['job_num'] == info['job_num'] for f in all_job_folders):
                        print(f"  ✅ 找到 job_{info['job_num']} (在 batch 中): {folder}")
                        all_job_folders.append((folder, info))
    
    # 3. 如果指定了 eval_files（旧格式），使用旧的加载逻辑
    if eval_files:
        print(f"📄 加载指定的 CSV 文件: {len(eval_files)} 个")
        all_data = load_specific_data(eval_files)
        
        if all_data:
            # 使用旧格式的输出目录
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            total_models = sum(len(models) for models in all_data.values())
            report_dir = f'{output_dir}/reports/models_{total_models}_{timestamp}'
            
            report_output = generate_report_to_folder(all_data, report_dir)
            print_completion_summary(all_data, report_dir, report_output)
        else:
            print("⚠️  没有找到有效的评估数据")
        return
    
    # 4. 如果没有指定任何参数，使用旧的加载逻辑（load_all_data）
    if not job_nums and not batch_nums:
        print("📖 使用默认逻辑加载所有评估数据...")
        all_data = load_all_data()
        
        if all_data:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            total_models = sum(len(models) for models in all_data.values())
            report_dir = f'{output_dir}/reports/models_{total_models}_{timestamp}'
            
            report_output = generate_report_to_folder(all_data, report_dir)
            print_completion_summary(all_data, report_dir, report_output)
        else:
            print("⚠️  没有找到有效的评估数据")
        return
    
    # 5. 从 job 文件夹加载数据
    if not all_job_folders:
        print("⚠️  没有找到匹配的 job 文件夹")
        return
    
    # 去重（按 job_num）
    unique_jobs = {}
    for folder, info in all_job_folders:
        job_num = info['job_num']
        if job_num not in unique_jobs:
            unique_jobs[job_num] = (folder, info)
    all_job_folders = list(unique_jobs.values())
    
    print(f"\n📖 加载 {len(all_job_folders)} 个 job 的数据...")
    all_data = load_data_from_jobs(all_job_folders)
    
    if not all_data:
        print("⚠️  没有找到有效的评估数据")
        return
    
    print(f"✅ 找到 {len(all_data)} 个品牌")
    total_models = sum(len(models) for models in all_data.values())
    for brand, models in all_data.items():
        print(f"  - {brand}: {len(models)} 个模型/运行")
    
    # 6. 确定报告输出目录
    if len(target_batch_folders) == 1:
        # 如果只有一个 batch，输出到 batch 文件夹内
        batch_folder, batch_info = target_batch_folders[0]
        report_dir = os.path.join(batch_folder, 'report')
        report_title = f"Batch #{batch_info['batch_num']} Evaluation Report"
    elif len(target_batch_folders) > 1:
        # 如果有多个 batch，创建一个新的报告目录
        batch_nums_str = '_'.join(str(b[1]['batch_num']) for b in target_batch_folders)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        report_dir = f'{output_dir}/reports/batches_{batch_nums_str}_{timestamp}'
        report_title = f"Batches {batch_nums_str} Evaluation Report"
    elif job_nums and len(all_job_folders) == 1:
        # 如果只指定了一个 job，输出到 job 文件夹内
        job_folder = all_job_folders[0][0]
        report_dir = os.path.join(job_folder, 'report')
        report_title = f"Job #{all_job_folders[0][1]['job_num']} Evaluation Report"
    else:
        # 多个 job 或混合情况，创建新的报告目录
        job_nums_str = '_'.join(str(f[1]['job_num']) for f in all_job_folders[:5])
        if len(all_job_folders) > 5:
            job_nums_str += f'_plus{len(all_job_folders)-5}more'
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        report_dir = f'{output_dir}/reports/jobs_{job_nums_str}_{timestamp}'
        report_title = f"Jobs Evaluation Report"
    
    # 7. 生成报告
    report_output = generate_report_to_folder(all_data, report_dir, report_title)
    print_completion_summary(all_data, report_dir, report_output)


def print_completion_summary(all_data, report_dir, report_output):
    """打印完成摘要"""
    print("\n🎉 完成！")
    print(f"📁 报告目录: {report_dir}/")
    print(f"📄 HTML 报告: {report_output}")
    total_charts = 2 + 13 + len(all_data) * 2  # 全局图 + 类别图 + 品牌图
    print(f"🖼️  图表总数: {total_charts} 张")
    print(f"   - 品牌分组图表: {len(all_data)*2} 张")
    print(f"   - 全局总攻击率图: 2 张")
    print(f"   - 类别对比图: 13 张")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="生成包含图表的评估报告",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 为指定的 batch 生成报告（报告输出到 batch 文件夹内）
  python generate_report_with_charts.py --batches 4
  
  # 为多个 batch 生成报告
  python generate_report_with_charts.py --batches 3 4
  
  # 为指定的 job 生成报告
  python generate_report_with_charts.py --jobs 153 154 155
  
  # 同时指定 job 和 batch
  python generate_report_with_charts.py --jobs 153 --batches 4
  
  # 使用旧格式的 CSV 文件
  python generate_report_with_charts.py --files output/eval_*.csv
        """
    )
    parser.add_argument("--jobs", nargs='+', type=int, default=None,
                       help="指定要处理的 job 编号列表")
    parser.add_argument("--batches", nargs='+', type=int, default=None,
                       help="指定要处理的 batch 编号列表")
    parser.add_argument("--files", nargs='+', default=None,
                       help="指定要处理的评估 CSV 文件列表（旧格式）")
    parser.add_argument("--output-dir", default='output',
                       help="输出基础目录（默认: output）")
    
    args = parser.parse_args()
    
    main(eval_files=args.files, 
         job_nums=args.jobs, 
         batch_nums=args.batches,
         output_dir=args.output_dir)

