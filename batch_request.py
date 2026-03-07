#!/usr/bin/env python3
"""
批量运行 request.py 的脚本

通过配置 args_combo 列表，可以组合不同的参数运行多次 request.py

使用方式：
    python batch_request.py

配置说明：
    args_combo 是一个列表，每个元素代表一组参数变体
    - 如果元素是字符串，表示固定参数（所有组合都会使用）
    - 如果元素是列表，表示需要遍历的参数变体
    
    最终会生成所有变体的笛卡尔积组合

输出结构：
    output/batch_{first_job_num}_{timestamp}/
    ├── batch_summary.html          # 批次汇总报告
    ├── batch.log                   # 批次运行日志
    ├── job_{num}_tasks_{n}_.../    # 第一个任务
    ├── job_{num}_tasks_{n}_.../    # 第二个任务
    └── ...

示例：
    args_combo = [
        "--categories 12-Health_Consultation",  # 固定参数
        [  # 需要遍历的参数变体
            '--provider comt_vsp --model "qwen/qwen3-vl-235b-a22b-instruct"',
            '--provider openrouter --model "qwen/qwen3-vl-235b-a22b-instruct"',
        ],
    ]

    这会运行 2 次 request.py：
    1. --categories 12-Health_Consultation --provider comt_vsp --model "qwen/qwen3-vl-235b-a22b-instruct"
    2. --categories 12-Health_Consultation --provider openrouter --model "qwen/qwen3-vl-235b-a22b-instruct"

    也支持 --profile 简化参数：
    args_combo = [
        "--sampling_rate 0.12",
        [
            '--profile direct --model "qwen/qwen3-vl-235b-a22b-instruct"',
            '--profile comt_vsp_prebaked_ask',
            '--profile comt_vsp_prebaked_sd_good',
        ],
    ]
"""

import subprocess
import sys
import os
import shutil
import itertools
import re
import time
import base64
from io import BytesIO
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Optional, TextIO


# ============ 日志管理 ============

class TeeWriter:
    """同时写入多个输出流的类"""
    def __init__(self, *writers):
        self.writers = writers
    
    def write(self, text):
        for w in self.writers:
            w.write(text)
            w.flush()
    
    def flush(self):
        for w in self.writers:
            w.flush()


# 全局日志文件句柄
_log_file: Optional[TextIO] = None
_original_stdout = None


# ============ Batch Counter（单调递增的批次编号）============

BATCH_COUNTER_FILE = "output/.batch_counter"

def get_next_batch_num() -> int:
    """
    获取下一个批次编号（单调递增，从1开始）
    
    Returns:
        下一个可用的批次编号
    """
    os.makedirs("output", exist_ok=True)
    
    current_num = 0
    if os.path.exists(BATCH_COUNTER_FILE):
        try:
            with open(BATCH_COUNTER_FILE, 'r') as f:
                current_num = int(f.read().strip())
        except (ValueError, IOError):
            current_num = 0
    
    next_num = current_num + 1
    
    with open(BATCH_COUNTER_FILE, 'w') as f:
        f.write(str(next_num))
    
    return next_num


def setup_logging(log_path: str):
    """设置日志输出到文件和控制台"""
    global _log_file, _original_stdout
    
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    _log_file = open(log_path, 'w', encoding='utf-8')
    _original_stdout = sys.stdout
    sys.stdout = TeeWriter(_original_stdout, _log_file)
    
    return _log_file


def close_logging():
    """关闭日志文件"""
    global _log_file, _original_stdout
    
    if _original_stdout:
        sys.stdout = _original_stdout
    
    if _log_file:
        _log_file.close()
        _log_file = None


# ============ 配置区域 ============

# 参数组合配置
# - 字符串：固定参数（所有组合都会使用）
# - 列表：需要遍历的参数变体
args_combo = [
    # 固定参数
    "--tunnel cf --sampling_rate 0.01",
    # 需要遍历的参数变体：不同的 mode 组合
    [
        '--profile qwen235b',
        '--profile comt_vsp_prebaked_ask',
        '--profile comt_vsp_prebaked_sd_good',
        '--profile comt_vsp_prebaked_sd_bad',
    ],
]

# 是否显示详细输出
VERBOSE = True

# 是否在完成后生成报告
GENERATE_REPORT = True


# ============ 运行结果数据结构 ============

@dataclass
class RunResult:
    """单次运行的结果信息"""
    run_index: int                          # 运行序号
    args_str: str                           # 命令行参数
    success: bool                           # 是否成功
    start_time: datetime                    # 开始时间
    end_time: datetime                      # 结束时间
    duration: timedelta                     # 耗时
    task_num: Optional[int] = None          # 任务编号
    total_tasks: Optional[int] = None       # 总任务数
    job_folder: Optional[str] = None        # Job 文件夹路径
    output_file: Optional[str] = None       # 输出文件路径
    eval_file: Optional[str] = None         # 评估结果文件路径
    vsp_dir: Optional[str] = None           # VSP 详细输出目录
    summary_file: Optional[str] = None      # summary.html 路径
    error_message: Optional[str] = None     # 错误信息
    error_key: Optional[str] = None         # 错误类型键
    mode: Optional[str] = None              # Mode (direct/vsp/comt_vsp)
    provider: Optional[str] = None          # Provider (openai/openrouter)
    model: Optional[str] = None             # Model
    categories: Optional[str] = None        # Categories
    max_tasks_arg: Optional[int] = None     # Max tasks argument
    # VSP Postproc 相关参数
    vsp_postproc: Optional[bool] = None     # 是否启用 vsp_postproc
    vsp_postproc_backend: Optional[str] = None      # postproc backend (prebaked/sd)
    vsp_postproc_fallback: Optional[str] = None     # fallback method (ask/sd)
    vsp_postproc_method: Optional[str] = None       # postproc method (visual_mask/good/bad)
    vsp_postproc_sd_prompt: Optional[str] = None    # SD prompt
    comt_sample_id: Optional[str] = None    # COMT sample ID


def parse_args_str(args_str: str) -> dict:
    """从参数字符串中提取关键信息"""
    info = {}

    # 提取 mode
    mode_match = re.search(r'--mode\s+(\S+)', args_str)
    if mode_match:
        info['mode'] = mode_match.group(1)

    # 提取 provider
    provider_match = re.search(r'--provider\s+(\S+)', args_str)
    if provider_match:
        info['provider'] = provider_match.group(1)
    
    # 提取 model（可能带引号）
    model_match = re.search(r'--model\s+["\']?([^"\']+)["\']?', args_str)
    if model_match:
        info['model'] = model_match.group(1).strip()
    
    # 提取 categories
    categories_match = re.search(r'--categories\s+(\S+)', args_str)
    if categories_match:
        info['categories'] = categories_match.group(1)
    
    # 提取 max_tasks
    max_tasks_match = re.search(r'--max_tasks\s+(\d+)', args_str)
    if max_tasks_match:
        info['max_tasks_arg'] = int(max_tasks_match.group(1))
    
    # 提取 VSP Postproc 相关参数
    if '--vsp_postproc' in args_str:
        info['vsp_postproc'] = True
        
        # 提取 backend
        backend_match = re.search(r'--vsp_postproc_backend\s+(\S+)', args_str)
        if backend_match:
            info['vsp_postproc_backend'] = backend_match.group(1)
        
        # 提取 fallback
        fallback_match = re.search(r'--vsp_postproc_fallback\s+(\S+)', args_str)
        if fallback_match:
            info['vsp_postproc_fallback'] = fallback_match.group(1)
        
        # 提取 method
        method_match = re.search(r'--vsp_postproc_method\s+(\S+)', args_str)
        if method_match:
            info['vsp_postproc_method'] = method_match.group(1)
        
        # 提取 SD prompt（可能带引号）
        sd_prompt_match = re.search(r'--vsp_postproc_sd_prompt\s+["\']([^"\']+)["\']', args_str)
        if sd_prompt_match:
            info['vsp_postproc_sd_prompt'] = sd_prompt_match.group(1)
    
    # 提取 comt_sample_id
    comt_sample_id_match = re.search(r'--comt_sample_id\s+(\S+)', args_str)
    if comt_sample_id_match:
        info['comt_sample_id'] = comt_sample_id_match.group(1)
    
    return info


def parse_output(output: str) -> dict:
    """从 request.py 的输出中提取关键信息"""
    info = {}
    
    # 提取任务编号
    task_num_match = re.search(r'🔢 任务编号:\s*(\d+)', output)
    if task_num_match:
        info['task_num'] = int(task_num_match.group(1))
    
    # 提取 Job 文件夹路径（重命名后的）
    job_folder_match = re.search(r'✅ Job 文件夹已重命名:\s*(\S+)', output)
    if job_folder_match:
        info['job_folder'] = job_folder_match.group(1)
    else:
        # 尝试从创建临时文件夹的日志提取
        temp_folder_match = re.search(r'📁 创建临时 job 文件夹:\s*(\S+)', output)
        if temp_folder_match:
            info['job_folder'] = temp_folder_match.group(1)
    
    # 提取输出文件路径（重命名后的）
    output_file_match = re.search(r'✅ 文件已重命名:\s*(\S+\.jsonl)', output)
    if output_file_match:
        info['output_file'] = output_file_match.group(1)
    else:
        # 尝试从"输出文件:"行提取
        output_file_match2 = re.search(r'输出文件:\s*(\S+\.jsonl)', output)
        if output_file_match2:
            info['output_file'] = output_file_match2.group(1)
    
    # 提取 VSP 详细输出目录
    vsp_dir_match = re.search(r'✅ VSP 详细输出目录已重命名:\s*(\S+)', output)
    if vsp_dir_match:
        info['vsp_dir'] = vsp_dir_match.group(1)
    
    # 提取评估结果文件
    eval_file_match = re.search(r'✅ 评估指标已保存:\s*(\S+\.csv)', output)
    if eval_file_match:
        info['eval_file'] = eval_file_match.group(1)
    
    # 提取 Summary HTML
    summary_match = re.search(r'✅ Summary 已保存:\s*(\S+\.html)', output)
    if summary_match:
        info['summary_file'] = summary_match.group(1)
    
    # 提取总任务数
    total_tasks_match = re.search(r'总任务数:\s*(\d+)', output)
    if total_tasks_match:
        info['total_tasks'] = int(total_tasks_match.group(1))
    
    return info


def format_duration(td: timedelta) -> str:
    """格式化时间间隔"""
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    if hours > 0:
        return f"{hours}h{minutes}m{seconds}s"
    elif minutes > 0:
        return f"{minutes}m{seconds}s"
    else:
        return f"{seconds}s"


# ============ 主逻辑 ============

def generate_combinations(args_combo):
    """
    生成所有参数组合
    
    Args:
        args_combo: 参数组合配置列表
        
    Returns:
        参数组合列表，每个元素是完整的命令行参数字符串
    """
    # 将所有元素转换为列表格式
    normalized = []
    for item in args_combo:
        if isinstance(item, str):
            normalized.append([item])
        elif isinstance(item, list):
            normalized.append(item)
        else:
            raise ValueError(f"不支持的参数类型: {type(item)}")
    
    # 生成笛卡尔积
    combinations = list(itertools.product(*normalized))
    
    # 合并每个组合的参数
    result = []
    for combo in combinations:
        args = " ".join(combo)
        result.append(args)
    
    return result


def run_request(args_str: str, run_index: int, total_runs: int) -> RunResult:
    """
    运行一次 request.py
    
    Args:
        args_str: 命令行参数字符串
        run_index: 当前运行序号（从1开始）
        total_runs: 总运行次数
        
    Returns:
        RunResult 对象，包含运行结果的详细信息
    """
    print(f"\n{'='*80}")
    print(f"🚀 运行 [{run_index}/{total_runs}]")
    print(f"{'='*80}")
    print(f"📋 参数: {args_str}")
    print(f"{'='*80}\n")
    
    # 构建完整命令
    cmd = f"python request.py {args_str}"
    
    if VERBOSE:
        print(f"💻 执行命令: {cmd}\n")
    
    # 记录开始时间
    start_time = datetime.now()
    
    # 解析参数
    args_info = parse_args_str(args_str)
    
    try:
        # 设置环境变量，禁用 Python 的输出缓冲
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        
        # 运行命令，捕获输出同时显示在终端
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        
        # 实时输出并收集
        output_lines = []
        for line in process.stdout:
            print(line, end='')  # 实时显示
            sys.stdout.flush()   # 确保立即刷新到屏幕和日志
            output_lines.append(line)
        
        process.wait()
        output = ''.join(output_lines)
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        # 检测 STOP_REASON（由 request.py 内部自动停止时打印）
        stop_match = re.search(r"自动停止原因:\s*(.+)", output)
        error_msg = None
        error_key = None
        if stop_match:
            error_msg = stop_match.group(1).strip()
            error_key = "stop_reason"
        
        output_info = parse_output(output)
        
        success_flag = (process.returncode == 0) and (error_key is None)
        
        if success_flag:
            print(f"\n✅ 运行 [{run_index}/{total_runs}] 完成 (耗时: {format_duration(duration)})")
        else:
            print(f"\n❌ 运行 [{run_index}/{total_runs}] 失败")
            if process.returncode != 0:
                print(f"   退出码: {process.returncode}")
            if error_msg:
                print(f"   检测到错误: {error_msg}")
        
        return RunResult(
            run_index=run_index,
            args_str=args_str,
            success=success_flag,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            task_num=output_info.get('task_num'),
            total_tasks=output_info.get('total_tasks'),
            job_folder=output_info.get('job_folder'),
            output_file=output_info.get('output_file'),
            eval_file=output_info.get('eval_file'),
            vsp_dir=output_info.get('vsp_dir'),
            summary_file=output_info.get('summary_file'),
            error_message=error_msg or (f"退出码: {process.returncode}" if process.returncode != 0 else None),
            error_key=error_key,
            mode=args_info.get('mode'),
            provider=args_info.get('provider'),
            model=args_info.get('model'),
            categories=args_info.get('categories'),
            max_tasks_arg=args_info.get('max_tasks_arg'),
            vsp_postproc=args_info.get('vsp_postproc'),
            vsp_postproc_backend=args_info.get('vsp_postproc_backend'),
            vsp_postproc_fallback=args_info.get('vsp_postproc_fallback'),
            vsp_postproc_method=args_info.get('vsp_postproc_method'),
            vsp_postproc_sd_prompt=args_info.get('vsp_postproc_sd_prompt'),
            comt_sample_id=args_info.get('comt_sample_id'),
        )

    except Exception as e:
        end_time = datetime.now()
        duration = end_time - start_time

        print(f"\n❌ 运行 [{run_index}/{total_runs}] 异常")
        print(f"   错误: {e}")

        return RunResult(
            run_index=run_index,
            args_str=args_str,
            success=False,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            error_message=str(e),
            mode=args_info.get('mode'),
            provider=args_info.get('provider'),
            model=args_info.get('model'),
            categories=args_info.get('categories'),
            max_tasks_arg=args_info.get('max_tasks_arg'),
            vsp_postproc=args_info.get('vsp_postproc'),
            vsp_postproc_backend=args_info.get('vsp_postproc_backend'),
            vsp_postproc_fallback=args_info.get('vsp_postproc_fallback'),
            vsp_postproc_method=args_info.get('vsp_postproc_method'),
            vsp_postproc_sd_prompt=args_info.get('vsp_postproc_sd_prompt'),
            comt_sample_id=args_info.get('comt_sample_id'),
        )


def move_job_to_batch(result: RunResult, batch_folder: str) -> RunResult:
    """
    将 job 文件夹移动到 batch 文件夹中，并更新 result 中的路径
    
    Args:
        result: 运行结果
        batch_folder: batch 文件夹路径
        
    Returns:
        更新后的 RunResult
    """
    if not result.job_folder or not os.path.exists(result.job_folder):
        return result
    
    job_basename = os.path.basename(result.job_folder)
    new_job_folder = os.path.join(batch_folder, job_basename)
    
    try:
        shutil.move(result.job_folder, new_job_folder)
        print(f"📦 已移动 job 文件夹: {result.job_folder} -> {new_job_folder}")
        
        # 更新所有路径
        old_folder = result.job_folder
        result.job_folder = new_job_folder
        
        if result.output_file:
            result.output_file = result.output_file.replace(old_folder, new_job_folder)
        if result.eval_file:
            result.eval_file = result.eval_file.replace(old_folder, new_job_folder)
        if result.vsp_dir:
            result.vsp_dir = result.vsp_dir.replace(old_folder, new_job_folder)
        if result.summary_file:
            result.summary_file = result.summary_file.replace(old_folder, new_job_folder)
            
    except Exception as e:
        print(f"⚠️  移动 job 文件夹失败: {e}")
    
    return result


def generate_batch_summary_html(
    batch_folder: str,
    batch_num: int,
    results: List[RunResult],
    batch_start: datetime,
    batch_end: datetime,
    stop_reason: Optional[str] = None
) -> str:
    """
    生成 batch_summary.html
    
    Args:
        batch_folder: batch 文件夹路径
        batch_num: 批次编号
        results: 所有运行结果
        batch_start: 批次开始时间
        batch_end: 批次结束时间
        stop_reason: 停止原因（如果有）
        
    Returns:
        生成的 HTML 文件路径
    """
    batch_duration = batch_end - batch_start
    success_count = sum(1 for r in results if r.success)
    fail_count = len(results) - success_count
    
    # 构建 job 列表 HTML
    jobs_html = ""
    for r in results:
        status_class = "success" if r.success else "failed"
        status_text = "✅ 成功" if r.success else "❌ 失败"
        
        # 构建链接（如果有 summary.html）
        summary_link = ""
        if r.summary_file and os.path.exists(r.summary_file):
            rel_path = os.path.basename(r.job_folder) + "/summary.html" if r.job_folder else ""
            summary_link = f'<a href="{rel_path}" class="summary-link">查看详情</a>'
        
        # 构建 VSP Postproc 信息
        vsp_postproc_info = ""
        if r.vsp_postproc:
            parts = []
            if r.vsp_postproc_method:
                parts.append(f"<span class='vsp-method'>{r.vsp_postproc_method}</span>")
            if r.vsp_postproc_backend:
                parts.append(f"backend: {r.vsp_postproc_backend}")
            if r.vsp_postproc_fallback:
                parts.append(f"fallback: {r.vsp_postproc_fallback}")
            if r.comt_sample_id:
                parts.append(f"id: {r.comt_sample_id}")
            if r.vsp_postproc_sd_prompt:
                parts.append(f'prompt: "{r.vsp_postproc_sd_prompt}"')
            vsp_postproc_info = "<br>".join(parts) if parts else "✓"
        else:
            vsp_postproc_info = "-"
        
        jobs_html += f'''
        <tr class="{status_class}">
            <td>{r.run_index}</td>
            <td>{r.task_num or "N/A"}</td>
            <td>{r.provider or "N/A"}</td>
            <td class="model-cell">{r.model or "N/A"}</td>
            <td>{r.total_tasks or "N/A"}</td>
            <td class="vsp-postproc-cell">{vsp_postproc_info}</td>
            <td>{format_duration(r.duration)}</td>
            <td><span class="status-badge {status_class}">{status_text}</span></td>
            <td>{summary_link}</td>
        </tr>'''
    
    html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Batch #{batch_num} Summary</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 100%); color: #e0e0e0; min-height: 100vh; padding: 20px; }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        h1 {{ text-align: center; color: #00d9ff; margin-bottom: 10px; font-size: 2.2em; text-shadow: 0 0 20px rgba(0, 217, 255, 0.3); }}
        .subtitle {{ text-align: center; color: #888; margin-bottom: 30px; font-size: 0.9em; }}
        .section {{ background: rgba(255, 255, 255, 0.05); border-radius: 15px; padding: 25px; margin-bottom: 25px; border: 1px solid rgba(255, 255, 255, 0.1); }}
        .section h2 {{ color: #00d9ff; margin-bottom: 20px; font-size: 1.3em; border-bottom: 1px solid rgba(255, 255, 255, 0.1); padding-bottom: 10px; }}
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin-bottom: 20px; }}
        .stat-card {{ background: rgba(255, 255, 255, 0.05); border-radius: 12px; padding: 20px; text-align: center; border: 1px solid rgba(255, 255, 255, 0.1); }}
        .stat-card h3 {{ font-size: 2em; margin-bottom: 5px; color: #ffd93d; }}
        .stat-card.success h3 {{ color: #00ff88; }}
        .stat-card.failed h3 {{ color: #ff6b6b; }}
        .stat-card.duration h3 {{ color: #00d9ff; }}
        .stat-card p {{ color: #888; font-size: 0.85em; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
        th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid rgba(255, 255, 255, 0.1); }}
        th {{ background: rgba(0, 0, 0, 0.3); color: #00d9ff; font-weight: 600; }}
        tr:hover {{ background: rgba(255, 255, 255, 0.05); }}
        tr.failed {{ background: rgba(255, 107, 107, 0.1); }}
        .model-cell {{ max-width: 300px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
        .vsp-postproc-cell {{ font-size: 0.85em; color: #bbb; line-height: 1.6; max-width: 350px; }}
        .vsp-method {{ background: rgba(138, 43, 226, 0.3); color: #da70d6; padding: 2px 8px; border-radius: 8px; font-weight: bold; display: inline-block; margin-bottom: 2px; }}
        .status-badge {{ padding: 4px 10px; border-radius: 12px; font-size: 0.85em; font-weight: bold; }}
        .status-badge.success {{ background: rgba(0, 255, 136, 0.2); color: #00ff88; }}
        .status-badge.failed {{ background: rgba(255, 107, 107, 0.2); color: #ff6b6b; }}
        .summary-link {{ color: #00d9ff; text-decoration: none; }}
        .summary-link:hover {{ text-decoration: underline; }}
        .time-info {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
        .time-info p {{ margin-bottom: 8px; }}
        .time-info strong {{ color: #00d9ff; }}
        .stop-reason {{ background: rgba(255, 193, 7, 0.2); color: #ffc107; padding: 10px 15px; border-radius: 8px; margin-top: 15px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Batch #{batch_num} Summary</h1>
        <p class="subtitle">{os.path.basename(batch_folder)}</p>
        
        <div class="section">
            <h2>Overview</h2>
            <div class="stats">
                <div class="stat-card"><h3>{len(results)}</h3><p>Total Runs</p></div>
                <div class="stat-card success"><h3>{success_count}</h3><p>Successful</p></div>
                <div class="stat-card failed"><h3>{fail_count}</h3><p>Failed</p></div>
                <div class="stat-card duration"><h3>{format_duration(batch_duration)}</h3><p>Total Duration</p></div>
            </div>
            <div class="time-info">
                <div>
                    <p><strong>Start Time:</strong> {batch_start.strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p><strong>End Time:</strong> {batch_end.strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                <div>
                    <p><strong>Batch Folder:</strong> {os.path.basename(batch_folder)}</p>
                </div>
            </div>
            {f'<div class="stop-reason">⚠️ Stop Reason: {stop_reason}</div>' if stop_reason else ''}
        </div>
        
        <div class="section">
            <h2>Job Details</h2>
            <table>
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Job Num</th>
                        <th>Provider</th>
                        <th>Model</th>
                        <th>Tasks</th>
                        <th>VSP Postproc</th>
                        <th>Duration</th>
                        <th>Status</th>
                        <th>Details</th>
                    </tr>
                </thead>
                <tbody>
                    {jobs_html}
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>'''
    
    summary_path = os.path.join(batch_folder, "batch_summary.html")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✅ Batch summary 已生成: {summary_path}")
    return summary_path


def print_results_summary(results: List[RunResult], batch_start: datetime, batch_end: datetime, stop_reason: Optional[str] = None):
    """打印所有运行结果的详细汇总"""
    batch_duration = batch_end - batch_start
    
    success_count = sum(1 for r in results if r.success)
    fail_count = sum(1 for r in results if not r.success)
    
    print(f"\n{'='*100}")
    print(f"{'='*100}")
    print(f"📊 批量运行结果汇总")
    print(f"{'='*100}")
    print(f"{'='*100}")
    
    # 总体统计
    print(f"\n📈 总体统计")
    print(f"{'─'*50}")
    print(f"  开始时间:     {batch_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  结束时间:     {batch_end.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  总耗时:       {format_duration(batch_duration)}")
    print(f"  总运行次数:   {len(results)}")
    print(f"  成功:         {success_count}")
    print(f"  失败:         {fail_count}")
    if stop_reason:
        print(f"  停止原因:     {stop_reason}")
    
    # 每次运行的详细信息
    print(f"\n{'='*100}")
    print(f"📋 各任务详细信息")
    print(f"{'='*100}")
    
    for r in results:
        status_icon = "✅" if r.success else "❌"
        print(f"\n{status_icon} 运行 #{r.run_index}")
        print(f"{'─'*80}")
        
        # 基本信息
        print(f"  状态:         {'成功' if r.success else '失败'}")
        print(f"  耗时:         {format_duration(r.duration)}")
        print(f"  开始时间:     {r.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  结束时间:     {r.end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 参数信息
        print(f"\n  📌 请求参数:")
        if r.provider:
            print(f"     Provider:    {r.provider}")
        if r.model:
            print(f"     Model:       {r.model}")
        if r.categories:
            print(f"     Categories:  {r.categories}")
        if r.max_tasks_arg:
            print(f"     Max Tasks:   {r.max_tasks_arg}")
        print(f"     完整参数:   {r.args_str}")
        
        # 输出信息
        if r.success:
            print(f"\n  📁 输出文件:")
            if r.task_num:
                print(f"     任务编号:   {r.task_num}")
            if r.total_tasks:
                print(f"     实际任务数: {r.total_tasks}")
            if r.output_file:
                print(f"     JSONL 文件: {r.output_file}")
            if r.eval_file:
                print(f"     评估结果:   {r.eval_file}")
            if r.vsp_dir:
                print(f"     VSP 目录:   {r.vsp_dir}")
        else:
            print(f"\n  ⚠️ 错误信息:")
            print(f"     {r.error_message or '未知错误'}")
    
    # 输出文件汇总表
    successful_results = [r for r in results if r.success]
    if successful_results:
        print(f"\n{'='*100}")
        print(f"📁 输出文件汇总")
        print(f"{'='*100}")
        
        # 表头
        print(f"\n  {'#':<4} {'任务编号':<8} {'Provider':<12} {'Model':<35} {'耗时':<12} {'输出文件'}")
        print(f"  {'─'*4} {'─'*8} {'─'*12} {'─'*35} {'─'*12} {'─'*50}")
        
        for r in successful_results:
            task_num_str = str(r.task_num) if r.task_num else "N/A"
            provider_str = r.provider or "N/A"
            model_str = (r.model[:32] + "...") if r.model and len(r.model) > 35 else (r.model or "N/A")
            duration_str = format_duration(r.duration) if r.duration else "N/A"
            output_str = r.output_file or "N/A"
            
            print(f"  {r.run_index:<4} {task_num_str:<8} {provider_str:<12} {model_str:<35} {duration_str:<12} {output_str}")
    
    # 失败任务汇总
    failed_results = [r for r in results if not r.success]
    if failed_results:
        print(f"\n{'='*100}")
        print(f"❌ 失败任务汇总")
        print(f"{'='*100}")
        
        for r in failed_results:
            print(f"\n  运行 #{r.run_index}:")
            print(f"    参数: {r.args_str}")
            print(f"    错误: {r.error_message or '未知错误'}")
    
    print(f"\n{'='*100}")
    print(f"🏁 批量运行完成")
    print(f"{'='*100}\n")


def generate_batch_report(results: List[RunResult], batch_folder: str, batch_num: int):
    """
    调用 generate_report_with_charts.py 生成批量结果报告
    
    Args:
        results: 所有运行结果列表
        batch_folder: batch 文件夹路径
        batch_num: batch 编号
    """
    # 收集所有成功的 eval 文件
    eval_files = [r.eval_file for r in results if r.success and r.eval_file]
    
    if not eval_files:
        print("⚠️  没有找到评估结果文件，跳过图表报告生成")
        return
    
    print(f"\n{'='*80}")
    print(f"📊 生成批量评估报告（带图表）")
    print(f"{'='*80}")
    print(f"找到 {len(eval_files)} 个评估结果文件:")
    for f in eval_files:
        print(f"  - {f}")
    print()
    
    try:
        # 设置环境变量，禁用输出缓冲
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        
        # 使用新的 --batches 参数，让 generate_report_with_charts.py 自动处理输出路径
        # 报告会生成在 batch_folder/report/ 目录下
        cmd = f'python3 generate_report_with_charts.py --batches {batch_num}'
        
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        
        # 实时输出
        for line in process.stdout:
            print(line, end='')
            sys.stdout.flush()
        
        process.wait()
        
        if process.returncode == 0:
            # 报告会生成在 batch_folder/report/ 目录下
            report_output = os.path.join(batch_folder, "report", "evaluation_report.html")
            print(f"\n✅ 图表报告生成完成")
            if os.path.exists(report_output):
                print(f"📄 HTML 报告: {report_output}")
            else:
                print(f"📁 报告目录: {os.path.join(batch_folder, 'report')}")
        else:
            print(f"\n⚠️  图表报告生成失败，退出码: {process.returncode}")
            
    except Exception as e:
        print(f"\n❌ 生成报告时发生错误: {e}")


def show_batch_config(combinations):
    """
    调用 request.py --show-config 获取每个组合的解析配置，以表格形式对比展示。
    只显示组合之间有差异的参数列，以及 profile 列。
    """
    import json as _json
    import shutil as _shutil

    configs = []
    for combo in combinations:
        cmd = f"python request.py {combo} --show-config"
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=10
            )
            # 从输出中提取 JSON（最后一个 { ... } 块）
            stdout = result.stdout
            # 找到 JSON 起始位置
            json_start = stdout.rfind("{\n")
            if json_start >= 0:
                cfg = _json.loads(stdout[json_start:])
                configs.append(cfg)
            else:
                configs.append({"_error": f"无法解析输出"})
        except Exception as e:
            configs.append({"_error": str(e)})

    if not configs:
        print("❌ 没有配置可显示")
        return

    # 收集所有键
    all_keys = []
    seen = set()
    for cfg in configs:
        for k in cfg:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    # 找出有差异的键 + 始终显示的键
    always_show = {"profile", "mode", "provider", "model"}
    diff_keys = []
    for k in all_keys:
        if k.startswith("_"):
            continue
        vals = [str(cfg.get(k, "")) for cfg in configs]
        if k in always_show or len(set(vals)) > 1:
            diff_keys.append(k)

    if not diff_keys:
        diff_keys = all_keys  # fallback: 全部显示

    # 计算列宽
    term_width = _shutil.get_terminal_size((120, 40)).columns
    col_data = {}  # key -> [header, val1, val2, ...]
    for k in diff_keys:
        vals = []
        for cfg in configs:
            v = cfg.get(k)
            if v is None:
                vals.append("-")
            elif isinstance(v, bool):
                vals.append("yes" if v else "no")
            elif isinstance(v, list):
                vals.append(",".join(str(x) for x in v))
            else:
                vals.append(str(v))
        col_data[k] = vals

    # 计算每列宽度（header 和值的最大长度）
    col_widths = {}
    for k in diff_keys:
        w = len(k)
        for v in col_data[k]:
            w = max(w, len(v))
        col_widths[k] = w

    # 行号列
    idx_width = max(3, len(str(len(configs))))

    # 打印表头
    header = f"{'#':>{idx_width}}"
    for k in diff_keys:
        header += f"  {k:<{col_widths[k]}}"
    print(f"\n📊 Batch 配置对比（{len(configs)} 个组合，仅显示差异列）:\n")
    print(header)
    print("-" * len(header))

    # 打印每行
    for i, cfg in enumerate(configs):
        row = f"{i+1:>{idx_width}}"
        for k in diff_keys:
            row += f"  {col_data[k][i]:<{col_widths[k]}}"
        print(row)

    print()


def main():
    """主函数"""
    batch_start = datetime.now()
    timestamp_str = batch_start.strftime('%m%d_%H%M%S')
    
    # 生成所有参数组合
    combinations = generate_combinations(args_combo)
    total_runs = len(combinations)

    # --show-config: 显示所有组合的解析配置并退出
    if "--show-config" in sys.argv or "--show_config" in sys.argv:
        show_batch_config(combinations)
        sys.exit(0)

    # 确保 output 目录存在
    os.makedirs("output", exist_ok=True)
    
    # 获取批次编号（独立计数，从1开始）
    batch_num = get_next_batch_num()
    
    # 创建 batch 文件夹
    batch_folder = f"output/batch_{batch_num}_{timestamp_str}"
    os.makedirs(batch_folder, exist_ok=True)
    
    # 创建日志文件（直接放在 batch 文件夹）
    log_path = os.path.join(batch_folder, "batch.log")
    log_file = setup_logging(log_path)
    
    try:
        print(f"\n{'='*80}")
        print(f"🔧 批量运行 request.py")
        print(f"{'='*80}")
        print(f"批次编号: {batch_num}")
        print(f"开始时间: {batch_start.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"总运行次数: {total_runs}")
        print(f"Batch 文件夹: {batch_folder}")
        print(f"{'='*80}\n")
        
        # 显示所有组合
        print("📋 将运行以下组合:")
        for i, combo in enumerate(combinations, 1):
            print(f"   [{i}] {combo}")
        print()
        
        # 运行每个组合
        results: List[RunResult] = []
        stop_reason: Optional[str] = None
        
        for i, args_str in enumerate(combinations, 1):
            result = run_request(args_str, i, total_runs)
            
            # 将 job 文件夹移动到 batch 文件夹
            result = move_job_to_batch(result, batch_folder)
            
            results.append(result)
        
        # 记录结束时间
        batch_end = datetime.now()
        
        # 打印详细汇总
        print_results_summary(results, batch_start, batch_end, stop_reason)
        
        # 生成 batch_summary.html
        generate_batch_summary_html(batch_folder, batch_num, results, batch_start, batch_end, stop_reason)
        
        # 生成批量结果报告（带图表的 evaluation_report.html）
        if GENERATE_REPORT:
            generate_batch_report(results, batch_folder, batch_num)
        
        # 关闭日志文件
        close_logging()
        
        # 打印最终信息
        print(f"\n{'='*80}")
        print(f"🎉 批量运行完成！")
        print(f"{'='*80}")
        print(f"📁 Batch 文件夹: {batch_folder}")
        print(f"📊 Summary: {os.path.join(batch_folder, 'batch_summary.html')}")
        print(f"📝 日志: {log_path}")
        print(f"{'='*80}\n")
        
        # 返回退出码
        fail_count = sum(1 for r in results if not r.success)
        # 若内部自动停止则使用特殊退出码 2，便于上层监控
        if stop_reason:
            sys.exit(2)
        sys.exit(0 if fail_count == 0 else 1)
        
    except Exception as e:
        # 确保异常时也关闭日志
        close_logging()
        print(f"\n❌ 批量运行发生异常: {e}")
        raise


if __name__ == "__main__":
    main()

