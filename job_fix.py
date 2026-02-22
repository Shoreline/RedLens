"""
Job Fix - 断点重传

根据已完成 job 的 results.jsonl，识别失败任务并重新执行。
成功结果保持不变，只重跑失败的部分，最后合并回原文件。

用法:
    python job_fix.py 182
    python job_fix.py 182 --consumers 3
    python job_fix.py 182 --skip_eval
    python job_fix.py 182 --dry_run          # 只查看失败任务，不执行
"""

import os, re, json, sys, glob, time, asyncio, argparse, shutil, random
from datetime import datetime
from typing import List, Dict, Any, Optional

from request import (
    Item, RunConfig, Task, create_prompt, send_with_retry,
    detect_error_from_answer, build_record_for_disk, write_jsonl,
    consumer, format_time, ensure_ssh_tunnels, clean_sensitive_paths,
    MMSB_IMAGE_QUESTION_MAP, provider_to_camelcase, get_folder_label,
)
from provider import get_provider


# ============ Job 目录定位 ============

def find_job_folder(job_num: int) -> str:
    pattern = f"output/job_{job_num}_tasks_*"
    matches = glob.glob(pattern)
    if not matches:
        print(f"❌ 未找到 job {job_num} 的输出目录")
        print(f"   搜索模式: {pattern}")
        sys.exit(1)
    if len(matches) > 1:
        print(f"⚠️  找到多个匹配目录，使用最新的:")
        for m in sorted(matches):
            print(f"   {m}")
    return sorted(matches)[-1]


# ============ 配置恢复 ============

def load_run_config(job_folder: str) -> Optional[Dict[str, Any]]:
    config_path = os.path.join(job_folder, "run_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def infer_mode_provider_from_old(old_provider: str):
    """
    从旧 run_config.json 的 provider 字段推导 mode 和 provider。
    旧格式: "openai"/"openrouter"/"vsp"/"comt_vsp"/"qwen"
    新格式: mode="direct"/"vsp"/"comt_vsp", provider="openai"/"openrouter"
    """
    if old_provider in ("vsp", "comt_vsp"):
        return old_provider, "openrouter"  # 旧 VSP 默认使用 OpenRouter
    elif old_provider in ("openai", "openrouter"):
        return "direct", old_provider
    else:
        # qwen or unknown → treat as direct + openrouter
        return "direct", "openrouter"


def parse_job_folder_name(folder_name: str):
    """从 job 文件夹名提取 mode, provider, model（run_config.json 不存在时的回退方案）"""
    basename = os.path.basename(folder_name)
    # job_182_tasks_153_ComtVsp_Qwen3-VL-8B-Instruct_0219_142044
    match = re.match(r'job_\d+_tasks_\d+_([A-Z][a-zA-Z]*)_(.+)_(\d{4}_\d{6})$', basename)
    if not match:
        return None, None, None
    label_camel = match.group(1)
    model = match.group(2)
    # CamelCase -> snake_case: ComtVsp -> comt_vsp
    label_snake = re.sub(r'(?<!^)([A-Z])', r'_\1', label_camel).lower()
    mode, provider = infer_mode_provider_from_old(label_snake)
    return mode, provider, model


def detect_comt_sample_id(job_folder: str) -> Optional[str]:
    """从 console.log 中检测 comt_sample_id"""
    console_log = os.path.join(job_folder, "console.log")
    if not os.path.exists(console_log):
        return None
    with open(console_log, "r", encoding="utf-8") as f:
        for line in f:
            m = re.search(r'使用指定的CoMT样本:\s*(\S+)', line)
            if m:
                return m.group(1)
    return None


# ============ Results 读取与分析 ============

def load_results(jsonl_path: str):
    """读取 results.jsonl，返回 (success_records, failed_records)"""
    success, failed = [], []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"⚠️  第 {line_num} 行 JSON 解析失败: {e}")
                continue
            if record.get("error_key") is not None:
                failed.append(record)
            else:
                success.append(record)
    return success, failed


def records_to_items(records: List[Dict]) -> List[Item]:
    items = []
    for r in records:
        origin = r.get("origin", {})
        image_path = origin.get("image_path", "")
        if image_path.startswith("~"):
            image_path = os.path.expanduser(image_path)
        items.append(Item(
            index=origin.get("index", ""),
            category=origin.get("category", ""),
            question=origin.get("question", ""),
            image_path=image_path,
            image_type=origin.get("image_type", "SD"),
        ))
    return items


# ============ 重试 Pipeline ============

async def run_fix_pipeline(items: List[Item], cfg: RunConfig):
    """简化版 pipeline：直接处理给定的 items 列表"""

    provider = get_provider(cfg)
    q: asyncio.Queue = asyncio.Queue()
    start_time = time.time()

    progress_state = {
        'completed': 0,
        'total': len(items),
        'start_time': start_time,
        'total_task_time': 0.0,
        'errors': 0,
        'seen': 0,
        'consecutive_error_key': None,
        'consecutive_error_count': 0,
        'stop': False,
        'stop_reason': None,
    }
    progress_lock = asyncio.Lock()

    for item in items:
        prompt_struct = create_prompt(item, mode=cfg.mode)
        await q.put(Task(item=item, prompt_struct=prompt_struct))

    for _ in range(cfg.consumer_size):
        await q.put(None)

    print(f"\n{'='*80}")
    print(f"🔄 开始重试失败任务")
    print(f"{'='*80}")
    print(f"失败任务数: {len(items)}")
    print(f"并发数: {cfg.consumer_size}")
    print(f"模型: {cfg.model}")
    print(f"输出路径: {cfg.save_path}")
    print(f"{'='*80}\n")

    consumers = [
        asyncio.create_task(
            consumer(i, q, provider, cfg, None, progress_state, progress_lock)
        )
        for i in range(cfg.consumer_size)
    ]

    await q.join()
    await asyncio.gather(*consumers)

    total_time = time.time() - start_time
    avg_time = progress_state['total_task_time'] / len(items) if items else 0

    print(f"\n{'='*80}")
    print(f"🎉 重试完成！")
    print(f"{'='*80}")
    print(f"重试任务数: {len(items)}")
    print(f"总耗时: {format_time(total_time)}")
    print(f"平均每任务: {avg_time:.2f}s")
    print(f"{'='*80}\n")

    return progress_state.get('stop_reason')


# ============ 结果合并 ============

def merge_results(original_success: List[Dict], retry_jsonl_path: str, output_path: str):
    """合并原始成功记录和重试结果，按 (category, index) 排序后写回"""
    merged = {}
    for r in original_success:
        key = (r["origin"]["category"], r["origin"]["index"])
        merged[key] = r

    retry_records = []
    if os.path.exists(retry_jsonl_path):
        with open(retry_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    retry_records.append(record)
                    key = (record["origin"]["category"], record["origin"]["index"])
                    merged[key] = record

    def sort_key(r):
        idx = r["origin"]["index"]
        return (r["origin"]["category"], int(idx) if idx.isdigit() else idx)

    sorted_records = sorted(merged.values(), key=sort_key)

    with open(output_path, "w", encoding="utf-8") as f:
        for r in sorted_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    new_success = sum(1 for r in retry_records if r.get("error_key") is None)
    still_failed = sum(1 for r in retry_records if r.get("error_key") is not None)
    return len(retry_records), new_success, still_failed


# ============ 主流程 ============

def main():
    parser = argparse.ArgumentParser(
        description="Job Fix - 断点重传：重试失败的任务",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
用法示例:
  python job_fix.py 182                                # 重试 job 182 的失败任务
  python job_fix.py 182 --consumers 3                  # 指定并发数
  python job_fix.py 182 --comt_sample_id deletion-0107 # 指定 CoMT 样本
  python job_fix.py 182 --skip_eval                    # 跳过评估
  python job_fix.py 182 --dry_run                      # 只显示失败任务，不执行
        """,
    )
    parser.add_argument("job_number", type=int, help="Job 编号")
    parser.add_argument("--consumers", type=int, default=None,
                        help="并发数（默认: 从原始配置读取，或 VSP 默认 3）")
    parser.add_argument("--comt_sample_id", default=None,
                        help="CoMT 样本ID（默认: 从原始配置 / console.log 自动检测）")
    parser.add_argument("--skip_eval", action="store_true", help="跳过评估步骤")
    parser.add_argument("--dry_run", action="store_true", help="只显示失败任务列表，不执行重试")
    parser.add_argument("--eval_model", default=None, help="评估模型（默认: 从原始配置读取）")
    parser.add_argument("--llm_base_url", default=None, help="自定义 LLM API base URL")
    parser.add_argument("--llm_api_key", default=None, help="自定义 LLM API key")
    parser.add_argument("--no-ssh-tunnel", action="store_true",
                       help="跳过自动 SSH tunnel（适用于本地运行或手动管理隧道）")
    args = parser.parse_args()

    # ---- 1. 定位 Job 目录 ----
    job_folder = find_job_folder(args.job_number)
    print(f"📁 Job 目录: {job_folder}")

    # ---- 2. 恢复原始运行配置 ----
    saved_cfg = load_run_config(job_folder)
    if saved_cfg:
        # 新格式 run_config.json 包含 mode 字段
        if "mode" in saved_cfg:
            mode = saved_cfg["mode"]
            provider = saved_cfg["provider"]
        else:
            # 旧格式：从 provider 推导 mode
            mode, provider = infer_mode_provider_from_old(saved_cfg["provider"])
        model = saved_cfg["model"]
        print(f"📋 配置来源: run_config.json")
    else:
        mode, provider, model = parse_job_folder_name(job_folder)
        if not mode:
            print(f"❌ 无法从目录名解析配置: {os.path.basename(job_folder)}")
            sys.exit(1)
        print(f"📋 配置来源: 目录名解析（旧 job 无 run_config.json）")
    print(f"   Mode: {mode}, Provider: {provider}, Model: {model}")

    # ---- 3. 读取并分析 results.jsonl ----
    jsonl_path = os.path.join(job_folder, "results.jsonl")
    if not os.path.exists(jsonl_path):
        print(f"❌ 未找到 results.jsonl: {jsonl_path}")
        sys.exit(1)

    success_records, failed_records = load_results(jsonl_path)
    total = len(success_records) + len(failed_records)

    print(f"\n📊 任务统计:")
    print(f"   总计: {total}")
    print(f"   ✅ 成功: {len(success_records)}")
    print(f"   ❌ 失败: {len(failed_records)}")

    if not failed_records:
        print(f"\n✅ 没有失败的任务，无需重试！")
        return

    # 错误分类统计
    error_summary: Dict[str, int] = {}
    for r in failed_records:
        ek = r.get("error_key", "unknown")
        error_summary[ek] = error_summary.get(ek, 0) + 1
    print(f"\n   错误类型分布:")
    for ek, count in sorted(error_summary.items(), key=lambda x: -x[1]):
        print(f"     {ek}: {count} 个")

    # ---- dry_run 模式 ----
    if args.dry_run:
        print(f"\n🔍 失败任务详情:")
        for r in failed_records:
            o = r.get("origin", {})
            em = (r.get("error_message") or "")[:80]
            print(f"   [{o.get('category')}/{o.get('index')}] {em}")
        return

    # ---- 4. 构建 RunConfig ----
    if saved_cfg:
        temperature = saved_cfg.get("temperature", 0.0)
        top_p = saved_cfg.get("top_p", 1.0)
        max_tokens = saved_cfg.get("max_tokens", 2048)
        seed = saved_cfg.get("seed")
        llm_base_url = args.llm_base_url or saved_cfg.get("llm_base_url")
        eval_model = args.eval_model or saved_cfg.get("eval_model", "gpt-5-mini")
        eval_concurrency = saved_cfg.get("eval_concurrency", 20)
    else:
        sample_meta = (success_records[0] if success_records else failed_records[0]).get("meta", {})
        params = sample_meta.get("params", {})
        temperature = params.get("temperature", 0.0)
        top_p = params.get("top_p", 1.0)
        max_tokens = params.get("max_tokens", 2048)
        seed = params.get("seed")
        llm_base_url = args.llm_base_url
        eval_model = args.eval_model or "gpt-5-mini"
        eval_concurrency = 20

    # 并发数
    consumers = args.consumers
    if consumers is None:
        if saved_cfg:
            consumers = saved_cfg.get("consumers", 3 if mode in ("vsp", "comt_vsp") else 10)
        else:
            consumers = 3 if mode in ("vsp", "comt_vsp") else 10

    # CoMT sample ID
    comt_sample_id = args.comt_sample_id
    if not comt_sample_id and saved_cfg:
        comt_sample_id = saved_cfg.get("comt_sample_id")
    if not comt_sample_id and mode == "comt_vsp":
        comt_sample_id = detect_comt_sample_id(job_folder)
        if comt_sample_id:
            print(f"🎯 从 console.log 检测到 CoMT 样本: {comt_sample_id}")
    if not comt_sample_id and mode == "comt_vsp":
        print(f"❌ CoMT-VSP 模式需要 comt_sample_id，请通过 --comt_sample_id 指定")
        sys.exit(1)

    # 重试结果临时文件
    retry_jsonl = os.path.join(job_folder, "retry_results.jsonl")
    if os.path.exists(retry_jsonl):
        os.remove(retry_jsonl)

    # OpenRouter provider routing
    openrouter_provider = saved_cfg.get("openrouter_provider") if saved_cfg else None

    cfg = RunConfig(
        mode=mode,
        provider=provider,
        model=model,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        seed=seed,
        consumer_size=consumers,
        save_path=retry_jsonl,
        comt_sample_id=comt_sample_id,
        job_folder=job_folder,
        llm_base_url=llm_base_url,
        llm_api_key=args.llm_api_key,
        openrouter_provider=openrouter_provider,
    )

    # ---- 5. 环境准备 ----
    if mode in ("vsp", "comt_vsp") and not args.no_ssh_tunnel:
        if not ensure_ssh_tunnels():
            print("❌ SSH tunnels required but could not be established.")
            sys.exit(1)
        cfg.vsp_batch_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        print(f"🔧 VSP 时间戳: {cfg.vsp_batch_timestamp}")

    # ---- 6. 执行重试 ----
    failed_items = records_to_items(failed_records)
    print(f"\n🚀 开始重试 {len(failed_items)} 个失败任务...\n")

    retry_start = time.time()
    stop_reason = asyncio.run(run_fix_pipeline(failed_items, cfg))
    retry_duration = time.time() - retry_start

    if stop_reason:
        print(f"\n⚠️  自动停止: {stop_reason}")

    # ---- 7. 备份并合并结果 ----
    backup_path = jsonl_path + ".bak"
    shutil.copy2(jsonl_path, backup_path)
    print(f"💾 原始文件已备份: {backup_path}")

    retry_count, new_success, still_failed = merge_results(
        success_records, retry_jsonl, jsonl_path
    )

    print(f"\n📊 重试结果:")
    print(f"   重试任务: {retry_count}")
    print(f"   新增成功: {new_success}")
    print(f"   仍然失败: {still_failed}")
    print(f"   最终成功: {len(success_records) + new_success}/{total}")

    if os.path.exists(retry_jsonl):
        os.remove(retry_jsonl)

    # ---- 8. 清理路径 ----
    if mode in ("vsp", "comt_vsp") and os.path.exists(job_folder):
        clean_stats = clean_sensitive_paths(job_folder)
        if clean_stats["replacements"] > 0:
            print(f"🧹 路径清理: {clean_stats['replacements']} 处")

    # ---- 9. 评估（可选）----
    if not args.skip_eval and not stop_reason and new_success > 0:
        from mmsb_eval import perform_eval_async, cal_metric, add_vsp_tool_usage_field

        print(f"\n{'='*80}")
        print(f"🔍 重新评估全部结果")
        print(f"{'='*80}\n")

        eval_start = time.time()
        asyncio.run(perform_eval_async(
            jsonl_file_path=jsonl_path,
            scenario=None,
            model=eval_model,
            max_tasks=None,
            concurrency=eval_concurrency,
            override=True,
        ))
        eval_duration = time.time() - eval_start
        print(f"   评估耗时: {format_time(eval_duration)}")

        if mode in ("vsp", "comt_vsp"):
            add_vsp_tool_usage_field(jsonl_path)

        cal_metric(jsonl_path, scenario=None)

    # ---- 完成 ----
    print(f"\n{'='*80}")
    print(f"✅ Job {args.job_number} 修复完成！")
    print(f"{'='*80}")
    print(f"   结果文件: {jsonl_path}")
    print(f"   备份文件: {backup_path}")
    print(f"   重试耗时: {format_time(retry_duration)}")
    if still_failed > 0:
        print(f"   ⚠️  仍有 {still_failed} 个任务失败，可再次运行 job_fix.py 重试")
    print()


if __name__ == "__main__":
    main()
