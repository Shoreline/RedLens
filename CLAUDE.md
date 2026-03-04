# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Mediator is a Python framework for inference and evaluation on the MM-SafetyBench multimodal safety dataset. It sends questions with images to LLMs, collects responses, then uses a GPT judge to evaluate whether responses are safe or unsafe, computing attack rates by category.

## Quick Commands

```bash
# Setup
python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt

# Quick test (default: --mode direct --provider openrouter)
python request.py --max_tasks 10

# Direct mode with OpenAI
python request.py --provider openai --model "gpt-5" --max_tasks 50

# Direct mode with custom LLM endpoint
python request.py --llm_base_url "http://autodl:8000/v1" --model "Qwen3-VL-8B" --max_tasks 50

# OpenRouter with pinned upstream provider
python request.py --model "qwen/qwen3-vl-8b-instruct" --openrouter_provider alibaba --max_tasks 50

# VSP mode (use lower concurrency)
python request.py --mode vsp --consumers 3 --max_tasks 50

# VSP mode with Cloudflare Tunnel (替代 SSH，跨国加速 ~400x)
python tools/cf_tunnel.py start                          # 自动检测 Named/Quick Tunnel
python tools/cf_tunnel.py start --quick                  # 强制 Quick Tunnel
python request.py --mode vsp --tunnel cf --max_tasks 50  # 使用 CF tunnel
python tools/cf_tunnel.py status                         # 查看状态
python tools/cf_tunnel.py stop                           # 停止
python tools/cf_tunnel.py setup                          # Named Tunnel 配置指南

# CoMT-VSP dual-task mode
python request.py --mode comt_vsp --comt_sample_id "deletion-0107" --max_tasks 20

# Retry failed tasks in a job (checkpoint-restart)
python job_fix.py 182
python job_fix.py 182 --dry_run  # preview only
python job_fix.py 182 --skip_eval 

# Evaluation only
python mmsb_eval.py --jsonl_file output/job_*/results.jsonl

# Batch runs / Report
python batch_request.py
python generate_report_with_charts.py

# Tests
python -m pytest tests/
```

完整参数参考见 docs/cli_reference.md。

## Architecture

### Pipeline Flow

`request.py` orchestrates a three-stage pipeline: **Request** (call LLM) → **Eval** (GPT safety judge via `mmsb_eval.py`) → **Metrics** (attack rates by category). Use `--skip_eval` to only generate answers.

### Provider System (`provider.py`)

`BaseProvider` defines the async interface: `send(prompt_struct, cfg) -> str`. Concrete providers:

- **OpenAIProvider** / **OpenRouterProvider** — API-based, use OpenAI SDK
- **QwenProvider** — local/remote Qwen via aiohttp
- **VSPProvider** — spawns VisualSketchpad as subprocess, extracts answers from debug logs
- **ComtVspProvider** — sequential dual-task VSP (CoMT object detection first, then safety evaluation as follow-up in the same conversation)

Execution mode is selected via `--mode` (`direct`/`vsp`/`comt_vsp`), and LLM provider via `--provider` (`openai`/`openrouter`, default `openrouter`). Factory function `get_provider()` dispatches based on mode first, then provider. OpenRouterProvider supports `--openrouter_provider` to pin a specific upstream provider (e.g. `together`, `alibaba`). All modes support `--llm_base_url` to override the default LLM endpoint (highest priority). When using a custom endpoint (`--llm_base_url`) that returns `hidden_state` in the API response, all modes (including direct) automatically capture and save hidden states as `.npy` files in `{job_folder}/hidden_states/`. Direct mode captures from the API response's extra fields; VSP/CoMT-VSP modes capture from the subprocess's `hidden_states.json`.

### Tunnel 传输 (`--tunnel`)

VSP/CoMT-VSP 模式需要与 AutoDL 远程主机上的服务通信。`--tunnel` 参数控制传输方式：

- **`ssh`**（默认）— SSH 端口转发，跨国场景受 GFW 限速（~3-5 KB/s）
- **`cf`** — Cloudflare Tunnel，通过 CDN 中继，跨国 ~2 MB/s。自动检测：存在 `.cf_named_tunnel.json` 则使用 Named Tunnel（稳定 URL、单进程），否则回退到 Quick Tunnel（随机 URL）
- **`none`** — 不建立 tunnel（服务已可直接访问时使用）

CF Tunnel 通过 `tools/cf_tunnel.py` 管理，在 AutoDL 上启动 `cloudflared` 进程暴露服务端口。Named Tunnel 使用固定子域名（如 `llm.yuantian.me`），Quick Tunnel 分配随机 `*.trycloudflare.com` URL。运行时配置保存在 `.cf_tunnels.json`，通过环境变量传递给 VSP 子进程。详见 docs/cf_tunnel.md。

### Request Processing (`request.py`)

Uses async producer-consumer pattern with configurable `--consumers` count. Items are loaded from MM-SafetyBench JSON files, images encoded to base64, then queued for concurrent LLM calls. Includes failure detection (regex-based) with auto-retry. Supports `--sampling_rate` for partial dataset runs and `--rate_limit_qps` for rate limiting.

### VSP Post-Processor

VSP 推理完成后可启用后处理（`--vsp_postproc`），支持 `ask`/`sd`/`prebaked` 三种后端和多种图片处理方法。详见 docs/vsp_postprocessor.md。

### Output Organization

All output goes to `output/`. Detailed structure documented in docs/output_structure.md.

- **Job 目录**: `job_{num}_tasks_{total}_{Provider}_{model}_{MMDD_HHMMSS}/` — 包含 results.jsonl, eval.csv, console.log, summary.html, details/ 等
- **Batch 目录**: `batch_{num}_{timestamp}/` — 包含多个 job 目录 + batch_summary.html + report/

### Key Modules

| File | Purpose |
|------|---------|
| `request.py` | Main entry point — inference + evaluation pipeline |
| `provider.py` | LLM provider abstraction (OpenAI, OpenRouter, Qwen, VSP, CoMT-VSP) |
| `mmsb_eval.py` | Safety evaluation engine (GPT as judge) |
| `job_fix.py` | Resume failed tasks in a job (checkpoint-restart) |
| `batch_request.py` | Runs parameter combinations, generates batch reports |
| `pseudo_random_sampler.py` | Deterministic per-category sampling for reproducible experiments |
| `generate_report_with_charts.py` | HTML report with matplotlib charts |
| `check_vsp_tool_usage.py` | Analyzes VSP tool usage in results |
| `copy_sd_pictures.py` | Copies SD pictures to VSP prebaked_images directory |
| `cleanup_output.py` | Cleans up output directories |
| `view_jsonl.py` | JSONL file viewer/converter |
| `compare_hidden_states.py` | Cross-job hidden states difference direction analysis |
| `tools/cf_tunnel.py` | Cloudflare Tunnel 管理（start/stop/status） |

## Configuration

API keys are managed via `.env` file (loaded by python-dotenv). Key variables:
- `OPENAI_API_KEY` — for OpenAI and OpenRouter providers
- `OPENROUTER_API_KEY` — OpenRouter-specific key
- `QWEN_ENDPOINT`, `QWEN_API_KEY` — Qwen provider
- `VSP_PATH` — path to VisualSketchpad installation
- `COMT_DATA_PATH` — local CoMT dataset path (auto-downloads from HuggingFace if missing)

AutoDL 远程日志路径（通过 `ssh seetacloud` 访问）：
- Qwen/LLM server log: `/root/projects/hidden_states/logs/qwen.log`

## Conventions

- Documentation and code comments are in Chinese; code identifiers are in English.
- The project uses no build system or package manager beyond pip — scripts are run directly.
- Tests use unittest and can be run individually as scripts (they auto-configure `sys.path`).
- The JSONL format includes `pred` (model response), `origin` (input metadata), `eval_result` (safe/unsafe), and `meta` (model/params/timestamp).

## Detailed Documentation

- docs/cli_reference.md — request.py 全部命令行参数
- docs/output_structure.md — output 目录详细结构和字段说明
- docs/vsp_postprocessor.md — VSP 后处理器指南（简要索引）
- docs/vsp_post_processing.md — VSP 后处理器详细文档（参数、后端、数据流、调试）
- docs/cf_tunnel.md — Cloudflare Tunnel 使用与原理
- docs/compare_hidden_states.md — Hidden States 差异方向分析（流程、指标、输出）
- COMT_GUIDE.md — CoMT 模式完整指南
