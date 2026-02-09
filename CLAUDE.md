# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Mediator is a Python framework for inference and evaluation on the MM-SafetyBench multimodal safety dataset. It sends questions with images to LLMs, collects responses, then uses a GPT judge to evaluate whether responses are safe or unsafe, computing attack rates by category.

## Commands

### Setup
```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### Running Inference
```bash
# Quick test (10 samples, default provider/model)
python request.py --max_tasks 10

# Full run with specific provider
python request.py --provider openai --model "gpt-4o" --max_tasks 50

# Skip evaluation step
python request.py --max_tasks 10 --skip_eval

# VSP provider (local multimodal tool, use lower concurrency)
python request.py --provider vsp --consumers 3 --max_tasks 50

# CoMT-VSP dual-task mode
python request.py --provider comt_vsp --comt_sample_id "deletion-0107" --max_tasks 20
```

### Running Evaluation Separately
```bash
python mmsb_eval.py --jsonl_file output/job_*/results.jsonl
```

### Batch Runs (parameter combinations)
```bash
python batch_request.py
```

### Report Generation
```bash
python generate_report_with_charts.py
```

### Running Tests
```bash
# Single test
python tests/test_failed_answer_detection.py

# All tests (if pytest installed)
python -m pytest tests/

# Or without pytest
for test in tests/test_*.py; do python "$test"; done
```

## Architecture

### Pipeline Flow

`request.py` orchestrates a three-stage pipeline: **Request** (call LLM) → **Eval** (GPT safety judge via `mmsb_eval.py`) → **Metrics** (attack rates by category). By default all three run automatically; use `--skip_eval` to only generate answers.

### Provider System (`provider.py`)

`BaseProvider` defines the async interface: `send(prompt_struct, cfg) -> str`. Concrete providers:

- **OpenAIProvider** / **OpenRouterProvider** — API-based, use OpenAI SDK
- **QwenProvider** — local/remote Qwen via aiohttp
- **VSPProvider** — spawns VisualSketchpad as subprocess, extracts answers from debug logs
- **ComtVspProvider** — dual-task VSP (CoMT object detection + safety evaluation)

Provider is selected via `--provider` flag; factory function `get_provider()` instantiates it.

### Request Processing (`request.py`)

Uses async producer-consumer pattern with configurable `--consumers` count. Items are loaded from MM-SafetyBench JSON files, images encoded to base64, then queued for concurrent LLM calls. Includes failure detection (regex-based) with auto-retry.

### Output Organization

All output goes to `output/`. Each run creates a job folder: `job_{num}_tasks_{total}_{Provider}_{model}_{MMDD_HHMMSS}/` containing `results.jsonl`, `eval.csv`, `console.log`, `summary.html`. A global counter in `output/.task_counter` provides monotonically increasing job numbers. Batch runs group jobs under `batch_{num}_{timestamp}/`.

### Key Modules

| File | Purpose |
|------|---------|
| `request.py` | Main entry point — inference + evaluation pipeline |
| `provider.py` | LLM provider abstraction (OpenAI, OpenRouter, Qwen, VSP, CoMT-VSP) |
| `mmsb_eval.py` | Safety evaluation engine (GPT as judge) |
| `batch_request.py` | Runs parameter combinations, generates batch reports |
| `pseudo_random_sampler.py` | Deterministic per-category sampling for reproducible experiments |
| `generate_report_with_charts.py` | HTML report with matplotlib charts |
| `check_vsp_tool_usage.py` | Analyzes VSP tool usage in results |
| `cleanup_output.py` | Cleans up output directories |
| `view_jsonl.py` | JSONL file viewer/converter |

## Configuration

API keys are managed via `.env` file (loaded by python-dotenv). Key variables:
- `OPENAI_API_KEY` — for OpenAI and OpenRouter providers
- `OPENROUTER_API_KEY` — OpenRouter-specific key
- `QWEN_ENDPOINT`, `QWEN_API_KEY` — Qwen provider
- `VSP_PATH` — path to VisualSketchpad installation
- `COMT_DATA_PATH` — local CoMT dataset path (auto-downloads from HuggingFace if missing)

## Conventions

- Documentation and code comments are in Chinese; code identifiers are in English.
- The project uses no build system or package manager beyond pip — scripts are run directly.
- Tests use unittest and can be run individually as scripts (they auto-configure `sys.path`).
- The JSONL format includes `pred` (model response), `origin` (input metadata), `eval_result` (safe/unsafe), and `meta` (model/params/timestamp).
