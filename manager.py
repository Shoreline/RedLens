#!/usr/bin/env python3
"""
RedLens Manager — 统一管理 Dashboard

功能:
  - 浏览 output/ 和 output_persisted/ 下的所有结果（job, batch, hs_comp, refusal_dir）
  - Delete / Persist / Retry 操作
  - 从 UI 启动 job / batch / hs_comp / refusal_dir
  - 查看 job/batch 详情（eval 数据、config）
  - 比较同类型 item 的 config
  - 后台进程监控

用法:
    python manager.py              # 默认 port 8765
    python manager.py --port 9000  # 自定义端口
"""

import os
import sys
import csv
import json
import time
import shutil
import asyncio
import mimetypes
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any

import yaml
from aiohttp import web

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
PERSISTED_DIR = BASE_DIR / "output_persisted"
PROFILES_FILE = BASE_DIR / "profiles.yaml"


# ============ Utility ============

def get_dir_size(path: str) -> int:
    total = 0
    for dirpath, _, filenames in os.walk(path, followlinks=False):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                if not os.path.islink(fp):
                    total += os.path.getsize(fp)
            except OSError:
                pass
    return total


def format_file_size(size_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def resolve_profiles() -> dict:
    """Load profiles.yaml, resolve _inherit, return {defaults, profiles}."""
    try:
        raw = yaml.safe_load(PROFILES_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {"defaults": {}, "profiles": {}}

    defaults = raw.get("defaults", {})
    profiles = {}
    for name, vals in raw.items():
        if name == "defaults":
            continue
        resolved = dict(defaults)
        # handle _inherit
        parent = vals.get("_inherit")
        if parent and parent in raw:
            parent_vals = dict(defaults)
            parent_vals.update({k: v for k, v in raw[parent].items() if k != "_inherit"})
            resolved.update(parent_vals)
        resolved.update({k: v for k, v in vals.items() if k != "_inherit"})
        profiles[name] = resolved
    return {"defaults": defaults, "profiles": profiles}


def _is_safe_path(target_path: Path) -> bool:
    """Check path is under output/ or output_persisted/."""
    try:
        target_path.resolve().relative_to(OUTPUT_DIR.resolve())
        return True
    except ValueError:
        pass
    try:
        target_path.resolve().relative_to(PERSISTED_DIR.resolve())
        return True
    except ValueError:
        return False


def _cleanup_empty_batches(parent: Path):
    """删除 parent 目录下不再包含任何 job 链接/目录的 batch 目录"""
    for batch_dir in list(parent.iterdir()):
        if not batch_dir.name.startswith("batch_") or not batch_dir.is_dir():
            continue
        has_job = any(
            e.name.startswith("job_")
            for e in batch_dir.iterdir()
            if e.is_symlink() or e.is_dir()
        )
        if not has_job:
            shutil.rmtree(str(batch_dir))


# ============ Scanners ============

def _extract_job_num(name: str) -> Optional[int]:
    """Extract job number from directory name like job_243_tasks_..."""
    import re
    m = re.match(r"^job_(\d+)", name)
    return int(m.group(1)) if m else None


def scan_job(dirpath: Path, persisted: bool) -> Optional[dict]:
    info: Dict[str, Any] = {
        "type": "job",
        "name": dirpath.name,
        "path": str(dirpath),
        "persisted": persisted,
        "mtime": os.path.getmtime(str(dirpath)),
        "job_num": _extract_job_num(dirpath.name),
    }

    cfg = read_json(dirpath / "run_config.json")
    if cfg:
        info["mode"] = cfg.get("mode", "")
        info["provider"] = cfg.get("provider", "")
        info["model"] = cfg.get("model", "")
        info["temperature"] = cfg.get("temperature")
        info["sampling_rate"] = cfg.get("sampling_rate")
        info["max_tasks"] = cfg.get("max_tasks")
        info["profile"] = cfg.get("profile")
    else:
        info["mode"] = ""
        info["provider"] = ""
        info["model"] = ""

    eval_csv = dirpath / "eval.csv"
    if eval_csv.exists():
        try:
            with open(eval_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                total_unsafe = 0
                total_evaluated = 0
                for row in reader:
                    try:
                        total_unsafe += int(row.get("Unsafe", 0))
                        total_evaluated += int(row.get("Evaluated", 0))
                    except (ValueError, TypeError):
                        pass
                if total_evaluated > 0:
                    info["attack_rate"] = round(total_unsafe / total_evaluated * 100, 2)
                else:
                    info["attack_rate"] = None
        except Exception:
            info["attack_rate"] = None
    else:
        info["attack_rate"] = None

    jsonl = dirpath / "results.jsonl"
    if jsonl.exists():
        try:
            total = 0
            failed = 0
            with open(jsonl, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    total += 1
                    try:
                        rec = json.loads(line)
                        if rec.get("error_key") is not None:
                            failed += 1
                    except json.JSONDecodeError:
                        pass
            info["total_tasks"] = total
            info["failed_tasks"] = failed
        except Exception:
            info["total_tasks"] = None
            info["failed_tasks"] = None
    else:
        info["total_tasks"] = None
        info["failed_tasks"] = None

    info["has_hidden_states"] = (dirpath / "hidden_states").is_dir()
    info["has_eval"] = eval_csv.exists()
    info["has_summary_html"] = (dirpath / "summary.html").exists()

    size = get_dir_size(str(dirpath))
    info["size_bytes"] = size
    info["size_human"] = format_file_size(size)
    return info


def scan_batch(dirpath: Path, persisted: bool) -> Optional[dict]:
    info: Dict[str, Any] = {
        "type": "batch",
        "name": dirpath.name,
        "path": str(dirpath),
        "persisted": persisted,
        "mtime": os.path.getmtime(str(dirpath)),
    }

    state = read_json(dirpath / "batch_state.json")
    if state:
        info["batch_num"] = state.get("batch_num")
        info["created_at"] = state.get("created_at", "")
        info["total_runs"] = state.get("total_runs", 0)
        runs = state.get("runs", [])
        info["completed"] = sum(1 for r in runs if r.get("status") == "completed")
        info["failed"] = sum(1 for r in runs if r.get("status") == "failed")
    else:
        info["batch_num"] = None
        info["total_runs"] = 0
        info["completed"] = 0
        info["failed"] = 0

    size = get_dir_size(str(dirpath))
    info["size_bytes"] = size
    info["size_human"] = format_file_size(size)
    return info


def scan_refusal_dir(dirpath: Path, persisted: bool) -> Optional[dict]:
    info: Dict[str, Any] = {
        "type": "refusal_dir",
        "name": dirpath.name,
        "path": str(dirpath),
        "persisted": persisted,
        "mtime": os.path.getmtime(str(dirpath)),
    }

    summary = read_json(dirpath / "summary.json")
    if summary:
        info["refdir_num"] = summary.get("refdir_num")
        info["timestamp"] = summary.get("timestamp", "")
        overall = summary.get("overall", {})
        info["auc_roc"] = overall.get("auc_roc")
        info["accuracy"] = overall.get("accuracy_at_optimal")
        ds = summary.get("data_stats", {})
        info["total_samples"] = ds.get("total", 0)
        info["safe"] = ds.get("safe", 0)
        info["unsafe"] = ds.get("unsafe", 0)
        train_jobs = summary.get("train_jobs", [])
        info["train_model"] = train_jobs[0].get("model", "") if train_jobs else ""
        params = summary.get("params", {})
        info["sub_task"] = params.get("sub_task", "")
        info["turn"] = params.get("turn", "")
    else:
        info["refdir_num"] = None
        info["auc_roc"] = None

    info["has_report_html"] = (dirpath / "report.html").exists()
    info["has_roc_curve"] = (dirpath / "roc_curve.png").exists()

    size = get_dir_size(str(dirpath))
    info["size_bytes"] = size
    info["size_human"] = format_file_size(size)
    return info


def _load_job_config(job_dir_name: str) -> Optional[dict]:
    """尝试从 output/ 或 output_persisted/ 加载 job 的 run_config.json。"""
    for base in (OUTPUT_DIR, PERSISTED_DIR):
        cfg_path = base / job_dir_name / "run_config.json"
        if cfg_path.exists():
            return read_json(cfg_path)
    return None


def _find_job_path(job_dir_name: str) -> Optional[str]:
    """返回 job 目录的相对路由前缀（output 或 output_persisted）。"""
    for base, prefix in ((OUTPUT_DIR, "output"), (PERSISTED_DIR, "output_persisted")):
        if (base / job_dir_name).is_dir():
            return prefix
    return None


def scan_hs_comp(dirpath: Path, persisted: bool) -> Optional[dict]:
    info: Dict[str, Any] = {
        "type": "hs_comp",
        "name": dirpath.name,
        "path": str(dirpath),
        "persisted": persisted,
        "mtime": os.path.getmtime(str(dirpath)),
    }

    summary = read_json(dirpath / "summary.json")
    if summary:
        info["comp_num"] = summary.get("comp_num")
        info["timestamp"] = summary.get("timestamp", "")
        info["matched_tasks"] = summary.get("matched_tasks", 0)
        j1 = summary.get("job1", {})
        j2 = summary.get("job2", {})
        info["job1_num"] = j1.get("num")
        info["job2_num"] = j2.get("num")
        info["job1_sub"] = j1.get("sub_task", "")
        info["job1_turn"] = j1.get("turn", "")
        info["job1_dir"] = j1.get("dir", "")
        info["job2_sub"] = j2.get("sub_task", "")
        info["job2_turn"] = j2.get("turn", "")
        info["job2_dir"] = j2.get("dir", "")

        # 加载 job configs
        for side in ("job1", "job2"):
            dir_name = info.get(f"{side}_dir", "")
            cfg = _load_job_config(dir_name) if dir_name else None
            route_prefix = _find_job_path(dir_name) if dir_name else None
            if cfg:
                info[f"{side}_config"] = cfg
            if route_prefix:
                info[f"{side}_route"] = route_prefix

        results = summary.get("results", {})
        all_cos = []
        for cat_data in results.values():
            cos = cat_data.get("cos_sim", {})
            vals = cos.get("values", [])
            all_cos.extend(vals)
        info["mean_cosine"] = round(sum(all_cos) / len(all_cos), 4) if all_cos else None
    else:
        info["comp_num"] = None
        info["mean_cosine"] = None

    info["has_summary_png"] = (dirpath / "hs_diff_summary.png").exists()

    size = get_dir_size(str(dirpath))
    info["size_bytes"] = size
    info["size_human"] = format_file_size(size)
    return info


def scan_all() -> List[dict]:
    items = []
    for base, persisted in [(OUTPUT_DIR, False), (PERSISTED_DIR, True)]:
        if not base.is_dir():
            continue
        for entry in base.iterdir():
            if not entry.is_dir():
                continue
            name = entry.name
            if name.startswith("."):
                continue
            try:
                if name.startswith("job_"):
                    item = scan_job(entry, persisted)
                elif name.startswith("batch_"):
                    item = scan_batch(entry, persisted)
                elif name.startswith("refusal_dir_"):
                    item = scan_refusal_dir(entry, persisted)
                elif name.startswith("hidden_state_comp_"):
                    item = scan_hs_comp(entry, persisted)
                else:
                    continue
                if item:
                    items.append(item)
            except Exception:
                pass

    items.sort(key=lambda x: x.get("mtime", 0), reverse=True)
    return items


# ============ Process Manager ============

class ProcessManager:
    """Manages multiple concurrent subprocess launches."""

    def __init__(self):
        self._processes: Dict[str, dict] = {}
        self._counter = 0

    def _next_id(self) -> str:
        self._counter += 1
        return f"proc_{self._counter}"

    async def launch(self, proc_type: str, cmd: List[str], label: str) -> str:
        pid = self._next_id()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(BASE_DIR),
            text=True,
            bufsize=1,
        )
        entry = {
            "id": pid,
            "type": proc_type,
            "label": label,
            "cmd": cmd,
            "process": proc,
            "output_lines": [],
            "started_at": time.time(),
        }
        self._processes[pid] = entry
        asyncio.create_task(self._read_output(pid))
        return pid

    async def _read_output(self, pid: str):
        entry = self._processes.get(pid)
        if not entry:
            return
        proc = entry["process"]
        loop = asyncio.get_event_loop()
        while proc.poll() is None:
            line = await loop.run_in_executor(None, proc.stdout.readline)
            if line:
                entry["output_lines"].append(line.rstrip("\n"))
        if proc.stdout:
            for line in proc.stdout:
                entry["output_lines"].append(line.rstrip("\n"))

    def _is_running(self, entry: dict) -> bool:
        return entry["process"].poll() is None

    @staticmethod
    def _extract_result_link(entry: dict) -> Optional[str]:
        """从进程输出中提取结果目录，返回可点击的链接路径。"""
        import re
        # 匹配 "输出目录: xxx" 模式（所有脚本共用）
        for line in reversed(entry["output_lines"]):
            m = re.search(r"输出目录:\s*(\S+)", line)
            if m:
                dirname = m.group(1)
                # 根据目录名选择最佳结果页面
                base = OUTPUT_DIR / dirname
                for candidate in ("report.html", "summary.html", "hs_diff_summary.png"):
                    if (base / candidate).exists():
                        return f"/output/{dirname}/{candidate}"
                # 目录存在但无已知结果文件，链接到目录本身
                if base.is_dir():
                    return f"/output/{dirname}/"
                return None
        return None

    def _entry_to_dict(self, entry: dict) -> dict:
        running = self._is_running(entry)
        rc = entry["process"].returncode if not running else None
        d = {
            "id": entry["id"],
            "type": entry["type"],
            "label": entry["label"],
            "running": running,
            "return_code": rc,
            "started_at": entry["started_at"],
            "elapsed": round(time.time() - entry["started_at"], 1),
            "output_tail": entry["output_lines"][-80:],
        }
        if not running and rc == 0:
            d["result_link"] = self._extract_result_link(entry)
        return d

    def status_all(self) -> List[dict]:
        return [self._entry_to_dict(e) for e in self._processes.values()]

    def status_one(self, pid: str) -> Optional[dict]:
        entry = self._processes.get(pid)
        if not entry:
            return None
        return self._entry_to_dict(entry)

    def clear_finished(self):
        to_del = [pid for pid, e in self._processes.items() if not self._is_running(e)]
        for pid in to_del:
            del self._processes[pid]


proc_manager = ProcessManager()


# ============ API Handlers ============

async def handle_index(request: web.Request) -> web.Response:
    return web.Response(text=DASHBOARD_HTML, content_type="text/html")


async def handle_items(request: web.Request) -> web.Response:
    items = await asyncio.get_event_loop().run_in_executor(None, scan_all)
    return web.json_response(items)


async def handle_delete(request: web.Request) -> web.Response:
    data = await request.json()
    target_path = Path(data.get("path", ""))
    if not _is_safe_path(target_path):
        return web.json_response({"error": "Invalid path"}, status=400)
    if not target_path.exists() and not target_path.is_symlink():
        return web.json_response({"error": "Path not found"}, status=404)

    deleted_name = target_path.name

    if deleted_name.startswith("batch_"):
        # 删除 batch：只删 batch 目录自身（含 symlink），不删真实 job 目录
        # shutil.rmtree 默认不跟随顶层 symlink 子项，但 os.walk 会。
        # 所以先手动移除 symlink，再 rmtree 剩余文件。
        for child in target_path.iterdir():
            if child.is_symlink():
                child.unlink()
        shutil.rmtree(str(target_path))
    elif deleted_name.startswith("job_"):
        # 删除 job：先清理所有 batch 中指向它的 symlink，再删空 batch
        parent = target_path.parent
        real_path = target_path.resolve()
        for batch_dir in parent.iterdir():
            if batch_dir.name.startswith("batch_") and batch_dir.is_dir():
                link = batch_dir / deleted_name
                if link.is_symlink():
                    try:
                        if link.resolve() == real_path:
                            link.unlink()
                    except Exception:
                        pass
        shutil.rmtree(str(target_path))
        # 清理不再包含任何 job 的 batch 目录
        _cleanup_empty_batches(parent)
    else:
        shutil.rmtree(str(target_path))

    return web.json_response({"ok": True, "deleted": str(target_path)})


async def handle_persist(request: web.Request) -> web.Response:
    data = await request.json()
    target_path = Path(data.get("path", ""))
    if not target_path.exists():
        return web.json_response({"error": "Path not found"}, status=404)

    def _relink_job_in_batches(old_path: Path, new_path: Path):
        """移动 job 后，更新所有 batch 中指向它的 symlink"""
        job_name = old_path.name
        if not job_name.startswith("job_"):
            return
        old_resolved = old_path.resolve()
        for search_dir in [OUTPUT_DIR, PERSISTED_DIR]:
            if not search_dir.is_dir():
                continue
            for batch_dir in search_dir.iterdir():
                if batch_dir.name.startswith("batch_") and batch_dir.is_dir():
                    link = batch_dir / job_name
                    if link.is_symlink():
                        try:
                            if link.resolve() == old_resolved:
                                link.unlink()
                                link.symlink_to(new_path.resolve())
                        except Exception:
                            pass

    try:
        target_path.resolve().relative_to(OUTPUT_DIR.resolve())
        PERSISTED_DIR.mkdir(exist_ok=True)
        dest = PERSISTED_DIR / target_path.name
        _relink_job_in_batches(target_path, dest)
        shutil.move(str(target_path), str(dest))
        return web.json_response({"ok": True, "action": "persisted", "new_path": str(dest)})
    except ValueError:
        pass

    try:
        target_path.resolve().relative_to(PERSISTED_DIR.resolve())
        dest = OUTPUT_DIR / target_path.name
        _relink_job_in_batches(target_path, dest)
        shutil.move(str(target_path), str(dest))
        return web.json_response({"ok": True, "action": "unpersisted", "new_path": str(dest)})
    except ValueError:
        return web.json_response({"error": "Invalid path"}, status=400)


async def handle_retry(request: web.Request) -> web.Response:
    data = await request.json()
    job_num = data.get("job_num")
    if job_num is None:
        return web.json_response({"error": "job_num required"}, status=400)
    try:
        cmd = [sys.executable, "job_fix.py", str(int(job_num))]
        pid = await proc_manager.launch("retry", cmd, f"Retry job {job_num}")
        return web.json_response({"ok": True, "job_num": job_num, "process_id": pid})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


# ---- Profiles API ----

async def handle_profiles(request: web.Request) -> web.Response:
    data = await asyncio.get_event_loop().run_in_executor(None, resolve_profiles)
    return web.json_response(data)


# ---- Job Detail API ----

async def handle_job_detail(request: web.Request) -> web.Response:
    path = request.query.get("path", "")
    dirpath = Path(path)
    if not dirpath.is_dir() or not _is_safe_path(dirpath):
        return web.json_response({"error": "Invalid path"}, status=400)

    result: Dict[str, Any] = {"path": path, "name": dirpath.name}

    # run_config.json
    cfg = read_json(dirpath / "run_config.json")
    result["config"] = cfg or {}

    # eval.csv
    eval_csv = dirpath / "eval.csv"
    eval_rows = []
    if eval_csv.exists():
        try:
            with open(eval_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    eval_rows.append(dict(row))
        except Exception:
            pass
    result["eval"] = eval_rows

    # results.jsonl stats
    jsonl = dirpath / "results.jsonl"
    stats = {"total": 0, "success": 0, "failed": 0, "error_types": {}}
    if jsonl.exists():
        try:
            with open(jsonl, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    stats["total"] += 1
                    try:
                        rec = json.loads(line)
                        ek = rec.get("error_key")
                        if ek is not None:
                            stats["failed"] += 1
                            stats["error_types"][str(ek)] = stats["error_types"].get(str(ek), 0) + 1
                        else:
                            stats["success"] += 1
                    except json.JSONDecodeError:
                        pass
        except Exception:
            pass
    result["results_stats"] = stats

    result["has_summary_html"] = (dirpath / "summary.html").exists()
    result["has_hidden_states"] = (dirpath / "hidden_states").is_dir()
    result["has_console_log"] = (dirpath / "console.log").exists()

    return web.json_response(result)


# ---- Batch Detail API ----

async def handle_batch_detail(request: web.Request) -> web.Response:
    path = request.query.get("path", "")
    dirpath = Path(path)
    if not dirpath.is_dir() or not _is_safe_path(dirpath):
        return web.json_response({"error": "Invalid path"}, status=400)

    result: Dict[str, Any] = {"path": path, "name": dirpath.name}

    state = read_json(dirpath / "batch_state.json")
    result["batch_state"] = state or {}

    # Per-run eval summary
    runs_summary = []
    if state:
        for run in state.get("runs", []):
            rs: Dict[str, Any] = {
                "index": run.get("index"),
                "args_str": run.get("args_str", ""),
                "status": run.get("status", ""),
                "duration_secs": run.get("duration_secs"),
                "job_folder": run.get("job_folder", ""),
            }
            # Try to get attack rate from sub-job eval.csv
            jf = run.get("job_folder", "")
            if jf:
                job_eval = dirpath / jf / "eval.csv"
                if not job_eval.exists():
                    # try absolute
                    job_eval = Path(jf) / "eval.csv"
                if job_eval.exists():
                    try:
                        total_unsafe = total_evaluated = 0
                        with open(job_eval, "r", encoding="utf-8") as f:
                            reader = csv.DictReader(f)
                            for row in reader:
                                try:
                                    total_unsafe += int(row.get("Unsafe", 0))
                                    total_evaluated += int(row.get("Evaluated", 0))
                                except (ValueError, TypeError):
                                    pass
                        if total_evaluated > 0:
                            rs["attack_rate"] = round(total_unsafe / total_evaluated * 100, 2)
                    except Exception:
                        pass
            runs_summary.append(rs)
    result["runs"] = runs_summary

    result["has_batch_summary"] = (dirpath / "batch_summary.html").exists()
    result["has_eval_report"] = (dirpath / "report" / "evaluation_report.html").exists()

    return web.json_response(result)


# ---- Config Compare API ----

async def handle_compare_configs(request: web.Request) -> web.Response:
    data = await request.json()
    paths = data.get("paths", [])
    if len(paths) < 2:
        return web.json_response({"error": "Need at least 2 paths"}, status=400)

    configs = []
    labels = []
    for p in paths:
        dp = Path(p)
        cfg = None
        if (dp / "run_config.json").exists():
            cfg = read_json(dp / "run_config.json")
        elif (dp / "batch_state.json").exists():
            bs = read_json(dp / "batch_state.json")
            if bs:
                # Use run_configs.json first, fallback to batch_state
                rc = read_json(dp / "run_configs.json")
                if rc and isinstance(rc, list) and rc:
                    cfg = rc[0]
                else:
                    cfg = {k: v for k, v in bs.items() if k != "runs"}
        elif (dp / "summary.json").exists():
            cfg = read_json(dp / "summary.json")
        configs.append(cfg or {})
        labels.append(dp.name)

    # Find all keys
    all_keys = set()
    for c in configs:
        all_keys.update(c.keys())

    diff_keys = []
    same_keys = []
    for key in sorted(all_keys):
        values = [c.get(key) for c in configs]
        # Convert to comparable form
        str_values = [json.dumps(v, sort_keys=True, default=str) for v in values]
        if len(set(str_values)) > 1:
            diff_keys.append({"key": key, "values": values})
        else:
            same_keys.append({"key": key, "value": values[0]})

    return web.json_response({
        "labels": labels,
        "diff_keys": diff_keys,
        "same_keys": same_keys,
    })


# ---- Launch APIs ----

async def handle_launch_job(request: web.Request) -> web.Response:
    data = await request.json()
    profile = data.get("profile")
    overrides = data.get("overrides", {})

    cmd = [sys.executable, "request.py"]
    if profile:
        cmd += ["--profile", profile]

    # Map overrides to CLI args
    flag_args = {"skip_eval", "vsp_postproc", "show_config"}
    list_args = {"image_types", "categories"}
    for key, val in overrides.items():
        if val is None or val == "":
            continue
        flag = f"--{key}"
        if key in flag_args:
            if val:
                cmd.append(flag)
        elif key in list_args:
            if isinstance(val, list) and val:
                cmd.append(flag)
                cmd.extend(str(v) for v in val)
        else:
            cmd += [flag, str(val)]

    label = f"Job: {profile or 'custom'}"
    if overrides.get("model"):
        label += f" ({overrides['model']})"
    if overrides.get("max_tasks"):
        label += f" max={overrides['max_tasks']}"

    pid = await proc_manager.launch("job", cmd, label)
    return web.json_response({"ok": True, "process_id": pid, "cmd": " ".join(cmd)})


async def handle_launch_batch(request: web.Request) -> web.Response:
    data = await request.json()
    cmd = [sys.executable, "batch_request.py"]
    resume = data.get("resume")
    if resume:
        cmd += ["--resume", str(resume)]
    pid = await proc_manager.launch("batch", cmd, f"Batch{' resume=' + str(resume) if resume else ''}")
    return web.json_response({"ok": True, "process_id": pid, "cmd": " ".join(cmd)})


async def handle_launch_hs_comp(request: web.Request) -> web.Response:
    data = await request.json()
    job1 = data.get("job1")
    job2 = data.get("job2")
    if not job1 or not job2:
        return web.json_response({"error": "job1 and job2 required"}, status=400)

    cmd = [sys.executable, "compare_hidden_states.py", str(job1), str(job2)]
    for key in ("sub_task", "turn", "sub_task1", "sub_task2", "turn1", "turn2"):
        if data.get(key):
            cmd += [f"--{key}", str(data[key])]
    if data.get("detailed") is False:
        cmd.append("--no-detailed")

    label = f"HS Comp: {job1} vs {job2}"
    pid = await proc_manager.launch("hs_comp", cmd, label)
    return web.json_response({"ok": True, "process_id": pid, "cmd": " ".join(cmd)})


async def handle_launch_refusal_dir(request: web.Request) -> web.Response:
    data = await request.json()
    job_nums = data.get("job_nums", [])
    cmd = [sys.executable, "refusal_direction.py"] + [str(j) for j in job_nums]

    list_args = {"batch", "test_job", "test_batch"}
    for key in list_args:
        vals = data.get(key)
        if vals:
            cmd.append(f"--{key}")
            cmd.extend(str(v) for v in (vals if isinstance(vals, list) else [vals]))

    for key in ("sub_task", "turn", "split_ratio", "n_folds", "score_method", "seed", "load_direction"):
        if data.get(key) is not None and data.get(key) != "":
            cmd += [f"--{key}", str(data[key])]

    if data.get("save_direction"):
        cmd.append("--save_direction")

    label = f"Refusal Dir: jobs={job_nums}"
    pid = await proc_manager.launch("refusal_dir", cmd, label)
    return web.json_response({"ok": True, "process_id": pid, "cmd": " ".join(cmd)})


# ---- Process Status API ----

async def handle_process_status(request: web.Request) -> web.Response:
    pid = request.query.get("id")
    if pid:
        s = proc_manager.status_one(pid)
        return web.json_response(s or {"error": "Not found"})
    return web.json_response(proc_manager.status_all())


async def handle_process_clear(request: web.Request) -> web.Response:
    proc_manager.clear_finished()
    return web.json_response({"ok": True})


# ---- Static file serving ----

async def handle_static_file(request: web.Request) -> web.Response:
    """Serve files from output/ or output_persisted/."""
    rel = request.match_info.get("path", "")
    # Determine base
    route_prefix = request.path.split("/")[1]  # "output" or "output_persisted"
    if route_prefix == "output_persisted":
        base = PERSISTED_DIR
    else:
        base = OUTPUT_DIR

    file_path = (base / rel).resolve()
    # Security check
    try:
        file_path.relative_to(base.resolve())
    except ValueError:
        return web.Response(status=403, text="Forbidden")

    if not file_path.exists() or not file_path.is_file():
        return web.Response(status=404, text="Not found")

    content_type, _ = mimetypes.guess_type(str(file_path))
    if content_type is None:
        content_type = "application/octet-stream"

    return web.FileResponse(file_path)


# ============ HTML Dashboard ============

DASHBOARD_HTML = (
    '<!DOCTYPE html>\n'
    '<html lang="en">\n'
    '<head>\n'
    '<meta charset="UTF-8">\n'
    '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
    '<title>RedLens Manager</title>\n'
    '<style>\n'
    '* { margin: 0; padding: 0; box-sizing: border-box; }\n'
    'body {\n'
    '    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, sans-serif;\n'
    '    background: #f0f2f5; color: #2d3436; min-height: 100vh;\n'
    '}\n'
    '.header {\n'
    '    background: #fff; border-bottom: 1px solid #e1e4e8;\n'
    '    padding: 16px 24px; position: sticky; top: 0; z-index: 100;\n'
    '    box-shadow: 0 1px 3px rgba(0,0,0,0.06);\n'
    '}\n'
    '.header-top { display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px; }\n'
    '.header h1 { font-size: 20px; font-weight: 700; color: #2d3436; display: flex; align-items: center; gap: 8px; }\n'
    '.header h1 .logo { color: #d63031; }\n'
    '.stats-row { display: flex; gap: 16px; font-size: 13px; color: #636e72; flex-wrap: wrap; margin-top: 8px; }\n'
    '.stats-row span { display: flex; align-items: center; gap: 4px; }\n'
    '.stat-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }\n'
    '.btn { padding: 6px 14px; border-radius: 6px; border: 1px solid #ddd; background: #fff; cursor: pointer; font-size: 13px; font-weight: 500; transition: all 0.15s; display: inline-flex; align-items: center; gap: 4px; }\n'
    '.btn:hover { background: #f8f9fa; border-color: #ccc; }\n'
    '.btn-primary { background: #0984e3; color: #fff; border-color: #0984e3; }\n'
    '.btn-primary:hover { background: #0773c5; }\n'
    '.btn-danger { color: #d63031; border-color: #fab1a0; }\n'
    '.btn-danger:hover { background: #fff5f5; border-color: #d63031; }\n'
    '.btn-success { color: #00b894; border-color: #a3e4d7; }\n'
    '.btn-success:hover { background: #f0fff4; }\n'
    '.btn-sm { padding: 4px 10px; font-size: 12px; }\n'
    '.btn-warning { background: #fdcb6e; color: #2d3436; border-color: #f9ca24; }\n'
    '.btn-warning:hover { background: #f9ca24; }\n'
    # Nav tabs
    '.nav-tabs { display: flex; gap: 0; margin-top: 12px; }\n'
    '.nav-tab { padding: 8px 20px; cursor: pointer; font-size: 13px; font-weight: 600; border: 1px solid #e1e4e8; border-bottom: none; background: #f0f2f5; color: #636e72; border-radius: 8px 8px 0 0; margin-right: -1px; transition: all 0.15s; }\n'
    '.nav-tab.active { background: #fff; color: #2d3436; border-bottom: 1px solid #fff; margin-bottom: -1px; z-index: 1; }\n'
    '.nav-tab:hover:not(.active) { background: #e8eaed; }\n'
    # Toolbar
    '.toolbar { display: flex; align-items: center; gap: 8px; padding: 12px 24px; background: #fff; border-bottom: 1px solid #e1e4e8; flex-wrap: wrap; }\n'
    '.rd-eval-tab { border: 1px solid #ddd; background: #fff; font-size: 11px; padding: 4px 10px; cursor: pointer; transition: all 0.15s; }\n'
    '.rd-eval-tab:hover { background: #f0f2f5; }\n'
    '.rd-eval-tab.active { background: #0984e3; color: #fff; border-color: #0984e3; }\n'
    '.filter-btn { padding: 5px 12px; border-radius: 20px; border: 1px solid #ddd; background: #fff; cursor: pointer; font-size: 12px; font-weight: 500; transition: all 0.15s; }\n'
    '.filter-btn:hover { background: #f0f2f5; }\n'
    '.filter-btn.active { background: #0984e3; color: #fff; border-color: #0984e3; }\n'
    '.search-box { flex: 1; min-width: 200px; padding: 6px 12px; border: 1px solid #ddd; border-radius: 6px; font-size: 13px; outline: none; }\n'
    '.search-box:focus { border-color: #0984e3; box-shadow: 0 0 0 2px rgba(9,132,227,0.15); }\n'
    '.container { max-width: 1200px; margin: 0 auto; padding: 16px 24px; }\n'
    # Cards
    '.card { background: #fff; border: 1px solid #e1e4e8; border-radius: 10px; padding: 16px 20px; margin-bottom: 10px; transition: box-shadow 0.15s; display: flex; align-items: flex-start; gap: 16px; }\n'
    '.card:hover { box-shadow: 0 2px 8px rgba(0,0,0,0.08); }\n'
    '.card.persisted { border-left: 3px solid #00b894; }\n'
    '.card-left { flex: 1; min-width: 0; cursor: pointer; }\n'
    '.card-header { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; margin-bottom: 6px; }\n'
    '.badge { padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; text-transform: uppercase; white-space: nowrap; }\n'
    '.badge-job { background: #dfe6fd; color: #0652DD; }\n'
    '.badge-batch { background: #fdebd0; color: #e67e22; }\n'
    '.badge-refusal_dir { background: #fadbd8; color: #c0392b; }\n'
    '.badge-hs_comp { background: #d5f5e3; color: #1e8449; }\n'
    '.badge-persisted { background: #d5f5e3; color: #00b894; font-size: 10px; }\n'
    '.badge-temp { background: #ffeaa7; color: #d35400; font-size: 10px; }\n'
    '.card-title { font-size: 14px; font-weight: 600; color: #2d3436; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }\n'
    '.card-meta { display: flex; gap: 16px; flex-wrap: wrap; font-size: 12px; color: #636e72; margin-bottom: 4px; }\n'
    '.card-meta .metric { display: flex; align-items: center; gap: 4px; }\n'
    '.mv { font-weight: 600; color: #2d3436; }\n'
    '.md { color: #d63031; font-weight: 600; }\n'
    '.ms { color: #00b894; font-weight: 600; }\n'
    '.card-actions { display: flex; gap: 6px; align-items: flex-start; flex-shrink: 0; flex-wrap: wrap; }\n'
    '.card-check { width: 18px; height: 18px; cursor: pointer; margin-top: 2px; flex-shrink: 0; }\n'
    # Modal
    '.modal-overlay { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.4); z-index: 200; justify-content: center; align-items: flex-start; padding-top: 40px; }\n'
    '.modal-overlay.visible { display: flex; }\n'
    '.modal { background: #fff; border-radius: 12px; width: 90%; max-width: 900px; max-height: 85vh; overflow-y: auto; box-shadow: 0 8px 30px rgba(0,0,0,0.15); }\n'
    '.modal-header { display: flex; justify-content: space-between; align-items: center; padding: 16px 24px; border-bottom: 1px solid #e1e4e8; position: sticky; top: 0; background: #fff; border-radius: 12px 12px 0 0; z-index: 1; }\n'
    '.modal-header h2 { font-size: 16px; font-weight: 700; }\n'
    '.modal-close { background: none; border: none; font-size: 20px; cursor: pointer; color: #636e72; padding: 4px 8px; }\n'
    '.modal-body { padding: 20px 24px; }\n'
    # Tables
    '.data-table { width: 100%; border-collapse: collapse; font-size: 13px; margin: 12px 0; }\n'
    '.data-table th { text-align: left; padding: 8px 12px; background: #f8f9fa; border: 1px solid #e1e4e8; font-weight: 600; white-space: nowrap; }\n'
    '.data-table td { padding: 8px 12px; border: 1px solid #e1e4e8; }\n'
    '.data-table tr:hover { background: #f8f9fa; }\n'
    '.diff-val { background: #fff3cd; }\n'
    '.same-val { color: #b2bec3; }\n'
    '.rate-high { color: #d63031; font-weight: 700; }\n'
    '.rate-low { color: #00b894; font-weight: 700; }\n'
    # Launch panel
    '.launch-panel { display: none; }\n'
    '.launch-panel.visible { display: block; }\n'
    '.launch-section { background: #fff; border: 1px solid #e1e4e8; border-radius: 10px; padding: 20px; margin-bottom: 16px; }\n'
    '.launch-section h3 { font-size: 15px; font-weight: 700; margin-bottom: 12px; display: flex; align-items: center; gap: 8px; }\n'
    '.form-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 12px; }\n'
    '.form-group { display: flex; flex-direction: column; gap: 4px; }\n'
    '.form-group label { font-size: 12px; font-weight: 600; color: #636e72; }\n'
    '.form-group input, .form-group select { padding: 6px 10px; border: 1px solid #ddd; border-radius: 6px; font-size: 13px; outline: none; }\n'
    '.form-group input:focus, .form-group select:focus { border-color: #0984e3; box-shadow: 0 0 0 2px rgba(9,132,227,0.15); }\n'
    '.form-group-wide { grid-column: 1 / -1; }\n'
    '.form-section-title { font-size: 13px; font-weight: 700; color: #0984e3; margin: 12px 0 4px 0; grid-column: 1 / -1; border-bottom: 1px solid #e1e4e8; padding-bottom: 4px; }\n'
    '.form-actions { margin-top: 16px; display: flex; gap: 8px; }\n'
    '.collapse-toggle { cursor: pointer; color: #0984e3; font-size: 12px; margin-left: auto; }\n'
    '.collapsible { display: none; }\n'
    '.collapsible.open { display: grid; }\n'
    # Process monitor
    '.proc-monitor { position: fixed; bottom: 0; left: 0; right: 0; background: #fff; border-top: 1px solid #e1e4e8; z-index: 150; max-height: 300px; overflow-y: auto; transition: max-height 0.3s; }\n'
    '.proc-monitor.collapsed { max-height: 36px; overflow: hidden; }\n'
    '.proc-monitor-header { display: flex; justify-content: space-between; align-items: center; padding: 8px 16px; cursor: pointer; background: #2d3436; color: #fff; font-size: 13px; font-weight: 600; }\n'
    '.proc-item { padding: 8px 16px; border-bottom: 1px solid #f0f2f5; font-size: 12px; display: flex; align-items: center; gap: 12px; }\n'
    '.proc-status { display: inline-block; width: 8px; height: 8px; border-radius: 50%; }\n'
    '.proc-status.running { background: #0984e3; animation: pulse 1.5s infinite; }\n'
    '.proc-status.done { background: #00b894; }\n'
    '.proc-status.failed { background: #d63031; }\n'
    '@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.4; } }\n'
    '.proc-label { flex: 1; }\n'
    '.proc-log { background: #1e272e; color: #dfe6e9; font-family: Menlo, Consolas, monospace; font-size: 11px; padding: 8px 16px; max-height: 150px; overflow-y: auto; white-space: pre-wrap; word-break: break-all; display: none; }\n'
    '.proc-log.visible { display: block; }\n'
    # Compare bar
    '.compare-bar { display: none; position: fixed; bottom: 44px; left: 0; right: 0; background: #fff; border-top: 1px solid #e1e4e8; padding: 10px 24px; z-index: 140; box-shadow: 0 -2px 8px rgba(0,0,0,0.06); text-align: center; }\n'
    '.compare-bar.visible { display: block; }\n'
    '.empty-state { text-align: center; padding: 60px 20px; color: #b2bec3; font-size: 15px; }\n'
    '.loading { text-align: center; padding: 40px; color: #636e72; font-size: 14px; }\n'
    '</style>\n'
    '</head>\n'
    '<body>\n'

    # ---- Header ----
    '<div class="header">\n'
    '  <div class="header-top">\n'
    '    <h1><span class="logo">&#9679;</span> RedLens Manager</h1>\n'
    '    <div style="display:flex;gap:8px">\n'
    '      <button class="btn btn-primary" onclick="fetchItems()">&#8635; Reload</button>\n'
    '    </div>\n'
    '  </div>\n'
    '  <div class="stats-row" id="stats-row"></div>\n'
    '  <div class="nav-tabs">\n'
    '    <div class="nav-tab active" data-tab="dashboard" onclick="switchTab(\'dashboard\',this)">Dashboard</div>\n'
    '    <div class="nav-tab" data-tab="launch" onclick="switchTab(\'launch\',this)">Launch</div>\n'
    '  </div>\n'
    '</div>\n'

    # ---- Dashboard tab (toolbar + items) ----
    '<div id="tab-dashboard">\n'
    '<div class="toolbar">\n'
    '  <button class="filter-btn active" data-filter="all" onclick="setFilter(\'all\',this)">All</button>\n'
    '  <button class="filter-btn" data-filter="job" onclick="setFilter(\'job\',this)">Jobs</button>\n'
    '  <button class="filter-btn" data-filter="batch" onclick="setFilter(\'batch\',this)">Batches</button>\n'
    '  <button class="filter-btn" data-filter="hs_comp" onclick="setFilter(\'hs_comp\',this)">HS Comp</button>\n'
    '  <button class="filter-btn" data-filter="refusal_dir" onclick="setFilter(\'refusal_dir\',this)">Refusal Dir</button>\n'
    '  <input class="search-box" type="text" placeholder="Search model, job number..." id="search-box" oninput="applyFilters()">\n'
    '</div>\n'
    '<div class="container">\n'
    '  <div id="items-container"><div class="loading">Loading...</div></div>\n'
    '</div>\n'
    '</div>\n'

    # ---- Launch tab ----
    '<div id="tab-launch" class="launch-panel">\n'
    '<div class="container">\n'

    # Job launch section
    '  <div class="launch-section">\n'
    '    <h3>Launch Job</h3>\n'
    '    <div class="form-grid" id="job-form">\n'
    '      <div class="form-group">\n'
    '        <label>Profile</label>\n'
    '        <select id="lj-profile" onchange="onProfileChange()"><option value="">-- No Profile --</option></select>\n'
    '      </div>\n'
    '      <div class="form-group">\n'
    '        <label>Mode</label>\n'
    '        <select id="lj-mode"><option value="direct">direct</option><option value="vsp">vsp</option><option value="comt_vsp">comt_vsp</option></select>\n'
    '      </div>\n'
    '      <div class="form-group">\n'
    '        <label>Provider</label>\n'
    '        <select id="lj-provider"><option value="openrouter">openrouter</option><option value="openai">openai</option><option value="self">self</option></select>\n'
    '      </div>\n'
    '      <div class="form-group">\n'
    '        <label>Model</label>\n'
    '        <input type="text" id="lj-model" placeholder="gpt-5">\n'
    '      </div>\n'
    # Sampling
    '      <div class="form-section-title">Sampling</div>\n'
    '      <div class="form-group">\n'
    '        <label>Temperature</label>\n'
    '        <input type="number" id="lj-temp" step="0.1" min="0" max="2" placeholder="0.0">\n'
    '      </div>\n'
    '      <div class="form-group">\n'
    '        <label>Top P</label>\n'
    '        <input type="number" id="lj-top_p" step="0.1" min="0" max="1" placeholder="1.0">\n'
    '      </div>\n'
    '      <div class="form-group">\n'
    '        <label>Max Tokens</label>\n'
    '        <input type="number" id="lj-max_tokens" min="1" placeholder="2048">\n'
    '      </div>\n'
    '      <div class="form-group">\n'
    '        <label>Seed</label>\n'
    '        <input type="number" id="lj-seed" placeholder="">\n'
    '      </div>\n'
    # Data
    '      <div class="form-section-title">Data</div>\n'
    '      <div class="form-group">\n'
    '        <label>Sampling Rate</label>\n'
    '        <input type="number" id="lj-sampling_rate" step="0.01" min="0" max="1" placeholder="1.0">\n'
    '      </div>\n'
    '      <div class="form-group">\n'
    '        <label>Sampling Seed</label>\n'
    '        <input type="number" id="lj-sampling_seed" placeholder="42">\n'
    '      </div>\n'
    '      <div class="form-group">\n'
    '        <label>Max Tasks</label>\n'
    '        <input type="number" id="lj-max_tasks" min="1" placeholder="">\n'
    '      </div>\n'
    '      <div class="form-group">\n'
    '        <label>Image Types</label>\n'
    '        <input type="text" id="lj-image_types" placeholder="SD">\n'
    '      </div>\n'
    '      <div class="form-group">\n'
    '        <label>Categories (space-sep)</label>\n'
    '        <input type="text" id="lj-categories" placeholder="">\n'
    '      </div>\n'
    # Network
    '      <div class="form-section-title">Network</div>\n'
    '      <div class="form-group">\n'
    '        <label>Tunnel</label>\n'
    '        <select id="lj-tunnel"><option value="ssh">ssh</option><option value="cf">cf</option><option value="none">none</option></select>\n'
    '      </div>\n'
    '      <div class="form-group">\n'
    '        <label>Consumers</label>\n'
    '        <input type="number" id="lj-consumers" min="1" placeholder="20">\n'
    '      </div>\n'
    '      <div class="form-group">\n'
    '        <label>LLM Base URL</label>\n'
    '        <input type="text" id="lj-llm_base_url" placeholder="">\n'
    '      </div>\n'
    '      <div class="form-group">\n'
    '        <label>OpenRouter Provider</label>\n'
    '        <input type="text" id="lj-openrouter_provider" placeholder="">\n'
    '      </div>\n'
    # Self-provider remote image
    '      <div class="form-section-title">Self-Provider 远程图片</div>\n'
    '      <div class="form-group">\n'
    '        <label>Remote Image Base URL</label>\n'
    '        <input type="text" id="lj-remote_image_base_url" placeholder="http://localhost:8001">\n'
    '      </div>\n'
    '      <div class="form-group">\n'
    '        <label>Remote VSP Override URL</label>\n'
    '        <input type="text" id="lj-remote_vsp_override_url" placeholder="http://localhost:8002">\n'
    '      </div>\n'
    '      <div class="form-group">\n'
    '        <label>Remote VSP Override SSH</label>\n'
    '        <input type="text" id="lj-remote_vsp_override_ssh" placeholder="seetacloud:/root/vsp_override/">\n'
    '      </div>\n'
    # CoMT
    '      <div class="form-section-title">CoMT / Eval</div>\n'
    '      <div class="form-group">\n'
    '        <label>CoMT Sample ID</label>\n'
    '        <input type="text" id="lj-comt_sample_id" placeholder="">\n'
    '      </div>\n'
    '      <div class="form-group">\n'
    '        <label>Eval Model</label>\n'
    '        <input type="text" id="lj-eval_model" placeholder="gpt-5-mini">\n'
    '      </div>\n'
    '      <div class="form-group">\n'
    '        <label>Skip Eval</label>\n'
    '        <select id="lj-skip_eval"><option value="">No</option><option value="1">Yes</option></select>\n'
    '      </div>\n'
    '    </div>\n'
    '    <div class="form-actions">\n'
    '      <button class="btn btn-primary" onclick="launchJob()">Launch Job</button>\n'
    '    </div>\n'
    '  </div>\n'

    # Quick Launch section
    '  <div class="launch-section">\n'
    '    <h3>Quick Launch</h3>\n'
    '    <div style="display:flex;gap:24px;flex-wrap:wrap">\n'
    # HS Comp
    '      <div style="flex:1;min-width:280px">\n'
    '        <h4 style="font-size:13px;margin-bottom:8px">Hidden States Comparison</h4>\n'
    '        <div class="form-grid" style="grid-template-columns:1fr 1fr">\n'
    '          <div class="form-group"><label>Job 1</label><input type="number" id="ql-hs-job1" placeholder="e.g. 243"></div>\n'
    '          <div class="form-group"><label>Job 2</label><input type="number" id="ql-hs-job2" placeholder="e.g. 245"></div>\n'
    '          <div class="form-group"><label>Job 1 q/t</label><input type="text" id="ql-hs-sub1" value="q1" style="width:45%;display:inline" placeholder="q"><input type="text" id="ql-hs-turn1" value="t0" style="width:45%;display:inline;margin-left:5%" placeholder="t"></div>\n'
    '          <div class="form-group"><label>Job 2 q/t</label><input type="text" id="ql-hs-sub2" value="q1" style="width:45%;display:inline" placeholder="q"><input type="text" id="ql-hs-turn2" value="t0" style="width:45%;display:inline;margin-left:5%" placeholder="t"></div>\n'
    '        </div>\n'
    '        <div class="form-actions"><button class="btn btn-primary btn-sm" onclick="launchHsComp()">Launch HS Comp</button></div>\n'
    '      </div>\n'
    # Refusal Dir
    '      <div style="flex:1;min-width:280px">\n'
    '        <h4 style="font-size:13px;margin-bottom:8px">Refusal Direction</h4>\n'
    '        <div class="form-grid" style="grid-template-columns:1fr 1fr">\n'
    '          <div class="form-group"><label>Train Jobs (space-sep)</label><input type="text" id="ql-rd-jobs" placeholder="e.g. 243 245"></div>\n'
    '          <div class="form-group"><label>Train Batches (space-sep)</label><input type="text" id="ql-rd-batch" placeholder="e.g. 17"></div>\n'
    '          <div class="form-group"><label>Sub Task</label><input type="text" id="ql-rd-sub" value="q0"></div>\n'
    '          <div class="form-group"><label>Turn</label><input type="text" id="ql-rd-turn" value="t0"></div>\n'
    '          <div class="form-group"><label>Score Method</label><select id="ql-rd-score"><option value="dot">dot</option><option value="cosine">cosine</option></select></div>\n'
    '          <div class="form-group" style="visibility:hidden"></div>\n'
    '        </div>\n'
    # Eval mode selector
    '        <div style="margin:8px 0 4px"><label style="font-size:12px;font-weight:600;color:#2d3436">Eval Mode</label></div>\n'
    '        <div id="ql-rd-eval-tabs" style="display:flex;gap:0;margin-bottom:8px">\n'
    '          <button class="btn btn-sm rd-eval-tab active" data-mode="split" onclick="setRdEvalMode(\'split\')" style="border-radius:6px 0 0 6px">Random Split</button>\n'
    '          <button class="btn btn-sm rd-eval-tab" data-mode="kfold" onclick="setRdEvalMode(\'kfold\')" style="border-radius:0">K-fold CV</button>\n'
    '          <button class="btn btn-sm rd-eval-tab" data-mode="cross" onclick="setRdEvalMode(\'cross\')" style="border-radius:0 6px 6px 0">Cross-Job Test</button>\n'
    '        </div>\n'
    # Split mode fields
    '        <div id="ql-rd-mode-split" class="form-grid" style="grid-template-columns:1fr 1fr">\n'
    '          <div class="form-group"><label>Split Ratio</label><input type="number" id="ql-rd-split" value="0.7" step="0.1" min="0.1" max="0.9"></div>\n'
    '          <div class="form-group"><label>Seed</label><input type="number" id="ql-rd-seed-split" value="42"></div>\n'
    '        </div>\n'
    # K-fold mode fields
    '        <div id="ql-rd-mode-kfold" class="form-grid" style="grid-template-columns:1fr 1fr;display:none">\n'
    '          <div class="form-group"><label>K-fold</label><input type="number" id="ql-rd-folds" value="5" min="2"></div>\n'
    '          <div class="form-group"><label>Seed</label><input type="number" id="ql-rd-seed-kfold" value="42"></div>\n'
    '        </div>\n'
    # Cross-job mode fields
    '        <div id="ql-rd-mode-cross" class="form-grid" style="grid-template-columns:1fr 1fr;display:none">\n'
    '          <div class="form-group"><label>Test Jobs (space-sep)</label><input type="text" id="ql-rd-test-jobs" placeholder="e.g. 250 251"></div>\n'
    '          <div class="form-group"><label>Test Batches (space-sep)</label><input type="text" id="ql-rd-test-batch" placeholder="e.g. 18"></div>\n'
    '        </div>\n'
    '        <div class="form-actions"><button class="btn btn-primary btn-sm" onclick="launchRefusalDir()">Launch Refusal Dir</button></div>\n'
    '      </div>\n'
    # Batch
    '      <div style="flex:1;min-width:280px">\n'
    '        <h4 style="font-size:13px;margin-bottom:8px">Batch Request</h4>\n'
    '        <p style="font-size:12px;color:#636e72;margin-bottom:8px">Runs batch_request.py with its current hardcoded args_combo config.</p>\n'
    '        <div class="form-grid" style="grid-template-columns:1fr">\n'
    '          <div class="form-group"><label>Resume Batch # (optional)</label><input type="number" id="ql-batch-resume" placeholder=""></div>\n'
    '        </div>\n'
    '        <div class="form-actions"><button class="btn btn-primary btn-sm" onclick="launchBatch()">Launch Batch</button></div>\n'
    '      </div>\n'
    '    </div>\n'
    '  </div>\n'
    '</div>\n'
    '</div>\n'

    # ---- Modal ----
    '<div class="modal-overlay" id="modal-overlay" onclick="if(event.target===this)closeModal()">\n'
    '  <div class="modal">\n'
    '    <div class="modal-header">\n'
    '      <h2 id="modal-title">Detail</h2>\n'
    '      <button class="modal-close" onclick="closeModal()">&times;</button>\n'
    '    </div>\n'
    '    <div class="modal-body" id="modal-body"></div>\n'
    '  </div>\n'
    '</div>\n'

    # ---- Compare bar ----
    '<div class="compare-bar" id="compare-bar">\n'
    '  <span id="compare-count">0</span> items selected &nbsp;\n'
    '  <button class="btn btn-primary btn-sm" onclick="compareSelected()">Compare Configs</button>\n'
    '  <button class="btn btn-sm" onclick="clearSelection()">Clear</button>\n'
    '</div>\n'

    # ---- Process monitor ----
    '<div class="proc-monitor collapsed" id="proc-monitor">\n'
    '  <div class="proc-monitor-header" onclick="toggleProcMonitor()">\n'
    '    <span>Processes <span id="proc-running-count"></span></span>\n'
    '    <span id="proc-toggle-icon">&#9650;</span>\n'
    '  </div>\n'
    '  <div id="proc-list"></div>\n'
    '</div>\n'

    # ---- JavaScript ----
    '<script>\n'
    '"use strict";\n'
    'let allItems = [];\n'
    'let currentFilter = "all";\n'
    'let profilesData = null;\n'
    'let selectedPaths = new Set();\n'
    'let procPollTimer = null;\n'
    '\n'

    # Tab switching
    'function switchTab(tab, el) {\n'
    '    document.querySelectorAll(".nav-tab").forEach(t => t.classList.remove("active"));\n'
    '    el.classList.add("active");\n'
    '    document.getElementById("tab-dashboard").style.display = tab === "dashboard" ? "" : "none";\n'
    '    const lp = document.getElementById("tab-launch");\n'
    '    if (tab === "launch") { lp.classList.add("visible"); lp.style.display = ""; loadProfiles(); }\n'
    '    else { lp.classList.remove("visible"); lp.style.display = "none"; }\n'
    '}\n'
    '\n'

    # Fetch items
    'async function fetchItems() {\n'
    '    try {\n'
    '        const res = await fetch("/api/items");\n'
    '        allItems = await res.json();\n'
    '        applyFilters();\n'
    '        updateStats();\n'
    '    } catch (e) {\n'
    '        document.getElementById("items-container").innerHTML =\n'
    '            "<div class=\\"empty-state\\">Failed to load: " + e.message + "</div>";\n'
    '    }\n'
    '}\n'
    '\n'

    # Stats
    'function updateStats() {\n'
    '    const counts = {};\n'
    '    let totalSize = 0, persistedCount = 0;\n'
    '    allItems.forEach(item => {\n'
    '        counts[item.type] = (counts[item.type] || 0) + 1;\n'
    '        totalSize += item.size_bytes || 0;\n'
    '        if (item.persisted) persistedCount++;\n'
    '    });\n'
    '    const colors = { job: "#0652DD", batch: "#e67e22", refusal_dir: "#c0392b", hs_comp: "#1e8449" };\n'
    '    const labels = { job: "Jobs", batch: "Batches", refusal_dir: "Refusal Dir", hs_comp: "HS Comp" };\n'
    '    let html = "";\n'
    '    for (const [type, label] of Object.entries(labels)) {\n'
    '        html += `<span><span class="stat-dot" style="background:${colors[type]}"></span>${label}: ${counts[type]||0}</span>`;\n'
    '    }\n'
    '    html += `<span>Total: ${fmtSize(totalSize)}</span>`;\n'
    '    if (persistedCount > 0) html += `<span>Persisted: ${persistedCount}</span>`;\n'
    '    document.getElementById("stats-row").innerHTML = html;\n'
    '}\n'
    '\n'
    'function fmtSize(bytes) {\n'
    '    const units = ["B", "KB", "MB", "GB"];\n'
    '    let i = 0;\n'
    '    while (bytes >= 1024 && i < units.length - 1) { bytes /= 1024; i++; }\n'
    '    return bytes.toFixed(1) + " " + units[i];\n'
    '}\n'
    '\n'
    'function setFilter(filter, btn) {\n'
    '    currentFilter = filter;\n'
    '    document.querySelectorAll(".filter-btn").forEach(b => b.classList.remove("active"));\n'
    '    btn.classList.add("active");\n'
    '    applyFilters();\n'
    '}\n'
    '\n'
    'function applyFilters() {\n'
    '    const query = document.getElementById("search-box").value.toLowerCase();\n'
    '    const filtered = allItems.filter(item => {\n'
    '        if (currentFilter !== "all" && item.type !== currentFilter) return false;\n'
    '        if (query) {\n'
    '            const s = [item.name, item.model||"", item.provider||"", item.mode||"",\n'
    '                item.type, item.train_model||"", String(item.batch_num||""),\n'
    '                String(item.comp_num||""), String(item.refdir_num||"")].join(" ").toLowerCase();\n'
    '            if (!s.includes(query)) return false;\n'
    '        }\n'
    '        return true;\n'
    '    });\n'
    '    renderItems(filtered);\n'
    '}\n'
    '\n'

    # Render items
    'function renderItems(items) {\n'
    '    const c = document.getElementById("items-container");\n'
    '    if (!items.length) { c.innerHTML = "<div class=\\"empty-state\\">No items found</div>"; return; }\n'
    '    c.innerHTML = items.map(item => renderCard(item)).join("");\n'
    '}\n'
    '\n'
    'function esc(s) { const d = document.createElement("div"); d.textContent = s; return d.innerHTML; }\n'
    '\n'
    'function m(label, value, cls) {\n'
    '    cls = cls === "d" ? "md" : cls === "s" ? "ms" : "mv";\n'
    '    return `<span class="metric">${label}: <span class="${cls}">${esc(String(value!=null?value:"-"))}</span></span>`;\n'
    '}\n'
    '\n'
    'function renderCard(item) {\n'
    '    const tl = { job: "JOB", batch: "BATCH", refusal_dir: "REFUSAL DIR", hs_comp: "HS COMP" };\n'
    '    let badges = `<span class="badge badge-${item.type}">${tl[item.type]}</span>`;\n'
    '    if (item.persisted) badges += " <span class=\\"badge badge-persisted\\">PERSISTED</span>";\n'
    '\n'
    '    let title = item.name;\n'
    '    let metrics = "";\n'
    '\n'
    '    if (item.type === "job") {\n'
    '        const jid = item.job_num != null ? "#" + item.job_num + " " : "";\n'
    '        title = esc(jid) + esc(item.model || item.name);\n'
    '        metrics += m("Mode", item.mode);\n'
    '        metrics += m("Provider", item.provider);\n'
    '        metrics += m("Tasks", item.total_tasks);\n'
    '        if (item.failed_tasks > 0) metrics += m("Failed", item.failed_tasks, "d");\n'
    '        if (item.attack_rate != null)\n'
    '            metrics += m("Attack Rate", item.attack_rate + "%", item.attack_rate > 50 ? "d" : "s");\n'
    '        if (item.has_hidden_states) metrics += m("HS", "\u2713", "s");\n'
    '    } else if (item.type === "batch") {\n'
    '        title = esc("Batch " + (item.batch_num || "?"));\n'
    '        metrics += m("Runs", item.total_runs);\n'
    '        metrics += m("Done", item.completed, "s");\n'
    '        if (item.failed > 0) metrics += m("Failed", item.failed, "d");\n'
    '    } else if (item.type === "refusal_dir") {\n'
    '        title = esc("Refusal Dir " + (item.refdir_num || "?"));\n'
    '        if (item.auc_roc != null) metrics += m("AUC-ROC", (item.auc_roc*100).toFixed(1)+"%", "s");\n'
    '        if (item.accuracy != null) metrics += m("Accuracy", (item.accuracy*100).toFixed(1)+"%");\n'
    '        if (item.train_model) metrics += m("Model", item.train_model);\n'
    '        if (item.sub_task) metrics += m("Sub", item.sub_task+"/"+(item.turn||""));\n'
    '    } else if (item.type === "hs_comp") {\n'
    '        title = esc("HS Comp " + (item.comp_num || "?"));\n'
    '        if (item.job1_num && item.job2_num) {\n'
    '            let jobsLabel = item.job1_num + (item.job1_sub ? " ("+item.job1_sub+"/"+item.job1_turn+")" : "") + " vs " + item.job2_num + (item.job2_sub ? " ("+item.job2_sub+"/"+item.job2_turn+")" : "");\n'
    '            metrics += m("Jobs", jobsLabel);\n'
    '        }\n'
    '        metrics += m("Matched", item.matched_tasks);\n'
    '        if (item.mean_cosine != null) metrics += m("Cos Sim", item.mean_cosine.toFixed(4), "s");\n'
    '    } else { title = esc(title); }\n'
    '\n'
    '    const timeStr = item.mtime ? new Date(item.mtime * 1000).toLocaleString() : "";\n'
    '\n'
    '    const dp = encodeURIComponent(item.path);\n'
    '    const dn = encodeURIComponent(item.name);\n'
    '    const persistLabel = item.persisted ? "Unpersist" : "Persist";\n'
    '    const persistCls = item.persisted ? "btn-sm" : "btn-sm btn-success";\n'
    '\n'
    '    let actions = "";\n'
    '    actions += `<button class="btn btn-sm" data-action="view" data-path="${dp}" data-type="${item.type}">View</button>`;\n'
    '    actions += `<button class="btn ${persistCls}" data-action="persist" data-path="${dp}">${persistLabel}</button>`;\n'
    '\n'
    '    if (item.type === "job" && !item.persisted && item.failed_tasks > 0) {\n'
    '        const jm = item.name.match(/^job_(\\d+)/);\n'
    '        if (jm) actions += `<button class="btn btn-sm btn-primary" data-action="retry" data-jobnum="${jm[1]}">Retry</button>`;\n'
    '    }\n'
    '    actions += `<button class="btn btn-sm btn-danger" data-action="delete" data-path="${dp}" data-name="${dn}">Delete</button>`;\n'
    '\n'
    '    const checked = selectedPaths.has(item.path) ? "checked" : "";\n'
    '\n'
    '    return `<div class="card${item.persisted ? " persisted" : ""}">`\n'
    '        + `<input type="checkbox" class="card-check" data-path="${dp}" ${checked} onchange="onCheckChange(this)">`\n'
    '        + `<div class="card-left" data-action="view" data-path="${dp}" data-type="${item.type}">`\n'
    '        + `<div class="card-header">${badges} <span class="card-title">${title}</span></div>`\n'
    '        + `<div class="card-meta">${metrics}`\n'
    '        + `<span class="metric">${esc(item.size_human)}</span>`\n'
    '        + `<span class="metric" style="color:#b2bec3">${esc(timeStr)}</span>`\n'
    '        + `</div></div>`\n'
    '        + `<div class="card-actions">${actions}</div></div>`;\n'
    '}\n'
    '\n'

    # Checkbox selection
    'function onCheckChange(cb) {\n'
    '    const path = decodeURIComponent(cb.dataset.path);\n'
    '    if (cb.checked) selectedPaths.add(path); else selectedPaths.delete(path);\n'
    '    updateCompareBar();\n'
    '}\n'
    'function updateCompareBar() {\n'
    '    const bar = document.getElementById("compare-bar");\n'
    '    document.getElementById("compare-count").textContent = selectedPaths.size;\n'
    '    bar.classList.toggle("visible", selectedPaths.size >= 2);\n'
    '}\n'
    'function clearSelection() {\n'
    '    selectedPaths.clear();\n'
    '    document.querySelectorAll(".card-check").forEach(c => c.checked = false);\n'
    '    updateCompareBar();\n'
    '}\n'
    '\n'

    # Compare
    'async function compareSelected() {\n'
    '    const paths = Array.from(selectedPaths);\n'
    '    try {\n'
    '        const res = await fetch("/api/compare_configs", {\n'
    '            method: "POST", headers: {"Content-Type":"application/json"},\n'
    '            body: JSON.stringify({paths})\n'
    '        });\n'
    '        const data = await res.json();\n'
    '        if (data.error) { alert(data.error); return; }\n'
    '        showCompareModal(data);\n'
    '    } catch (e) { alert("Compare failed: " + e.message); }\n'
    '}\n'
    '\n'
    'function showCompareModal(data) {\n'
    '    document.getElementById("modal-title").textContent = "Config Comparison";\n'
    '    let html = "";\n'
    '    if (data.diff_keys.length) {\n'
    '        html += "<h3 style=\\"font-size:14px;margin-bottom:8px\\">Differences</h3>";\n'
    '        html += "<table class=\\"data-table\\"><tr><th>Key</th>";\n'
    '        data.labels.forEach(l => { html += "<th>" + esc(l) + "</th>"; });\n'
    '        html += "</tr>";\n'
    '        data.diff_keys.forEach(d => {\n'
    '            html += "<tr><td><b>" + esc(d.key) + "</b></td>";\n'
    '            d.values.forEach(v => { html += "<td class=\\"diff-val\\">" + esc(JSON.stringify(v)) + "</td>"; });\n'
    '            html += "</tr>";\n'
    '        });\n'
    '        html += "</table>";\n'
    '    }\n'
    '    if (data.same_keys.length) {\n'
    '        html += "<h3 style=\\"font-size:14px;margin:16px 0 8px\\">Same</h3>";\n'
    '        html += "<table class=\\"data-table\\"><tr><th>Key</th><th>Value</th></tr>";\n'
    '        data.same_keys.forEach(s => {\n'
    '            html += "<tr><td class=\\"same-val\\">" + esc(s.key) + "</td><td class=\\"same-val\\">" + esc(JSON.stringify(s.value)) + "</td></tr>";\n'
    '        });\n'
    '        html += "</table>";\n'
    '    }\n'
    '    document.getElementById("modal-body").innerHTML = html;\n'
    '    document.getElementById("modal-overlay").classList.add("visible");\n'
    '}\n'
    '\n'

    # Event delegation
    'document.getElementById("items-container").addEventListener("click", function(e) {\n'
    '    const el = e.target.closest("[data-action]");\n'
    '    if (!el) return;\n'
    '    const action = el.dataset.action;\n'
    '    const path = el.dataset.path ? decodeURIComponent(el.dataset.path) : null;\n'
    '    if (action === "delete" && path) {\n'
    '        const name = el.dataset.name ? decodeURIComponent(el.dataset.name) : path;\n'
    '        if (!confirm("Delete \\"" + name + "\\"? This cannot be undone.")) return;\n'
    '        apiPost("/api/delete", { path }).then(d => { if (d && d.ok) fetchItems(); });\n'
    '    } else if (action === "persist" && path) {\n'
    '        apiPost("/api/persist", { path }).then(d => { if (d && d.ok) fetchItems(); });\n'
    '    } else if (action === "retry") {\n'
    '        const jobNum = parseInt(el.dataset.jobnum);\n'
    '        if (!confirm("Retry failed tasks in job " + jobNum + "?")) return;\n'
    '        apiPost("/api/retry", { job_num: jobNum }).then(d => { if (d && d.ok) startProcPolling(); });\n'
    '    } else if (action === "view" && path) {\n'
    '        const type = el.dataset.type;\n'
    '        if (type === "job") showJobDetail(path);\n'
    '        else if (type === "batch") showBatchDetail(path);\n'
    '        else if (type === "refusal_dir") showRefusalDirDetail(path);\n'
    '        else if (type === "hs_comp") showHsCompDetail(path);\n'
    '    }\n'
    '});\n'
    '\n'

    # API helper
    'async function apiPost(url, body) {\n'
    '    try {\n'
    '        const res = await fetch(url, { method: "POST", headers: {"Content-Type":"application/json"}, body: JSON.stringify(body) });\n'
    '        const data = await res.json();\n'
    '        if (!data.ok && data.error) alert("Error: " + data.error);\n'
    '        return data;\n'
    '    } catch (e) { alert("Request failed: " + e.message); return {}; }\n'
    '}\n'
    '\n'

    # Modal
    'function closeModal() { document.getElementById("modal-overlay").classList.remove("visible"); }\n'
    'document.addEventListener("keydown", e => { if (e.key === "Escape") closeModal(); });\n'
    '\n'

    # Job detail
    'async function showJobDetail(path) {\n'
    '    document.getElementById("modal-title").textContent = "Job Detail";\n'
    '    document.getElementById("modal-body").innerHTML = "<div class=\\"loading\\">Loading...</div>";\n'
    '    document.getElementById("modal-overlay").classList.add("visible");\n'
    '    try {\n'
    '        const res = await fetch("/api/job_detail?path=" + encodeURIComponent(path));\n'
    '        const d = await res.json();\n'
    '        if (d.error) { document.getElementById("modal-body").innerHTML = esc(d.error); return; }\n'
    '        let html = "<h3 style=\\"font-size:14px;margin-bottom:8px\\">" + esc(d.name) + "</h3>";\n'
    # Links
    '        html += "<div style=\\"margin-bottom:12px;display:flex;gap:8px\\">";\n'
    '        if (d.has_summary_html) {\n'
    '            const rel = d.path.includes("output_persisted") ? "output_persisted" : "output";\n'
    '            html += `<a href="/${rel}/${d.name}/summary.html" target="_blank" class="btn btn-sm">Open Summary</a>`;\n'
    '        }\n'
    '        if (d.has_console_log) {\n'
    '            const rel = d.path.includes("output_persisted") ? "output_persisted" : "output";\n'
    '            html += `<a href="/${rel}/${d.name}/console.log" target="_blank" class="btn btn-sm">Console Log</a>`;\n'
    '        }\n'
    '        html += "</div>";\n'
    # Config table
    '        html += "<h4 style=\\"font-size:13px;margin-bottom:6px\\">Config</h4>";\n'
    '        html += "<table class=\\"data-table\\"><tr><th>Key</th><th>Value</th></tr>";\n'
    '        const cfg = d.config || {};\n'
    '        for (const [k,v] of Object.entries(cfg)) {\n'
    '            html += "<tr><td>" + esc(k) + "</td><td>" + esc(JSON.stringify(v)) + "</td></tr>";\n'
    '        }\n'
    '        html += "</table>";\n'
    # Results stats
    '        const st = d.results_stats;\n'
    '        html += "<h4 style=\\"font-size:13px;margin:12px 0 6px\\">Results (" + st.total + " total, " + st.success + " success, " + st.failed + " failed)</h4>";\n'
    '        if (st.failed > 0 && Object.keys(st.error_types).length) {\n'
    '            html += "<table class=\\"data-table\\"><tr><th>Error Type</th><th>Count</th></tr>";\n'
    '            for (const [et, cnt] of Object.entries(st.error_types)) {\n'
    '                html += "<tr><td>" + esc(et) + "</td><td>" + cnt + "</td></tr>";\n'
    '            }\n'
    '            html += "</table>";\n'
    '        }\n'
    # Eval table
    '        if (d.eval.length) {\n'
    '            html += "<h4 style=\\"font-size:13px;margin:12px 0 6px\\">Eval Results</h4>";\n'
    '            const cols = Object.keys(d.eval[0]);\n'
    '            html += "<table class=\\"data-table\\"><tr>";\n'
    '            cols.forEach(c => { html += "<th>" + esc(c) + "</th>"; });\n'
    '            html += "</tr>";\n'
    '            d.eval.forEach(row => {\n'
    '                html += "<tr>";\n'
    '                cols.forEach(c => {\n'
    '                    let val = row[c] || "";\n'
    '                    let cls = "";\n'
    '                    if (c.includes("Attack_Rate") || c.includes("attack")) {\n'
    '                        const n = parseFloat(val);\n'
    '                        if (!isNaN(n)) cls = n > 50 ? "rate-high" : "rate-low";\n'
    '                    }\n'
    '                    html += `<td class="${cls}">` + esc(val) + "</td>";\n'
    '                });\n'
    '                html += "</tr>";\n'
    '            });\n'
    '            html += "</table>";\n'
    '        }\n'
    '        document.getElementById("modal-body").innerHTML = html;\n'
    '    } catch (e) { document.getElementById("modal-body").innerHTML = "Error: " + esc(e.message); }\n'
    '}\n'
    '\n'

    # Batch detail
    'async function showBatchDetail(path) {\n'
    '    document.getElementById("modal-title").textContent = "Batch Detail";\n'
    '    document.getElementById("modal-body").innerHTML = "<div class=\\"loading\\">Loading...</div>";\n'
    '    document.getElementById("modal-overlay").classList.add("visible");\n'
    '    try {\n'
    '        const res = await fetch("/api/batch_detail?path=" + encodeURIComponent(path));\n'
    '        const d = await res.json();\n'
    '        if (d.error) { document.getElementById("modal-body").innerHTML = esc(d.error); return; }\n'
    '        let html = "<h3 style=\\"font-size:14px;margin-bottom:8px\\">" + esc(d.name) + "</h3>";\n'
    # Links
    '        html += "<div style=\\"margin-bottom:12px;display:flex;gap:8px\\">";\n'
    '        if (d.has_batch_summary) {\n'
    '            const rel = d.path.includes("output_persisted") ? "output_persisted" : "output";\n'
    '            html += `<a href="/${rel}/${d.name}/batch_summary.html" target="_blank" class="btn btn-sm">Batch Summary</a>`;\n'
    '        }\n'
    '        if (d.has_eval_report) {\n'
    '            const rel = d.path.includes("output_persisted") ? "output_persisted" : "output";\n'
    '            html += `<a href="/${rel}/${d.name}/report/evaluation_report.html" target="_blank" class="btn btn-sm">Eval Report</a>`;\n'
    '        }\n'
    '        html += "</div>";\n'
    # Batch state info
    '        const bs = d.batch_state;\n'
    '        if (bs.batch_num != null) {\n'
    '            html += "<p style=\\"font-size:13px;margin-bottom:8px\\">Batch #" + bs.batch_num + " | Created: " + esc(bs.created_at||"") + " | Total runs: " + (bs.total_runs||0) + "</p>";\n'
    '        }\n'
    # Runs table
    '        if (d.runs.length) {\n'
    '            html += "<h4 style=\\"font-size:13px;margin:12px 0 6px\\">Runs</h4>";\n'
    '            html += "<table class=\\"data-table\\"><tr><th>#</th><th>Args</th><th>Status</th><th>Duration</th><th>Attack Rate</th></tr>";\n'
    '            d.runs.forEach(r => {\n'
    '                const dur = r.duration_secs != null ? (r.duration_secs/60).toFixed(1) + "m" : "-";\n'
    '                const ar = r.attack_rate != null ? r.attack_rate + "%" : "-";\n'
    '                const arCls = r.attack_rate != null ? (r.attack_rate > 50 ? "rate-high" : "rate-low") : "";\n'
    '                const stCls = r.status === "completed" ? "ms" : r.status === "failed" ? "md" : "";\n'
    '                html += "<tr>";\n'
    '                html += "<td>" + (r.index != null ? r.index : "-") + "</td>";\n'
    '                html += "<td style=\\"max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap\\" title=\\"" + esc(r.args_str) + "\\">" + esc(r.args_str) + "</td>";\n'
    '                html += `<td class="${stCls}">` + esc(r.status) + "</td>";\n'
    '                html += "<td>" + dur + "</td>";\n'
    '                html += `<td class="${arCls}">` + ar + "</td>";\n'
    '                html += "</tr>";\n'
    '            });\n'
    '            html += "</table>";\n'
    '        }\n'
    '        document.getElementById("modal-body").innerHTML = html;\n'
    '    } catch (e) { document.getElementById("modal-body").innerHTML = "Error: " + esc(e.message); }\n'
    '}\n'
    '\n'

    # Refusal Dir detail
    'function showRefusalDirDetail(path) {\n'
    '    const item = allItems.find(i => i.path === path);\n'
    '    if (!item) return;\n'
    '    document.getElementById("modal-title").textContent = "Refusal Dir #" + (item.refdir_num || "?");\n'
    '    const rel = path.includes("output_persisted") ? "output_persisted" : "output";\n'
    '    let html = "";\n'
    # Links
    '    html += "<div style=\\"margin-bottom:12px;display:flex;gap:8px;flex-wrap:wrap\\">";\n'
    '    if (item.has_report_html) html += `<a href="/${rel}/${item.name}/report.html" target="_blank" class="btn btn-sm">Report HTML</a>`;\n'
    '    if (item.has_roc_curve) {\n'
    '        html += `<a href="/${rel}/${item.name}/roc_curve.png" target="_blank" class="btn btn-sm">ROC Curve</a>`;\n'
    '        html += `<a href="/${rel}/${item.name}/score_distribution.png" target="_blank" class="btn btn-sm">Score Distribution</a>`;\n'
    '        html += `<a href="/${rel}/${item.name}/category_auc.png" target="_blank" class="btn btn-sm">Category AUC</a>`;\n'
    '        html += `<a href="/${rel}/${item.name}/pca_scatter.png" target="_blank" class="btn btn-sm">PCA Scatter</a>`;\n'
    '    }\n'
    '    html += "</div>";\n'
    # Metrics
    '    html += "<table class=\\"data-table\\"><tr><th>Metric</th><th>Value</th></tr>";\n'
    '    if (item.auc_roc != null) html += "<tr><td>AUC-ROC</td><td>" + (item.auc_roc*100).toFixed(2) + "%</td></tr>";\n'
    '    if (item.accuracy != null) html += "<tr><td>Accuracy</td><td>" + (item.accuracy*100).toFixed(2) + "%</td></tr>";\n'
    '    if (item.train_model) html += "<tr><td>Model</td><td>" + esc(item.train_model) + "</td></tr>";\n'
    '    if (item.sub_task) html += "<tr><td>Sub Task / Turn</td><td>" + esc(item.sub_task) + " / " + esc(item.turn||"") + "</td></tr>";\n'
    '    html += "<tr><td>Samples</td><td>Total: " + (item.total_samples||0) + " (Safe: " + (item.safe||0) + ", Unsafe: " + (item.unsafe||0) + ")</td></tr>";\n'
    '    if (item.timestamp) html += "<tr><td>Timestamp</td><td>" + esc(item.timestamp) + "</td></tr>";\n'
    '    html += "</table>";\n'
    # Inline images
    '    if (item.has_roc_curve) {\n'
    '        html += "<h4 style=\\"font-size:13px;margin:16px 0 8px\\">ROC Curve</h4>";\n'
    '        html += `<img src="/${rel}/${item.name}/roc_curve.png" style="max-width:100%;border:1px solid #e1e4e8;border-radius:8px">`;\n'
    '    }\n'
    '    document.getElementById("modal-body").innerHTML = html;\n'
    '    document.getElementById("modal-overlay").classList.add("visible");\n'
    '}\n'
    '\n'

    # HS Comp detail — helper to render one job's info card
    'function _hsJobCard(item, side) {\n'
    '    const num = item[side+"_num"];\n'
    '    const sub = item[side+"_sub"] || "?";\n'
    '    const turn = item[side+"_turn"] || "?";\n'
    '    const dir = item[side+"_dir"] || "";\n'
    '    const cfg = item[side+"_config"] || {};\n'
    '    const route = item[side+"_route"];\n'
    '    const label = side === "job1" ? "Job 1" : "Job 2";\n'
    '    let h = "<div style=\\"flex:1;min-width:260px;border:1px solid #e1e4e8;border-radius:8px;padding:12px\\">";\n'
    '    h += "<div style=\\"display:flex;align-items:center;justify-content:space-between;margin-bottom:8px\\">";\n'
    '    h += "<strong style=\\"font-size:14px\\">" + label + " — #" + (num||"?") + "</strong>";\n'
    '    if (route && dir) h += ` <a href="/${route}/${dir}/summary.html" target="_blank" class="btn btn-sm" style="font-size:11px">Open Job</a>`;\n'
    '    h += "</div>";\n'
    '    h += "<table class=\\"data-table\\" style=\\"font-size:12px\\">";\n'
    '    h += "<tr><td style=\\"width:110px\\">q / t</td><td>" + esc(sub) + " / " + esc(turn) + "</td></tr>";\n'
    '    h += "<tr><td>Mode</td><td>" + esc(cfg.mode||"—") + "</td></tr>";\n'
    '    h += "<tr><td>Provider</td><td>" + esc(cfg.provider||"—") + "</td></tr>";\n'
    '    h += "<tr><td>Model</td><td>" + esc(cfg.model||"—") + "</td></tr>";\n'
    '    if (cfg.vsp_override_images_dir) h += "<tr><td>Override</td><td style=\\"word-break:break-all\\">" + esc(cfg.vsp_override_images_dir) + "</td></tr>";\n'
    '    h += "</table>";\n'
    # More Details collapsible
    '    const detailId = side + "_details_" + (item.comp_num||0);\n'
    '    h += "<details style=\\"margin-top:6px\\"><summary style=\\"cursor:pointer;font-size:12px;color:#586069\\">More Details</summary>";\n'
    '    h += "<table class=\\"data-table\\" style=\\"font-size:11px;margin-top:4px\\">";\n'
    '    if (cfg.profile) h += "<tr><td>Profile</td><td>" + esc(cfg.profile) + "</td></tr>";\n'
    '    if (cfg.temperature != null) h += "<tr><td>Temperature</td><td>" + cfg.temperature + "</td></tr>";\n'
    '    if (cfg.llm_base_url) h += "<tr><td>LLM URL</td><td style=\\"word-break:break-all\\">" + esc(cfg.llm_base_url) + "</td></tr>";\n'
    '    if (cfg.comt_sample_id) h += "<tr><td>CoMT Sample</td><td>" + esc(cfg.comt_sample_id) + "</td></tr>";\n'
    '    if (cfg.sampling_rate) h += "<tr><td>Sampling Rate</td><td>" + cfg.sampling_rate + "</td></tr>";\n'
    '    if (cfg.tunnel) h += "<tr><td>Tunnel</td><td>" + esc(cfg.tunnel) + "</td></tr>";\n'
    '    if (cfg.image_types) h += "<tr><td>Image Types</td><td>" + esc(cfg.image_types.join(", ")) + "</td></tr>";\n'
    '    if (cfg.max_tokens) h += "<tr><td>Max Tokens</td><td>" + cfg.max_tokens + "</td></tr>";\n'
    '    if (cfg.openrouter_provider) h += "<tr><td>OR Provider</td><td>" + esc(cfg.openrouter_provider) + "</td></tr>";\n'
    '    h += "<tr><td>Dir</td><td style=\\"word-break:break-all;font-size:10px\\">" + esc(dir) + "</td></tr>";\n'
    '    h += "</table></details>";\n'
    '    h += "</div>";\n'
    '    return h;\n'
    '}\n'
    '\n'

    # HS Comp detail — main function
    'function showHsCompDetail(path) {\n'
    '    const item = allItems.find(i => i.path === path);\n'
    '    if (!item) return;\n'
    '    document.getElementById("modal-title").textContent = "HS Comp #" + (item.comp_num || "?");\n'
    '    const rel = path.includes("output_persisted") ? "output_persisted" : "output";\n'
    '    let html = "";\n'
    # Links
    '    html += "<div style=\\"margin-bottom:12px;display:flex;gap:8px;flex-wrap:wrap\\">";\n'
    '    if (item.has_summary_png) {\n'
    '        html += `<a href="/${rel}/${item.name}/hs_diff_summary.png" target="_blank" class="btn btn-sm">Summary PNG</a>`;\n'
    '    }\n'
    '    html += "</div>";\n'
    # Job info cards side by side
    '    html += "<div style=\\"display:flex;gap:12px;flex-wrap:wrap;margin-bottom:16px\\">";\n'
    '    html += _hsJobCard(item, "job1");\n'
    '    html += _hsJobCard(item, "job2");\n'
    '    html += "</div>";\n'
    # Comparison metrics
    '    html += "<table class=\\"data-table\\"><tr><th>Metric</th><th>Value</th></tr>";\n'
    '    html += "<tr><td>Matched Tasks</td><td>" + (item.matched_tasks||0) + "</td></tr>";\n'
    '    if (item.mean_cosine != null) html += "<tr><td>Mean Cosine Sim</td><td>" + item.mean_cosine.toFixed(4) + "</td></tr>";\n'
    '    if (item.timestamp) html += "<tr><td>Timestamp</td><td>" + esc(item.timestamp) + "</td></tr>";\n'
    '    html += "</table>";\n'
    # Inline image
    '    if (item.has_summary_png) {\n'
    '        html += "<h4 style=\\"font-size:13px;margin:16px 0 8px\\">Summary</h4>";\n'
    '        html += `<img src="/${rel}/${item.name}/hs_diff_summary.png" style="max-width:100%;border:1px solid #e1e4e8;border-radius:8px">`;\n'
    '    }\n'
    '    document.getElementById("modal-body").innerHTML = html;\n'
    '    document.getElementById("modal-overlay").classList.add("visible");\n'
    '}\n'
    '\n'

    # Profiles loading
    'async function loadProfiles() {\n'
    '    if (profilesData) return;\n'
    '    try {\n'
    '        const res = await fetch("/api/profiles");\n'
    '        profilesData = await res.json();\n'
    '        const sel = document.getElementById("lj-profile");\n'
    '        for (const name of Object.keys(profilesData.profiles)) {\n'
    '            const opt = document.createElement("option");\n'
    '            opt.value = name; opt.textContent = name;\n'
    '            sel.appendChild(opt);\n'
    '        }\n'
    '    } catch (e) { console.error("Failed to load profiles:", e); }\n'
    '}\n'
    '\n'
    'function onProfileChange() {\n'
    '    const name = document.getElementById("lj-profile").value;\n'
    '    if (!name || !profilesData) return;\n'
    '    const p = profilesData.profiles[name];\n'
    '    if (!p) return;\n'
    '    const fieldMap = {\n'
    '        mode: "lj-mode", provider: "lj-provider", model: "lj-model",\n'
    '        temp: "lj-temp", top_p: "lj-top_p", max_tokens: "lj-max_tokens",\n'
    '        sampling_rate: "lj-sampling_rate", sampling_seed: "lj-sampling_seed",\n'
    '        tunnel: "lj-tunnel", consumers: "lj-consumers",\n'
    '        llm_base_url: "lj-llm_base_url", openrouter_provider: "lj-openrouter_provider",\n'
    '        comt_sample_id: "lj-comt_sample_id", image_types: "lj-image_types",\n'
    '    };\n'
    '    for (const [key, elId] of Object.entries(fieldMap)) {\n'
    '        const el = document.getElementById(elId);\n'
    '        if (!el) continue;\n'
    '        let val = p[key];\n'
    '        if (val == null) val = "";\n'
    '        if (Array.isArray(val)) val = val.join(" ");\n'
    '        el.value = String(val);\n'
    '    }\n'
    '}\n'
    '\n'

    # Launch functions
    'function launchJob() {\n'
    '    const profile = document.getElementById("lj-profile").value || null;\n'
    '    const overrides = {};\n'
    '    const fields = [\n'
    '        ["lj-mode","mode"], ["lj-provider","provider"], ["lj-model","model"],\n'
    '        ["lj-temp","temp"], ["lj-top_p","top_p"], ["lj-max_tokens","max_tokens"],\n'
    '        ["lj-seed","seed"], ["lj-sampling_rate","sampling_rate"], ["lj-sampling_seed","sampling_seed"],\n'
    '        ["lj-max_tasks","max_tasks"], ["lj-tunnel","tunnel"], ["lj-consumers","consumers"],\n'
    '        ["lj-llm_base_url","llm_base_url"], ["lj-openrouter_provider","openrouter_provider"],\n'
    '        ["lj-remote_image_base_url","remote_image_base_url"],\n'
    '        ["lj-remote_vsp_override_url","remote_vsp_override_url"],\n'
    '        ["lj-remote_vsp_override_ssh","remote_vsp_override_ssh"],\n'
    '        ["lj-comt_sample_id","comt_sample_id"], ["lj-eval_model","eval_model"],\n'
    '    ];\n'
    '    fields.forEach(([elId, key]) => {\n'
    '        const v = document.getElementById(elId).value.trim();\n'
    '        if (v) overrides[key] = v;\n'
    '    });\n'
    '    // Handle image_types and categories as arrays\n'
    '    const it = document.getElementById("lj-image_types").value.trim();\n'
    '    if (it) overrides.image_types = it.split(/\\s+/);\n'
    '    const cats = document.getElementById("lj-categories").value.trim();\n'
    '    if (cats) overrides.categories = cats.split(/\\s+/);\n'
    '    // skip_eval\n'
    '    if (document.getElementById("lj-skip_eval").value === "1") overrides.skip_eval = true;\n'
    '\n'
    '    apiPost("/api/launch_job", { profile, overrides }).then(d => {\n'
    '        if (d && d.ok) startProcPolling();\n'
    '    });\n'
    '}\n'
    '\n'
    'function launchHsComp() {\n'
    '    const job1 = parseInt(document.getElementById("ql-hs-job1").value);\n'
    '    const job2 = parseInt(document.getElementById("ql-hs-job2").value);\n'
    '    if (!job1 || !job2) { alert("Enter both job numbers"); return; }\n'
    '    const body = { job1, job2,\n'
    '        sub_task1: document.getElementById("ql-hs-sub1").value || "q1",\n'
    '        turn1: document.getElementById("ql-hs-turn1").value || "t0",\n'
    '        sub_task2: document.getElementById("ql-hs-sub2").value || "q1",\n'
    '        turn2: document.getElementById("ql-hs-turn2").value || "t0"\n'
    '    };\n'
    '    apiPost("/api/launch_hs_comp", body).then(d => {\n'
    '        if (d && d.ok) startProcPolling();\n'
    '    });\n'
    '}\n'
    '\n'
    'let rdEvalMode = "split";\n'
    'function setRdEvalMode(mode) {\n'
    '    rdEvalMode = mode;\n'
    '    document.querySelectorAll(".rd-eval-tab").forEach(b => b.classList.toggle("active", b.dataset.mode === mode));\n'
    '    document.getElementById("ql-rd-mode-split").style.display = mode === "split" ? "" : "none";\n'
    '    document.getElementById("ql-rd-mode-kfold").style.display = mode === "kfold" ? "" : "none";\n'
    '    document.getElementById("ql-rd-mode-cross").style.display = mode === "cross" ? "" : "none";\n'
    '}\n'
    '\n'
    'function launchRefusalDir() {\n'
    '    const jobsStr = document.getElementById("ql-rd-jobs").value.trim();\n'
    '    const batchStr = document.getElementById("ql-rd-batch").value.trim();\n'
    '    const job_nums = jobsStr ? jobsStr.split(/\\s+/).map(Number).filter(n => !isNaN(n)) : [];\n'
    '    const batch = batchStr ? batchStr.split(/\\s+/).map(Number).filter(n => !isNaN(n)) : [];\n'
    '    if (!job_nums.length && !batch.length) { alert("Enter train job or batch numbers"); return; }\n'
    '    const body = { job_nums, batch,\n'
    '        sub_task: document.getElementById("ql-rd-sub").value || "q0",\n'
    '        turn: document.getElementById("ql-rd-turn").value || "t0",\n'
    '        score_method: document.getElementById("ql-rd-score").value\n'
    '    };\n'
    '    if (rdEvalMode === "cross") {\n'
    '        const tj = document.getElementById("ql-rd-test-jobs").value.trim();\n'
    '        const tb = document.getElementById("ql-rd-test-batch").value.trim();\n'
    '        const test_job = tj ? tj.split(/\\s+/).map(Number).filter(n => !isNaN(n)) : [];\n'
    '        const test_batch = tb ? tb.split(/\\s+/).map(Number).filter(n => !isNaN(n)) : [];\n'
    '        if (!test_job.length && !test_batch.length) { alert("Cross-Job mode: enter test job or batch numbers"); return; }\n'
    '        if (test_job.length) body.test_job = test_job;\n'
    '        if (test_batch.length) body.test_batch = test_batch;\n'
    '    } else if (rdEvalMode === "kfold") {\n'
    '        const folds = parseInt(document.getElementById("ql-rd-folds").value);\n'
    '        if (!folds || folds < 2) { alert("K-fold requires >= 2"); return; }\n'
    '        body.n_folds = folds;\n'
    '        const seed = parseInt(document.getElementById("ql-rd-seed-kfold").value);\n'
    '        if (!isNaN(seed)) body.seed = seed;\n'
    '    } else {\n'
    '        const splitRatio = parseFloat(document.getElementById("ql-rd-split").value);\n'
    '        if (splitRatio > 0 && splitRatio < 1) body.split_ratio = splitRatio;\n'
    '        const seed = parseInt(document.getElementById("ql-rd-seed-split").value);\n'
    '        if (!isNaN(seed)) body.seed = seed;\n'
    '    }\n'
    '    apiPost("/api/launch_refusal_dir", body).then(d => {\n'
    '        if (d && d.ok) startProcPolling();\n'
    '    });\n'
    '}\n'
    '\n'
    'function launchBatch() {\n'
    '    const resume = document.getElementById("ql-batch-resume").value.trim();\n'
    '    const body = {};\n'
    '    if (resume) body.resume = parseInt(resume);\n'
    '    apiPost("/api/launch_batch", body).then(d => {\n'
    '        if (d && d.ok) startProcPolling();\n'
    '    });\n'
    '}\n'
    '\n'

    # Process monitor
    'function toggleProcMonitor() {\n'
    '    const m = document.getElementById("proc-monitor");\n'
    '    m.classList.toggle("collapsed");\n'
    '    document.getElementById("proc-toggle-icon").innerHTML = m.classList.contains("collapsed") ? "&#9650;" : "&#9660;";\n'
    '}\n'
    '\n'
    'function startProcPolling() {\n'
    '    const m = document.getElementById("proc-monitor");\n'
    '    m.classList.remove("collapsed");\n'
    '    document.getElementById("proc-toggle-icon").innerHTML = "&#9660;";\n'
    '    if (!procPollTimer) {\n'
    '        pollProcs();\n'
    '        procPollTimer = setInterval(pollProcs, 3000);\n'
    '    }\n'
    '}\n'
    '\n'
    'let expandedProc = null;\n'
    'async function pollProcs() {\n'
    '    try {\n'
    '        const res = await fetch("/api/process_status");\n'
    '        const procs = await res.json();\n'
    '        const list = document.getElementById("proc-list");\n'
    '        const running = procs.filter(p => p.running).length;\n'
    '        document.getElementById("proc-running-count").textContent = running ? "(" + running + " running)" : "";\n'
    '        if (!procs.length) {\n'
    '            list.innerHTML = "<div style=\\"padding:12px 16px;color:#b2bec3;font-size:12px\\">No processes</div>";\n'
    '            if (!running && procPollTimer) { clearInterval(procPollTimer); procPollTimer = null; fetchItems(); }\n'
    '            return;\n'
    '        }\n'
    '        let html = "";\n'
    '        procs.forEach(p => {\n'
    '            const sc = p.running ? "running" : (p.return_code === 0 ? "done" : "failed");\n'
    '            const sl = p.running ? "Running" : (p.return_code === 0 ? "Done" : "Failed (code " + p.return_code + ")");\n'
    '            const elapsed = p.elapsed > 60 ? (p.elapsed/60).toFixed(1) + "m" : p.elapsed.toFixed(0) + "s";\n'
    '            const expanded = expandedProc === p.id;\n'
    '            html += `<div class="proc-item" style="cursor:pointer" onclick="toggleProcLog(\'${p.id}\')">`;\n'
    '            html += `<span class="proc-status ${sc}"></span>`;\n'
    '            html += `<span class="proc-label"><b>${esc(p.type)}</b>: ${esc(p.label)} (${elapsed})</span>`;\n'
    '            let statusHtml = `<span style="font-size:11px;color:#636e72">${sl}</span>`;\n'
    '            if (p.result_link) statusHtml += ` <a href="${p.result_link}" target="_blank" onclick="event.stopPropagation()" style="font-size:11px;margin-left:6px">View Result</a>`;\n'
    '            html += statusHtml;\n'
    '            html += "</div>";\n'
    '            html += `<div class="proc-log${expanded ? " visible" : ""}" id="proc-log-${p.id}">`;\n'
    '            html += esc((p.output_tail||[]).join("\\n"));\n'
    '            html += "</div>";\n'
    '        });\n'
    '        list.innerHTML = html;\n'
    '        // auto scroll expanded log\n'
    '        if (expandedProc) {\n'
    '            const lg = document.getElementById("proc-log-" + expandedProc);\n'
    '            if (lg) lg.scrollTop = lg.scrollHeight;\n'
    '        }\n'
    '        if (!running && procPollTimer) { clearInterval(procPollTimer); procPollTimer = null; fetchItems(); }\n'
    '    } catch (e) {}\n'
    '}\n'
    '\n'
    'function toggleProcLog(id) {\n'
    '    if (expandedProc === id) { expandedProc = null; }\n'
    '    else { expandedProc = id; }\n'
    '    document.querySelectorAll(".proc-log").forEach(el => {\n'
    '        el.classList.toggle("visible", el.id === "proc-log-" + expandedProc);\n'
    '    });\n'
    '}\n'
    '\n'

    # Init
    'fetchItems();\n'
    '// Start polling if processes exist\n'
    'fetch("/api/process_status").then(r => r.json()).then(procs => {\n'
    '    if (procs.some(p => p.running)) startProcPolling();\n'
    '    else if (procs.length) { pollProcs(); }\n'
    '}).catch(() => {});\n'
    '</script>\n'
    '</body></html>\n'
)


# ============ App Setup ============

def create_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/", handle_index)
    app.router.add_get("/api/items", handle_items)
    app.router.add_post("/api/delete", handle_delete)
    app.router.add_post("/api/persist", handle_persist)
    app.router.add_post("/api/retry", handle_retry)
    app.router.add_get("/api/profiles", handle_profiles)
    app.router.add_get("/api/job_detail", handle_job_detail)
    app.router.add_get("/api/batch_detail", handle_batch_detail)
    app.router.add_post("/api/compare_configs", handle_compare_configs)
    app.router.add_post("/api/launch_job", handle_launch_job)
    app.router.add_post("/api/launch_batch", handle_launch_batch)
    app.router.add_post("/api/launch_hs_comp", handle_launch_hs_comp)
    app.router.add_post("/api/launch_refusal_dir", handle_launch_refusal_dir)
    app.router.add_get("/api/process_status", handle_process_status)
    app.router.add_post("/api/process_clear", handle_process_clear)
    # Static file serving
    app.router.add_get("/output/{path:.*}", handle_static_file)
    app.router.add_get("/output_persisted/{path:.*}", handle_static_file)
    return app


def main():
    parser = argparse.ArgumentParser(description="RedLens Manager")
    parser.add_argument("--port", type=int, default=8765, help="Server port (default: 8765)")
    args = parser.parse_args()

    print(f"\n  RedLens Manager")
    print(f"  http://localhost:{args.port}")
    print(f"  Output dir: {OUTPUT_DIR}")
    print()

    app = create_app()
    web.run_app(app, host="0.0.0.0", port=args.port, print=None)


if __name__ == "__main__":
    main()
