#!/usr/bin/env python3
"""CF Tunnel 网速测试

通过 curl 精确计时，测量本地到 AutoDL 经 CF Quick Tunnel 的延迟和吞吐量。
分离 DNS/TCP/TLS/TTFB/传输 各阶段耗时。

用法:
    python tools/bench_tunnel.py                  # 从 .cf_tunnels.json 读取
    python tools/bench_tunnel.py --url https://xxx.trycloudflare.com
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import statistics

ROUNDS = 3

SIZES = [
    ("ping",    0),
    ("10 KB",   10_240),
    ("100 KB",  102_400),
    ("500 KB",  512_000),
    ("1 MB",    1_048_576),
    ("3 MB",    3_145_728),
]

# curl 计时格式
CURL_FMT = json.dumps({
    "dns":        "%{time_namelookup}",
    "tcp":        "%{time_connect}",
    "tls":        "%{time_appconnect}",
    "ttfb":       "%{time_starttransfer}",
    "total":      "%{time_total}",
    "upload_speed": "%{speed_upload}",
    "download_speed": "%{speed_download}",
    "size_upload": "%{size_upload}",
    "size_download": "%{size_download}",
    "http_code":  "%{http_code}",
})


def load_cf_url():
    cfg_path = os.path.join(os.path.dirname(__file__), "..", ".cf_tunnels.json")
    try:
        with open(cfg_path) as f:
            return json.load(f)["tunnels"]["llm"]["url"]
    except (FileNotFoundError, KeyError):
        return None


def curl_once(url, data_file=None):
    """执行一次 curl 并返回计时 dict"""
    cmd = [
        "curl", "-s", "-o", "/dev/null",
        "-w", CURL_FMT,
        "--max-time", "30",
    ]
    if data_file:
        cmd += ["-X", "POST", "-d", f"@{data_file}",
                "-H", "Content-Type: application/octet-stream"]
    cmd.append(url)

    r = subprocess.run(cmd, capture_output=True, text=True, timeout=35)
    if r.returncode != 0 and not r.stdout:
        return None
    try:
        return {k: float(v) if k != "http_code" else int(float(v))
                for k, v in json.loads(r.stdout).items()}
    except (json.JSONDecodeError, ValueError):
        return None


def fmt_ms(s):
    return f"{s*1000:>7.0f}ms"


def fmt_speed(bps):
    """bytes/s → 人类可读"""
    if bps < 1024:
        return f"{bps:.0f} B/s"
    elif bps < 1024 * 1024:
        return f"{bps/1024:.0f} KB/s"
    else:
        return f"{bps/1024/1024:.1f} MB/s"


def bench(base_url, rounds):
    # 用 /v1/models 做 GET 测试端点, 用 /v1/chat/completions 做 POST 测试
    get_url = base_url.rstrip("/") + "/v1/models"
    post_url = base_url.rstrip("/") + "/v1/chat/completions"

    print(f"  目标: {base_url}")
    print(f"  轮数: {rounds}\n")

    # 连通性
    r = curl_once(get_url)
    if not r:
        print("  连通性: 失败 ❌")
        sys.exit(1)
    print(f"  连通性: HTTP {r['http_code']} ✅")
    print(f"  首次: DNS={r['dns']*1000:.0f}ms TCP={r['tcp']*1000:.0f}ms "
          f"TLS={r['tls']*1000:.0f}ms TTFB={r['ttfb']*1000:.0f}ms Total={r['total']*1000:.0f}ms\n")

    # ── Latency 测试 (GET /v1/models) ──
    print("  ── Latency (GET /v1/models) ──\n")
    ttfbs = []
    for i in range(rounds):
        r = curl_once(get_url)
        if r:
            ttfbs.append(r["ttfb"])
            print(f"    #{i+1}  TTFB={fmt_ms(r['ttfb'])}  Total={fmt_ms(r['total'])}")
    if ttfbs:
        print(f"    →  TTFB 平均={fmt_ms(statistics.mean(ttfbs))}  "
              f"最小={fmt_ms(min(ttfbs))}")

    # ── Upload 测试 (POST 递增 payload) ──
    print(f"\n  ── Upload (POST 不同大小 payload) ──\n")
    print(f"  {'档位':>8}  {'TTFB':>9}  {'Total':>9}  {'上传速率':>12}  {'可视化'}")
    print(f"  {'─'*60}")

    results = []
    for name, size in SIZES:
        # 创建临时数据文件
        data_file = None
        if size > 0:
            f = tempfile.NamedTemporaryFile(delete=False, suffix=".bin")
            f.write(os.urandom(size))
            f.close()
            data_file = f.name

        totals = []
        ttfbs_up = []
        speeds = []
        for _ in range(rounds):
            url = post_url if data_file else get_url
            r = curl_once(url, data_file)
            if r:
                totals.append(r["total"])
                ttfbs_up.append(r["ttfb"])
                if r["upload_speed"] > 0:
                    speeds.append(r["upload_speed"])

        if data_file:
            os.unlink(data_file)

        if not totals:
            print(f"  {name:>8}  超时 ❌")
            continue

        avg_total = statistics.mean(totals)
        min_total = min(totals)
        avg_ttfb = statistics.mean(ttfbs_up)
        avg_speed = statistics.mean(speeds) if speeds else 0

        bar_width = int(min(avg_total * 4, 30))
        bar = "█" * bar_width

        speed_str = fmt_speed(avg_speed) if avg_speed > 0 else "—"
        print(f"  {name:>8}  {fmt_ms(avg_ttfb)}  {fmt_ms(avg_total)}  {speed_str:>12}  {bar}")

        results.append({
            "name": name, "size": size,
            "avg_total": avg_total, "min_total": min_total,
            "avg_ttfb": avg_ttfb, "avg_speed": avg_speed,
        })

    # 估算
    if len(results) >= 2:
        small = next((r for r in results if r["size"] == 0), results[0])
        big = max(results, key=lambda r: r["size"])
        delta_bytes = big["size"] - small["size"]
        delta_time = big["min_total"] - small["min_total"]
        if delta_bytes > 0 and delta_time > 0:
            bw = delta_bytes / delta_time
            print(f"\n  ── 估算 ──")
            print(f"  基础延迟: ~{small['min_total']*1000:.0f}ms")
            print(f"  有效上传带宽: ~{fmt_speed(bw)}")
            for label, sz in [("500KB", 500*1024), ("1MB", 1024**2), ("5MB", 5*1024**2)]:
                est = small["min_total"] + sz / bw
                print(f"  → 传 {label} 预计: {est:.2f}s")


def main():
    parser = argparse.ArgumentParser(description="CF Tunnel 网速测试")
    parser.add_argument("--url", help="CF Tunnel URL")
    parser.add_argument("--rounds", type=int, default=ROUNDS,
                        help=f"每档测试轮数 (默认 {ROUNDS})")
    args = parser.parse_args()

    url = args.url or load_cf_url()
    if not url:
        print("❌ 请用 --url 指定 URL 或确保 .cf_tunnels.json 存在")
        sys.exit(1)

    print(f"\n🚀 CF Tunnel 网速测试\n")
    bench(url, args.rounds)
    print()


if __name__ == "__main__":
    main()
