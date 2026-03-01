#!/usr/bin/env python3
"""
Cloudflare Tunnel 管理脚本

在 autodl 远程主机上通过 cloudflared quick tunnel 暴露服务端口，
替代 SSH port forwarding 实现更快的跨国传输（~2 MB/s vs SSH 3-5 KB/s）。

用法:
    python tools/cf_tunnel.py start          # 启动所有 tunnels
    python tools/cf_tunnel.py retry          # 重试失败的 tunnels
    python tools/cf_tunnel.py stop           # 停止所有 tunnels
    python tools/cf_tunnel.py status         # 查看 tunnel 状态
    python tools/cf_tunnel.py start --ports 8000 7860  # 只启动指定端口
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

# ============ 配置 ============

SSH_HOST = "seetacloud"
CONFIG_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cf_tunnels.json")

# 服务端口 → 名称映射（与 request.py 中 AUTODL_TUNNEL_PORTS 对应）
SERVICE_PORTS = {
    8000: "llm",
    7860: "grounding_dino",
    7861: "depth_anything",
    7862: "som",
}

# ============ 远程命令（单次 SSH 调用，避免多次连接开销）============

def _ssh_run(cmd, timeout=60):
    """在远程主机上执行命令"""
    return subprocess.run(
        ["ssh", "-o", "ConnectTimeout=30", SSH_HOST, cmd],
        capture_output=True, text=True, timeout=timeout,
    )

# ============ 启动 Tunnel（核心：单次 SSH 执行所有操作）============

REMOTE_START_SCRIPT = r'''
#!/bin/bash
# 在远程执行：启动 cloudflared tunnels 并输出 JSON 结果
# 参数: 空格分隔的端口号列表

PORTS="$@"
RESULT='{'
FIRST=true

for PORT in $PORTS; do
    # 只杀掉该端口上的 quick tunnel 旧进程（不影响 named tunnel）
    pkill -f "cloudflared.*--url.*localhost:$PORT" 2>/dev/null || true
    sleep 0.3

    LOG="/tmp/cf_tunnel_${PORT}.log"
    rm -f "$LOG"

    # 启动 cloudflared（后台）
    # --protocol http2: 某些网络（如中国大陆）会阻止 QUIC/UDP，必须用 HTTP/2
    nohup cloudflared tunnel --url "http://localhost:$PORT" --protocol http2 --no-autoupdate > "$LOG" 2>&1 &
    PID=$!

    # 等待 URL 出现（最多 45 秒）
    URL=""
    for i in $(seq 1 45); do
        sleep 1
        if [ -f "$LOG" ]; then
            URL=$(grep -oP 'https://[a-zA-Z0-9-]+\.trycloudflare\.com' "$LOG" | head -1)
            if [ -n "$URL" ]; then
                break
            fi
        fi
    done

    # 输出 JSON 片段
    if [ "$FIRST" = true ]; then
        FIRST=false
    else
        RESULT="${RESULT},"
    fi

    if [ -n "$URL" ]; then
        RESULT="${RESULT}\"${PORT}\":{\"url\":\"${URL}\",\"pid\":${PID}}"
        echo "OK port=$PORT url=$URL pid=$PID" >&2
    else
        RESULT="${RESULT}\"${PORT}\":{\"url\":null,\"pid\":${PID}}"
        echo "FAIL port=$PORT pid=$PID" >&2
    fi
done

RESULT="${RESULT}}"
echo "$RESULT"
'''

def cmd_start(ports=None, merge_existing=False):
    """启动 Cloudflare tunnels（通过单次 SSH 调用）

    Args:
        ports: 要启动的端口列表，None 表示全部
        merge_existing: 为 True 时将新结果合并到现有配置（retry 模式）
    """
    print("🚀 启动 Cloudflare Tunnels\n")

    # 检查 cloudflared
    try:
        result = _ssh_run("cloudflared --version 2>/dev/null", timeout=15)
        if result.returncode == 0:
            print(f"✅ cloudflared: {result.stdout.strip().split(chr(10))[0]}")
        else:
            print("📦 正在远程安装 cloudflared...")
            install_cmd = (
                "wget -q 'https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb' "
                "-O /tmp/cloudflared.deb && dpkg -i /tmp/cloudflared.deb && rm /tmp/cloudflared.deb && cloudflared --version"
            )
            result = _ssh_run(install_cmd, timeout=120)
            if result.returncode != 0:
                print(f"❌ 安装失败: {result.stderr.strip()}")
                sys.exit(1)
            print(f"✅ 安装成功: {result.stdout.strip().split(chr(10))[0]}")
    except subprocess.TimeoutExpired:
        print("❌ SSH 连接超时，请检查 SSH 配置")
        sys.exit(1)

    # 确定要启动的端口
    if ports:
        targets = {p: SERVICE_PORTS.get(p, "port_%d" % p) for p in ports}
    else:
        targets = SERVICE_PORTS.copy()

    port_list = " ".join(str(p) for p in targets.keys())
    print(f"\n📡 启动 {len(targets)} 个 tunnel（单次 SSH 调用）...")
    print(f"   端口: {port_list}")
    print(f"   每个 tunnel 需要 5-30 秒获取 URL，请耐心等待...\n")

    # 单次 SSH 调用执行远程脚本
    remote_cmd = f"bash -c '{REMOTE_START_SCRIPT}' -- {port_list}"
    try:
        proc = subprocess.Popen(
            ["ssh", SSH_HOST, remote_cmd],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )

        # 实时打印 stderr（进度信息）
        import selectors
        sel = selectors.DefaultSelector()
        sel.register(proc.stderr, selectors.EVENT_READ)
        sel.register(proc.stdout, selectors.EVENT_READ)

        stdout_data = ""
        stderr_lines = []
        timeout_sec = 60 * len(targets)  # 每个 tunnel 最多 60 秒
        deadline = time.time() + timeout_sec

        while proc.poll() is None and time.time() < deadline:
            events = sel.select(timeout=1.0)
            for key, _ in events:
                chunk = key.fileobj.read(4096)
                if not chunk:
                    continue
                if key.fileobj is proc.stderr:
                    for line in chunk.strip().split("\n"):
                        if line.strip():
                            print(f"  📡 {line.strip()}")
                            stderr_lines.append(line.strip())
                else:
                    stdout_data += chunk

        sel.close()

        # 读取剩余输出
        remaining_out, remaining_err = proc.communicate(timeout=10)
        stdout_data += remaining_out or ""
        if remaining_err:
            for line in remaining_err.strip().split("\n"):
                if line.strip():
                    print(f"  📡 {line.strip()}")

    except subprocess.TimeoutExpired:
        proc.kill()
        print(f"\n❌ 超时（{timeout_sec}s），请检查远程网络")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ SSH 执行失败: {e}")
        sys.exit(1)

    # 解析 JSON 结果
    stdout_data = stdout_data.strip()
    if not stdout_data:
        print("\n❌ 远程脚本未返回结果")
        sys.exit(1)

    try:
        remote_result = json.loads(stdout_data)
    except json.JSONDecodeError:
        print(f"\n❌ 无法解析远程输出: {stdout_data[:200]}")
        sys.exit(1)

    # 构建配置
    tunnels = {}
    for port_str, info in remote_result.items():
        port = int(port_str)
        name = targets.get(port, "port_%d" % port)
        url = info.get("url")
        if url:
            tunnels[name] = {"local_port": port, "url": url}
            print(f"\n  ✅ {name}: {url}")
        else:
            print(f"\n  ❌ {name} (port {port}): 未获取到 URL")

    if not tunnels:
        print("\n❌ 没有成功启动任何 tunnel")
        sys.exit(1)

    # 保存配置（merge 模式下保留已有的成功 tunnel）
    if merge_existing and os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE) as f:
                existing_config = json.load(f)
            existing_tunnels = existing_config.get("tunnels", {})
            existing_tunnels.update(tunnels)
            tunnels = existing_tunnels
        except (json.JSONDecodeError, IOError):
            pass

    config = {
        "tunnels": tunnels,
        "started_at": datetime.now().isoformat(),
        "ssh_host": SSH_HOST,
    }
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*60}")
    print(f"✅ 成功启动 {len(tunnels)}/{len(targets)} 个 tunnel")
    print(f"📄 配置已保存: {CONFIG_FILE}")
    print(f"\n使用方式:")
    print(f"  python request.py --mode vsp --tunnel cf --max_tasks 10")
    print(f"{'='*60}")


# ============ 重试失败的 Tunnel ============

def cmd_retry():
    """检测上次 start 中失败的端口并重试"""
    if not os.path.exists(CONFIG_FILE):
        print("❌ 未找到 tunnel 配置，请先运行 `python tools/cf_tunnel.py start`")
        sys.exit(1)

    with open(CONFIG_FILE) as f:
        config = json.load(f)

    existing_tunnels = config.get("tunnels", {})
    # 已成功的端口
    ok_ports = {info["local_port"] for info in existing_tunnels.values()}
    # 需要重试的端口
    failed_ports = [p for p in SERVICE_PORTS if p not in ok_ports]

    if not failed_ports:
        print("✅ 所有 tunnel 都已成功，无需重试")
        for name, info in existing_tunnels.items():
            print(f"  {name}: {info['url']}")
        return

    failed_names = [SERVICE_PORTS.get(p, str(p)) for p in failed_ports]
    print(f"🔄 检测到 {len(failed_ports)} 个失败的 tunnel: {', '.join(failed_names)}")
    print(f"   端口: {', '.join(str(p) for p in failed_ports)}\n")

    cmd_start(ports=failed_ports, merge_existing=True)


# ============ 停止 Tunnel ============

def cmd_stop():
    """停止远程所有 cloudflared 进程"""
    print("🛑 停止 Cloudflare Tunnels...")

    try:
        # 只杀掉带 --url 参数的 quick tunnel 进程，不影响 named tunnel（带 --token）
        result = _ssh_run("pkill -f 'cloudflared tunnel --url' 2>/dev/null; rm -f /tmp/cf_tunnel_*.log; echo 'done'", timeout=15)
        print("✅ 已发送停止信号")
    except subprocess.TimeoutExpired:
        print("⚠️  SSH 超时，但停止命令可能已执行")

    # 删除本地配置
    if os.path.exists(CONFIG_FILE):
        os.remove(CONFIG_FILE)
        print(f"✅ 已删除本地配置: {CONFIG_FILE}")

    print("✅ 完成")


# ============ 查看状态 ============

def cmd_status():
    """查看 tunnel 状态"""
    if not os.path.exists(CONFIG_FILE):
        print("❌ 未找到 tunnel 配置（.cf_tunnels.json）")
        print("   运行 `python tools/cf_tunnel.py start` 来启动 tunnels")
        return

    with open(CONFIG_FILE) as f:
        config = json.load(f)

    print(f"📄 配置文件: {CONFIG_FILE}")
    print(f"⏰ 启动时间: {config.get('started_at', 'unknown')}")
    print()

    tunnels = config.get("tunnels", {})
    for name, info in tunnels.items():
        url = info.get("url", "N/A")
        port = info.get("local_port", "?")
        try:
            import urllib.request
            # 用 GET 而非 HEAD（部分服务不支持 HEAD）
            resp = urllib.request.urlopen(url, timeout=10)
            status = "✅ 可达"
        except urllib.error.HTTPError as e:
            # HTTP 错误但连接成功（如 404, 405），说明 tunnel 可达
            status = "✅ 可达" if e.code < 500 else "⚠️  HTTP %d" % e.code
        except Exception:
            status = "❌ 不可达"

        print(f"  {name:20s}  port:{port}  {status}")
        print(f"  {'':20s}  {url}")
        print()


# ============ 工具函数（供 request.py 导入）============

def load_tunnel_config():
    """
    读取 .cf_tunnels.json 并返回 {service_name: url} 映射。
    返回 None 表示未配置或配置无效。
    """
    if not os.path.exists(CONFIG_FILE):
        return None

    try:
        with open(CONFIG_FILE) as f:
            config = json.load(f)
    except (json.JSONDecodeError, IOError):
        return None

    tunnels = config.get("tunnels", {})
    if not tunnels:
        return None

    return {name: info["url"] for name, info in tunnels.items() if "url" in info}


# ============ 入口 ============

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cloudflare Tunnel 管理")
    parser.add_argument("command", choices=["start", "stop", "status", "retry"],
                        help="start: 启动 tunnels, stop: 停止, status: 查看状态, retry: 重试失败的 tunnels")
    parser.add_argument("--ports", type=int, nargs="+", default=None,
                        help="只启动指定端口（默认: 全部）")
    args = parser.parse_args()

    if args.command == "start":
        cmd_start(args.ports)
    elif args.command == "retry":
        cmd_retry()
    elif args.command == "stop":
        cmd_stop()
    elif args.command == "status":
        cmd_status()
