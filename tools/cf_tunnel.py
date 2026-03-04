#!/usr/bin/env python3
"""
Cloudflare Tunnel 管理脚本

支持两种模式：
  - Named Tunnel（推荐）：稳定 URL、单进程、Dashboard 监控
  - Quick Tunnel（fallback）：匿名、随机 URL、每端口一个进程

自动检测：如果存在 .cf_named_tunnel.json 则使用 Named Tunnel，否则回退到 Quick Tunnel。

用法:
    python tools/cf_tunnel.py start          # 自动检测 Named/Quick
    python tools/cf_tunnel.py start --quick  # 强制 Quick Tunnel
    python tools/cf_tunnel.py retry          # 重试失败的 tunnels
    python tools/cf_tunnel.py stop           # 停止所有 tunnels
    python tools/cf_tunnel.py status         # 查看 tunnel 状态
    python tools/cf_tunnel.py setup          # Named Tunnel 配置指南
    python tools/cf_tunnel.py start --ports 8000 7860  # Quick: 只启动指定端口
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
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_FILE = os.path.join(_PROJECT_ROOT, ".cf_tunnels.json")
NAMED_TUNNEL_CONFIG_FILE = os.path.join(_PROJECT_ROOT, ".cf_named_tunnel.json")

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


def _check_cloudflared(auto_install=True):
    """检查远程 cloudflared 是否可用，可选自动安装。返回 True 表示可用。"""
    try:
        result = _ssh_run("cloudflared --version 2>/dev/null", timeout=15)
        if result.returncode == 0:
            print(f"✅ cloudflared: {result.stdout.strip().split(chr(10))[0]}")
            return True
        if not auto_install:
            print("❌ cloudflared 未安装")
            return False
        print("📦 正在远程安装 cloudflared...")
        install_cmd = (
            "wget -q 'https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb' "
            "-O /tmp/cloudflared.deb && dpkg -i /tmp/cloudflared.deb && rm /tmp/cloudflared.deb && cloudflared --version"
        )
        result = _ssh_run(install_cmd, timeout=120)
        if result.returncode != 0:
            print(f"❌ 安装失败: {result.stderr.strip()}")
            return False
        print(f"✅ 安装成功: {result.stdout.strip().split(chr(10))[0]}")
        return True
    except subprocess.TimeoutExpired:
        print("❌ SSH 连接超时，请检查 SSH 配置")
        return False


# ============ Named Tunnel 配置 ============

def load_named_tunnel_config() -> Optional[dict]:
    """加载 .cf_named_tunnel.json。返回配置 dict 或 None。"""
    if not os.path.exists(NAMED_TUNNEL_CONFIG_FILE):
        return None
    try:
        with open(NAMED_TUNNEL_CONFIG_FILE) as f:
            config = json.load(f)
    except (json.JSONDecodeError, IOError):
        return None

    for field in ("tunnel_name", "domain", "services"):
        if field not in config:
            print(f"⚠️  .cf_named_tunnel.json 缺少 '{field}' 字段，回退到 Quick Tunnel")
            return None
    return config


def _resolve_named_tunnel_urls(config: dict) -> Dict[str, str]:
    """从 Named Tunnel 配置推导 {service_name: url}。URL 是确定性的，无需解析日志。"""
    domain = config["domain"]
    return {
        name: f"https://{info['subdomain']}.{domain}"
        for name, info in config["services"].items()
    }


# ============ Named Tunnel 启动 ============

REMOTE_NAMED_START_SCRIPT = r'''
#!/bin/bash
# 启动 Named Tunnel（单进程服务所有 ingress）
# 参数: $1 = config.yml 路径

CONFIG_FILE="${1:-/root/.cloudflared/config.yml}"

# 停止已有的 named tunnel 进程
pkill -f 'cloudflared tunnel.*run' 2>/dev/null || true
sleep 1

LOG="/tmp/cf_named_tunnel.log"
rm -f "$LOG"

# 启动（后台）
nohup cloudflared tunnel --config "$CONFIG_FILE" run > "$LOG" 2>&1 &
PID=$!

# 等待连接注册（最多 30 秒）
for i in $(seq 1 30); do
    sleep 1
    if grep -q 'Registered tunnel connection' "$LOG" 2>/dev/null; then
        echo "OK pid=$PID" >&2
        echo "{\"status\":\"ok\",\"pid\":$PID}"
        exit 0
    fi
    # 检查进程是否还活着
    if ! kill -0 $PID 2>/dev/null; then
        echo "DEAD pid=$PID" >&2
        echo "{\"status\":\"dead\",\"pid\":$PID}"
        exit 1
    fi
done

echo "TIMEOUT pid=$PID" >&2
echo "{\"status\":\"timeout\",\"pid\":$PID}"
exit 1
'''


def cmd_start_named(config: dict):
    """启动 Named Tunnel（单进程，所有服务共享）"""
    tunnel_name = config["tunnel_name"]
    config_file = config.get("config_file", "/root/.cloudflared/config.yml")
    tunnel_urls = _resolve_named_tunnel_urls(config)

    print("🚀 启动 Named Tunnel\n")
    print(f"   Tunnel:  {tunnel_name}")
    print(f"   Config:  {config_file}")
    print(f"   Domain:  {config['domain']}")
    for name, url in tunnel_urls.items():
        port = config["services"][name]["local_port"]
        print(f"   {name}: {url} → localhost:{port}")
    print()

    if not _check_cloudflared(auto_install=False):
        print("请先安装 cloudflared: python tools/cf_tunnel.py start --quick")
        sys.exit(1)

    # 单次 SSH 启动
    remote_cmd = f"bash -c '{REMOTE_NAMED_START_SCRIPT}' -- {config_file}"
    try:
        result = _ssh_run(remote_cmd, timeout=60)
    except subprocess.TimeoutExpired:
        print("❌ 超时（60s），请检查远程网络")
        sys.exit(1)

    stdout = result.stdout.strip()
    if result.stderr:
        for line in result.stderr.strip().split("\n"):
            if line.strip():
                print(f"  📡 {line.strip()}")

    try:
        remote_result = json.loads(stdout)
    except json.JSONDecodeError:
        print(f"❌ 无法解析远程输出: {stdout[:200]}")
        sys.exit(1)

    status = remote_result.get("status")
    if status != "ok":
        print(f"\n❌ Named Tunnel 启动失败 (status={status})")
        print("   查看远程日志: ssh seetacloud cat /tmp/cf_named_tunnel.log")
        sys.exit(1)

    # 生成 .cf_tunnels.json（与 Quick Tunnel 格式兼容，load_tunnel_config() 无需修改）
    tunnels_dict = {}
    for name, info in config["services"].items():
        tunnels_dict[name] = {
            "local_port": info["local_port"],
            "url": tunnel_urls[name],
        }

    runtime_config = {
        "tunnel_type": "named",
        "tunnel_name": tunnel_name,
        "tunnels": tunnels_dict,
        "started_at": datetime.now().isoformat(),
        "ssh_host": SSH_HOST,
    }
    with open(CONFIG_FILE, "w") as f:
        json.dump(runtime_config, f, indent=2)

    print(f"\n{'='*60}")
    print(f"✅ Named Tunnel '{tunnel_name}' 启动成功")
    print(f"📄 配置已保存: {CONFIG_FILE}")
    print(f"\n使用方式:")
    print(f"  python request.py --mode vsp --tunnel cf --max_tasks 10")
    print(f"{'='*60}")


# ============ Quick Tunnel 启动（原有逻辑，完整保留）============

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

def cmd_start_quick(ports=None, merge_existing=False):
    """启动 Quick Tunnels（通过单次 SSH 调用）

    Args:
        ports: 要启动的端口列表，None 表示全部
        merge_existing: 为 True 时将新结果合并到现有配置（retry 模式）
    """
    print("🚀 启动 Quick Tunnels\n")

    if not _check_cloudflared(auto_install=True):
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
        "tunnel_type": "quick",
        "tunnels": tunnels,
        "started_at": datetime.now().isoformat(),
        "ssh_host": SSH_HOST,
    }
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*60}")
    print(f"✅ 成功启动 {len(tunnels)}/{len(targets)} 个 Quick Tunnel")
    print(f"📄 配置已保存: {CONFIG_FILE}")
    print(f"\n使用方式:")
    print(f"  python request.py --mode vsp --tunnel cf --max_tasks 10")
    print(f"{'='*60}")


# ============ 统一入口：自动检测 Named/Quick ============

def cmd_start(ports=None, force_quick=False):
    """启动 Cloudflare Tunnels。自动检测 Named/Quick 模式。"""
    if not force_quick:
        named_config = load_named_tunnel_config()
        if named_config:
            if ports:
                print("⚠️  Named Tunnel 不支持 --ports（单进程服务所有端口），忽略该参数")
            cmd_start_named(named_config)
            return

    cmd_start_quick(ports)


# ============ 重试失败的 Tunnel ============

def cmd_retry():
    """检测上次 start 中失败的端口并重试"""
    # Named Tunnel: 单进程无部分失败，直接重新启动
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE) as f:
                config = json.load(f)
            if config.get("tunnel_type") == "named":
                print("🔄 Named Tunnel 重新启动（单进程，无部分失败模式）\n")
                named_config = load_named_tunnel_config()
                if named_config:
                    cmd_start_named(named_config)
                else:
                    print("❌ .cf_named_tunnel.json 不存在或无效")
                return
        except (json.JSONDecodeError, IOError):
            pass

    # Quick Tunnel: 原有重试逻辑
    if not os.path.exists(CONFIG_FILE):
        print("❌ 未找到 tunnel 配置，请先运行 `python tools/cf_tunnel.py start`")
        sys.exit(1)

    with open(CONFIG_FILE) as f:
        config = json.load(f)

    existing_tunnels = config.get("tunnels", {})
    ok_ports = {info["local_port"] for info in existing_tunnels.values()}
    failed_ports = [p for p in SERVICE_PORTS if p not in ok_ports]

    if not failed_ports:
        print("✅ 所有 tunnel 都已成功，无需重试")
        for name, info in existing_tunnels.items():
            print(f"  {name}: {info['url']}")
        return

    failed_names = [SERVICE_PORTS.get(p, str(p)) for p in failed_ports]
    print(f"🔄 检测到 {len(failed_ports)} 个失败的 tunnel: {', '.join(failed_names)}")
    print(f"   端口: {', '.join(str(p) for p in failed_ports)}\n")

    cmd_start_quick(ports=failed_ports, merge_existing=True)


# ============ 停止 Tunnel ============

def cmd_stop():
    """停止远程 cloudflared 进程（自动检测 Named/Quick）"""
    # 检测当前运行的 tunnel 类型
    tunnel_type = "quick"
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE) as f:
                config = json.load(f)
            tunnel_type = config.get("tunnel_type", "quick")
        except (json.JSONDecodeError, IOError):
            pass

    if tunnel_type == "named":
        print("🛑 停止 Named Tunnel...")
        try:
            _ssh_run(
                "pkill -f 'cloudflared tunnel.*run' 2>/dev/null; rm -f /tmp/cf_named_tunnel.log; echo 'done'",
                timeout=15,
            )
            print("✅ 已发送停止信号")
        except subprocess.TimeoutExpired:
            print("⚠️  SSH 超时，但停止命令可能已执行")
    else:
        print("🛑 停止 Quick Tunnels...")
        try:
            _ssh_run(
                "pkill -f 'cloudflared tunnel --url' 2>/dev/null; rm -f /tmp/cf_tunnel_*.log; echo 'done'",
                timeout=15,
            )
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

    tunnel_type = config.get("tunnel_type", "quick")
    type_label = "Named Tunnel" if tunnel_type == "named" else "Quick Tunnel"

    print(f"📄 配置文件: {CONFIG_FILE}")
    print(f"🏷️  类型: {type_label}")
    if tunnel_type == "named":
        print(f"🔗 Tunnel: {config.get('tunnel_name', 'N/A')}")
    print(f"⏰ 启动时间: {config.get('started_at', 'unknown')}")
    print()

    tunnels = config.get("tunnels", {})
    for name, info in tunnels.items():
        url = info.get("url", "N/A")
        port = info.get("local_port", "?")
        try:
            import urllib.request
            resp = urllib.request.urlopen(url, timeout=10)
            status = "✅ 可达"
        except urllib.error.HTTPError as e:
            status = "✅ 可达" if e.code < 500 else "⚠️  HTTP %d" % e.code
        except Exception:
            status = "❌ 不可达"

        print(f"  {name:20s}  port:{port}  {status}")
        print(f"  {'':20s}  {url}")
        print()


# ============ Named Tunnel 配置指南 ============

def cmd_setup():
    """打印 Named Tunnel 一次性配置步骤"""
    print("📋 Named Tunnel 配置指南")
    print("=" * 60)
    print()
    print("前置条件:")
    print("  1. Cloudflare 免费账户 + 已添加域名")
    print("  2. SSH 可连通 AutoDL (ssh seetacloud)")
    print()

    print("步骤 1/5: 安装 cloudflared（如已安装可跳过）")
    print("  ssh seetacloud")
    print("  wget -q 'https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb' \\")
    print("    -O /tmp/cloudflared.deb && dpkg -i /tmp/cloudflared.deb")
    print()

    print("步骤 2/5: 登录 Cloudflare")
    print("  cloudflared tunnel login")
    print("  # 输出一个 URL → 复制到本地浏览器打开 → 选择域名 → 授权")
    print()

    print("步骤 3/5: 创建 Tunnel")
    print("  cloudflared tunnel create autodl-mediator")
    print("  # 记下输出的 Tunnel UUID（如 f5984285-05a1-47f5-87db-dfe408623a38）")
    print()

    print("步骤 4/5: 创建 DNS 记录 + config.yml")
    print("  # DNS 记录（将 yourdomain.com 替换为你的域名）:")
    print("  cloudflared tunnel route dns autodl-mediator llm.yourdomain.com")
    print("  cloudflared tunnel route dns autodl-mediator dino.yourdomain.com")
    print("  cloudflared tunnel route dns autodl-mediator depth.yourdomain.com")
    print("  cloudflared tunnel route dns autodl-mediator som.yourdomain.com")
    print()
    print("  # 创建 /root/.cloudflared/config.yml:")
    print("  tunnel: autodl-mediator")
    print("  credentials-file: /root/.cloudflared/<UUID>.json")
    print("  protocol: http2")
    print("  ingress:")
    print("    - hostname: llm.yourdomain.com")
    print("      service: http://localhost:8000")
    print("    - hostname: dino.yourdomain.com")
    print("      service: http://localhost:7860")
    print("    - hostname: depth.yourdomain.com")
    print("      service: http://localhost:7861")
    print("    - hostname: som.yourdomain.com")
    print("      service: http://localhost:7862")
    print("    - service: http_status:404")
    print()

    print("步骤 5/5: 创建本地配置")
    print(f"  在项目根目录创建 {NAMED_TUNNEL_CONFIG_FILE}:")
    print('  {')
    print('    "tunnel_name": "autodl-mediator",')
    print('    "config_file": "/root/.cloudflared/config.yml",')
    print('    "domain": "yourdomain.com",')
    print('    "services": {')
    print('      "llm":            {"subdomain": "llm",   "local_port": 8000},')
    print('      "grounding_dino": {"subdomain": "dino",  "local_port": 7860},')
    print('      "depth_anything": {"subdomain": "depth", "local_port": 7861},')
    print('      "som":            {"subdomain": "som",   "local_port": 7862}')
    print('    }')
    print('  }')
    print()

    print("验证:")
    print("  python tools/cf_tunnel.py start    # 自动检测 Named Tunnel")
    print("  python tools/cf_tunnel.py status   # 检查可达性")
    print()
    print("详见 docs/cf_tunnel.md")


# ============ 工具函数（供 request.py 导入）============

def load_tunnel_config():
    """
    读取 .cf_tunnels.json 并返回 {service_name: url} 映射。
    返回 None 表示未配置或配置无效。
    兼容 Named Tunnel 和 Quick Tunnel 两种格式。
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
    parser.add_argument("command", choices=["start", "stop", "status", "retry", "setup"],
                        help="start: 启动, stop: 停止, status: 查看状态, retry: 重试, setup: 配置指南")
    parser.add_argument("--ports", type=int, nargs="+", default=None,
                        help="Quick Tunnel: 只启动指定端口（默认: 全部）")
    parser.add_argument("--quick", action="store_true",
                        help="强制使用 Quick Tunnel（跳过 Named Tunnel 自动检测）")
    args = parser.parse_args()

    if args.command == "start":
        cmd_start(args.ports, force_quick=args.quick)
    elif args.command == "retry":
        cmd_retry()
    elif args.command == "stop":
        cmd_stop()
    elif args.command == "status":
        cmd_status()
    elif args.command == "setup":
        cmd_setup()
