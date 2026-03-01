# Cloudflare Tunnel 使用指南

## 背景

Mac（美国）通过 SSH 端口转发访问 AutoDL（中国）上的服务时，受 GFW 影响带宽仅 3-5 KB/s。每个 VSP 请求携带 base64 图片（~200KB-1.5MB），单个请求需要数分钟。

Cloudflare Quick Tunnel 利用 CDN 全球节点中继流量，实测 AutoDL → Cloudflare 带宽约 **2 MB/s**，相比 SSH 提速约 400 倍。

| 指标 | SSH Tunnel | CF Tunnel |
|------|-----------|-----------|
| 带宽 | 3-5 KB/s | ~2 MB/s |
| 单请求（200KB 图片） | ~40 秒 | **~4 秒**（含推理） |
| 1000 tasks | ~11 小时 | **~1 小时** |

## 快速开始

```bash
# 1. 启动 tunnels（首次使用自动在 AutoDL 安装 cloudflared）
python tools/cf_tunnel.py start

# 2. 查看状态
python tools/cf_tunnel.py status

# 3. 使用 CF tunnel 运行任务
python request.py --mode vsp --tunnel cf --max_tasks 50

# 4. 停止 tunnels
python tools/cf_tunnel.py stop
```

## 工作原理

### 架构

```
Mac (美国)                        Cloudflare CDN                    AutoDL (中国)
┌──────────┐                     ┌──────────────┐                 ┌──────────────┐
│request.py│──HTTPS──→│ CF Edge  │──HTTP/2──→│ cloudflared  │
│          │          │ *.trycloudflare.com  │              │ (quick tunnel)│
│provider  │          └──────────────┘                 │    ↓           │
│  .py     │                                           │ localhost:PORT │
│          │                                           │  (LLM/DINO/..)│
└──────────┘                                           └──────────────┘
```

1. `cf_tunnel.py start` 通过 SSH 在 AutoDL 上为每个服务端口启动一个 `cloudflared` 进程
2. 每个进程创建一个 Quick Tunnel，获得唯一的 `*.trycloudflare.com` URL
3. URL 保存到本地 `.cf_tunnels.json` 配置文件
4. `request.py --tunnel cf` 读取配置，通过环境变量将 URL 传递给 VSP 子进程

### 服务端口映射

| 端口 | 服务 | 环境变量 |
|------|------|---------|
| 8000 | LLM (vLLM/Qwen) | `LLM_BASE_URL` |
| 7860 | GroundingDINO | `VSP_GROUNDING_DINO_ADDRESS` |
| 7861 | Depth Anything | `VSP_DEPTH_ANYTHING_ADDRESS` |
| 7862 | SOM | `VSP_SOM_ADDRESS` |

端口映射定义在 `tools/cf_tunnel.py` 的 `SERVICE_PORTS` 字典中。

### 配置文件格式

`.cf_tunnels.json`（项目根目录，已 gitignore）：

```json
{
  "tunnels": {
    "llm": {"local_port": 8000, "url": "https://abc-xyz.trycloudflare.com"},
    "grounding_dino": {"local_port": 7860, "url": "https://def-uvw.trycloudflare.com"},
    "depth_anything": {"local_port": 7861, "url": "https://ghi-rst.trycloudflare.com"},
    "som": {"local_port": 7862, "url": "https://jkl-opq.trycloudflare.com"}
  },
  "started_at": "2026-02-25T15:30:00",
  "ssh_host": "seetacloud"
}
```

## 命令详解

### `start`

```bash
python tools/cf_tunnel.py start              # 启动所有 4 个服务的 tunnel
python tools/cf_tunnel.py start --ports 8000 7860  # 只启动指定端口
```

执行流程：
1. 检查远程是否已安装 `cloudflared`，未安装则自动下载 deb 包安装
2. 通过**单次 SSH 调用**执行远程 bash 脚本（避免多次 SSH 连接的开销）
3. 对每个端口：终止旧的 quick tunnel 进程 → 启动新 `cloudflared` → 等待 URL（最多 45 秒）
4. 解析返回的 JSON，保存到 `.cf_tunnels.json`

关键参数：
- `--protocol http2`：AutoDL 网络封锁 QUIC/UDP，必须用 HTTP/2
- `--no-autoupdate`：禁止 cloudflared 自动更新

### `stop`

```bash
python tools/cf_tunnel.py stop
```

- 通过 SSH 执行 `pkill -f 'cloudflared tunnel --url'` 终止 quick tunnel 进程
- 使用 `--url` 匹配模式，**不会影响**用户可能运行的 Named Tunnel（带 `--token` 参数）
- 删除本地 `.cf_tunnels.json`

### `status`

```bash
python tools/cf_tunnel.py status
```

- 读取 `.cf_tunnels.json`，对每个 URL 发送 HTTP GET 检查可达性
- HTTP 状态码 < 500 视为可达（部分服务对 GET 返回 404/405 但 tunnel 本身正常）

## `--tunnel` 参数（request.py）

| 值 | 行为 |
|----|------|
| `ssh`（默认） | 使用 SSH 端口转发，自动建立 tunnel |
| `cf` | 使用 Cloudflare Tunnel，读取 `.cf_tunnels.json` |
| `none` | 不建立任何 tunnel（等价于已废弃的 `--no-ssh-tunnel`） |

仅 `vsp` 和 `comt_vsp` 模式会使用 tunnel。`direct` 模式直接通过 `--llm_base_url` 访问 LLM，不需要 tunnel。

## 涉及的代码文件

| 文件 | 修改内容 |
|------|---------|
| `tools/cf_tunnel.py` | Tunnel 管理脚本（start/stop/status） |
| `request.py` | `--tunnel` 参数、`ensure_cf_tunnels()` 函数、RunConfig.tunnel_urls |
| `provider.py` | VSPProvider 启动子进程时传递 tunnel URL 环境变量 |
| `VisualSketchpad/agent/config.py` | 工具端点地址支持环境变量覆盖 |

## 注意事项

1. **URL 不稳定**：Quick Tunnel 的 URL 每次 `start` 都会变化。如需稳定 URL，可注册 Cloudflare 免费账号使用 Named Tunnel。

2. **SSH 仍需可用**：`cf_tunnel.py start` 通过 SSH 在远程启动 cloudflared，因此 SSH 连接仍需配置正确（`~/.ssh/config` 中的 `seetacloud` host）。

3. **AutoDL 开机端口变化**：AutoDL 每次开机可能分配不同的 SSH 端口，需更新 `~/.ssh/config`。

4. **QUIC 被封**：AutoDL 网络环境下 QUIC/UDP 协议不通，脚本已强制使用 `--protocol http2`。

5. **Named Tunnel 共存**：如果 AutoDL 上已运行 Named Tunnel（带 `--token`），`stop` 命令不会影响它，因为 pkill 只匹配 `cloudflared tunnel --url` 模式。

## 故障排查

| 现象 | 可能原因 | 解决方法 |
|------|---------|---------|
| `start` 超时 | SSH 连接慢或不通 | 检查 `ssh seetacloud` 是否正常 |
| URL 获取失败 | cloudflared 启动失败 | 查看远程日志 `ssh seetacloud cat /tmp/cf_tunnel_PORT.log` |
| `status` 显示不可达 | Tunnel 进程已退出 | 重新 `start` |
| HTTP 530 错误 | Tunnel 连接断开 | 重新 `start`；检查远程服务是否在对应端口运行 |
| 请求缓慢 | CDN 路由不佳 | 属偶发现象，可重启 tunnel 获取新 URL |
