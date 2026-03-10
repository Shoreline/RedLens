# Cloudflare Tunnel 使用指南

## 背景

Mac（美国）通过 SSH 端口转发访问 AutoDL（中国）上的服务时，受 GFW 影响带宽仅 3-5 KB/s。每个 VSP 请求携带 base64 图片（~200KB-1.5MB），单个请求需要数分钟。

Cloudflare Tunnel 利用 CDN 全球节点中继流量，实测 AutoDL → Cloudflare 带宽约 **2 MB/s**，相比 SSH 提速约 400 倍。

| 指标 | SSH Tunnel | CF Tunnel |
|------|-----------|-----------|
| 带宽 | 3-5 KB/s | ~2 MB/s |
| 单请求（200KB 图片） | ~40 秒 | **~4 秒**（含推理） |
| 1000 tasks | ~11 小时 | **~1 小时** |

## 两种模式

脚本支持两种 Cloudflare Tunnel 模式，`start` 时自动检测：

| | Named Tunnel（推荐） | Quick Tunnel（fallback） |
|---|---|---|
| 前置条件 | Cloudflare 账户 + 自有域名 | 无需账户 |
| URL | 固定子域名（如 `llm.yuantian.me`） | 每次随机 `*.trycloudflare.com` |
| 进程数 | 1 个 cloudflared 路由所有服务 | 每个端口 1 个（共 4 个） |
| 启动速度 | ~10 秒 | ~2 分钟（逐个等 URL） |
| 可靠性 | 自动重连、Dashboard 监控 | 偶尔断开需重启 |
| URL 管理 | URL 已知，无需解析日志 | 需从日志解析，可能超时失败 |

**自动检测逻辑**：项目根目录存在 `.cf_named_tunnel.json` → Named Tunnel，否则 → Quick Tunnel。

## 快速开始

```bash
# 1. 启动 tunnels（自动检测 Named/Quick）
python tools/cf_tunnel.py start

# 2. 查看状态
python tools/cf_tunnel.py status

# 3. 使用 CF tunnel 运行任务
python request.py --mode vsp --tunnel cf --max_tasks 50

# 4. 停止 tunnels
python tools/cf_tunnel.py stop
```

---

## Named Tunnel 配置（一次性）

### 前置条件

- Cloudflare 免费账户，已添加域名（如 `yuantian.me`）
- SSH 可连通 AutoDL（`ssh seetacloud`）

### 步骤 1: 安装 cloudflared

```bash
ssh seetacloud

# 如已安装可跳过
wget -q 'https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb' \
  -O /tmp/cloudflared.deb && dpkg -i /tmp/cloudflared.deb
cloudflared --version
```

### 步骤 2: 登录 Cloudflare

```bash
# 在 AutoDL 上执行
cloudflared tunnel login
```

输出一个 URL，复制到本地浏览器打开，选择域名并授权。授权成功后会在 `/root/.cloudflared/` 下生成证书文件。

### 步骤 3: 创建 Tunnel

```bash
cloudflared tunnel create autodl-redlens
# 输出: Created tunnel autodl-redlens with id <UUID>
# 凭证文件: /root/.cloudflared/<UUID>.json
```

记下 **Tunnel UUID**（如 `f5984285-05a1-47f5-87db-dfe408623a38`）。

### 步骤 4: 创建 DNS 记录

将 `yourdomain.com` 替换为你的实际域名：

```bash
cloudflared tunnel route dns autodl-redlens llm.yourdomain.com
cloudflared tunnel route dns autodl-redlens dino.yourdomain.com
cloudflared tunnel route dns autodl-redlens depth.yourdomain.com
cloudflared tunnel route dns autodl-redlens som.yourdomain.com
```

每条命令会在 Cloudflare DNS 中自动创建 CNAME 记录。如果某条因网络超时失败，重试即可。

### 步骤 5: 创建 config.yml

在 AutoDL 上创建 `/root/.cloudflared/config.yml`（将 `<UUID>` 和域名替换为实际值）：

```yaml
tunnel: autodl-redlens
credentials-file: /root/.cloudflared/<UUID>.json
protocol: http2

ingress:
  - hostname: llm.yourdomain.com
    service: http://localhost:8000
  - hostname: dino.yourdomain.com
    service: http://localhost:7860
  - hostname: depth.yourdomain.com
    service: http://localhost:7861
  - hostname: som.yourdomain.com
    service: http://localhost:7862
  - service: http_status:404
```

关键配置说明：
- `protocol: http2`：AutoDL 网络封锁 QUIC/UDP，必须用 HTTP/2
- 最后一条 `service: http_status:404` 是 catch-all 规则（cloudflared 必需）

### 步骤 6: 创建本地配置

在 RedLens 项目根目录创建 `.cf_named_tunnel.json`（已 gitignore）：

```json
{
  "tunnel_name": "autodl-redlens",
  "config_file": "/root/.cloudflared/config.yml",
  "domain": "yourdomain.com",
  "services": {
    "llm":            {"subdomain": "llm",   "local_port": 8000},
    "grounding_dino": {"subdomain": "dino",  "local_port": 7860},
    "depth_anything": {"subdomain": "depth", "local_port": 7861},
    "som":            {"subdomain": "som",   "local_port": 7862}
  }
}
```

URL 由 `https://{subdomain}.{domain}` 确定性推导（如 `https://llm.yuantian.me`），无需解析日志。

### 步骤 7: 验证

```bash
# 前台测试（看日志，出现 "Registered tunnel connection" 即成功）
ssh seetacloud "cloudflared tunnel --config /root/.cloudflared/config.yml run"

# 或通过脚本启动
python tools/cf_tunnel.py start    # 自动检测到 Named Tunnel
python tools/cf_tunnel.py status   # 检查可达性

# Mac 上 curl 测试
curl -s -o /dev/null -w "%{http_code}" https://llm.yourdomain.com
# 返回任何 HTTP 状态码（如 404）说明 tunnel 通了
```

---

## 工作原理

### Named Tunnel 架构

```
Mac (美国)                        Cloudflare CDN                    AutoDL (中国)
┌──────────┐                     ┌──────────────┐                 ┌──────────────┐
│request.py│──HTTPS──→│ CF Edge  │──HTTP/2──→│ cloudflared  │
│          │          │ llm.yourdomain.com   │              │ (named tunnel)│
│provider  │          │ dino.yourdomain.com  │ 1 个进程     │
│  .py     │          │ depth.yourdomain.com │ 路由所有服务 │
│          │          │ som.yourdomain.com   │    ↓          │
└──────────┘          └──────────────┘       │ localhost:PORT│
                                             └──────────────┘
```

1. `cf_tunnel.py start` 检测 `.cf_named_tunnel.json` 存在 → Named Tunnel 模式
2. 通过 SSH 在 AutoDL 上启动单个 `cloudflared tunnel run` 进程
3. cloudflared 根据 `config.yml` 的 ingress 规则将不同 hostname 路由到不同本地端口
4. 生成 `.cf_tunnels.json`（与 Quick Tunnel 格式兼容），URL 为固定子域名
5. `request.py --tunnel cf` 读取配置，通过环境变量传递给 VSP 子进程

### Quick Tunnel 架构（fallback）

```
Mac (美国)                        Cloudflare CDN                    AutoDL (中国)
┌──────────┐                     ┌──────────────┐                 ┌──────────────┐
│request.py│──HTTPS──→│ CF Edge  │──HTTP/2──→│ cloudflared  │
│          │          │ *.trycloudflare.com  │              │ (4 个进程)    │
│provider  │          └──────────────┘                 │    ↓           │
│  .py     │                                           │ localhost:PORT │
│          │                                           │  (LLM/DINO/..)│
└──────────┘                                           └──────────────┘
```

1. `cf_tunnel.py start` 未找到 `.cf_named_tunnel.json` → Quick Tunnel 模式
2. 通过 SSH 为每个服务端口启动一个 `cloudflared` 进程（共 4 个）
3. 每个进程获得随机 `*.trycloudflare.com` URL（需从日志解析）
4. URL 保存到 `.cf_tunnels.json`
5. 后续流程与 Named Tunnel 相同

### 服务端口映射

| 端口 | 服务 | 环境变量 | Named Tunnel 子域名 |
|------|------|---------|---------------------|
| 8000 | LLM (vLLM/Qwen) | `LLM_BASE_URL` | `llm` |
| 7860 | GroundingDINO | `VSP_GROUNDING_DINO_ADDRESS` | `dino` |
| 7861 | Depth Anything | `VSP_DEPTH_ANYTHING_ADDRESS` | `depth` |
| 7862 | SOM | `VSP_SOM_ADDRESS` | `som` |

端口映射定义在 `tools/cf_tunnel.py` 的 `SERVICE_PORTS` 字典中。

### 运行时配置文件

`.cf_tunnels.json`（项目根目录，gitignore，`start` 时自动生成）：

```json
{
  "tunnel_type": "named",
  "tunnel_name": "autodl-redlens",
  "tunnels": {
    "llm": {"local_port": 8000, "url": "https://llm.yuantian.me"},
    "grounding_dino": {"local_port": 7860, "url": "https://dino.yuantian.me"},
    "depth_anything": {"local_port": 7861, "url": "https://depth.yuantian.me"},
    "som": {"local_port": 7862, "url": "https://som.yuantian.me"}
  },
  "started_at": "2026-03-04T15:30:00",
  "ssh_host": "seetacloud"
}
```

Quick Tunnel 模式下 `tunnel_type` 为 `"quick"`，URL 为 `*.trycloudflare.com`。两种格式均被 `load_tunnel_config()` 兼容读取，下游代码（`request.py`、`provider.py`）无需区分。

## 命令详解

### `start`

```bash
python tools/cf_tunnel.py start              # 自动检测 Named/Quick
python tools/cf_tunnel.py start --quick      # 强制 Quick Tunnel
python tools/cf_tunnel.py start --ports 8000 7860  # Quick: 只启动指定端口
```

**Named Tunnel 模式**：
1. 读取 `.cf_named_tunnel.json` 获取 tunnel 名称和 config 路径
2. 通过 SSH 执行 `cloudflared tunnel --config ... run`（单进程）
3. 等待 "Registered tunnel connection"（最多 30 秒）
4. 生成 `.cf_tunnels.json`

**Quick Tunnel 模式**（`.cf_named_tunnel.json` 不存在或 `--quick`）：
1. 检查远程 `cloudflared`，未安装则自动下载安装
2. 通过单次 SSH 调用执行远程 bash 脚本
3. 对每个端口：终止旧进程 → 启动 `cloudflared` → 等待 URL（最多 45 秒）
4. 解析返回的 JSON，保存到 `.cf_tunnels.json`

关键参数：
- `--protocol http2`：AutoDL 网络封锁 QUIC/UDP，必须用 HTTP/2
- `--no-autoupdate`：禁止 cloudflared 自动更新（仅 Quick Tunnel）

### `stop`

```bash
python tools/cf_tunnel.py stop
```

自动检测当前运行的 tunnel 类型（读取 `.cf_tunnels.json` 的 `tunnel_type` 字段）：
- **Named**：`pkill -f 'cloudflared tunnel.*run'`
- **Quick**：`pkill -f 'cloudflared tunnel --url'`
- 删除本地 `.cf_tunnels.json`

### `status`

```bash
python tools/cf_tunnel.py status
```

- 显示 tunnel 类型（Named/Quick）和 tunnel 名称
- 读取 `.cf_tunnels.json`，对每个 URL 发送 HTTP GET 检查可达性
- HTTP 状态码 < 500 视为可达（部分服务对 GET 返回 404/405 但 tunnel 本身正常）

### `retry`

```bash
python tools/cf_tunnel.py retry
```

- **Named Tunnel**：单进程无部分失败，直接重新启动
- **Quick Tunnel**：检测失败端口，仅重试失败的部分

### `setup`

```bash
python tools/cf_tunnel.py setup
```

打印 Named Tunnel 一次性配置步骤指南。

## `--tunnel` 参数（request.py）

| 值 | 行为 |
|----|------|
| `ssh`（默认） | 使用 SSH 端口转发，自动建立 tunnel |
| `cf` | 使用 Cloudflare Tunnel，读取 `.cf_tunnels.json`（Named 或 Quick 均可） |
| `none` | 不建立任何 tunnel（等价于已废弃的 `--no-ssh-tunnel`） |

仅 `vsp` 和 `comt_vsp` 模式会使用 tunnel。`direct` 模式直接通过 `--llm_base_url` 访问 LLM，不需要 tunnel。

## 涉及的代码文件

| 文件 | 内容 |
|------|------|
| `tools/cf_tunnel.py` | Tunnel 管理脚本（start/stop/status/retry/setup，支持 Named + Quick） |
| `.cf_named_tunnel.json` | Named Tunnel 静态配置（用户创建，gitignore） |
| `.cf_tunnels.json` | 运行时配置（start 自动生成，gitignore） |
| `request.py` | `--tunnel` 参数、`ensure_cf_tunnels()` 函数、RunConfig.tunnel_urls |
| `provider.py` | VSPProvider 启动子进程时传递 tunnel URL 环境变量 |
| `VisualSketchpad/agent/config.py` | 工具端点地址支持环境变量覆盖 |

## 注意事项

1. **Quick Tunnel URL 不稳定**：每次 `start` URL 都会变化。Named Tunnel 的 URL 是固定子域名，无此问题。

2. **SSH 仍需可用**：无论 Named 还是 Quick，`cf_tunnel.py start` 都通过 SSH 在远程启动 cloudflared，因此 SSH 连接仍需配置正确（`~/.ssh/config` 中的 `seetacloud` host）。

3. **AutoDL 开机端口变化**：AutoDL 每次开机可能分配不同的 SSH 端口，需更新 `~/.ssh/config`。

4. **QUIC 被封**：AutoDL 网络环境下 QUIC/UDP 协议不通。Named Tunnel 在 `config.yml` 中设置 `protocol: http2`，Quick Tunnel 由脚本传递 `--protocol http2`。

5. **Named Tunnel 凭证持久化**：确保 `/root/.cloudflared/` 目录在 AutoDL 的持久存储上（通常 `/root/` 持久化）。如果使用临时系统盘，需将凭证存放到数据盘并创建软链接。

6. **`--quick` 回退**：如果 Named Tunnel 出现问题，可用 `python tools/cf_tunnel.py start --quick` 临时回退到 Quick Tunnel。

## 故障排查

| 现象 | 可能原因 | 解决方法 |
|------|---------|---------|
| `start` 超时 | SSH 连接慢或不通 | 检查 `ssh seetacloud` 是否正常 |
| Named Tunnel 启动失败 | 凭证过期或 config.yml 有误 | `ssh seetacloud cat /tmp/cf_named_tunnel.log` |
| Quick Tunnel URL 获取失败 | cloudflared 启动失败 | `ssh seetacloud cat /tmp/cf_tunnel_PORT.log` |
| `status` 显示不可达 | Tunnel 进程已退出 | 重新 `start` |
| HTTP 530 错误 | Tunnel 连接断开 | 重新 `start`；检查远程服务是否在对应端口运行 |
| DNS 记录创建超时 | Cloudflare API 网络抖动 | 重试 `cloudflared tunnel route dns` 命令 |
| 请求缓慢 | CDN 路由不佳 | Named: 稳定路由，少见；Quick: 可重启获取新 URL |
