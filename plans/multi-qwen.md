# 多 LLM 实例并行推理支持

## Context

A800 80GB 显存跑 Qwen 3 VL-8B 只用 ~20GB，可同时跑 3 个实例来加速推理。当前 server.py 是自定义 FastAPI 服务（HuggingFace transformers 直接推理，**无 batching**），每次只能处理一个请求，所以多实例会有接近线性的提速。

当前项目只支持单个 `--llm_base_url`，需要两端改造：
1. **AutoDL 端**：改造 `qwen.sh` 支持多实例管理
2. **RedLens 端**：支持多端点 round-robin 负载均衡

## 一、AutoDL 端：qwen.sh 多实例支持

当前 `qwen.sh` 用单个 PID 文件 (`qwen.pid`) 管理一个实例。改造为支持按端口区分的多实例管理。

### 改造要点

- PID 文件改为 `qwen_$PORT.pid`（如 `qwen_8000.pid`, `qwen_8001.pid`）
- 日志文件改为 `qwen_$PORT.log`
- 新增 `start-multi` 子命令：一键启动 N 个实例（端口 8000, 8001, 8002...）
- 新增 `stop-all` 子命令：停止所有实例
- `status` 显示所有实例状态
- 原有 `start/stop/restart` 保持兼容，默认操作端口 8000

### CLI 用法

```bash
# 一键启动 3 个实例
./qwen.sh start-multi 3

# 查看所有实例状态
./qwen.sh status
# [+] Qwen server (port 8000): RUNNING (pid 1234)
# [+] Qwen server (port 8001): RUNNING (pid 1235)
# [+] Qwen server (port 8002): RUNNING (pid 1236)

# 停止所有
./qwen.sh stop-all

# 原有用法不变
./qwen.sh start --port 8001
./qwen.sh stop --port 8001
```

### 关键修改

文件：AutoDL 上 `~/qwen.sh`

1. PID/日志 文件路径函数化，按端口区分：
```bash
pid_file()  { echo "$PID_DIR/qwen_${1:-$DEFAULT_PORT}.pid"; }
log_file()  { echo "$LOG_DIR/qwen_${1:-$DEFAULT_PORT}.log"; }
```

2. `is_running` / `do_start` / `do_stop` 接受 port 参数

3. 新增 `do_start_multi N`：循环调用 `do_start --port $((DEFAULT_PORT + i))`

4. 新增 `do_stop_all`：遍历 PID 目录中所有 `qwen_*.pid`，逐个停止

5. `do_status`：列出所有 `qwen_*.pid` 的状态

## 二、RedLens 端

### CLI 用法

```bash
# 方式 1: --llm_instances 自动递增端口（推荐）
python request.py --llm_base_url "http://localhost:18000/v1" --llm_instances 3
# → localhost:18000, localhost:18001, localhost:18002

# 方式 2: 逗号分隔手动指定
python request.py --llm_base_url "http://localhost:18000/v1,http://localhost:18001/v1,http://localhost:18002/v1"

# 方式 3: 单端点（完全向后兼容）
python request.py --llm_base_url "http://localhost:18000/v1"
```

### 1. request.py — RunConfig + CLI

**RunConfig** (line 285):
- `llm_base_url: Optional[str]` → `llm_base_urls: Optional[List[str]]`
- 添加 `@property llm_base_url` 兼容属性，返回首个 URL

**Argparse** (line 1481):
- `--llm_base_url` 保持 string 类型，支持逗号分隔
- 新增 `--llm_instances N`（默认 1）：从 base URL 端口递增生成 N 个 URL

**解析逻辑** (args → RunConfig, ~line 1572):
- 含逗号 → split 成列表
- `--llm_instances > 1` → 端口递增生成列表
- 两者互斥校验

**SSH Tunnel** (line 61): `AUTODL_TUNNEL_PORTS` 动态扩展
- 根据 `llm_instances` 添加 18001→8001、18002→8002 等额外映射

### 2. provider.py — MultiEndpointProvider

新增类（放在 OpenRouterProvider 之后）:
```python
class MultiEndpointProvider(BaseProvider):
    """Round-robin 分发请求到多个 LLM 端点"""
    def __init__(self, providers: List[OpenRouterProvider]):
        self.providers = providers
        self._counter = 0

    async def send(self, prompt_struct, cfg) -> str:
        provider = self.providers[self._counter % len(self.providers)]
        self._counter += 1
        return await provider.send(prompt_struct, cfg)
```

**get_provider()** (line 1120):
- 多 URL → 为每个创建 OpenRouterProvider，包装为 MultiEndpointProvider
- 单 URL → 现有逻辑不变

### 3. provider.py — VSP/CoMT-VSP 模式

VSPProvider._call_vsp (line 399): round-robin 选择 URL 传给子进程:
```python
if cfg and getattr(cfg, 'llm_base_urls', None):
    urls = cfg.llm_base_urls
    url = urls[self._rr_counter % len(urls)]
    self._rr_counter += 1
    env["LLM_BASE_URL"] = url
```

### 4. tools/cf_tunnel.py — 多 LLM 端口

`SERVICE_PORTS` 预定义 8001/8002:
```python
SERVICE_PORTS = {
    8000: "llm",
    8001: "llm_1",   # 多实例扩展
    8002: "llm_2",   # 多实例扩展
    7860: "grounding_dino",
    7861: "depth_anything",
    7862: "som",
}
```
`start` 时只为实际监听的端口创建 tunnel（已有端口检测逻辑）。

### 5. request.py — 启动日志

多实例时显示:
```
🔗 LLM 端点: 3 个实例 (round-robin)
   - http://localhost:18000/v1
   - http://localhost:18001/v1
   - http://localhost:18002/v1
```

### 6. job_fix.py — 向后兼容

读取 `run_config.json` 时兼容旧格式 `llm_base_url`（string）和新格式 `llm_base_urls`（list）。

## 不修改的部分

- `mmsb_eval.py` — 评估用 GPT，不涉及自部署端点
- `batch_request.py` — 间接通过 request.py 获得支持
- Hidden states 捕获 — 每个 OpenRouterProvider 实例独立捕获，无需额外修改

## 验证

1. AutoDL: `./qwen.sh start-multi 3` → `./qwen.sh status` 确认 3 个实例运行
2. 单端点向后兼容: `python request.py --llm_base_url "http://localhost:18000/v1" --max_tasks 5`
3. 多实例: `python request.py --llm_base_url "http://localhost:18000/v1" --llm_instances 3 --max_tasks 30`
4. 检查日志中 round-robin 分发是否均匀
5. 检查 hidden states 是否正常保存
