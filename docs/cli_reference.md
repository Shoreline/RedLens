# request.py 命令行参数参考

## 基础参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mode` | `"direct"` | 执行模式：`direct`（直接调用 LLM API）、`vsp`（VSP 子进程）、`comt_vsp`（CoMT 双任务 + VSP） |
| `--provider` | `"openrouter"` | LLM 提供商：`openai`、`openrouter`、`self`（自部署，需配合 `--llm_base_url`） |
| `--model` | `"gpt-5"` | 模型名称 |
| `--temp` | `0.0` | Temperature |
| `--top_p` | `1.0` | Top-p sampling |
| `--max_tokens` | `2048` | 最大生成 token 数 |
| `--seed` | `None` | 随机种子 |
| `--proxy` | `None` | HTTP/HTTPS 代理地址 |

### `--mode` 与 `--provider` 的关系

- **`--mode direct`**：RedLens 直接调用 LLM API。`--provider` 选择端点（openai / openrouter），`--llm_base_url` 覆盖默认端点
- **`--mode vsp` / `--mode comt_vsp`**：通过 VSP 子进程执行。`--provider` 控制 VSP 内部使用的 LLM 端点
- **优先级**：`--llm_base_url`（最高）> `--provider`（默认端点）

## 数据加载

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--json_glob` | `~/code/MM-SafetyBench/data/processed_questions/*.json` | 数据集 JSON 文件 glob 路径 |
| `--image_base` | `~/Downloads/MM-SafetyBench_imgs/` | 图片根目录 |
| `--image_types` | `["SD"]` | 图片类型，可选：`SD`, `SD_TYPO`, `TYPO` |
| `--categories` | `None` | 指定类别（空格分隔），默认全部 |
| `--save_path` | 自动生成 | 结果保存路径 |

## 并发与任务控制

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--consumers` | `20` | 并发消费者数量（VSP 建议降至 3） |
| `--max_tasks` | `None` | 最大任务数（不设则处理全部） |
| `--rate_limit_qps` | `None` | 每秒请求数限制 |

## 采样

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--sampling_rate` | `1.0` | 数据采样比例（0-1） |
| `--sampling_seed` | `42` | 采样随机种子 |

## 评估

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--skip_eval` | `false` | 跳过评估步骤 |
| `--eval_model` | `"gpt-5-mini"` | 评估用模型 |
| `--eval_concurrency` | `20` | 评估并发数 |

## CoMT-VSP 专用

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--comt_data_path` | `None` | CoMT 数据集路径（缺失时从 HuggingFace 下载） |
| `--comt_sample_id` | `None` | CoMT 样本 ID，如 `"deletion-0107"` |

## VSP 后处理器

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--vsp_postproc` | `false` | 启用 VSP 后处理 |
| `--vsp_postproc_backend` | `"ask"` | 后处理后端：`ask`, `sd`, `prebaked` |
| `--vsp_postproc_method` | `None` | 后处理方法：`visual_mask`, `visual_edit`, `zoom_in`, `blur`, `good`, `bad` |
| `--vsp_postproc_fallback` | `"ask"` | 后备方案：`ask`, `sd` |

### Stable Diffusion 后处理参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--vsp_postproc_sd_model` | `"lucataco/sdxl-inpainting"` | SD inpainting 模型 |
| `--vsp_postproc_sd_prompt` | `"remove the objects, fill with natural background"` | SD 正向提示词 |
| `--vsp_postproc_sd_negative_prompt` | `"blurry, distorted, artifacts"` | SD 负向提示词 |
| `--vsp_postproc_sd_num_steps` | `50` | SD 推理步数 |
| `--vsp_postproc_sd_guidance_scale` | `7.5` | SD guidance scale |

## OpenRouter 提供商路由

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--openrouter_provider` | `None` | 指定 OpenRouter 底层提供商 slug（`--provider openrouter` 时有效，VSP 模式下传递给子进程） |

OpenRouter 同一模型可能有多个底层提供商（如 `together`, `parasail`, `novita`, `alibaba` 等），默认由 OpenRouter 自动路由。使用此参数可锁定到指定提供商，不允许 fallback。在 VSP/CoMT-VSP 模式下，此参数通过 `OPENROUTER_PROVIDER` 环境变量传递给 VSP 子进程。

## 自定义 LLM 端点

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--llm_base_url` | `None` | 自定义 LLM API 地址（所有模式通用，优先级最高） |
| `--llm_api_key` | `None` | 自定义 LLM API 密钥（自部署通常不需要） |

- **direct 模式**：`--llm_base_url` 覆盖 OpenAI/OpenRouter 的默认端点，使用 OpenAI-compatible ChatCompletion 客户端
- **vsp/comt_vsp 模式**：`--llm_base_url` 传递给 VSP 子进程的 `LLM_BASE_URL` 环境变量

## Profile 系统

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--profile` | `None` | 使用预定义的参数 profile（如 `comt_vsp`、`autodl_qwen`），CLI 参数覆盖 profile 值 |
| `--profile-file` | `"profiles.yaml"` | Profile 配置文件路径 |
| `--list-profiles` | `false` | 列出所有可用的 profile 并退出 |
| `--show-config` | `false` | 显示解析后的完整运行配置（JSON）并退出，不实际运行 |

所有参数支持连字符和下划线两种写法（如 `--show-config` 和 `--show_config`）。

详见 docs/profiles.md。

## Tunnel

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--tunnel` | `"ssh"` | Tunnel 模式：`ssh`（SSH port forwarding）、`cf`（Cloudflare Tunnel）、`none`（跳过） |
| `--no-ssh-tunnel` | `false` | [已废弃] 等价于 `--tunnel none` |

## 远程图片（provider=self 专用）

仅在 `--provider self` 时有意义。用于绕过 GFW 上传瓶颈：将图片放在 LLM 所在主机，LLM 直接读取本地 HTTP 服务，无需从本机跨国传输。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--remote_image_base_url` | `None` | AutoDL 上 MMSB 图片 HTTP 服务根 URL（如 `http://localhost:8001`）。设置后 LLM 通过 URL 读取图片，跳过本机 base64 上传。一次性准备：`rsync ~/Downloads/MM-SafetyBench_imgs/ seetacloud:/root/MM-SafetyBench_imgs/`，然后在 AutoDL 上运行 `python -m http.server 8001 --directory /root/MM-SafetyBench_imgs/` |
| `--remote_vsp_override_url` | `None` | （VSP 模式专用）AutoDL 上 override 图片 HTTP 服务根 URL（如 `http://localhost:8002`）。必须与 `--remote_vsp_override_ssh` 同时使用 |
| `--remote_vsp_override_ssh` | `None` | （VSP 模式专用）override 图片的 rsync 目标（如 `seetacloud:/root/vsp_override/`）。设置后每次 job 开始前自动将 `--vsp_override_images_dir` 内容 rsync 到 AutoDL |

**约束**：
- 以上三个参数仅对 `--provider self` 有效，其他 provider 下设置这些参数会报错退出
- `--remote_vsp_override_url` 和 `--remote_vsp_override_ssh` 必须同时设置或同时不设置
- `--remote_vsp_override_url` 需与 `--vsp_override_images_dir` 同时使用

**一次性准备（AutoDL）**：
```bash
# MMSB 图片服务（常驻）
python -m http.server 8001 --directory /root/MM-SafetyBench_imgs/

# Override 图片服务（常驻，内容由 RedLens 自动 rsync）
mkdir -p /root/vsp_override/
python -m http.server 8002 --directory /root/vsp_override/
```

**推荐做法**：将这三个参数打包到 `profiles.yaml` 的 self-provider profile 中，避免每次命令行重复填写。参见 `autodl_qwen` / `autodl_comt_vsp` profile。

## 常用命令示例

```bash
# 快速测试（默认: --mode direct --provider openrouter）
python request.py --max_tasks 10

# Direct + OpenAI
python request.py --provider openai --model "gpt-5" --max_tasks 50

# Direct + OpenRouter + 指定底层提供商
python request.py --model "qwen/qwen3-vl-8b-instruct" --openrouter_provider alibaba --max_tasks 50

# Direct + 自部署 LLM（llm_base_url 覆盖默认端点）
python request.py --llm_base_url "http://autodl:8000/v1" --model "Qwen3-VL-8B" --max_tasks 50

# VSP 模式（降低并发）
python request.py --mode vsp --consumers 3 --max_tasks 50

# VSP + OpenRouter + 固定上游提供商
python request.py --mode vsp --openrouter_provider alibaba --model "qwen/qwen3-vl-8b-instruct"

# VSP + 自部署 LLM
python request.py --mode vsp --llm_base_url "http://autodl:8000/v1" --model "Qwen3-VL-8B"

# CoMT-VSP 双任务模式
python request.py --mode comt_vsp --comt_sample_id "deletion-0107" --max_tasks 20

# CoMT-VSP + 后处理
python request.py --mode comt_vsp --comt_sample_id "deletion-0107" \
    --vsp_postproc --vsp_postproc_backend sd --max_tasks 20

# 采样 + 速率限制
python request.py --sampling_rate 0.5 --rate_limit_qps 10 --max_tasks 100
```
