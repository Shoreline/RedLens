# request.py 命令行参数参考

## 基础参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--provider` | `"openai"` | Provider 类型：`openai`, `openrouter`, `qwen`, `vsp`, `comt_vsp` |
| `--model` | `"gpt-5"` | 模型名称 |
| `--temp` | `0.0` | Temperature |
| `--top_p` | `1.0` | Top-p sampling |
| `--max_tokens` | `2048` | 最大生成 token 数 |
| `--seed` | `None` | 随机种子 |
| `--proxy` | `None` | HTTP/HTTPS 代理地址 |

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

## 自定义 LLM 端点

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--llm_base_url` | `None` | 自定义 LLM API 地址 |
| `--llm_api_key` | `None` | 自定义 LLM API 密钥 |

## 常用命令示例

```bash
# 快速测试
python request.py --max_tasks 10

# 指定模型和 provider
python request.py --provider openrouter --model "qwen/qwen3-vl-235b-a22b-instruct" --max_tasks 50

# VSP（降低并发）
python request.py --provider vsp --consumers 3 --max_tasks 50

# CoMT-VSP + 后处理
python request.py --provider comt_vsp --comt_sample_id "deletion-0107" \
    --vsp_postproc --vsp_postproc_backend sd --max_tasks 20

# 采样 + 速率限制
python request.py --sampling_rate 0.5 --rate_limit_qps 10 --max_tasks 100

# 自定义 LLM 端点
python request.py --llm_base_url "http://localhost:8000/v1" --llm_api_key "sk-xxx"
```
