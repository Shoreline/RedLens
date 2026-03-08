# Batch Request 模型选取指南

## 模型选取标准

1. **视觉能力**: 模型必须支持图像输入（VL / multimodal），用于 MM-SafetyBench 多模态安全评估
2. **同系列多参数规模**: 优先选取同一系列的不同参数量变体（如 Qwen3-VL 8B / 32B / 235B），便于分析参数量对安全性能的影响
3. **最新版本优先**: 同一模型有多个版本时选最新的
4. **多厂商覆盖**: 跨厂商选取，避免单一来源偏差
5. **Instruct 变体**: 选 instruct（非 thinking），与安全评估场景匹配
6. **同系列同 provider**: 同一模型系列使用相同的 `--openrouter_provider`，控制上游推理环境一致

## 运行模式

每个模型运行两次：

| 模式 | 说明 | 关键参数 |
|------|------|----------|
| `direct` | 直接向 LLM 发送带图像的 prompt | `--mode direct` |
| `comt_vsp` | CoMT 目标检测 + VSP 安全评估（无后处理） | `--mode comt_vsp --comt_sample_id "deletion-0107"` |

固定参数：`--sampling_rate 0.12 --sampling_seed 42 --tunnel cf`

## 示例：跨模型对比 batch（2026-03）

8 个模型 × 2 模式 = 16 个 job

### 选取的模型

| 系列 | Model ID | 参数量 | openrouter_provider |
|------|----------|--------|---------------------|
| Qwen3-VL | `qwen/qwen3-vl-8b-instruct` | 8B | alibaba |
| Qwen3-VL | `qwen/qwen3-vl-32b-instruct` | 32B (dense) | alibaba |
| Qwen3-VL | `qwen/qwen3-vl-235b-a22b-instruct` | 235B (MoE, 22B active) | alibaba |
| Gemma 3 | `google/gemma-3-12b-it` | 12B | deepinfra |
| Gemma 3 | `google/gemma-3-27b-it` | 27B | deepinfra |
| Llama 4 | `meta-llama/llama-4-scout` | 109B (17B active, 16E) | deepinfra |
| Llama 4 | `meta-llama/llama-4-maverick` | 400B (17B active, 128E) | deepinfra |
| Pixtral | `mistralai/pixtral-large-2411` | 124B | mistral |

> **注意**: `mistralai/pixtral-12b` 已无活跃 endpoint（2026-03 验证），已移除。
> Google 不在 OpenRouter 上提供 Gemma 开源模型，Together 不再提供 Llama 4，均改为 deepinfra。

### args_combo 配置

```python
args_combo = [
    "--tunnel cf --sampling_rate 0.12 --sampling_seed 42",
    [
        # ── Qwen3-VL 系列 (openrouter_provider: alibaba) ──
        '--mode direct --model "qwen/qwen3-vl-8b-instruct" --openrouter_provider alibaba',
        '--mode comt_vsp --model "qwen/qwen3-vl-8b-instruct" --openrouter_provider alibaba --comt_sample_id "deletion-0107"',
        '--mode direct --model "qwen/qwen3-vl-32b-instruct" --openrouter_provider alibaba',
        '--mode comt_vsp --model "qwen/qwen3-vl-32b-instruct" --openrouter_provider alibaba --comt_sample_id "deletion-0107"',
        '--mode direct --model "qwen/qwen3-vl-235b-a22b-instruct" --openrouter_provider alibaba',
        '--mode comt_vsp --model "qwen/qwen3-vl-235b-a22b-instruct" --openrouter_provider alibaba --comt_sample_id "deletion-0107"',

        # ── Gemma 3 系列 (openrouter_provider: deepinfra) ──
        '--mode direct --model "google/gemma-3-12b-it" --openrouter_provider deepinfra',
        '--mode comt_vsp --model "google/gemma-3-12b-it" --openrouter_provider deepinfra --comt_sample_id "deletion-0107"',
        '--mode direct --model "google/gemma-3-27b-it" --openrouter_provider deepinfra',
        '--mode comt_vsp --model "google/gemma-3-27b-it" --openrouter_provider deepinfra --comt_sample_id "deletion-0107"',

        # ── Llama 4 系列 (openrouter_provider: deepinfra) ──
        '--mode direct --model "meta-llama/llama-4-scout" --openrouter_provider deepinfra',
        '--mode comt_vsp --model "meta-llama/llama-4-scout" --openrouter_provider deepinfra --comt_sample_id "deletion-0107"',
        '--mode direct --model "meta-llama/llama-4-maverick" --openrouter_provider deepinfra',
        '--mode comt_vsp --model "meta-llama/llama-4-maverick" --openrouter_provider deepinfra --comt_sample_id "deletion-0107"',

        # ── Pixtral / Mistral 系列 (openrouter_provider: mistral) ──
        '--mode direct --model "mistralai/pixtral-large-2411" --openrouter_provider mistral',
        '--mode comt_vsp --model "mistralai/pixtral-large-2411" --openrouter_provider mistral --comt_sample_id "deletion-0107"',
    ],
]
```
