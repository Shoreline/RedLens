# Output 目录结构详解

所有运行输出都存放在 `output/` 目录下。

## 单次运行（Job）

命名格式：`job_{num}_tasks_{total}_{Provider}_{model}_{MMDD_HHMMSS}/`

- `{num}` — 全局递增编号，由 `output/.task_counter` 维护
- `{total}` — 本次运行的任务总数
- `{Provider}` — Provider 类名（如 `ComtVsp`、`Openrouter`）
- `{model}` — 模型名（如 `Qwen3-VL-8B-Instruct`）
- `{MMDD_HHMMSS}` — 启动时间

### Job 目录内容

```
job_171_tasks_1_ComtVsp_Qwen3-VL-8B-Instruct_0216_192859/
├── results.jsonl               # 推理结果（每行一个样本）
├── eval.csv                    # 评估指标（按类别统计，仅在评估后生成）
├── console.log                 # 控制台完整日志
├── summary.html                # 可视化汇总报告
├── prebaked_report_data.json   # prebaked processor 数据（启用时生成）
├── hidden_states/              # Hidden states（仅自定义 LLM 端点且模型返回时生成）
│   ├── meta.json               # 全局元信息 {layer, hidden_dim, dtype, model}
│   ├── {cat}_{index}_q0_t0.npy  # 子任务0 第0轮 last-token hidden state, shape: (hidden_dim,)
│   ├── {cat}_{index}_q0_t1.npy  # 子任务0 第1轮（cat=category编号如08, index=问题编号）
│   ├── {cat}_{index}_q1_t0.npy  # 子任务1 第0轮（THOUGHT编号重置时自动分割子任务）
│   ├── {cat}_{index}_turns.json # 轮次元数据（question + turn 编号 + 对话内容摘要）
│   └── ...
└── details/                    # VSP/CoMT-VSP 详细输出（仅 VSP 类 provider）
    └── vsp_{YYYY-MM-DD_HH-MM-SS}/
        └── {category}/
            └── {index}/
                ├── input/
                │   ├── request.json            # VSP 任务输入（含 query + follow_up_queries）
                │   └── image_*.jpg
                ├── output/
                │   ├── vsp_debug.log
                │   ├── input/          # VSP 工作目录（含 output.json 等）
                │   │   ├── output.json
                │   │   ├── usage_summary.json
                │   │   └── hidden_states.json  # 中间格式（仅自定义端点）
                │   │   └── ...
                │   └── ...
                └── mediator_metadata.json  # VSP 执行元数据（精简版）
```

### JSONL 字段说明

每行 JSON 包含：
- `pred` — 模型原始回答
- `origin` — 输入元数据（类别、问题文本、图片路径等）
- `eval_result` — 评估结果（`"safe"` / `"unsafe"`，评估后填充）
- `meta` — 运行元数据（模型名、参数、时间戳等）

### Hidden States（自定义 LLM 端点）

当使用 `--llm_base_url` 指向自部署模型（如 Qwen）且服务端返回 `hidden_state` 字段时，**所有模式（direct / vsp / comt_vsp）** 均会自动捕获并保存：

- **Direct 模式**：从 API 响应的 `model_extra`（pydantic extra fields）中提取 `hidden_state`，每个 task 保存一个 `.npy` 文件
- **VSP / CoMT-VSP 模式**：VSP 子进程将 hidden states 写入 `hidden_states.json`，Mediator 读取后按轮次保存为多个 `.npy` 文件

每个 task 的每轮 LLM 调用生成一个 `.npy` 文件，命名为 `{cat}_{index}_q{question}_t{turn}.npy`：
- `{cat}` — 类别编号（如 `08`）
- `{index}` — MM-SafetyBench 样本编号
- `{question}` — 子任务编号（direct 模式固定为 0）
- `{turn}` — 该子任务中的第几轮 LLM 调用（从 0 开始）

`{index}_turns.json` 记录轮次与对话的对应关系：
```json
[
    {"turn": 0, "content_preview": "THOUGHT 0: I need to analyze..."},
    {"turn": 1, "content_preview": "THOUGHT 1: The detection results..."},
    {"turn": 2, "content_preview": "ANSWER: I cannot assist..."}
]
```

`meta.json` 记录全局信息（所有 task 共享）：
```json
{"layer": -1, "hidden_dim": 4096, "dtype": "float32", "model": "Qwen3-VL-8B-Instruct"}
```

加载示例：
```python
import numpy as np, json
from pathlib import Path

hs_dir = Path("output/job_.../hidden_states")

# 加载某个 task 的所有轮次
turns = json.loads((hs_dir / "17_turns.json").read_text())
vectors = [np.load(hs_dir / f"17_t{t['turn']}.npy") for t in turns]

# 加载所有 task 的最终轮 hidden state
final_hs = {}
for f in hs_dir.glob("*_turns.json"):
    tid = f.stem.replace("_turns", "")
    meta = json.loads(f.read_text())
    final_hs[tid] = np.load(hs_dir / f"{tid}_t{meta[-1]['turn']}.npy")
```

### mediator_metadata.json（VSP/CoMT-VSP）

精简版元数据，位于每个 task 的 `details/` 子目录下：
- `extracted_answer` — 从 VSP debug log 提取的最终答案文本
- `task_data` — 精简的任务数据（query 文本、图片数量、CoMT 任务信息）
- `timestamp` — 执行时间戳

## 批量运行（Batch）

命名格式：`batch_{num}_{MMDD_HHMMSS}/`

- `{num}` — 批次编号，由 `output/.batch_counter` 维护

### Batch 目录内容

```
batch_7_0208_104209/
├── batch.log                   # 批次运行完整日志
├── batch_summary.html          # 批次汇总报告（含配置对比表）
├── batch_state.json            # 运行状态（用于断点续传 --resume）
├── run_configs.json            # 各 job 运行配置汇总（便于对比差异）
├── job_164_tasks_202_.../      # 各个子任务的 job 目录
├── job_165_tasks_202_.../
├── ...
└── report/                     # 跨任务对比报表
    ├── evaluation_report.html  # HTML 评估报告
    ├── chart_1_*.png           # 按模型/方法的汇总图表
    ├── chart_1_*_overall.png   # 总体攻击率图表
    ├── chart_category_*.png    # 按类别的攻击率图表
    ├── chart_global_*.png      # 全局对比图表
    └── ...
```

### run_configs.json

汇总各 job 的 `run_config.json`，以数组形式存放（按运行顺序排列）。每个元素包含该 job 的完整运行参数（profile、mode、provider、model、sampling_rate 等），以及附加的元信息字段：

- `_job_folder` — job 文件夹名
- `_job_num` — job 编号
- `_status` — 运行状态（`"success"` / `"failed"`）

`batch_summary.html` 中的 **Run Configs Comparison** 区域自动对比所有 job 的配置：
- 仅展示有差异的参数行（颜色区分不同值）
- 相同参数折叠在可展开的 `<details>` 中

## 全局文件

- `output/.task_counter` — Job 编号计数器（纯数字文本）
- `output/.batch_counter` — Batch 编号计数器（纯数字文本）
- `output/reports/` — generate_report_with_charts.py 生成的独立报表
