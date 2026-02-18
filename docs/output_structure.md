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
├── mediator_metadata.json      # VSP 执行元数据（VSP/CoMT-VSP 运行时生成）
├── prebaked_report_data.json   # prebaked processor 数据（启用时生成）
└── details/                    # VSP/CoMT-VSP 详细输出（仅 VSP 类 provider）
    └── vsp_{YYYY-MM-DD_HH-MM-SS}/
        └── {category}/
            └── {index}/
                ├── input/
                │   ├── request.json
                │   └── image_*.jpg
                └── output/
                    ├── vsp_debug.log
                    └── output.json
```

### JSONL 字段说明

每行 JSON 包含：
- `pred` — 模型原始回答
- `origin` — 输入元数据（类别、问题文本、图片路径等）
- `eval_result` — 评估结果（`"safe"` / `"unsafe"`，评估后填充）
- `meta` — 运行元数据（模型名、参数、时间戳等）

## 批量运行（Batch）

命名格式：`batch_{num}_{MMDD_HHMMSS}/`

- `{num}` — 批次编号，由 `output/.batch_counter` 维护

### Batch 目录内容

```
batch_7_0208_104209/
├── batch.log                   # 批次运行完整日志
├── batch_summary.html          # 批次汇总报告
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

## 全局文件

- `output/.task_counter` — Job 编号计数器（纯数字文本）
- `output/.batch_counter` — Batch 编号计数器（纯数字文本）
- `output/reports/` — generate_report_with_charts.py 生成的独立报表
