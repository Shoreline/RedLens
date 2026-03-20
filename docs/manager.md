# RedLens Manager 使用指南

## 概述

`manager.py` 是 RedLens 的 Web 管理界面，提供统一的 Dashboard 用于浏览实验结果、启动任务和监控后台进程。

```bash
python manager.py              # 默认端口 8765
python manager.py --port 9000  # 自定义端口
```

启动后在浏览器访问 `http://localhost:8765`。

---

## 功能模块

### 1. 结果浏览（Items 标签页）

扫描 `output/` 和 `output_persisted/` 目录，展示所有实验结果：

- **Job**：单次推理任务，显示模型、攻击率、总任务数、失败数
- **Batch**：批量参数组合实验，显示包含的 job 数和整体攻击率
- **HS Comp**：Hidden States 对比分析结果
- **Refusal Dir**：Refusal Direction（Difference in Means）分析结果

**操作**：
- **Delete**：删除 job/batch 目录（批量 job 会一并清理空 batch）
- **Persist**：将 job 移动到 `output_persisted/`（保留重要实验结果，不被 cleanup 删除）
- **Retry**：对 job 重新执行 `job_fix.py`（修复失败任务）
- **Detail**：查看 job/batch 详情弹窗（eval 数据、run config、链接到 HTML 报告）

**筛选**：按类型、provider、模型、attac rate 范围过滤，支持按字段排序。

### 2. 配置对比（Config Compare 按钮）

多选若干 job/batch 后点击 Compare，展示各实验的 `run_config.json` 差异字段和相同字段，便于排查参数差异。

### 3. Launch Job 表单

从 UI 直接启动推理任务，无需命令行。表单参数：

| 区域 | 字段 | 说明 |
|------|------|------|
| 基础 | Profile | 从 `profiles.yaml` 选择预设参数组合 |
| 基础 | Mode | `direct` / `vsp` / `comt_vsp` |
| 基础 | Provider | `openrouter` / `openai` / `self` |
| 基础 | Model | 模型名称 |
| Sampling | Temperature / Top P / Max Tokens / Seed | 采样参数 |
| Data | Sampling Rate / Sampling Seed / Max Tasks | 数据参数 |
| Data | Image Types / Categories | 图片类型（空格分隔），类别过滤 |
| Network | Tunnel / Consumers | `ssh`/`cf`/`none`；并发数 |
| Network | LLM Base URL / OpenRouter Provider | 自定义端点；指定 OpenRouter 上游 |
| Self-Provider 远程图片 | Remote Image Base URL | AutoDL MMSB 图片 HTTP 服务根 URL（`provider=self` 专用） |
| Self-Provider 远程图片 | Remote VSP Override URL | AutoDL override 图片 HTTP 服务根 URL（VSP 模式 + `provider=self`） |
| Self-Provider 远程图片 | Remote VSP Override SSH | override 图片 rsync 目标，如 `seetacloud:/root/vsp_override/` |
| CoMT / Eval | CoMT Sample ID / Eval Model / Skip Eval | CoMT 模式参数 |

**Profile 优先级**：选了 Profile 后，表单中非空字段会覆盖 profile 值（与 CLI `--profile` 行为一致）。

### 4. Quick Launch

快速启动辅助分析任务，无需主推理任务：

- **Hidden States Comparison**：对比两个 job 的 hidden states（指定 job 编号、sub_task、turn）
- **Refusal Direction**：对多个 job/batch 的 hidden states 训练 Refusal Direction 分类器

### 5. 后台进程监控

页面右侧 Processes 面板实时显示所有通过 Manager 启动的后台进程：

- 状态：`running` / `done` / `error`
- 启动命令
- stdout/stderr 输出（可展开查看）
- Clear Finished：清除已完成的进程记录

---

## API 接口

Manager 提供 REST API，供外部脚本或自动化调用：

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/items` | 列出所有 job/batch/hs_comp/refusal_dir |
| POST | `/api/delete` | 删除指定路径 |
| POST | `/api/persist` | 将 job 移至 output_persisted/ |
| POST | `/api/retry` | 对 job 执行 job_fix.py |
| GET | `/api/profiles` | 返回 profiles.yaml 解析结果 |
| GET | `/api/job_detail?path=...` | 返回 job 详细信息（eval, config） |
| GET | `/api/batch_detail?path=...` | 返回 batch 详细信息 |
| POST | `/api/compare_configs` | 对比多个 job/batch 的 config |
| POST | `/api/launch_job` | 启动 request.py 推理任务 |
| POST | `/api/launch_batch` | 启动 batch_request.py |
| POST | `/api/launch_hs_comp` | 启动 compare_hidden_states.py |
| POST | `/api/launch_refusal_dir` | 启动 refusal_direction.py |
| GET | `/api/process_status` | 查询后台进程状态 |
| POST | `/api/process_clear` | 清除已完成进程记录 |
| GET | `/output/{path}` | 静态文件服务（output/ 目录） |
| GET | `/output_persisted/{path}` | 静态文件服务（output_persisted/ 目录） |

### launch_job 请求体示例

```json
{
  "profile": "autodl_qwen",
  "overrides": {
    "max_tasks": 50,
    "consumers": 5,
    "remote_image_base_url": "http://localhost:8001"
  }
}
```

`overrides` 中的字段直接映射为 `request.py` 的 CLI 参数，会覆盖 profile 值。

---

## 注意事项

- 所有文件操作（delete/persist）限制在 `output/` 和 `output_persisted/` 目录内，防止路径穿越
- 后台进程使用 `asyncio.create_subprocess_exec` 启动，stdout/stderr 实时缓冲
- `output_persisted/` 内的 job 不会被 `cleanup_output.py` 删除
