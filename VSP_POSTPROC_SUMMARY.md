# VSP Postproc 信息显示功能

## 概述

在 `batch_request.py` 中添加了 VSP postproc 相关信息的显示功能，现在 batch summary HTML 会展示每个 job 的 VSP postproc 配置。

## 修改内容

### 1. RunResult 数据类扩展

添加了以下字段来存储 VSP postproc 相关信息：
- `vsp_postproc`: 是否启用 vsp_postproc (bool)
- `vsp_postproc_backend`: 后处理 backend (prebaked/sd)
- `vsp_postproc_fallback`: fallback 方法 (ask/sd)
- `vsp_postproc_method`: 后处理方法 (visual_mask/good/bad)
- `vsp_postproc_sd_prompt`: Stable Diffusion prompt
- `comt_sample_id`: COMT sample ID

### 2. 参数解析功能增强

`parse_args_str()` 函数现在能够从命令行参数中提取以下信息：
- `--vsp_postproc` 标志
- `--vsp_postproc_backend`
- `--vsp_postproc_fallback`
- `--vsp_postproc_method`
- `--vsp_postproc_sd_prompt` (支持带引号的字符串)
- `--comt_sample_id`

### 3. HTML 报告更新

在 `generate_batch_summary_html()` 中：
- 添加了 "VSP Postproc" 列
- 为每个 job 显示完整的 postproc 配置信息
- 使用彩色标签高亮显示 method (visual_mask/good/bad)
- 对于没有启用 postproc 的 job，显示 "-"

### 4. CSS 样式

新增样式：
- `.vsp-postproc-cell`: postproc 列的样式（小字体，灰色）
- `.vsp-method`: method 标签的样式（紫色背景）

## 显示效果

### 无 VSP Postproc 的 Job
```
VSP Postproc
-
```

### 带 VSP Postproc 的 Job
```
VSP Postproc
[visual_mask]           ← 紫色标签
backend: prebaked
fallback: ask
id: deletion-0107
```

### 带 SD prompt 的 Job
```
VSP Postproc
[good]                  ← 紫色标签
backend: prebaked
fallback: sd
id: deletion-0107
prompt: "remove the boxed objects"
```

## 测试

参考 `batch_summary_example.html` 查看完整的显示效果示例。

## 兼容性

所有新增字段都是 Optional，因此不会影响现有代码的运行。对于没有 vsp_postproc 参数的 job，这些字段将保持为 None。
