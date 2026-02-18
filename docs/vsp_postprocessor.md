# VSP 后处理器指南

VSP（VisualSketchpad）后处理器在 VSP 推理完成后，对图片进行二次处理以增强实验效果。

## 概述

后处理流程：VSP 推理 → 提取工具使用 → 后处理图片 → 二次推理

## 后端类型

### `ask` 后端（默认）
直接使用 LLM 对 VSP 输出进行二次判断，不修改图片。

### `sd` 后端（Stable Diffusion）
使用 SD inpainting 模型修改图片，然后重新推理：
- 模型默认：`lucataco/sdxl-inpainting`
- 支持自定义 prompt、步数、guidance scale

### `prebaked` 后端
使用预先生成好的处理结果，适合复现实验。
- 数据存储在 `prebaked_report_data.json`

## 后处理方法

| 方法 | 说明 |
|------|------|
| `visual_mask` | 视觉遮罩 |
| `visual_edit` | 视觉编辑 |
| `zoom_in` | 放大关键区域 |
| `blur` | 模糊处理 |
| `good` | SD 后端正面生成 |
| `bad` | SD 后端负面生成 |

## 相关文件

| 文件 | 说明 |
|------|------|
| `copy_sd_pictures.py` | 将 SD 图片复制到 VSP prebaked_images 目录 |
| `check_vsp_tool_usage.py` | 分析 VSP 工具使用统计 |
| `test_sd_postproc.sh` | SD 后处理集成测试脚本 |

## 详细参数

参见 [CLI 参考 - VSP 后处理器](cli_reference.md#vsp-后处理器)。

## 已有文档

项目根目录下还有以下相关文档（历史遗留，内容可能较旧）：
- `VSP_SD_POSTPROCESSOR_GUIDE.md` — SD 后处理器完整指南
- `VSP_POSTPROCESSOR_QUICKSTART.md` — 快速开始
- `VSP_POSTPROCESSOR_USAGE.md` — 使用详情
- `VSP_POSTPROCESSOR_INTEGRATION_SUMMARY.md` — 集成总结
- `VSP_POSTPROC_SUMMARY.md` — 功能概要
- `COMT_GUIDE.md` — CoMT 模式完整指南
