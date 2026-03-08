# VSP 后处理器（Post-Processor）

VSP 推理完成后可启用后处理，对检测到的目标区域进行图片修改（遮罩、inpainting、缩放等），然后将修改后的图片送回 LLM 进行二次推理。后处理对 LLM 完全透明——模型不知道图片经过了修改。

## 快速开始

```bash
# 视觉遮罩（黑色矩形覆盖检测区域）
python request.py --mode vsp --max_tasks 10 --vsp_postproc --vsp_postproc_method visual_mask

# OpenCV inpainting（移除检测到的目标）
python request.py --mode vsp --max_tasks 10 --vsp_postproc --vsp_postproc_method visual_edit

# Stable Diffusion inpainting（高质量移除）
python request.py --mode vsp --max_tasks 5 --vsp_postproc --vsp_postproc_backend sd

# 使用预生成结果（复现实验）
python request.py --mode comt_vsp --comt_sample_id "deletion-0107" --max_tasks 10 \
    --vsp_postproc --vsp_postproc_backend prebaked --vsp_postproc_method visual_mask

# 不启用后处理（默认）
python request.py --mode vsp --max_tasks 10
```

## 后端与方法

### 三种后端

| 后端 | 说明 | 速度 | 成本 |
|------|------|------|------|
| **`ask`**（默认） | 本地图像处理（PIL/OpenCV） | 极快 | 免费 |
| **`sd`** | Stable Diffusion inpainting（Replicate API） | 慢（10-30s/张） | ~$0.05/张 |
| **`prebaked`** | 使用预先生成的处理结果 | 极快 | 免费 |

### 后处理方法

| 方法 | 适用后端 | 效果 |
|------|----------|------|
| `visual_mask` | ask / prebaked | 在检测区域绘制黑色矩形 |
| `visual_edit` | ask / prebaked | OpenCV inpainting，自动填充背景 |
| `zoom_in` | ask / prebaked | 裁剪并放大首个检测区域 |
| `blur` | ask / prebaked | 对检测区域进行模糊处理 |
| `good` | sd / prebaked | SD 正面生成（自然背景填充） |
| `bad` | sd / prebaked | SD 负面生成 |

## 命令行参数

### 基础参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--vsp_postproc` | `false` | 启用后处理（flag） |
| `--vsp_postproc_backend` | `"ask"` | 后端类型：`ask`, `sd`, `prebaked` |
| `--vsp_postproc_method` | `None` | 后处理方法（见上表） |
| `--vsp_postproc_fallback` | `"ask"` | `prebaked` 缓存未命中时的回退后端：`ask` 或 `sd` |

### Stable Diffusion 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--vsp_postproc_sd_model` | `"lucataco/sdxl-inpainting"` | Replicate 上的 SD 模型 |
| `--vsp_postproc_sd_prompt` | `"remove the objects, fill with natural background"` | 正向提示词 |
| `--vsp_postproc_sd_negative_prompt` | `"blurry, distorted, artifacts"` | 负向提示词 |
| `--vsp_postproc_sd_num_steps` | `50` | 推理步数（越高质量越好，越慢） |
| `--vsp_postproc_sd_guidance_scale` | `7.5` | Guidance scale（1-20） |

## 适用模式

- `--mode vsp` — 支持
- `--mode comt_vsp` — 支持
- `--mode direct` — 不支持（参数会被忽略）

## 工作原理

### 数据流

```
request.py (CLI 参数)
    ↓
RunConfig (vsp_postproc_* 字段)
    ↓
VSPProvider._call_vsp() → 设置环境变量
    ↓
VSP subprocess (config.py 读取环境变量 → POST_PROCESSOR_CONFIG)
    ↓
VSP vision tools (detection / segment_and_mark)
    ↓  调用 apply_postprocess()
修改后的图片 → LLM 二次推理
```

### 环境变量传递

Mediator 通过环境变量将配置传递给 VSP 子进程：

| 环境变量 | 对应参数 |
|----------|----------|
| `VSP_POSTPROC_ENABLED` | `--vsp_postproc`（"1" / "0"） |
| `VSP_POSTPROC_BACKEND` | `--vsp_postproc_backend` |
| `VSP_POSTPROC_METHOD` | `--vsp_postproc_method` |
| `VSP_POSTPROC_FALLBACK` | `--vsp_postproc_fallback` |
| `VSP_POSTPROC_SD_MODEL` | `--vsp_postproc_sd_model` |
| `VSP_POSTPROC_SD_PROMPT` | `--vsp_postproc_sd_prompt` |
| `VSP_POSTPROC_SD_NEGATIVE_PROMPT` | `--vsp_postproc_sd_negative_prompt` |
| `VSP_POSTPROC_SD_NUM_STEPS` | `--vsp_postproc_sd_num_steps` |
| `VSP_POSTPROC_SD_GUIDANCE_SCALE` | `--vsp_postproc_sd_guidance_scale` |

## 输出文件

启用后处理时，VSP 会自动保存处理前后的图片：

```
output/job_XXX/details/vsp_*/category/task_id/output/input/
├── image_0.jpg                            # 原始输入图片
├── before_postproc_detection_*.png        # 处理前：VSP 标注（bbox、label）
└── <hash>.png                             # 处理后：LLM 实际看到的图片
```

- **before** 图片用于验证 VSP 检测是否正确
- **after** 图片是 LLM 真正看到的内容

## 使用示例

### ASK 后端

```bash
# 遮罩：验证检测精度
python request.py --mode vsp --max_tasks 10 \
    --vsp_postproc --vsp_postproc_method visual_mask

# Inpainting：移除有害目标
python request.py --mode vsp --categories "01-Illegal_Activity" --max_tasks 20 \
    --vsp_postproc --vsp_postproc_method visual_edit

# 缩放：聚焦检测区域
python request.py --mode vsp --max_tasks 10 \
    --vsp_postproc --vsp_postproc_method zoom_in
```

### SD 后端

需要在环境中设置 `REPLICATE_API_TOKEN`。

```bash
# 基础用法
python request.py --mode comt_vsp --comt_sample_id "deletion-0107" --max_tasks 1 \
    --vsp_postproc --vsp_postproc_backend sd --skip_eval

# 自定义提示词和参数
python request.py --mode vsp --max_tasks 5 \
    --vsp_postproc --vsp_postproc_backend sd \
    --vsp_postproc_sd_prompt "remove detected objects, natural lighting" \
    --vsp_postproc_sd_num_steps 30 \
    --vsp_postproc_sd_guidance_scale 9.0
```

### Prebaked 后端

```bash
# 使用预生成结果，缓存未命中时回退到 ask
python request.py --mode comt_vsp --comt_sample_id "deletion-0107" --max_tasks 10 \
    --vsp_postproc --vsp_postproc_backend prebaked \
    --vsp_postproc_method visual_mask --vsp_postproc_fallback ask
```

### 对比实验

```bash
# 无后处理（基线）
python request.py --mode vsp --max_tasks 10

# ASK visual_mask
python request.py --mode vsp --max_tasks 10 --vsp_postproc --vsp_postproc_method visual_mask

# SD inpainting
python request.py --mode vsp --max_tasks 10 --vsp_postproc --vsp_postproc_backend sd
```

对比各 job 的 `eval.csv` 即可分析后处理对攻击率的影响。

## 调试

### 查看 VSP 日志

```bash
grep "POST_PROCESSOR" output/job_*/details/vsp_*/*/*/output/vsp_debug.log
```

正常输出示例：
```
[POST_PROCESSOR] ASK:visual_mask
[POST_PROCESSOR] SD:inpaint via Replicate
```

### 常见问题

**后处理没有生效：**
1. 确认使用了 `--mode vsp` 或 `--mode comt_vsp`
2. 确认添加了 `--vsp_postproc` flag
3. 检查 VSP 是否调用了 vision tools（后处理只在 detection/segment_and_mark 时触发）

**SD 报错 REPLICATE_API_TOKEN：**
```bash
export REPLICATE_API_TOKEN="your_token_here"
```

**SD 报错 429 Rate Limit：**
账户余额低于 $5 时有频率限制。代码已内置自动重试，或者充值 $5+ 解除限制。

**SD 质量不佳：**
- 调整 `--vsp_postproc_sd_prompt` 提供更具体的指令
- 增加 `--vsp_postproc_sd_num_steps`（如 50）
- 调整 `--vsp_postproc_sd_guidance_scale`（尝试 8-10）

## VSP Tool Override（图片替换）

独立于后处理的简化机制：直接替换 VSP vision tool 返回的图片，跳过远程 tool 调用。

### 参数

| 参数 | 说明 |
|------|------|
| `--vsp_override_images_dir` | 预备图片目录。启用后 vision tool 跳过远程调用，返回该目录中的图片 |

### 图片目录结构

支持三级粒度，按优先级查找（first match wins）：

```
override_images/
├── default.png                      # 全局兜底
├── 01-Illegal_Activity.png          # 类别级
├── 03-Physical_Harm/                # task 级
│   ├── default.png                  # 该类别兜底
│   ├── 0.png                        # task index 0
│   └── 1.png
└── ...
```

查找顺序：`{dir}/{category}/{index}.png` → `{dir}/{category}/default.png` → `{dir}/{category}.png` → `{dir}/default.png`

### 使用示例

```bash
# 所有 task 使用同一张全黑图
python request.py --mode comt_vsp --vsp_override_images_dir ~/data/override_black/ \
  --comt_sample_id "deletion-0107" --max_tasks 10

# 按类别使用不同图片
python request.py --mode comt_vsp --vsp_override_images_dir ~/data/override_by_category/ \
  --comt_sample_id "deletion-0107" --max_tasks 10
```

### 与后处理的关系

Override 在 tool 调用最前端拦截，跳过远程调用、tool cache 和 post-processing。`--vsp_postproc` 在 override 启用时会被忽略。

### 输出

启用 override 时，job 目录新增 `override_images/` 子目录，包含所有使用的 override 图片副本和 `override_info.json`。Batch 报告中显示 override 图片缩略图。

## 相关文件

| 文件 | 说明 |
|------|------|
| `request.py` | CLI 参数定义、RunConfig 字段 |
| `provider.py` | `VSPProvider._call_vsp()` 传递环境变量 |
| `copy_sd_pictures.py` | 将 SD 生成图片复制到 prebaked_images 目录 |
| `check_vsp_tool_usage.py` | 分析 VSP 工具使用统计 |
| `batch_request.py` | 批量运行时在 HTML 报告中展示 postproc/override 配置 |
| `test_sd_postproc.sh` | SD 后处理集成测试脚本 |
