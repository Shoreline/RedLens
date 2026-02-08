# Job Summary 按 Category 整合显示功能 ✅

## 功能说明

在每个 job 的 `summary.html` 中，将 **Statistics by Category** 和 **Detailed Entries** 整合在一起，按 category 分组显示。

### 新的布局结构

每个 category 现在是一个独立的 section，包含：

1. **Category Header** (标题栏，带蓝色背景)
   - Category 名称
   - 统计信息: Total Calls, Cache Hits, Hit Rate, Attack Rate

2. **Category Entries** (详细条目)
   - 该 category 下的所有 prebaked 图片
   - 每张图片显示: 缩略图、CoMT ID、Backend、Path

## 修改内容

### 文件：`request.py`

#### 1. 重构 prebaked_html 生成逻辑

**之前**: 分为两部分
- Statistics by Category 表格（统计信息）
- Detailed Entries（所有图片混在一起）

**现在**: 按 category 组织
- 每个 category 一个独立 section
- 统计信息显示在 category header
- 该 category 的所有 entries 显示在下方

#### 2. 代码改进

- 按 category 分组所有 prebaked_data
- 对同一张图片进行去重（记录使用次数）
- 为每个 category 生成完整的 HTML section
- 从 eval_metrics 获取 attack rate

#### 3. 新增 CSS 样式

```css
.category-sections    /* 所有 category sections 的容器 */
.category-section     /* 单个 category section */
.category-header      /* Category 标题栏（蓝色背景）*/
.category-stats-inline /* 统计信息的行内显示 */
.category-entries     /* 该 category 的所有 entries */
```

## 视觉效果示例

### Category Section 示例（Job 159）

```
┌─────────────────────────────────────────────────────────────┐
│  01-Illegal_Activitiy                                       │  ← 蓝色标题栏
│  Total: 10  Hits: 10  Hit Rate: 100.0%  Attack Rate: 0.0%  │
├─────────────────────────────────────────────────────────────┤
│  [Cache HIT] × 10  detection                                │
│  [图片缩略图]  CoMT ID: deletion-0107                       │
│                Backend: ask:visual_mask                     │
│                Path: ~/code/VisualSketchpad/...             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  02-HateSpeech                                              │  ← 蓝色标题栏
│  Total: 20  Hits: 20  Hit Rate: 100.0%  Attack Rate: 5.0%  │
├─────────────────────────────────────────────────────────────┤
│  [Cache HIT] × 20  detection                                │
│  [图片缩略图]  CoMT ID: deletion-0107                       │
│                Backend: ask:visual_mask                     │
│                Path: ~/code/VisualSketchpad/...             │
└─────────────────────────────────────────────────────────────┘

... (13 个 categories)
```

### 优势对比

**之前的布局** ❌
- 统计信息和详细信息分离
- 需要上下滚动对照
- 不容易找到特定 category 的图片

**新的布局** ✅
- 统计信息和详细图片整合在一起
- 按 category 清晰分组
- 一目了然，易于导航和对比

## 使用说明

### 新的 batch 运行

从现在开始，所有新的 batch 运行会自动在 summary.html 中使用新的整合布局。

### 查看示例

```bash
# 在浏览器中打开（推荐使用 Chrome/Firefox）
open output/batch_5_0208_005316/job_159_tasks_202_ComtVsp_qwen_qwen3-vl-235b-a22b-instruct_0208_005608/summary.html
```

### 视觉特点

- **Category Sections**: 每个 category 独立显示，带蓝色标题栏
- **统计信息**: 在标题栏内水平排列，清晰易读
- **图片缩略图**: 嵌入式显示，200x200 缩略图
- **详细信息**: CoMT ID、Backend、Path 信息紧邻图片
- **暗色主题**: 与整体 summary 风格一致

## 注意事项

1. **只有使用 prebaked processor 的 job 才会有这个表格**
   - openrouter 等 provider 不使用 prebaked，不会显示此表
   
2. **Attack Rate 来自 eval.csv**
   - 如果某个 category 在 eval.csv 中没有数据，会显示 "N/A"
   
3. **表格位置**
   - 位于 "Prebaked Processor Report" 部分
   - 在总体统计卡片之后
   - 在详细 entries 之前

## 已验证

✅ Job 159 (visual_mask) - 表格正常显示
✅ Job 160 (sd-good) - 表格正常显示  
✅ Job 161 (sd-bad) - 表格正常显示

所有 comt_vsp provider 的 job 都已成功添加此功能。
