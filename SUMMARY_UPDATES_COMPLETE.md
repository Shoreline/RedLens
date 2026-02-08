# Job Summary 更新完成 ✅

## 功能实现

在每个 job 的 `summary.html` 中添加了按 category 汇总的统计表格。

### 新增内容

每个使用 prebaked processor 的 job summary 现在包含：

#### "Statistics by Category" 表格

显示 13 个安全类别的详细统计：

| 列名 | 说明 |
|------|------|
| **Category** | 类别名称（如 01-Illegal_Activitiy） |
| **Total Calls** | 该类别的 prebaked processor 调用次数 |
| **Cache Hits** | 缓存命中次数 |
| **Cache Hit Rate** | 缓存命中率百分比 |
| **Attack Rate** | 攻击率百分比（从 eval.csv 读取） |

### 实现位置

表格插入在 "Prebaked Processor Report" 部分：
1. 总体统计卡片（Total Calls, Cache Hits, Generated, Hit Rate）
2. **🆕 Statistics by Category 表格**
3. Detailed Entries（按图片显示的详细信息）

## 代码修改

### `request.py`

**修改点 1**: 添加 category 统计逻辑
```python
# 按 category 分组统计（用于汇总表）
category_stats = defaultdict(lambda: {"total": 0, "hits": 0})
for entry in prebaked_data:
    category = entry.get("category", "Unknown")
    category_stats[category]["total"] += 1
    if entry.get("cache_hit"):
        category_stats[category]["hits"] += 1
```

**修改点 2**: 生成 HTML 表格
- 遍历所有 category
- 计算每个 category 的 hit rate
- 从 `eval_metrics['by_category']` 获取 attack rate
- 生成完整的 HTML 表格

**修改点 3**: 添加 CSS 样式
- `.category-table` - 表格主体样式
- `.category-table thead` - 表头样式
- `.category-table th/td` - 单元格样式
- 悬停效果

## 已测试

### Batch 5 的 CoMT/VSP Jobs

✅ **Job 159** (visual_mask + prebaked + fallback:ask)
- 13 个 category，全部显示
- Cache hit rate: 100%
- Attack rates 正确显示（0.0% - 88.9%）

✅ **Job 160** (sd-good + prebaked + fallback:sd)  
- 13 个 category，全部显示
- 正确显示各 category 的统计

✅ **Job 161** (sd-bad + prebaked + fallback:sd)
- 13 个 category，全部显示
- 正确显示各 category 的统计

### 示例数据（Job 159）

```
Category                 | Total | Hits | Hit Rate | Attack Rate
-------------------------|-------|------|----------|------------
01-Illegal_Activitiy     |  10   |  10  | 100.0%   | 0.0%
02-HateSpeech            |  20   |  20  | 100.0%   | 5.0%
03-Malware_Generation    |   5   |   5  | 100.0%   | 60.0%
08-Political_Lobbying    |  18   |  18  | 100.0%   | 88.9%
11-Financial_Advice      |  20   |  20  | 100.0%   | 75.0%
12-Health_Consultation   |  13   |  13  | 100.0%   | 61.5%
```

## 兼容性

### 向后兼容

- 所有现有功能保持不变
- 只是在 Prebaked Report 中添加了新的汇总表
- 不影响其他 provider（openrouter等）

### 条件显示

- **有 prebaked_data**: 显示完整的 Prebaked Report + Category 表格
- **无 prebaked_data**: 不显示 Prebaked Report（保持原样）

## 未来的 Batch

从现在开始，所有新运行的 batch 将自动在每个 job 的 summary.html 中包含这个统计表！

## 文件位置

### 查看示例

```bash
# 在浏览器中打开
open output/batch_5_0208_005316/job_159_tasks_202_ComtVsp_qwen_qwen3-vl-235b-a22b-instruct_0208_005608/summary.html

# 或其他 job
open output/batch_5_0208_005316/job_160_*/summary.html
open output/batch_5_0208_005316/job_161_*/summary.html
```

### 表格位置

滚动到 "Prebaked Processor Report" 部分，在总体统计卡片下方即可看到 "Statistics by Category" 表格。

## 优势

1. **快速概览**: 一眼看到所有 category 的表现
2. **数据关联**: 同时显示 cache hit rate 和 attack rate
3. **易于比较**: 不同 category 的数据对比清晰
4. **完整信息**: 替代了之前只在详细 entries 中显示的分散信息

## 注意事项

- Attack Rate 数据来自 `eval.csv`
- 如果某个 category 没有评估数据，会显示 "N/A"
- 表格按 category 名称字母顺序排序

---

✨ 功能已完成并测试通过！所有修改已提交到 `request.py`。
