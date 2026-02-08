# Batch Report 问题修复完成 ✅

## 修复结果

### ✅ 问题已解决

报告现在正确生成在：
```
output/batch_5_0208_005316/report/evaluation_report.html
```

之前错误的位置已清理：
```
output/batch_5_0208_005316/evaluation_report.html/  (已删除)
```

### ✅ 报告内容正确

报告包含了所有 4 个 job 的对比：

1. **Qwen3-VL-235B-Instruct** (baseline, openrouter)
   - 总攻击率: 37.6%
   
2. **Qwen3-VL-235B-Instruct + CoMT/VSP (visual_mask)**
   - 总攻击率: 32.7%
   - Postproc: visual_mask, backend: prebaked, fallback: ask
   
3. **Qwen3-VL-235B-Instruct + CoMT/VSP (sd-good)**
   - Postproc: good, backend: prebaked, fallback: sd
   
4. **Qwen3-VL-235B-Instruct + CoMT/VSP (sd-bad)**
   - Postproc: bad, backend: prebaked, fallback: sd

### ✅ 报告包含的图表

总共 17 张图表：
- 2 张全局总攻击率对比图（按job和按配置）
- 13 张分类别攻击率对比图
- 2 张品牌分组图表

## 代码修改

### 修改的文件

1. **batch_request.py**
   - `generate_batch_report()` 函数
     - 改用 `--batches` 参数而不是 `--files` 和 `--output`
     - 添加 `batch_num` 参数
     - 更新报告路径显示逻辑

### 核心修改

**之前:**
```python
cmd = f'python generate_report_with_charts.py --files {files_arg} --output "{report_output}"'
```

**修改后:**
```python
cmd = f'python3 generate_report_with_charts.py --batches {batch_num}'
```

## 如何使用

### 新的 batch 运行

运行 `batch_request.py` 时会自动生成正确的报告：
```bash
python batch_request.py
# 报告会自动生成在: output/batch_{num}_{timestamp}/report/evaluation_report.html
```

### 为已有 batch 重新生成报告

```bash
# 使用虚拟环境（推荐）
./venv/bin/python generate_report_with_charts.py --batches 5

# 或者激活虚拟环境后运行
source venv/bin/activate
python generate_report_with_charts.py --batches 5
```

### 查看报告

```bash
# 打开 HTML 报告
open output/batch_5_0208_005316/report/evaluation_report.html

# 或者查看目录
ls -lh output/batch_5_0208_005316/report/
```

## 报告特性

### 1. 全局对比
- 所有 job 的总攻击率横向对比
- 可以清楚看到不同配置的效果差异

### 2. 分类别对比
- 每个安全类别的详细对比
- 13 个类别的独立图表

### 3. VSP Postproc 配置显示
报告中会显示完整的配置信息：
- `visual_mask`: 使用预烘焙图像，fallback 为 ask
- `sd-good`: 使用 SD 生成"好"的图像
- `sd-bad`: 使用 SD 生成"坏"的图像

### 4. 交互式导航
- 左侧边栏快速导航
- 平滑滚动
- 响应式设计

## 验证

### 检查报告位置
```bash
ls output/batch_5_0208_005316/report/
# 应该看到:
# - evaluation_report.html
# - chart_*.png (17 个图表文件)
```

### 检查报告内容
```bash
# 检查是否包含所有 4 个 job
grep "stat-label" output/batch_5_0208_005316/report/evaluation_report.html

# 应该看到 4 个 job 的标签
```

## 注意事项

1. **虚拟环境**: 确保使用正确的 Python 环境（包含 matplotlib、pandas 等）
2. **权限**: 如果遇到权限问题，确保有写入权限
3. **旧报告**: 旧的错误目录结构已被清理

## 未来的 batch

从现在开始，所有新的 batch 运行都会自动生成正确格式的报告！✨
