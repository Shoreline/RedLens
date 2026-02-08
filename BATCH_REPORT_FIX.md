# Batch Report 生成问题修复

## 问题描述

运行 batch 后，`evaluation_report.html` 生成在了错误的位置：
```
output/batch_5_0208_005316/evaluation_report.html/reports/models_4_2026-02-08_01-17-25/evaluation_report.html
```

应该生成在：
```
output/batch_5_0208_005316/report/evaluation_report.html
```

## 根本原因

`batch_request.py` 使用了旧的调用方式（`--files` 和 `--output` 参数），但 `generate_report_with_charts.py` 已经更新为使用 `--batches` 参数的新逻辑。

## 修复内容

### 1. 修改 `batch_request.py` 的 `generate_batch_report()` 函数

**之前的代码：**
```python
def generate_batch_report(results: List[RunResult], batch_folder: str):
    # ...
    report_output = os.path.join(batch_folder, "evaluation_report.html")
    files_arg = ' '.join(f'"{f}"' for f in eval_files)
    cmd = f'python generate_report_with_charts.py --files {files_arg} --output "{report_output}"'
```

**修改后：**
```python
def generate_batch_report(results: List[RunResult], batch_folder: str, batch_num: int):
    # ...
    # 使用新的 --batches 参数，让 generate_report_with_charts.py 自动处理输出路径
    # 报告会生成在 batch_folder/report/ 目录下
    cmd = f'python3 generate_report_with_charts.py --batches {batch_num}'
```

### 2. 更新函数调用

在 `main()` 函数中添加 `batch_num` 参数：
```python
generate_batch_report(results, batch_folder, batch_num)
```

### 3. 更新输出路径显示

报告现在生成在 `{batch_folder}/report/` 目录下。

## 如何为已有的 batch 重新生成报告

### 方法 1: 使用 generate_report_with_charts.py（推荐）

```bash
cd /Users/yuantian/code/Mediator

# 为 batch_5 生成报告
python generate_report_with_charts.py --batches 5

# 报告会生成在：
# output/batch_5_0208_005316/report/evaluation_report.html
```

### 方法 2: 手动清理和重新运行

如果需要清理错误的目录：
```bash
# 删除错误的目录结构
rm -rf output/batch_5_0208_005316/evaluation_report.html

# 重新生成报告
python generate_report_with_charts.py --batches 5
```

## 新的报告目录结构

```
output/batch_5_0208_005316/
├── batch_summary.html          # Batch 概览（已正确）
├── report/                     # 新的报告目录
│   ├── evaluation_report.html  # 主报告文件
│   ├── chart_0_*.png          # 各个模型的图表
│   ├── chart_1_*.png
│   ├── chart_2_*.png
│   └── chart_3_*.png
├── job_158_*/                  # Job 文件夹
├── job_159_*/
├── job_160_*/
└── job_161_*/
```

## 报告功能对比

新的报告会：
1. ✅ 按 job 对比不同配置（openrouter vs comt_vsp）
2. ✅ 显示 VSP postproc 配置差异（visual_mask, good, bad）
3. ✅ 生成详细的攻击率对比图表
4. ✅ 提供分类别的详细分析

旧的报告问题：
1. ❌ 报告位置错误，嵌套太深
2. ❌ 使用了废弃的参数格式

## 验证修复

运行新的 batch 测试：
```bash
# 确保使用正确的 Python 环境（带 matplotlib）
python batch_request.py

# 检查报告位置
ls -la output/batch_*/report/
```

## 注意事项

1. 需要确保环境中安装了 `matplotlib`、`pandas` 等依赖
2. 如果使用虚拟环境，请激活后再运行
3. 旧的 batch 可以随时使用 `--batches` 参数重新生成报告
