# Tests 测试脚本

本目录包含 RedLens 项目的所有测试脚本。

## 测试文件列表

### 核心功能测试

- **`test_provider.py`** - 测试各种 LLM Provider（OpenAI, OpenRouter, Qwen, VSP）
- **`test_vsp_provider.py`** - 测试 VSPProvider 的基本功能
- **`test_extract_answer.py`** - 测试从 VSP 输出中提取答案
- **`test_failed_answer_detection.py`** - 测试失败答案检测功能

### VSP 相关测试

- **`test_vsp_batch.py`** - 测试 VSP 批量模式的目录结构
- **`test_vsp_concurrent.py`** - 测试 VSPProvider 的并发能力

### 数据加载测试

- **`test_mmsb_loader.py`** - 测试 MM-SafetyBench 数据加载器

### 采样功能测试

- **`test_pseudo_random_sampler.py`** - 测试伪随机采样器的所有功能
  - 采样掩码生成的确定性和正确性
  - 按类别独立采样功能
  - 边界情况和错误处理
  - MMSB 数据集场景的集成测试

## 使用方法

### 运行单个测试

从项目根目录运行：

```bash
# 方式 1: 从项目根目录运行
python tests/test_failed_answer_detection.py

# 方式 2: 进入 tests 目录运行
cd tests
python test_failed_answer_detection.py
```

### 运行所有测试

```bash
# 从项目根目录运行所有测试
python -m pytest tests/  # 如果安装了 pytest

# 或者手动运行每个测试
for test in tests/test_*.py; do python "$test"; done
```

## 注意事项

1. 所有测试脚本都已配置好导入路径，会自动将项目根目录添加到 `sys.path`
2. 某些测试（如 VSP 相关测试）需要 VSP 环境配置正确
3. 测试脚本会自动创建必要的输出目录和临时文件

## 测试日志

测试脚本运行时可能生成的日志文件（如 `test_vsp.log`）也会保存在本目录中。

