# Refusal Direction 分析

基于 Difference in Means 方法的 Hidden States 二分类器。用训练集中 safe/unsafe 样本的 hidden states 均值差作为"拒绝方向"向量，将新样本投影到该方向上，用 AUC-ROC 评估分类效果。

## 原理

1. 收集训练集中所有 safe 样本的 hidden states，取均值 `mean_safe`
2. 收集训练集中所有 unsafe 样本的 hidden states，取均值 `mean_unsafe`
3. 拒绝方向 = `normalize(mean_safe - mean_unsafe)`
4. 对新样本：`score = dot(h, direction)`，得分越高越接近 safe
5. 通过 Youden's J 找到最优分类阈值

## 用法

```bash
# 单 job，随机 70/30 split
python refusal_direction.py 236

# 指定 sub_task/turn（VSP 多轮场景）
python refusal_direction.py 243 --sub_task q1 --turn t0

# 多 job 合并训练
python refusal_direction.py 236 240

# batch 中所有 job 合并训练
python refusal_direction.py --batch 17

# job + batch 混合
python refusal_direction.py 236 --batch 17

# 5-fold 交叉验证
python refusal_direction.py 236 --n_folds 5

# 跨 job 测试（多个测试 job）
python refusal_direction.py 236 --test_job 237 238

# batch 作为测试集
python refusal_direction.py 236 --test_batch 17

# 训练和测试都用 batch
python refusal_direction.py --batch 15 --test_batch 17

# 保存方向向量
python refusal_direction.py 236 --save_direction

# 加载已有方向向量分类新数据
python refusal_direction.py 240 --load_direction output/refusal_dir_1/direction.npy

# 使用余弦相似度评分
python refusal_direction.py 236 --score_method cosine
```

## 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `job_nums` | 无 | 训练 job 编号（positional，可多个） |
| `--batch` | 无 | 训练 batch 编号（可多个，展开为其中所有已完成 job） |
| `--test_job` | 无 | 测试 job 编号（可多个） |
| `--test_batch` | 无 | 测试 batch 编号（可多个，展开为其中所有已完成 job） |
| `--sub_task` | `q0` | 子任务标识 |
| `--turn` | `t0` | 轮次标识 |
| `--split_ratio` | `0.7` | 训练集比例（单 job split 模式） |
| `--n_folds` | 无 | K-fold 折数（设置后用交叉验证） |
| `--save_direction` | 关 | 保存方向向量为 `.npy` |
| `--load_direction` | 无 | 加载已有方向向量（跳过训练） |
| `--score_method` | `dot` | 投影方式：`dot` 或 `cosine` |
| `--seed` | `42` | 随机种子 |

## 数据输入

### Job 和 Batch

- **Job**：直接指定 job 编号（positional 参数），可指定多个
- **Batch**：通过 `--batch` 指定 batch 编号，自动从 `batch_state.json` 读取所有已完成（`status=completed`）的 job
- 两者可混合使用，自动去重
- 没有 hidden states 的 job 会被优雅跳过

### 训练与测试分离

- `job_nums` / `--batch` 指定训练数据
- `--test_job` / `--test_batch` 指定测试数据
- 也可混合：`python refusal_direction.py 236 --batch 15 --test_job 240 --test_batch 17`

## 三种运行模式

### 1. Split（默认）

将数据按 `--split_ratio` 做 stratified split，训练集计算方向，测试集评估。

### 2. K-fold 交叉验证（`--n_folds`）

StratifiedKFold，每 fold 重新训练方向向量，汇总 out-of-fold 预测计算 AUC。报告各 fold AUC 均值和标准差。

### 3. 跨 job/batch 测试（`--test_job` / `--test_batch`）

用训练数据的全部样本训练方向，在测试数据上评估。适合测试方向向量的泛化能力。

## 输出

```
output/refusal_dir_{num}/
├── report.html           # 自包含 HTML 报告（内嵌图表 + 指标概览）
├── summary.json          # 元数据 + 所有指标 + PCA 信息
├── direction.npy         # 方向向量（--save_direction 时）
├── roc_curve.png         # ROC 曲线
├── score_distribution.png # safe/unsafe 得分分布直方图
├── category_auc.png      # 按类别 AUC 条形图
├── pca_scatter.png       # PCA PC1-PC2 散点图 + 拒绝方向箭头
└── pca_variance.png      # PCA 方差解释比例
```

### summary.json 关键字段

- `train_jobs` — 训练 job 列表（dir + model）
- `train_batches` — 训练 batch 编号列表
- `test_jobs` / `test_batches` — 测试 job/batch（仅跨 job 模式）
- `overall.auc_roc` — 整体 AUC
- `overall.optimal_threshold` — Youden's J 最优阈值
- `overall.accuracy_at_optimal` — 最优阈值下准确率
- `per_category.{cat}.auc_roc` — 各类别 AUC（单类别为 null）
- `cv.mean_auc` / `cv.std_auc` — K-fold 均值/标准差（仅交叉验证模式）
- `pca.explained_ratio` — PCA 各主成分方差解释比
- `pca.refusal_dir_in_pc` — 拒绝方向在前 3 个 PC 上的投影

## 数据要求

- Job 目录下需要有 `hidden_states/` 目录（使用 `--llm_base_url` 自部署模型时自动生成）
- Job 目录下需要有已评估的 `results.jsonl`（包含 `is_safe(gpt)` 字段）
- 训练集中 safe 和 unsafe 样本都不能为 0
- 没有 hidden states 或 results.jsonl 的 job 会被自动跳过

## 依赖

需要 `scikit-learn`（已加入 `requirements.txt`）。
