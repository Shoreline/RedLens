# Hidden States 差异方向分析

`compare_hidden_states.py` 用于对比两个 job 产生的 hidden states，分析同一输入在不同条件下的隐藏状态偏移方向是否一致。

## 核心思路

给定两个 job（例如同一模型在不同 prompt 下的推理结果），对每个共同 task 计算 hidden state 差值 `d = h1 - h2`，然后检验：**同一 category 内的各 task 偏移方向是否趋同？** 如果趋同，说明该 category 的输入引发了系统性的隐藏状态偏移，而非随机噪声。

## 流程

```
resolve_job_dir(job1, job2)          # 根据 job number 定位 output/job_{num}_* 目录
        ↓
find_matching_files(dir1, dir2)      # 找两个 job 中 (category, index) 相同的 .npy 文件
        ↓
compute_differences(matches)         # 逐 task 计算 d = h1 - h2，按 category 分组
        ↓
analyze_category(items)              # 对每个 category 做方向一致性分析
        ↓
print_summary_table / plot_results   # 终端表格 + 可视化图片
        ↓
write_summary(comp_dir)              # 保存 summary.json 到 output/hidden_state_comp_{num}/
```

## 文件匹配规则

Hidden state 文件命名格式：`{category}_{index}_{sub_task}_{turn}.npy`

- 默认匹配 `sub_task=q1, turn=t0`
- 支持两个 job 使用不同的 sub_task/turn（`--sub_task1/2`, `--turn1/2`）
- 只分析两个 job 中 `(category, index)` 交集部分

## 分析指标

### 核心指标：均值方向对齐度（默认输出）

对每个 category 内的所有差值向量：

1. 计算均值向量 `mean_vec = mean(d_1, d_2, ..., d_n)`
2. 归一化得到均值方向 `mean_dir`
3. 每个 task 的 `d_i` 与 `mean_dir` 计算 cosine similarity

输出每个 category 的 mean/std/min/max。值越接近 1 说明偏移方向越一致。

### 详细指标（`--detailed`）

- **Pairwise cosine similarity**：category 内所有 task 对之间的两两余弦相似度
- **PCA 分析**：对差值向量做 SVD，返回前 3 个主成分的方差解释比。PC1 占比越高说明偏移越集中在单一方向
- **跨 category baseline**：不同 category 之间的差值向量 pairwise cosine similarity，作为随机基线。如果同 category 内的一致性显著高于跨 category baseline，则说明偏移是 category-specific 的

## 输出

### 目录结构

```
output/hidden_state_comp_{num}/
├── summary.json                    # 元信息 + 每个 category 的统计结果
├── hs_diff_cosine_boxplot.png      # 箱线图：每个 category 各 task 与均值方向的 cosine sim
├── hs_diff_alignment_bar.png       # 柱状图：每个 category 的平均对齐度
└── hs_diff_intra_vs_cross.png      # (--detailed) 同 category vs 跨 category 分布直方图
```

编号通过 `output/.comp_counter` 文件单调递增管理。

### 终端表格

```
Category  Tasks     Mean      Std      Min      Max   [PCA-1  PCA-2]
```

后接逐 task 明细（每个 task 的 cosine similarity 值）。

## 用法

```bash
# 基本对比
python compare_hidden_states.py 177 178

# 指定 sub_task 和 turn
python compare_hidden_states.py 177 178 --sub_task q1 --turn t0

# 两个 job 使用不同的 sub_task/turn
python compare_hidden_states.py 177 178 --sub_task1 q1 --turn1 t0 --sub_task2 q2 --turn2 t0

# 详细分析（含 PCA + 跨 category baseline）
python compare_hidden_states.py 177 178 --detailed
```

## 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `job1`, `job2` | 必填 | 两个 job 的编号 |
| `--sub_task` | `q1` | 两个 job 共用的子任务标识 |
| `--turn` | `t0` | 两个 job 共用的轮次标识 |
| `--sub_task1/2` | 无 | 分别覆盖 Job1/Job2 的子任务标识 |
| `--turn1/2` | 无 | 分别覆盖 Job1/Job2 的轮次标识 |
| `--detailed` | `false` | 启用 PCA 分析 + 跨 category baseline |
