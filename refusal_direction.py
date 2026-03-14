#!/usr/bin/env python3
"""
Refusal Direction 分析 —— 基于 Difference in Means 的 Hidden States 分类器。

用训练集中 safe/unsafe 样本的 hidden states 均值差作为"拒绝方向"向量，
将样本投影到该方向上进行二分类，用 AUC-ROC 评估效果。

用法：
    python refusal_direction.py 236
    python refusal_direction.py 236 240                    # 多个 job 合并
    python refusal_direction.py --batch 17                 # batch 中所有 job
    python refusal_direction.py 236 --test_job 237 238     # 多个测试 job
    python refusal_direction.py 236 --test_batch 17        # batch 中所有 job 作为测试集
    python refusal_direction.py 236 --n_folds 5
    python refusal_direction.py 236 --save_direction
    python refusal_direction.py 237 --load_direction output/refusal_dir_1/direction.npy
    python refusal_direction.py 236 --sub_task q0 --turn t0
    python refusal_direction.py 236 --score_method cosine
"""

import numpy as np
from pathlib import Path
from collections import defaultdict
import argparse
import json
import sys

# ============================================================
# 可视化按需导入
# ============================================================

def _import_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
    import matplotlib.pyplot as plt
    return plt


# ============================================================
# Job 目录解析 & 输出目录
# ============================================================

OUTPUT_ROOT = Path(__file__).parent / "output"
REFDIR_COUNTER_FILE = OUTPUT_ROOT / ".refdir_counter"


def resolve_job_dir(job_num: int) -> Path:
    """根据 job number 在 output/ 下查找 job_{num}_* 目录。"""
    matches = sorted(OUTPUT_ROOT.glob(f"job_{job_num}_*"))
    matches = [m for m in matches if m.is_dir()]
    if len(matches) == 0:
        print(f"错误：在 {OUTPUT_ROOT} 下找不到 job_{job_num}_* 目录", file=sys.stderr)
        sys.exit(1)
    if len(matches) > 1:
        print(f"错误：在 {OUTPUT_ROOT} 下找到多个匹配 job_{job_num}_* 的目录：", file=sys.stderr)
        for m in matches:
            print(f"  {m.name}", file=sys.stderr)
        sys.exit(1)
    return matches[0]


def resolve_batch_dir(batch_num: int) -> Path:
    """根据 batch number 在 output/ 下查找 batch_{num}_* 目录。"""
    matches = sorted(OUTPUT_ROOT.glob(f"batch_{batch_num}_*"))
    matches = [m for m in matches if m.is_dir()]
    if len(matches) == 0:
        print(f"错误：在 {OUTPUT_ROOT} 下找不到 batch_{batch_num}_* 目录", file=sys.stderr)
        sys.exit(1)
    if len(matches) > 1:
        print(f"错误：在 {OUTPUT_ROOT} 下找到多个匹配 batch_{batch_num}_* 的目录：", file=sys.stderr)
        for m in matches:
            print(f"  {m.name}", file=sys.stderr)
        sys.exit(1)
    return matches[0]


def resolve_batch_jobs(batch_num: int) -> list[Path]:
    """从 batch 目录的 batch_state.json 中提取所有已完成的 job 目录。"""
    batch_dir = resolve_batch_dir(batch_num)
    state_path = batch_dir / "batch_state.json"
    if not state_path.exists():
        print(f"错误：{state_path} 不存在", file=sys.stderr)
        sys.exit(1)

    state = json.loads(state_path.read_text(encoding="utf-8"))
    job_dirs = []
    for run in state.get("runs", []):
        if run.get("status") != "completed":
            continue
        job_folder = run.get("job_folder", "")
        job_path = batch_dir / job_folder
        if not job_path.is_dir():
            # 也可能在 output/ 顶层
            job_path = OUTPUT_ROOT / job_folder
        if job_path.is_dir():
            job_dirs.append(job_path)
        else:
            print(f"  警告：跳过找不到的 job 目录 {job_folder}")
    return job_dirs


def resolve_job_dirs(job_nums: list[int] = None, batch_nums: list[int] = None) -> list[Path]:
    """解析 job_num 列表和 batch_num 列表，返回去重的 job 目录列表。"""
    dirs = []
    seen = set()

    if job_nums:
        for num in job_nums:
            d = resolve_job_dir(num)
            if d.name not in seen:
                dirs.append(d)
                seen.add(d.name)

    if batch_nums:
        for num in batch_nums:
            for d in resolve_batch_jobs(num):
                if d.name not in seen:
                    dirs.append(d)
                    seen.add(d.name)

    return dirs


def get_next_refdir_num() -> int:
    """获取下一个 refusal_dir 编号。"""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if REFDIR_COUNTER_FILE.exists():
        try:
            current = int(REFDIR_COUNTER_FILE.read_text().strip())
        except (ValueError, IOError):
            current = 0
    else:
        current = 0
    next_num = current + 1
    REFDIR_COUNTER_FILE.write_text(str(next_num))
    return next_num


def create_refdir_dir(num: int) -> Path:
    """创建 refusal_dir_{num}/ 目录。"""
    d = OUTPUT_ROOT / f"refusal_dir_{num}"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ============================================================
# 数据加载
# ============================================================

def load_labels(job_dir: Path) -> dict:
    """从 results.jsonl 读取 eval 标签。

    返回 {(cat_prefix, index): "safe"/"unsafe"}
    """
    jsonl_path = job_dir / "results.jsonl"
    if not jsonl_path.exists():
        print(f"错误：{jsonl_path} 不存在", file=sys.stderr)
        sys.exit(1)

    labels = {}
    n_error = 0
    n_missing = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            origin = record.get("origin", {})
            category = origin.get("category", "")
            index = str(origin.get("index", ""))
            label = record.get("is_safe(gpt)", "")

            if not category or not index:
                continue

            cat_prefix = category.split("-")[0]

            if label == "error":
                n_error += 1
                continue
            if label not in ("safe", "unsafe"):
                n_missing += 1
                continue

            labels[(cat_prefix, index)] = label

    if n_error:
        print(f"  跳过 {n_error} 条 error 标签")
    if n_missing:
        print(f"  跳过 {n_missing} 条无标签记录")
    return labels


def load_hidden_states(job_dir: Path, sub_task: str = "q0", turn: str = "t0",
                       layer: int = None) -> dict:
    """加载 hidden states .npy 文件。

    返回 {(cat_prefix, index): np.ndarray}，目录不存在时返回空 dict。

    layer 参数预留，当前未使用。
    """
    hs_dir = job_dir / "hidden_states"
    if not hs_dir.is_dir():
        return {}

    suffix = f"_{sub_task}_{turn}.npy"
    hidden_states = {}
    for f in hs_dir.glob(f"*{suffix}"):
        prefix = f.stem.removesuffix(f"_{sub_task}_{turn}")
        parts = prefix.split("_", 1)
        if len(parts) == 2:
            cat, index = parts
            hidden_states[(cat, index)] = np.load(f)
    return hidden_states


def load_meta(job_dir: Path) -> dict:
    """加载 hidden_states/meta.json。"""
    meta_path = job_dir / "hidden_states" / "meta.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    return {}


def pair_data(labels: dict, hidden_states: dict) -> list:
    """内连接标签和 hidden states。

    返回 [{"cat": str, "index": str, "label": str, "vector": np.ndarray}, ...]
    """
    common_keys = set(labels.keys()) & set(hidden_states.keys())
    n_label_only = len(labels) - len(common_keys)
    n_hs_only = len(hidden_states) - len(common_keys)

    if n_label_only:
        print(f"  {n_label_only} 条有标签但无 hidden state")
    if n_hs_only:
        print(f"  {n_hs_only} 条有 hidden state 但无标签")

    samples = []
    for key in sorted(common_keys):
        cat, index = key
        samples.append({
            "cat": cat,
            "index": index,
            "label": labels[key],
            "vector": hidden_states[key],
        })
    return samples


def load_multi_job(job_dirs: list[Path], sub_task: str, turn: str) -> tuple[list, list[dict]]:
    """从多个 job 目录加载并合并数据。

    返回 (合并后的 samples, 各 job 的 meta 列表)
    """
    all_samples = []
    metas = []
    for job_dir in job_dirs:
        meta = load_meta(job_dir)
        metas.append(meta)
        model_name = meta.get("model", "unknown")
        print(f"  {job_dir.name} ({model_name})")

        labels = load_labels(job_dir)
        hs = load_hidden_states(job_dir, sub_task, turn)
        if not hs:
            print(f"    跳过（无 hidden states）")
            continue
        samples = pair_data(labels, hs)
        # 标记来源 job
        for s in samples:
            s["job"] = job_dir.name
        all_samples.extend(samples)

    return all_samples, metas


# ============================================================
# 方向计算
# ============================================================

def compute_refusal_direction(samples: list) -> np.ndarray:
    """计算拒绝方向：mean(safe) - mean(unsafe)，归一化。"""
    safe_vecs = np.array([s["vector"] for s in samples if s["label"] == "safe"])
    unsafe_vecs = np.array([s["vector"] for s in samples if s["label"] == "unsafe"])

    if len(safe_vecs) == 0:
        print("错误：训练集中没有 safe 样本", file=sys.stderr)
        sys.exit(1)
    if len(unsafe_vecs) == 0:
        print("错误：训练集中没有 unsafe 样本", file=sys.stderr)
        sys.exit(1)

    n_safe, n_unsafe = len(safe_vecs), len(unsafe_vecs)
    ratio = max(n_safe, n_unsafe) / min(n_safe, n_unsafe)
    if ratio > 10:
        print(f"  警告：类别不平衡 safe={n_safe}, unsafe={n_unsafe} (比例 {ratio:.1f}:1)")

    direction = np.mean(safe_vecs, axis=0) - np.mean(unsafe_vecs, axis=0)
    norm = np.linalg.norm(direction)
    if norm < 1e-12:
        print("错误：safe 和 unsafe 均值完全相同，无法计算方向", file=sys.stderr)
        sys.exit(1)
    return direction / norm


# ============================================================
# 评分
# ============================================================

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """两个向量的余弦相似度。"""
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def score_samples(samples: list, direction: np.ndarray, method: str = "dot") -> np.ndarray:
    """将样本投影到拒绝方向上，返回得分数组。

    得分越高 → 越接近 safe。
    """
    vectors = np.array([s["vector"] for s in samples])
    if method == "cosine":
        scores = np.array([cosine_sim(v, direction) for v in vectors])
    else:  # dot
        scores = vectors @ direction
    return scores


# ============================================================
# 评估
# ============================================================

def evaluate(samples: list, scores: np.ndarray) -> dict:
    """计算整体和按 category 的 AUC-ROC。"""
    from sklearn.metrics import roc_auc_score, roc_curve

    # safe=1, unsafe=0
    labels = np.array([1 if s["label"] == "safe" else 0 for s in samples])
    cats = [s["cat"] for s in samples]

    result = {"overall": {}, "per_category": {}}

    # 整体 AUC
    n_safe = int(labels.sum())
    n_unsafe = int(len(labels) - n_safe)
    result["overall"]["n_safe"] = n_safe
    result["overall"]["n_unsafe"] = n_unsafe

    if n_safe > 0 and n_unsafe > 0:
        auc = float(roc_auc_score(labels, scores))
        fpr, tpr, thresholds = roc_curve(labels, scores)
        # Youden's J 最优阈值
        j_scores = tpr - fpr
        best_idx = int(np.argmax(j_scores))
        optimal_threshold = float(thresholds[best_idx])
        preds_at_optimal = (scores >= optimal_threshold).astype(int)
        accuracy = float(np.mean(preds_at_optimal == labels))

        result["overall"]["auc_roc"] = auc
        result["overall"]["optimal_threshold"] = optimal_threshold
        result["overall"]["accuracy_at_optimal"] = accuracy
        result["overall"]["roc_curve"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
    else:
        result["overall"]["auc_roc"] = None
        result["overall"]["optimal_threshold"] = None
        result["overall"]["accuracy_at_optimal"] = None

    # 按 category 分析
    cat_set = sorted(set(cats))
    for cat in cat_set:
        mask = np.array([c == cat for c in cats])
        cat_labels = labels[mask]
        cat_scores = scores[mask]
        n_s = int(cat_labels.sum())
        n_u = int(len(cat_labels) - n_s)

        cat_result = {"n_safe": n_s, "n_unsafe": n_u}
        if n_s > 0 and n_u > 0:
            cat_result["auc_roc"] = float(roc_auc_score(cat_labels, cat_scores))
        else:
            cat_result["auc_roc"] = None
        result["per_category"][cat] = cat_result

    return result


def evaluate_kfold(samples: list, direction_fn, score_method: str,
                   n_folds: int, seed: int) -> dict:
    """K-fold 交叉验证。

    direction_fn: compute_refusal_direction 函数引用
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    labels_arr = np.array([1 if s["label"] == "safe" else 0 for s in samples])
    n_safe = int(labels_arr.sum())
    n_unsafe = int(len(labels_arr) - n_safe)

    if n_safe < n_folds or n_unsafe < n_folds:
        print(f"警告：样本数 (safe={n_safe}, unsafe={n_unsafe}) 不足以支持 {n_folds}-fold CV",
              file=sys.stderr)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    fold_aucs = []
    all_scores = np.zeros(len(samples))
    all_tested = np.zeros(len(samples), dtype=bool)

    for fold_i, (train_idx, test_idx) in enumerate(skf.split(labels_arr, labels_arr)):
        train_samples = [samples[i] for i in train_idx]
        test_samples = [samples[i] for i in test_idx]

        direction = direction_fn(train_samples)
        fold_scores = score_samples(test_samples, direction, method=score_method)

        # 存储 out-of-fold 预测
        for j, idx in enumerate(test_idx):
            all_scores[idx] = fold_scores[j]
            all_tested[idx] = True

        # fold AUC
        fold_labels = labels_arr[test_idx]
        if fold_labels.sum() > 0 and (len(fold_labels) - fold_labels.sum()) > 0:
            fold_auc = float(roc_auc_score(fold_labels, fold_scores))
            fold_aucs.append(fold_auc)
            print(f"  Fold {fold_i + 1}/{n_folds}: AUC = {fold_auc:.4f}")
        else:
            print(f"  Fold {fold_i + 1}/{n_folds}: 单类别，跳过 AUC")

    # 汇总
    result = evaluate(samples, all_scores)
    result["cv"] = {
        "n_folds": n_folds,
        "fold_aucs": fold_aucs,
        "mean_auc": float(np.mean(fold_aucs)) if fold_aucs else None,
        "std_auc": float(np.std(fold_aucs)) if fold_aucs else None,
    }
    return result, all_scores


# ============================================================
# PCA 分析
# ============================================================

def pca_analyze(samples: list, direction: np.ndarray, n_components: int = 10):
    """对 hidden states 做 PCA，返回分析结果。"""
    vectors = np.array([s["vector"] for s in samples])
    labels = np.array([s["label"] for s in samples])

    # 中心化
    mean_vec = vectors.mean(axis=0)
    centered = vectors - mean_vec

    # SVD
    try:
        U, s, Vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None

    variance = s ** 2
    total_var = variance.sum()
    if total_var < 1e-12:
        return None

    explained_ratio = (variance / total_var)[:n_components]

    # 投影到前 2 个主成分
    pc_coords = centered @ Vt[:2].T  # (n_samples, 2)
    safe_mask = labels == "safe"

    # 方向在 PC 空间中的投影
    dir_in_pc = Vt[:min(3, len(Vt))] @ direction  # (3,)

    return {
        "explained_ratio": explained_ratio.tolist(),
        "pc_coords": pc_coords,
        "safe_mask": safe_mask,
        "dir_in_pc": dir_in_pc.tolist(),
        "Vt": Vt[:2],
    }


# ============================================================
# 可视化
# ============================================================

def plot_roc_curve(eval_result: dict, output_path: Path, title_suffix: str = ""):
    """绘制 ROC 曲线。"""
    plt = _import_matplotlib()

    overall = eval_result["overall"]
    if overall.get("auc_roc") is None:
        return

    roc = overall["roc_curve"]
    auc = overall["auc_roc"]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(roc["fpr"], roc["tpr"], color="#2C3E50", lw=2,
            label=f"ROC (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="#BDC3C7", lw=1, linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"Refusal Direction ROC{title_suffix}")
    ax.legend(loc="lower right")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    fig.savefig(output_path / "roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ROC 曲线: {output_path / 'roc_curve.png'}")


def plot_score_distribution(samples: list, scores: np.ndarray,
                            threshold: float, output_path: Path,
                            title_suffix: str = ""):
    """绘制 safe/unsafe 得分分布直方图。"""
    plt = _import_matplotlib()

    labels = np.array([s["label"] for s in samples])
    safe_scores = scores[labels == "safe"]
    unsafe_scores = scores[labels == "unsafe"]

    fig, ax = plt.subplots(figsize=(8, 5))

    # 确定 bin 范围
    all_scores = np.concatenate([safe_scores, unsafe_scores])
    bins = np.linspace(all_scores.min(), all_scores.max(), 40)

    ax.hist(safe_scores, bins=bins, alpha=0.5, color="#2ECC71", label=f"Safe (n={len(safe_scores)})",
            edgecolor="white", linewidth=0.5)
    ax.hist(unsafe_scores, bins=bins, alpha=0.5, color="#E74C3C", label=f"Unsafe (n={len(unsafe_scores)})",
            edgecolor="white", linewidth=0.5)

    if threshold is not None:
        ax.axvline(x=threshold, color="#2C3E50", linestyle="--", lw=1.5,
                   label=f"Threshold = {threshold:.3f}")

    ax.set_xlabel("Projection Score (→ safe)")
    ax.set_ylabel("Count")
    ax.set_title(f"Score Distribution{title_suffix}")
    ax.legend()

    fig.savefig(output_path / "score_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  得分分布: {output_path / 'score_distribution.png'}")


def plot_category_auc(eval_result: dict, output_path: Path, title_suffix: str = ""):
    """按类别 AUC 条形图。"""
    plt = _import_matplotlib()

    per_cat = eval_result["per_category"]
    # 过滤有 AUC 的类别
    valid = {c: v for c, v in per_cat.items() if v.get("auc_roc") is not None}
    if len(valid) < 2:
        return

    # 排序
    cats = sorted(valid.keys(), key=lambda c: valid[c]["auc_roc"])
    aucs = [valid[c]["auc_roc"] for c in cats]

    fig, ax = plt.subplots(figsize=(8, max(4, len(cats) * 0.4)))
    colors = ["#E74C3C" if a < 0.6 else "#F39C12" if a < 0.8 else "#2ECC71" for a in aucs]
    bars = ax.barh(range(len(cats)), aucs, color=colors, edgecolor="white", height=0.6)
    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels(cats)
    ax.set_xlabel("AUC-ROC")
    ax.set_title(f"Per-Category AUC{title_suffix}")
    ax.set_xlim([0, 1.05])

    # AUC 数值标注
    for i, (bar, auc) in enumerate(zip(bars, aucs)):
        ax.text(auc + 0.01, i, f"{auc:.3f}", va="center", fontsize=9)

    # 整体 AUC 参考线
    overall_auc = eval_result["overall"].get("auc_roc")
    if overall_auc is not None:
        ax.axvline(x=overall_auc, color="#2C3E50", linestyle="--", lw=1,
                   label=f"Overall AUC = {overall_auc:.3f}")
        ax.legend(loc="lower right", fontsize=9)

    fig.savefig(output_path / "category_auc.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  类别 AUC: {output_path / 'category_auc.png'}")


def plot_pca_scatter(pca_result: dict, output_path: Path, title_suffix: str = ""):
    """PCA 散点图：PC1 vs PC2，safe/unsafe 不同颜色 + 拒绝方向箭头。"""
    plt = _import_matplotlib()

    if pca_result is None:
        return

    coords = pca_result["pc_coords"]
    safe_mask = pca_result["safe_mask"]
    dir_in_pc = pca_result["dir_in_pc"]
    explained = pca_result["explained_ratio"]

    fig, ax = plt.subplots(figsize=(8, 7))

    # 散点
    ax.scatter(coords[safe_mask, 0], coords[safe_mask, 1],
               c="#2ECC71", alpha=0.5, s=20, label="Safe", edgecolors="white", linewidth=0.3)
    ax.scatter(coords[~safe_mask, 0], coords[~safe_mask, 1],
               c="#E74C3C", alpha=0.5, s=20, label="Unsafe", edgecolors="white", linewidth=0.3)

    # 拒绝方向箭头（在 PC1-PC2 空间）
    if len(dir_in_pc) >= 2:
        arrow_scale = max(np.abs(coords).max() * 0.3, 1.0)
        dx, dy = dir_in_pc[0] * arrow_scale, dir_in_pc[1] * arrow_scale
        ax.annotate("", xy=(dx, dy), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="#2C3E50", lw=2))
        ax.text(dx * 1.1, dy * 1.1, "Refusal Dir", fontsize=9, color="#2C3E50",
                ha="center", va="center")

    pc1_var = f"{explained[0]:.1%}" if len(explained) > 0 else "?"
    pc2_var = f"{explained[1]:.1%}" if len(explained) > 1 else "?"
    ax.set_xlabel(f"PC1 ({pc1_var} variance)")
    ax.set_ylabel(f"PC2 ({pc2_var} variance)")
    ax.set_title(f"PCA Projection{title_suffix}")
    ax.legend()

    fig.savefig(output_path / "pca_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  PCA 散点图: {output_path / 'pca_scatter.png'}")


def plot_pca_variance(pca_result: dict, output_path: Path, title_suffix: str = ""):
    """PCA 方差解释比例条形图。"""
    plt = _import_matplotlib()

    if pca_result is None:
        return

    explained = pca_result["explained_ratio"]
    if len(explained) < 2:
        return

    n = len(explained)
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.bar(range(1, n + 1), explained, color="#4ECDC4", edgecolor="white")
    cumulative = np.cumsum(explained)
    ax.plot(range(1, n + 1), cumulative, "o-", color="#E74C3C", markersize=5,
            label="Cumulative")

    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_title(f"PCA Explained Variance{title_suffix}")
    ax.set_xticks(range(1, n + 1))
    ax.legend()

    fig.savefig(output_path / "pca_variance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  PCA 方差: {output_path / 'pca_variance.png'}")


# ============================================================
# HTML 报告
# ============================================================

def generate_report_html(summary: dict, out_dir: Path):
    """生成 report.html 概览页，内嵌所有图表。"""
    import base64

    overall = summary.get("overall", {})
    params = summary.get("params", {})
    data_stats = summary.get("data_stats", {})
    per_cat = summary.get("per_category", {})
    cv = summary.get("cv")
    pca = summary.get("pca")
    train_jobs = summary.get("train_jobs", [])
    test_jobs = summary.get("test_jobs")

    auc = overall.get("auc_roc")
    acc = overall.get("accuracy_at_optimal")
    threshold = overall.get("optimal_threshold")

    def embed_img(filename):
        path = out_dir / filename
        if not path.exists():
            return ""
        data = base64.b64encode(path.read_bytes()).decode()
        return f'<img src="data:image/png;base64,{data}" style="width:100%; border-radius:8px;">'

    # 训练 job 列表
    train_list = "".join(
        f'<li>{j.get("dir", "?")} <span style="color:#888">({j.get("model", "?")})</span></li>'
        for j in train_jobs
    )

    # 测试 job 列表
    test_html = ""
    if test_jobs:
        test_list = "".join(f'<li>{j.get("dir", "?")}</li>' for j in test_jobs)
        test_html = f"""
        <div class="info-block">
            <h3>Test Jobs</h3>
            <ul>{test_list}</ul>
        </div>"""

    # CV 信息
    cv_html = ""
    if cv and cv.get("mean_auc") is not None:
        fold_strs = ", ".join(f'{a:.4f}' for a in cv.get("fold_aucs", []))
        cv_html = f"""
        <div class="stat-card">
            <h3>{cv['mean_auc']:.4f}</h3>
            <p>Mean AUC ({cv['n_folds']}-fold CV)</p>
        </div>
        <div class="stat-card">
            <h3>&plusmn;{cv['std_auc']:.4f}</h3>
            <p>Std AUC</p>
        </div>"""

    # PCA 信息
    pca_html = ""
    if pca:
        ratios = pca.get("explained_ratio", [])
        if len(ratios) >= 2:
            pca_html = f"""
        <div class="info-block">
            <h3>PCA</h3>
            <p>PC1 方差解释: <strong>{ratios[0]:.1%}</strong>, PC2: <strong>{ratios[1]:.1%}</strong></p>
            <p>Refusal dir in PC1-3: [{', '.join(f'{v:.4f}' for v in pca.get('refusal_dir_in_pc', []))}]</p>
        </div>"""

    # Category 表格行
    cat_rows = ""
    for cat in sorted(per_cat.keys()):
        r = per_cat[cat]
        auc_str = f'{r["auc_roc"]:.4f}' if r.get("auc_roc") is not None else '<span style="color:#666">N/A</span>'
        cat_rows += f"""
            <tr>
                <td>{cat}</td>
                <td>{r['n_safe']}</td>
                <td>{r['n_unsafe']}</td>
                <td>{auc_str}</td>
            </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Refusal Direction #{summary.get('refdir_num', '?')}</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); color: #e0e0e0; min-height: 100vh; padding: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ text-align: center; color: #00d9ff; margin-bottom: 6px; font-size: 2em; text-shadow: 0 0 20px rgba(0, 217, 255, 0.3); }}
        .subtitle {{ text-align: center; color: #888; margin-bottom: 30px; font-size: 0.9em; }}
        .section {{ background: rgba(255, 255, 255, 0.05); border-radius: 15px; padding: 25px; margin-bottom: 25px; border: 1px solid rgba(255, 255, 255, 0.1); }}
        .section h2 {{ color: #00d9ff; margin-bottom: 20px; font-size: 1.3em; border-bottom: 1px solid rgba(255, 255, 255, 0.1); padding-bottom: 10px; }}
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 15px; margin-bottom: 20px; }}
        .stat-card {{ background: rgba(255, 255, 255, 0.05); border-radius: 12px; padding: 18px; text-align: center; border: 1px solid rgba(255, 255, 255, 0.1); }}
        .stat-card h3 {{ font-size: 1.8em; margin-bottom: 4px; color: #ffd93d; }}
        .stat-card.auc h3 {{ color: #00ff88; }}
        .stat-card.safe h3 {{ color: #00ff88; }}
        .stat-card.unsafe h3 {{ color: #ff6b6b; }}
        .stat-card p {{ color: #888; font-size: 0.85em; }}
        .charts {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
        .charts img {{ width: 100%; border-radius: 8px; }}
        .chart-full {{ margin-top: 20px; }}
        .chart-full img {{ width: 100%; max-width: 700px; display: block; margin: 0 auto; border-radius: 8px; }}
        .info-block {{ margin-bottom: 15px; }}
        .info-block h3 {{ color: #00d9ff; font-size: 1em; margin-bottom: 8px; }}
        .info-block p {{ color: #bbb; font-size: 0.9em; line-height: 1.6; }}
        .info-block strong {{ color: #ffd93d; }}
        .info-block ul {{ list-style: none; padding: 0; }}
        .info-block li {{ color: #bbb; font-size: 0.85em; padding: 3px 0; }}
        .info-block li::before {{ content: "\\25B8 "; color: #00d9ff; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
        th {{ text-align: left; color: #00d9ff; padding: 8px 12px; border-bottom: 2px solid rgba(0, 217, 255, 0.3); font-size: 0.9em; }}
        td {{ padding: 6px 12px; border-bottom: 1px solid rgba(255, 255, 255, 0.06); font-size: 0.9em; }}
        tr:hover td {{ background: rgba(255, 255, 255, 0.03); }}
    </style>
</head>
<body>
<div class="container">
    <h1>Refusal Direction #{summary.get('refdir_num', '?')}</h1>
    <p class="subtitle">{summary.get('timestamp', '')}</p>

    <div class="section">
        <h2>Overview</h2>
        <div class="stats">
            <div class="stat-card auc">
                <h3>{f'{auc:.4f}' if auc is not None else 'N/A'}</h3>
                <p>AUC-ROC</p>
            </div>
            <div class="stat-card">
                <h3>{f'{acc:.1%}' if acc is not None else 'N/A'}</h3>
                <p>Accuracy @ Optimal</p>
            </div>
            <div class="stat-card">
                <h3>{data_stats.get('total', 0)}</h3>
                <p>Total Samples</p>
            </div>
            <div class="stat-card safe">
                <h3>{data_stats.get('safe', 0)}</h3>
                <p>Safe</p>
            </div>
            <div class="stat-card unsafe">
                <h3>{data_stats.get('unsafe', 0)}</h3>
                <p>Unsafe</p>
            </div>
            {cv_html}
        </div>
    </div>

    <div class="section">
        <h2>Configuration</h2>
        <div class="info-block">
            <h3>Train Jobs</h3>
            <ul>{train_list}</ul>
        </div>
        {test_html}
        <div class="info-block">
            <p>Sub-task: <strong>{params.get('sub_task', '?')}</strong> &nbsp; Turn: <strong>{params.get('turn', '?')}</strong> &nbsp; Score: <strong>{params.get('score_method', '?')}</strong> &nbsp; Layer: <strong>{params.get('layer', '?')}</strong></p>
            <p>Split: <strong>{params.get('split_ratio', '?')}</strong> &nbsp; Seed: <strong>{params.get('seed', '?')}</strong>{f" &nbsp; Folds: <strong>{params.get('n_folds')}</strong>" if params.get('n_folds') else ''}</p>
        </div>
        {pca_html}
    </div>

    <div class="section">
        <h2>Charts</h2>
        <div class="charts">
            <div>{embed_img('roc_curve.png')}</div>
            <div>{embed_img('score_distribution.png')}</div>
            <div>{embed_img('pca_scatter.png')}</div>
            <div>{embed_img('pca_variance.png')}</div>
        </div>
        <div class="chart-full">
            {embed_img('category_auc.png')}
        </div>
    </div>

    <div class="section">
        <h2>Per-Category AUC</h2>
        <table>
            <thead><tr><th>Category</th><th>Safe</th><th>Unsafe</th><th>AUC</th></tr></thead>
            <tbody>{cat_rows}</tbody>
        </table>
    </div>
</div>
</body>
</html>"""

    report_path = out_dir / "report.html"
    report_path.write_text(html, encoding="utf-8")
    print(f"  Report: {report_path}")


# ============================================================
# 终端输出
# ============================================================

def print_summary(eval_result: dict, cv_info: dict = None):
    """打印终端汇总。"""
    print()
    print("=" * 70)
    print("Refusal Direction 分析结果")
    print("=" * 70)

    overall = eval_result["overall"]
    auc = overall.get("auc_roc")
    acc = overall.get("accuracy_at_optimal")
    threshold = overall.get("optimal_threshold")

    print(f"  样本数: safe={overall['n_safe']}, unsafe={overall['n_unsafe']}")
    if auc is not None:
        print(f"  AUC-ROC: {auc:.4f}")
        print(f"  最优阈值: {threshold:.4f}")
        print(f"  最优阈值下准确率: {acc:.4f}")
    else:
        print("  AUC-ROC: 无法计算（单类别）")

    if cv_info:
        print(f"\n  交叉验证 ({cv_info['n_folds']}-fold):")
        if cv_info.get("mean_auc") is not None:
            print(f"    Mean AUC: {cv_info['mean_auc']:.4f} ± {cv_info['std_auc']:.4f}")
            print(f"    各 fold: {', '.join(f'{a:.4f}' for a in cv_info['fold_aucs'])}")

    # 按 category
    per_cat = eval_result["per_category"]
    cats = sorted(per_cat.keys())
    if cats:
        print()
        print(f"  {'Category':>10} {'Safe':>6} {'Unsafe':>6} {'AUC':>8}")
        print(f"  {'-' * 34}")
        for cat in cats:
            r = per_cat[cat]
            auc_str = f"{r['auc_roc']:.4f}" if r.get("auc_roc") is not None else "N/A"
            print(f"  {cat:>10} {r['n_safe']:>6} {r['n_unsafe']:>6} {auc_str:>8}")

    print("=" * 70)
    print()


# ============================================================
# Main
# ============================================================

def _format_job_label(job_dirs: list[Path]) -> str:
    """生成 job 列表的简短标签。"""
    if len(job_dirs) == 1:
        return job_dirs[0].name
    return f"{len(job_dirs)} jobs"


def main():
    parser = argparse.ArgumentParser(
        description="Refusal Direction (Difference in Means) Hidden States 分类器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python refusal_direction.py 236                          # 单 job
  python refusal_direction.py 236 240                      # 多 job 合并
  python refusal_direction.py --batch 17                   # batch 中所有 job
  python refusal_direction.py 236 --batch 17               # job + batch 混合
  python refusal_direction.py 236 --test_job 237 238       # 多个测试 job
  python refusal_direction.py 236 --test_batch 17          # batch 作为测试集
  python refusal_direction.py 236 --n_folds 5
  python refusal_direction.py 236 --save_direction
  python refusal_direction.py 237 --load_direction output/refusal_dir_1/direction.npy
        """,
    )
    parser.add_argument("job_nums", type=int, nargs="*", default=[],
                        help="训练 job 编号（可多个）")
    parser.add_argument("--batch", type=int, nargs="+", default=None,
                        help="训练 batch 编号（自动展开为其中所有已完成 job）")
    parser.add_argument("--test_job", type=int, nargs="+", default=None,
                        help="测试 job 编号（可多个）")
    parser.add_argument("--test_batch", type=int, nargs="+", default=None,
                        help="测试 batch 编号（自动展开为其中所有已完成 job）")
    parser.add_argument("--sub_task", default="q0", help="子任务标识（默认: q0）")
    parser.add_argument("--turn", default="t0", help="轮次标识（默认: t0）")
    parser.add_argument("--split_ratio", type=float, default=0.7,
                        help="训练集比例（默认: 0.7）")
    parser.add_argument("--n_folds", type=int, default=None,
                        help="K-fold 折数（设置后忽略 split_ratio）")
    parser.add_argument("--save_direction", action="store_true",
                        help="保存方向向量")
    parser.add_argument("--load_direction", type=str, default=None,
                        help="加载已有方向向量（跳过训练）")
    parser.add_argument("--score_method", choices=["dot", "cosine"], default="dot",
                        help="投影方式（默认: dot）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（默认: 42）")

    args = parser.parse_args()

    # ---------- 解析训练 job 目录 ----------
    train_dirs = resolve_job_dirs(args.job_nums or None, args.batch)
    if not train_dirs and not args.load_direction:
        parser.error("必须指定至少一个训练 job（positional）或 --batch，或使用 --load_direction")

    # ---------- 解析测试 job 目录 ----------
    test_dirs = resolve_job_dirs(args.test_job, args.test_batch)
    has_test = len(test_dirs) > 0

    # ---------- 加载训练数据 ----------
    print(f"训练数据 ({len(train_dirs)} jobs):")
    print(f"  子任务: {args.sub_task}, 轮次: {args.turn}")
    samples, train_metas = load_multi_job(train_dirs, args.sub_task, args.turn)

    n_safe = sum(1 for s in samples if s["label"] == "safe")
    n_unsafe = len(samples) - n_safe
    print(f"  合计: {len(samples)} 样本 (safe={n_safe}, unsafe={n_unsafe})")

    if len(samples) == 0 and not args.load_direction:
        print("错误：没有配对样本", file=sys.stderr)
        sys.exit(1)

    # 模型名（取第一个有效 meta）
    model_name = "unknown"
    for m in train_metas:
        if m.get("model"):
            model_name = m["model"]
            break

    # ---------- 方向计算或加载 ----------
    direction = None
    if args.load_direction:
        direction = np.load(args.load_direction)
        print(f"已加载方向向量: {args.load_direction}")

    # ---------- 构建标题后缀 ----------
    train_label = ", ".join(str(n) for n in args.job_nums) if args.job_nums else ""
    if args.batch:
        batch_label = ", ".join(f"B{n}" for n in args.batch)
        train_label = f"{train_label}, {batch_label}" if train_label else batch_label
    title_suffix = f"\n{model_name} | Train: {train_label}"

    cv_info = None

    # ---------- 分支：跨 job / K-fold / 单次 split ----------
    if has_test:
        # 跨 job 测试模式
        print(f"\n测试数据 ({len(test_dirs)} jobs):")
        test_samples, test_metas = load_multi_job(test_dirs, args.sub_task, args.turn)
        print(f"  合计: {len(test_samples)} 样本")

        if len(test_samples) == 0:
            print("错误：测试集没有配对样本", file=sys.stderr)
            sys.exit(1)

        if direction is None:
            direction = compute_refusal_direction(samples)

        scores = score_samples(test_samples, direction, method=args.score_method)
        eval_result = evaluate(test_samples, scores)
        eval_samples = test_samples

        test_label = ", ".join(str(n) for n in (args.test_job or []))
        if args.test_batch:
            tb = ", ".join(f"B{n}" for n in args.test_batch)
            test_label = f"{test_label}, {tb}" if test_label else tb
        title_suffix += f" → Test: {test_label}"

    elif args.n_folds is not None:
        # K-fold 交叉验证
        print(f"\n{args.n_folds}-fold 交叉验证:")
        if direction is not None:
            print("警告：K-fold 模式下 --load_direction 被忽略（每 fold 重新训练）")

        eval_result, scores = evaluate_kfold(
            samples, compute_refusal_direction, args.score_method,
            args.n_folds, args.seed
        )
        cv_info = eval_result.get("cv")
        eval_samples = samples

        direction = compute_refusal_direction(samples)
        title_suffix += f" | {args.n_folds}-fold CV"

    else:
        # 单次 train/test split
        rng = np.random.default_rng(args.seed)
        label_arr = np.array([s["label"] for s in samples])

        safe_idx = np.where(label_arr == "safe")[0]
        unsafe_idx = np.where(label_arr == "unsafe")[0]
        rng.shuffle(safe_idx)
        rng.shuffle(unsafe_idx)

        n_train_safe = max(1, int(len(safe_idx) * args.split_ratio))
        n_train_unsafe = max(1, int(len(unsafe_idx) * args.split_ratio))

        train_idx = np.concatenate([safe_idx[:n_train_safe], unsafe_idx[:n_train_unsafe]])
        test_idx = np.concatenate([safe_idx[n_train_safe:], unsafe_idx[n_train_unsafe:]])

        if len(test_idx) == 0:
            print("错误：测试集为空，请增加样本数或降低 split_ratio", file=sys.stderr)
            sys.exit(1)

        train_samples = [samples[i] for i in train_idx]
        test_samples = [samples[i] for i in test_idx]
        print(f"\n  训练集: {len(train_samples)}, 测试集: {len(test_samples)}")

        if direction is None:
            direction = compute_refusal_direction(train_samples)

        scores = score_samples(test_samples, direction, method=args.score_method)
        eval_result = evaluate(test_samples, scores)
        eval_samples = test_samples
        title_suffix += f" | split={args.split_ratio}"

    # ---------- 输出 ----------
    print_summary(eval_result, cv_info)

    refdir_num = get_next_refdir_num()
    out_dir = create_refdir_dir(refdir_num)
    print(f"输出目录: {out_dir.name}")

    print("生成图表:")
    plot_roc_curve(eval_result, out_dir, title_suffix=title_suffix)

    threshold = eval_result["overall"].get("optimal_threshold")
    plot_score_distribution(eval_samples, scores, threshold, out_dir, title_suffix=title_suffix)
    plot_category_auc(eval_result, out_dir, title_suffix=title_suffix)

    pca_data = samples if samples else eval_samples
    pca_result = pca_analyze(pca_data, direction)
    plot_pca_scatter(pca_result, out_dir, title_suffix=title_suffix)
    plot_pca_variance(pca_result, out_dir, title_suffix=title_suffix)

    if args.save_direction and direction is not None:
        np.save(out_dir / "direction.npy", direction)
        print(f"  方向向量: {out_dir / 'direction.npy'}")

    # summary.json
    from datetime import datetime
    summary = {
        "refdir_num": refdir_num,
        "timestamp": datetime.now().isoformat(),
        "train_jobs": [{"dir": d.name, "model": m.get("model", "unknown")}
                       for d, m in zip(train_dirs, train_metas)],
        "train_batches": args.batch,
        "test_jobs": [{"dir": d.name} for d in test_dirs] if has_test else None,
        "test_batches": args.test_batch,
        "params": {
            "sub_task": args.sub_task,
            "turn": args.turn,
            "split_ratio": args.split_ratio,
            "n_folds": args.n_folds,
            "score_method": args.score_method,
            "seed": args.seed,
            "layer": train_metas[0].get("layer", -1) if train_metas else -1,
        },
        "data_stats": {
            "total": len(samples),
            "safe": n_safe,
            "unsafe": n_unsafe,
        },
        "overall": {k: v for k, v in eval_result["overall"].items() if k != "roc_curve"},
        "per_category": eval_result["per_category"],
    }

    if cv_info:
        summary["cv"] = cv_info

    if pca_result:
        summary["pca"] = {
            "explained_ratio": pca_result["explained_ratio"],
            "refusal_dir_in_pc": pca_result["dir_in_pc"],
        }

    if args.save_direction:
        summary["direction_file"] = str(out_dir / "direction.npy")

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  Summary: {out_dir / 'summary.json'}")

    generate_report_html(summary, out_dir)


if __name__ == "__main__":
    main()
