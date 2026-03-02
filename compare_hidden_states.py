#!/usr/bin/env python3
"""
对比不同 job 的 Hidden States 差异方向。

用法：
    python compare_hidden_states.py 177 178
    python compare_hidden_states.py 177 178 --sub_task q1 --turn t0
    python compare_hidden_states.py 177 178 --sub_task1 q1 --turn1 t0 --sub_task2 q2 --turn2 t0
    python compare_hidden_states.py 177 178 --no-detailed
"""

import numpy as np
from pathlib import Path
from collections import defaultdict
from itertools import combinations
import argparse
import json
import sys

# 可视化按需导入
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
COMP_COUNTER_FILE = OUTPUT_ROOT / ".comp_counter"


def resolve_job_dir(job_num: int, output_root: Path = None) -> Path:
    """根据 job number 在 output/ 下查找 job_{num}_* 目录。"""
    if output_root is None:
        output_root = OUTPUT_ROOT
    matches = sorted(output_root.glob(f"job_{job_num}_*"))
    # 排除文件，只保留目录
    matches = [m for m in matches if m.is_dir()]
    if len(matches) == 0:
        print(f"错误：在 {output_root} 下找不到 job_{job_num}_* 目录", file=sys.stderr)
        sys.exit(1)
    if len(matches) > 1:
        print(f"错误：在 {output_root} 下找到多个匹配 job_{job_num}_* 的目录：", file=sys.stderr)
        for m in matches:
            print(f"  {m.name}", file=sys.stderr)
        sys.exit(1)
    return matches[0]


def get_next_comp_num() -> int:
    """获取下一个 comparison 编号（单调递增，从 1 开始）。"""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if COMP_COUNTER_FILE.exists():
        try:
            current = int(COMP_COUNTER_FILE.read_text().strip())
        except (ValueError, IOError):
            current = 0
    else:
        current = 0
    next_num = current + 1
    COMP_COUNTER_FILE.write_text(str(next_num))
    return next_num


def create_comp_dir(comp_num: int) -> Path:
    """创建 hidden_state_comp_{num}/ 目录。"""
    comp_dir = OUTPUT_ROOT / f"hidden_state_comp_{comp_num}"
    comp_dir.mkdir(parents=True, exist_ok=True)
    return comp_dir


def write_summary(comp_dir: Path, args, dir1: Path, dir2: Path, matches: dict,
                  d_by_cat: dict, results: dict, baseline_stats: dict = None):
    """写入 summary.json，记录本次 comparison 的元信息。"""
    from datetime import datetime
    summary = {
        "comp_num": int(comp_dir.name.split("_")[-1]),
        "timestamp": datetime.now().isoformat(),
        "job1": {"num": args.job1, "dir": dir1.name, "sub_task": args.sub_task1, "turn": args.turn1},
        "job2": {"num": args.job2, "dir": dir2.name, "sub_task": args.sub_task2, "turn": args.turn2},
        "params": {"detailed": args.detailed},
        "matched_tasks": len(matches),
        "categories": sorted(d_by_cat.keys()),
        "results": {},
    }
    for cat in sorted(results.keys()):
        r = dict(results[cat])
        r.pop("_vectors", None)
        summary["results"][cat] = r
    if baseline_stats:
        summary["cross_category_baseline"] = baseline_stats
    with open(comp_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Summary 已保存: {comp_dir / 'summary.json'}")


# ============================================================
# 数据加载
# ============================================================

def find_matching_files(dir1: Path, dir2: Path,
                        sub_task1: str = "q1", turn1: str = "t0",
                        sub_task2: str = None, turn2: str = None):
    """找到两个目录中共同 (cat, index) 的 hidden state 文件。

    Job1 使用 {sub_task1}_{turn1}，Job2 使用 {sub_task2}_{turn2}（默认与 Job1 相同）。
    返回 {(cat, index): (path1, path2)}
    """
    if sub_task2 is None:
        sub_task2 = sub_task1
    if turn2 is None:
        turn2 = turn1

    hs1 = dir1 / "hidden_states"
    hs2 = dir2 / "hidden_states"

    if not hs1.is_dir():
        print(f"错误：{hs1} 不存在", file=sys.stderr)
        sys.exit(1)
    if not hs2.is_dir():
        print(f"错误：{hs2} 不存在", file=sys.stderr)
        sys.exit(1)

    def collect_files(hs_dir: Path, sub_task: str, turn: str) -> dict:
        suffix = f"_{sub_task}_{turn}.npy"
        files = {}
        for f in hs_dir.glob(f"*{suffix}"):
            stem = f.stem
            prefix = stem.removesuffix(f"_{sub_task}_{turn}")
            parts = prefix.split("_", 1)
            if len(parts) == 2:
                cat, index = parts
                files[(cat, index)] = f
        return files

    files1 = collect_files(hs1, sub_task1, turn1)
    files2 = collect_files(hs2, sub_task2, turn2)

    # 求交集
    common_keys = set(files1.keys()) & set(files2.keys())
    if not common_keys:
        print("错误：两个 job 之间没有共同的 hidden state 文件", file=sys.stderr)
        sys.exit(1)

    matches = {k: (files1[k], files2[k]) for k in sorted(common_keys)}
    return matches


def compute_differences(matches: dict):
    """d = h1 - h2，按 category 分组。

    返回 {cat: [(index, d_vector), ...]}
    """
    d_by_cat = defaultdict(list)
    for (cat, index), (path1, path2) in matches.items():
        h1 = np.load(path1)
        h2 = np.load(path2)
        d = h1 - h2
        d_by_cat[cat].append((index, d))
    return dict(d_by_cat)


# ============================================================
# 分析函数
# ============================================================

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """两个向量的余弦相似度。"""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / (norm_a * norm_b))


def pairwise_cosine_similarity(vectors: list[np.ndarray]) -> list[float]:
    """N 个向量的 pairwise cosine similarity，返回上三角的值列表。"""
    sims = []
    for i, j in combinations(range(len(vectors)), 2):
        sims.append(cosine_sim(vectors[i], vectors[j]))
    return sims


def mean_direction_alignment(vectors: list[np.ndarray]) -> dict:
    """计算均值方向，返回每个向量与均值方向的 cosine similarity。"""
    mean_vec = np.mean(vectors, axis=0)
    mean_norm = np.linalg.norm(mean_vec)
    if mean_norm < 1e-12:
        return {"mean_alignment": float("nan"), "alignments": [float("nan")] * len(vectors)}
    mean_dir = mean_vec / mean_norm
    alignments = [cosine_sim(v, mean_dir) for v in vectors]
    return {
        "mean_alignment": float(np.nanmean(alignments)),
        "alignments": alignments,
    }


def pca_analysis(vectors: list[np.ndarray], n_components: int = 3) -> list[float]:
    """PCA 分析，返回 top-k explained variance ratios。"""
    if len(vectors) < 2:
        return []
    mat = np.stack(vectors)
    # 中心化
    mat = mat - mat.mean(axis=0)
    # SVD
    try:
        _, s, _ = np.linalg.svd(mat, full_matrices=False)
    except np.linalg.LinAlgError:
        return []
    var = s ** 2
    total_var = var.sum()
    if total_var < 1e-12:
        return [0.0] * min(n_components, len(var))
    ratios = (var / total_var).tolist()
    return ratios[:n_components]


def cross_category_baseline(d_by_cat: dict) -> list[float]:
    """跨 category 的 pairwise cosine similarity 作为 baseline。"""
    # 从不同 category 各取一个向量进行 pairwise
    cats = list(d_by_cat.keys())
    sims = []
    for ci, cj in combinations(range(len(cats)), 2):
        vecs_i = [v for _, v in d_by_cat[cats[ci]]]
        vecs_j = [v for _, v in d_by_cat[cats[cj]]]
        for vi in vecs_i:
            for vj in vecs_j:
                sims.append(cosine_sim(vi, vj))
    return sims


# ============================================================
# 单 category 分析
# ============================================================

def analyze_category(items: list[tuple], detailed: bool = False) -> dict:
    """对一个 category 做完整分析。

    items: [(index, d_vector), ...]

    主要指标：每个 task 的 d 向量与 category 均值方向的 cosine similarity（N 个值）。
    """
    indices = [idx for idx, _ in items]
    vectors = [v for _, v in items]
    n = len(vectors)

    result = {"n_tasks": n, "indices": indices}

    # 核心指标：均值方向对齐度（每个 task 一个值）
    if n >= 2:
        alignment = mean_direction_alignment(vectors)
        result["cos_sim"] = {
            "values": alignment["alignments"],
            "mean": float(np.nanmean(alignment["alignments"])),
            "std": float(np.nanstd(alignment["alignments"])),
            "min": float(np.nanmin(alignment["alignments"])),
            "max": float(np.nanmax(alignment["alignments"])),
        }
    elif n == 1:
        result["cos_sim"] = None
    else:
        result["cos_sim"] = None

    # Pairwise cosine similarity (detailed only)
    if detailed and n >= 2:
        sims = pairwise_cosine_similarity(vectors)
        valid_sims = [s for s in sims if not np.isnan(s)]
        if valid_sims:
            result["pairwise_cos"] = {
                "mean": float(np.mean(valid_sims)),
                "std": float(np.std(valid_sims)),
                "min": float(np.min(valid_sims)),
                "max": float(np.max(valid_sims)),
                "n_pairs": len(valid_sims),
            }
        else:
            result["pairwise_cos"] = None

    # PCA (detailed only)
    if detailed and n >= 2:
        result["pca_ratios"] = pca_analysis(vectors)

    return result


# ============================================================
# 输出
# ============================================================

def print_summary_table(results: dict, detailed: bool = False, baseline_stats: dict = None):
    """打印终端表格。"""
    print()
    print("=" * 90)
    print("Hidden States 差异方向分析")
    print("=" * 90)
    print("每个 task 的 d 向量与 category 均值方向的 cosine similarity")
    print()

    # 表头
    if detailed:
        header = f"{'Category':>10} {'Tasks':>5} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'PCA-1':>7} {'PCA-2':>7}"
    else:
        header = f"{'Category':>10} {'Tasks':>5} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}"
    print(header)
    print("-" * len(header))

    for cat in sorted(results.keys()):
        r = results[cat]
        n = r["n_tasks"]

        if r["cos_sim"]:
            cs = r["cos_sim"]
            cs_mean = f"{cs['mean']:.4f}"
            cs_std = f"{cs['std']:.4f}"
            cs_min = f"{cs['min']:.4f}"
            cs_max = f"{cs['max']:.4f}"
        else:
            cs_mean = cs_std = cs_min = cs_max = "N/A"

        row = f"{cat:>10} {n:>5} {cs_mean:>8} {cs_std:>8} {cs_min:>8} {cs_max:>8}"

        if detailed and "pca_ratios" in r and r["pca_ratios"]:
            pca = r["pca_ratios"]
            pca1 = f"{pca[0]:.4f}" if len(pca) > 0 else "N/A"
            pca2 = f"{pca[1]:.4f}" if len(pca) > 1 else "N/A"
            row += f" {pca1:>7} {pca2:>7}"

        print(row)

    # Baseline
    if baseline_stats:
        print("-" * len(header))
        bm = baseline_stats["mean"]
        bs = baseline_stats["std"]
        print(f"{'Cross-Cat':>10} {'':>5} {bm:>8.4f} {bs:>8.4f}")

    print("=" * len(header))

    # 逐 task 明细
    print()
    print("逐 task 明细:")
    for cat in sorted(results.keys()):
        r = results[cat]
        if r["cos_sim"] and r["cos_sim"]["values"]:
            print(f"  Category {cat}:")
            for idx, val in zip(r["indices"], r["cos_sim"]["values"]):
                v = f"{val:.4f}" if not np.isnan(val) else "N/A"
                print(f"    {cat}_{idx}: {v}")
    print()


def plot_results(results: dict, output_path: Path, detailed: bool = True, baseline_sims: list = None):
    """生成综合分析图（单张图包含 boxplot + 统计表格）。"""
    plt = _import_matplotlib()
    from matplotlib.gridspec import GridSpec

    cats = sorted(results.keys())

    # 收集数据
    box_data = []
    labels = []
    table_rows = []

    for cat in cats:
        r = results[cat]
        if not (r["cos_sim"] and r["cos_sim"]["values"]):
            continue
        valid = [s for s in r["cos_sim"]["values"] if not np.isnan(s)]
        if not valid:
            continue
        box_data.append(valid)
        labels.append(cat)

        cs = r["cos_sim"]
        row = [cat, str(r["n_tasks"]),
               f'{cs["mean"]:.3f} ± {cs["std"]:.3f}',
               f'[{cs["min"]:.3f}, {cs["max"]:.3f}]']

        if "pairwise_cos" in r and r["pairwise_cos"]:
            pc = r["pairwise_cos"]
            row.append(f'{pc["mean"]:.3f} ± {pc["std"]:.3f}')
        else:
            row.append("—")

        if "pca_ratios" in r and r["pca_ratios"]:
            pca = r["pca_ratios"]
            row.append(" / ".join(f"{v:.1%}" for v in pca[:3]))
        else:
            row.append("—")

        table_rows.append(row)

    if not box_data:
        return

    # Baseline 行
    if baseline_sims:
        valid_bl = [s for s in baseline_sims if not np.isnan(s)]
        if valid_bl:
            bl_mean, bl_std = np.mean(valid_bl), np.std(valid_bl)
            table_rows.append(["Cross-Cat", "", f"{bl_mean:.3f} ± {bl_std:.3f}", "", "", ""])

    # 布局：上方 boxplot，下方表格
    n_cats = len(labels)
    n_table_rows = len(table_rows) + 1  # +1 表头
    fig_w = max(10, n_cats * 1.8)
    fig_h = 5.5 + n_table_rows * 0.4
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[3.5, max(1, n_table_rows * 0.45)], hspace=0.15)
    ax = fig.add_subplot(gs[0])
    ax_tbl = fig.add_subplot(gs[1])

    # --- 箱线图 + 散点 ---
    bp = ax.boxplot(box_data, tick_labels=labels, patch_artist=True, widths=0.5,
                    showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="#FF6B6B",
                                   markeredgecolor="#FF6B6B", markersize=5))
    for patch in bp["boxes"]:
        patch.set_facecolor("#4ECDC4")
        patch.set_alpha(0.7)

    # 叠加散点显示每个 task
    rng = np.random.default_rng(42)
    for i, data in enumerate(box_data):
        x = rng.normal(i + 1, 0.04, size=len(data))
        ax.scatter(x, data, alpha=0.35, s=10, color="#2C3E50", zorder=3)

    ax.set_ylabel("Cosine Similarity (d_i · mean_dir)")
    ax.set_xlabel("Category")
    ax.set_title("Hidden States 差异方向分析")

    # Baseline 线
    if baseline_sims:
        valid_bl = [s for s in baseline_sims if not np.isnan(s)]
        if valid_bl:
            ax.axhline(y=np.mean(valid_bl), color="#FF6B6B", linestyle="--", alpha=0.7,
                       label=f"Cross-Cat baseline ({np.mean(valid_bl):.3f})")
            ax.legend(loc="lower left", fontsize=8)

    # --- 表格 ---
    ax_tbl.axis("off")
    col_labels = ["Category", "Tasks", "Align Mean±Std", "Min / Max", "Pairwise Cos", "PCA (1 / 2 / 3)"]
    tbl = ax_tbl.table(cellText=table_rows, colLabels=col_labels,
                       loc="center", cellLoc="center",
                       colWidths=[0.10, 0.07, 0.18, 0.16, 0.18, 0.22])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.4)

    # 表头样式
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#4ECDC4")
        tbl[0, j].set_text_props(weight="bold")
        tbl[0, j].set_alpha(0.4)

    # Cross-Cat 行高亮
    if baseline_sims and valid_bl:
        last_row = len(table_rows)
        for j in range(len(col_labels)):
            tbl[last_row, j].set_facecolor("#FFE0E0")

    fig_path = output_path / "hs_diff_summary.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"综合分析图已保存: {fig_path}")
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="对比不同 job 的 Hidden States 差异方向",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python compare_hidden_states.py 177 178
  python compare_hidden_states.py 177 178 --sub_task q1 --turn t0
  python compare_hidden_states.py 177 178 --sub_task1 q1 --turn1 t0 --sub_task2 q2 --turn2 t0
  python compare_hidden_states.py 177 178 --no-detailed
        """,
    )
    parser.add_argument("job1", type=int, help="第一个 job number")
    parser.add_argument("job2", type=int, help="第二个 job number")
    parser.add_argument("--sub_task", default="q1",
                        help="两个 job 共用的子任务标识（默认: q1），被 --sub_task1/2 覆盖")
    parser.add_argument("--turn", default="t0",
                        help="两个 job 共用的轮次标识（默认: t0），被 --turn1/2 覆盖")
    parser.add_argument("--sub_task1", default=None, help="Job1 的子任务标识（覆盖 --sub_task）")
    parser.add_argument("--sub_task2", default=None, help="Job2 的子任务标识（覆盖 --sub_task）")
    parser.add_argument("--turn1", default=None, help="Job1 的轮次标识（覆盖 --turn）")
    parser.add_argument("--turn2", default=None, help="Job2 的轮次标识（覆盖 --turn）")
    parser.add_argument("--detailed", action=argparse.BooleanOptionalAction, default=True,
                        help="PCA 分析 + 跨 category baseline（默认开启，--no-detailed 关闭）")
    args = parser.parse_args()

    # 合并共用 / 独立参数
    args.sub_task1 = args.sub_task1 or args.sub_task
    args.sub_task2 = args.sub_task2 or args.sub_task
    args.turn1 = args.turn1 or args.turn
    args.turn2 = args.turn2 or args.turn

    # 解析 job 目录
    dir1 = resolve_job_dir(args.job1)
    dir2 = resolve_job_dir(args.job2)
    print(f"Job {args.job1}: {dir1.name}  [{args.sub_task1}_{args.turn1}]")
    print(f"Job {args.job2}: {dir2.name}  [{args.sub_task2}_{args.turn2}]")

    # 找匹配文件
    matches = find_matching_files(dir1, dir2, args.sub_task1, args.turn1, args.sub_task2, args.turn2)
    print(f"共找到 {len(matches)} 个匹配的 task")

    # 计算差异
    d_by_cat = compute_differences(matches)
    print(f"覆盖 {len(d_by_cat)} 个 category: {', '.join(sorted(d_by_cat.keys()))}")

    # 分析每个 category
    results = {}
    for cat in sorted(d_by_cat.keys()):
        items = d_by_cat[cat]
        r = analyze_category(items, detailed=args.detailed)
        # 保留向量用于可视化
        r["_vectors"] = [v for _, v in items]
        results[cat] = r

    # 跨 category baseline
    baseline_sims = None
    baseline_stats = None
    if args.detailed and len(d_by_cat) >= 2:
        baseline_sims = cross_category_baseline(d_by_cat)
        valid = [s for s in baseline_sims if not np.isnan(s)]
        if valid:
            baseline_stats = {
                "mean": float(np.mean(valid)),
                "std": float(np.std(valid)),
            }

    # 创建输出目录
    comp_num = get_next_comp_num()
    comp_dir = create_comp_dir(comp_num)
    print(f"输出目录: {comp_dir.name}")

    # 输出
    print_summary_table(results, detailed=args.detailed, baseline_stats=baseline_stats)

    # 可视化
    plot_results(results, comp_dir, detailed=args.detailed, baseline_sims=baseline_sims)

    # Summary
    write_summary(comp_dir, args, dir1, dir2, matches, d_by_cat, results, baseline_stats)

    # 清理内部字段
    for r in results.values():
        r.pop("_vectors", None)


if __name__ == "__main__":
    main()
