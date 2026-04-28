import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parent.parent

PARAM_COLS = [
    "scale",
    "score_threshold",
    "nms_threshold",
    "hit_threshold",
    "shrink_factor",
    "grabcut_iters",
]
METRIC_COLS = [
    "det_score",
    "hole_score",
    "continuity",
    "texture",
    "heuristic_total",
    "hole_ratio",
    "boundary_mae",
]


def _load_trials(json_path: Path) -> pd.DataFrame:
    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    rows = payload.get("all_trials", [])
    if not rows:
        raise ValueError("Input JSON does not contain non-empty 'all_trials'.")
    df = pd.DataFrame(rows)
    if "det_metrics" in df.columns:
        det_df = pd.json_normalize(df["det_metrics"]).add_prefix("det_metrics.")
        df = pd.concat([df.drop(columns=["det_metrics"]), det_df], axis=1)
    if "heuristic_total" not in df.columns:
        hole_score = 1.0 - df.get("hole_ratio", pd.Series([0.0] * len(df))) * 5.0
        hole_score = np.clip(hole_score.astype(float), 0.0, 1.0)
        df["heuristic_total"] = (
            0.40 * df.get("det_score", 0.0).astype(float)
            + 0.25 * hole_score
            + 0.22 * df.get("continuity", 0.0).astype(float)
            + 0.13 * df.get("texture", 0.0).astype(float)
        )
    return df


def _save(fig: plt.Figure, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _safe_corr(a: pd.Series, b: pd.Series, method: str) -> float:
    valid = pd.concat([a, b], axis=1).dropna()
    if len(valid) < 3:
        return np.nan
    if valid.iloc[:, 0].nunique() <= 1 or valid.iloc[:, 1].nunique() <= 1:
        return np.nan
    return float(valid.iloc[:, 0].corr(valid.iloc[:, 1], method=method))


def _param_metric_correlation(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for p in PARAM_COLS:
        if p not in df.columns:
            continue
        for m in METRIC_COLS:
            if m not in df.columns:
                continue
            rows.append(
                {
                    "parameter": p,
                    "metric": m,
                    "pearson": _safe_corr(df[p], df[m], method="pearson"),
                    "spearman": _safe_corr(df[p], df[m], method="spearman"),
                }
            )
    out = pd.DataFrame(rows)
    out["abs_spearman"] = out["spearman"].abs()
    return out.sort_values("abs_spearman", ascending=False).reset_index(drop=True)


def plot_metric_param_heatmaps(df: pd.DataFrame, out_dir: Path):
    corr = _param_metric_correlation(df)
    corr.to_csv(out_dir / "param_metric_correlations.csv", index=False)

    pearson = corr.pivot(index="metric", columns="parameter", values="pearson")
    fig = plt.figure(figsize=(10, 5.5))
    sns.heatmap(pearson, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Pearson Correlation: Parameters -> Metrics")
    _save(fig, out_dir / "01_pearson_param_metric_heatmap.png")

    spearman = corr.pivot(index="metric", columns="parameter", values="spearman")
    fig = plt.figure(figsize=(10, 5.5))
    sns.heatmap(spearman, annot=True, fmt=".2f", cmap="vlag", center=0)
    plt.title("Spearman Correlation: Parameters -> Metrics")
    _save(fig, out_dir / "02_spearman_param_metric_heatmap.png")

    return corr


def plot_top_relationships(df: pd.DataFrame, corr_df: pd.DataFrame, out_dir: Path, top_k: int = 6):
    top = corr_df.dropna(subset=["spearman"]).head(top_k)
    if top.empty:
        return
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    for ax, (_, row) in zip(axes, top.iterrows()):
        p = row["parameter"]
        m = row["metric"]
        sns.regplot(
            data=df,
            x=p,
            y=m,
            scatter_kws={"alpha": 0.65, "s": 45},
            line_kws={"color": "#d62728"},
            ax=ax,
        )
        ax.set_title(f"{p} -> {m}\nr_s={row['spearman']:.2f}")
    for ax in axes[len(top) :]:
        ax.axis("off")
    fig.suptitle("Top Dependencies (by |Spearman|)", y=1.02)
    _save(fig, out_dir / "03_top_parameter_metric_relationships.png")


def plot_categorical_effects(df: pd.DataFrame, out_dir: Path):
    if "use_preprocessing" in df.columns:
        melted = df.melt(
            id_vars="use_preprocessing",
            value_vars=[m for m in ["det_score", "continuity", "texture", "heuristic_total"] if m in df.columns],
            var_name="metric",
            value_name="value",
        )
        fig = plt.figure(figsize=(11, 6))
        sns.boxplot(data=melted, x="metric", y="value", hue="use_preprocessing")
        plt.title("Metric Distribution by Preprocessing Usage")
        _save(fig, out_dir / "04_preprocessing_box_effects.png")

    if "win_stride" in df.columns:
        work = df.copy()
        work["win_stride_str"] = work["win_stride"].apply(
            lambda x: f"{x[0]}x{x[1]}" if isinstance(x, list) and len(x) == 2 else str(x)
        )
        fig = plt.figure(figsize=(8, 5))
        sns.boxplot(data=work, x="win_stride_str", y="heuristic_total")
        plt.title("Heuristic Total by Win Stride")
        plt.xlabel("win_stride")
        _save(fig, out_dir / "05_win_stride_vs_total_box.png")

    if "padding" in df.columns:
        work = df.copy()
        work["padding_str"] = work["padding"].apply(
            lambda x: f"{x[0]}x{x[1]}" if isinstance(x, list) and len(x) == 2 else str(x)
        )
        fig = plt.figure(figsize=(8, 5))
        sns.boxplot(data=work, x="padding_str", y="det_score")
        plt.title("Detection Score by Padding")
        plt.xlabel("padding")
        _save(fig, out_dir / "06_padding_vs_det_box.png")


def plot_interactions(df: pd.DataFrame, out_dir: Path):
    if {"score_threshold", "nms_threshold", "heuristic_total"} <= set(df.columns):
        pivot = df.pivot_table(
            index="score_threshold",
            columns="nms_threshold",
            values="heuristic_total",
            aggfunc="mean",
        )
        fig = plt.figure(figsize=(9, 6))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu")
        plt.title("Interaction: score_threshold x nms_threshold -> heuristic_total")
        _save(fig, out_dir / "07_interaction_score_nms_total.png")

    if {"grabcut_iters", "shrink_factor", "continuity"} <= set(df.columns):
        pivot = df.pivot_table(
            index="grabcut_iters",
            columns="shrink_factor",
            values="continuity",
            aggfunc="mean",
        )
        fig = plt.figure(figsize=(9, 6))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="magma")
        plt.title("Interaction: grabcut_iters x shrink_factor -> continuity")
        _save(fig, out_dir / "08_interaction_grabcut_shrink_continuity.png")


def build_report(df: pd.DataFrame, corr_df: pd.DataFrame, out_dir: Path):
    best = df.sort_values("heuristic_total", ascending=False).head(5).copy()
    keep_cols = [c for c in ["trial", "heuristic_total", "det_score", "continuity", "texture", "scale", "score_threshold", "nms_threshold", "grabcut_iters", "use_preprocessing"] if c in best.columns]
    best = best[keep_cols]
    best.to_csv(out_dir / "top5_trials.csv", index=False)

    strongest = corr_df.dropna(subset=["spearman"]).head(10).copy()
    strongest.to_csv(out_dir / "top10_strongest_dependencies.csv", index=False)

    top5_table = best.to_string(index=False)

    lines = [
        "# Smart Tune Deep Analysis",
        "",
        f"- Number of trials: {len(df)}",
        f"- Mean heuristic_total: {df['heuristic_total'].mean():.4f}",
        f"- Best heuristic_total: {df['heuristic_total'].max():.4f}",
        "",
        "## Strongest parameter -> metric dependencies (Spearman)",
    ]
    for _, row in strongest.iterrows():
        lines.append(
            f"- `{row['parameter']}` -> `{row['metric']}`: "
            f"spearman={row['spearman']:.3f}, pearson={row['pearson']:.3f}"
        )
    lines.extend(["", "## Top-5 trials", "```", top5_table, "```", ""])

    report_path = out_dir / "analysis_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Deep dependency analysis for smart_tune_results.json."
    )
    parser.add_argument("--input", default="smart_tune_results.json", help="Input JSON path.")
    parser.add_argument("--out-dir", default="analytics_output_deep_smart_tune", help="Output directory.")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.is_absolute():
        in_path = ROOT / in_path
    if not in_path.exists():
        raise FileNotFoundError(f"Cannot find input JSON: {in_path}")

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", context="talk")
    df = _load_trials(in_path)
    df.to_csv(out_dir / "all_trials_flat.csv", index=False)

    corr_df = plot_metric_param_heatmaps(df, out_dir)
    plot_top_relationships(df, corr_df, out_dir)
    plot_categorical_effects(df, out_dir)
    plot_interactions(df, out_dir)
    build_report(df, corr_df, out_dir)

    print(f"Deep smart_tune analysis saved to: {out_dir}")


if __name__ == "__main__":
    main()
