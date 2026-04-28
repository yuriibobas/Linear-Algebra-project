import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "artifacts" / "presentation_plots"


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save(fig: plt.Figure, name: str):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_pareto_and_trials(dataset_data: dict):
    all_trials = pd.DataFrame(dataset_data["all_trials"])
    pareto = pd.DataFrame(
        [
            {
                "trial": x["trial_id"],
                "f1": x["scores"]["f1"],
                "mean_iou": x["scores"]["mean_iou"],
            }
            for x in dataset_data["pareto_front"]
        ]
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(
        all_trials["f1"],
        all_trials["mean_iou"],
        s=55,
        alpha=0.5,
        label="All trials",
    )
    ax.scatter(
        pareto["f1"],
        pareto["mean_iou"],
        s=110,
        marker="*",
        color="#d62728",
        label="Pareto front",
    )
    for _, row in pareto.iterrows():
        ax.annotate(
            f"T{int(row['trial'])}",
            (row["f1"], row["mean_iou"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=9,
        )

    ax.set_title("Detection Tuning: F1 vs Mean IoU (Pareto Front)")
    ax.set_xlabel("F1 score")
    ax.set_ylabel("Mean IoU")
    ax.grid(alpha=0.25)
    ax.legend()
    _save(fig, "01_pareto_f1_vs_iou.png")


def plot_metric_distributions(dataset_data: dict):
    all_trials = pd.DataFrame(dataset_data["all_trials"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.histplot(all_trials["f1"], bins=12, kde=True, ax=axes[0], color="#1f77b4")
    axes[0].axvline(all_trials["f1"].max(), color="#d62728", linestyle="--", linewidth=2)
    axes[0].set_title("Distribution of F1 Across Trials")
    axes[0].set_xlabel("F1 score")

    sns.histplot(all_trials["mean_iou"], bins=12, kde=True, ax=axes[1], color="#2ca02c")
    axes[1].axvline(
        all_trials["mean_iou"].max(), color="#d62728", linestyle="--", linewidth=2
    )
    axes[1].set_title("Distribution of Mean IoU Across Trials")
    axes[1].set_xlabel("Mean IoU")

    for ax in axes:
        ax.grid(alpha=0.2)

    fig.suptitle("Quality Spread in 50 Bayesian Optimization Trials", y=1.03)
    _save(fig, "02_metric_distributions.png")


def plot_preprocessing_impact(dataset_data: dict):
    all_trials = pd.DataFrame(dataset_data["all_trials"])
    grouped = (
        all_trials.groupby("use_preprocessing")[["f1", "mean_iou", "precision", "recall"]]
        .mean()
        .reset_index()
    )
    grouped["use_preprocessing"] = grouped["use_preprocessing"].map(
        {True: "With preprocessing", False: "Without preprocessing"}
    )
    melted = grouped.melt(
        id_vars="use_preprocessing", var_name="metric", value_name="value"
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=melted, x="metric", y="value", hue="use_preprocessing", ax=ax)
    ax.set_title("Average Metrics: Effect of Image Preprocessing")
    ax.set_xlabel("")
    ax.set_ylabel("Average value")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(title="")
    _save(fig, "03_preprocessing_impact.png")


def plot_top_full_pipeline_scores(full_data: dict):
    top = pd.DataFrame(full_data["top20"]).copy()
    top = top.sort_values("total_score", ascending=False).head(10).reset_index(drop=True)
    top["rank"] = [f"#{i + 1}" for i in range(len(top))]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=top, y="rank", x="total_score", ax=ax, color="#4c78a8")
    ax.set_title("Top-10 Full Pipeline Configurations by Total Score")
    ax.set_xlabel("Total score")
    ax.set_ylabel("Rank")
    ax.grid(axis="x", alpha=0.2)

    for i, v in enumerate(top["total_score"]):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)

    _save(fig, "04_top10_full_pipeline_total_score.png")


def plot_best_components_breakdown(full_data: dict):
    best = full_data["best"]
    components = {
        "Detection score": best["det_score"],
        "Hole score": best["hole_score"],
        "Continuity score": best["continuity_score"],
        "Texture score": best["texture_score"],
    }
    comp_df = pd.DataFrame(
        {"component": list(components.keys()), "value": list(components.values())}
    )

    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.barplot(
        data=comp_df,
        x="component",
        y="value",
        hue="component",
        dodge=False,
        legend=False,
        ax=ax,
        palette="viridis",
    )
    ax.set_title("Best Configuration: Score Components")
    ax.set_xlabel("")
    ax.set_ylabel("Value")
    ax.grid(axis="y", alpha=0.2)
    ax.tick_params(axis="x", rotation=10)

    for i, v in enumerate(comp_df["value"]):
        ax.text(i, v + max(comp_df["value"]) * 0.02, f"{v:.3f}", ha="center", fontsize=9)

    _save(fig, "05_best_config_score_components.png")


def main():
    sns.set_theme(style="whitegrid", context="talk")

    dataset_path = ROOT / "dataset_tuning_results_optuna.json"
    full_path = ROOT / "full_pipeline_tuning_results_quick.json"

    if not dataset_path.exists() or not full_path.exists():
        missing = [str(p.name) for p in [dataset_path, full_path] if not p.exists()]
        raise FileNotFoundError(f"Missing data files: {', '.join(missing)}")

    dataset_data = _load_json(dataset_path)
    full_data = _load_json(full_path)

    plot_pareto_and_trials(dataset_data)
    plot_metric_distributions(dataset_data)
    plot_preprocessing_impact(dataset_data)
    plot_top_full_pipeline_scores(full_data)
    plot_best_components_breakdown(full_data)

    print(f"Saved plots to: {OUT_DIR}")


if __name__ == "__main__":
    main()
