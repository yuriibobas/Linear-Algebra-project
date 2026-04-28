import argparse
import csv
import json
import time
from itertools import product
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from detection import detect_humans, get_binary_masks
from tune_full_pipeline import parse_gt_boxes, score_config, swap_and_inpaint


DEFAULT_PRESETS = {
    "balanced": {
        "win_stride": [(8, 8), (4, 4)],
        "padding": [(8, 8), (16, 16)],
        "scale": [1.03, 1.05, 1.08],
        "score_threshold": [0.0, 0.1, 0.2],
        "nms_threshold": [0.2, 0.3, 0.4],
        "hit_threshold": [0.0, 0.2],
        "use_preprocessing": [True],
        "shrink_factor": [0.0, 0.05, 0.1],
        "grabcut_iters": [5, 7],
    },
    "detection_heavy": {
        "win_stride": [(4, 4)],
        "padding": [(16, 16), (32, 32)],
        "scale": [1.01, 1.03, 1.05],
        "score_threshold": [0.0, 0.05, 0.1],
        "nms_threshold": [0.25, 0.35],
        "hit_threshold": [0.0, 0.1],
        "use_preprocessing": [True, False],
        "shrink_factor": [0.0, 0.05],
        "grabcut_iters": [5, 7, 9],
    },
    "precision_heavy": {
        "win_stride": [(8, 8)],
        "padding": [(8, 8), (16, 16)],
        "scale": [1.05, 1.08, 1.1],
        "score_threshold": [0.2, 0.3, 0.4],
        "nms_threshold": [0.15, 0.2, 0.25],
        "hit_threshold": [0.1, 0.2, 0.3],
        "use_preprocessing": [True],
        "shrink_factor": [0.05, 0.1, 0.15],
        "grabcut_iters": [5, 7],
    },
    "inpaint_focus": {
        "win_stride": [(8, 8), (4, 4)],
        "padding": [(8, 8), (16, 16)],
        "scale": [1.03, 1.05],
        "score_threshold": [0.0, 0.1, 0.2],
        "nms_threshold": [0.25, 0.35],
        "hit_threshold": [0.0, 0.2],
        "use_preprocessing": [True],
        "shrink_factor": [0.08, 0.12, 0.16],
        "grabcut_iters": [7, 9, 11],
    },
}


def _require_analysis_libs():
    try:
        import pandas as pd
        import seaborn as sns
    except ImportError as e:
        raise ImportError(
            "This mode requires pandas and seaborn. Install dependencies: pip install -r requirements.txt"
        ) from e
    return pd, sns


def load_presets(path):
    if not path:
        return DEFAULT_PRESETS
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    presets = {}
    for name, cfg in raw.items():
        item = dict(cfg)
        item["win_stride"] = [tuple(v) for v in item["win_stride"]]
        item["padding"] = [tuple(v) for v in item["padding"]]
        presets[name] = item
    return presets


def expand_grid(grid):
    keys = [
        "win_stride",
        "padding",
        "scale",
        "score_threshold",
        "nms_threshold",
        "hit_threshold",
        "use_preprocessing",
        "shrink_factor",
        "grabcut_iters",
    ]
    for values in product(*(grid[k] for k in keys)):
        yield dict(zip(keys, values))


def maybe_sample(configs, max_configs, rng):
    if max_configs <= 0 or len(configs) <= max_configs:
        return configs
    idx = rng.choice(len(configs), size=max_configs, replace=False)
    return [configs[i] for i in idx]


def _evaluate_config(image, gt_boxes, cfg):
    pred_boxes = detect_humans(
        image,
        win_stride=cfg["win_stride"],
        padding=cfg["padding"],
        scale=cfg["scale"],
        score_threshold=cfg["score_threshold"],
        nms_threshold=cfg["nms_threshold"],
        hit_threshold=cfg["hit_threshold"],
        use_preprocessing=cfg["use_preprocessing"],
        shrink_factor=cfg["shrink_factor"],
    )
    if len(pred_boxes) < 2:
        return None
    masks_data = get_binary_masks(
        image,
        pred_boxes,
        top_k=2,
        grabcut_iters=cfg["grabcut_iters"],
    )
    if len(masks_data) < 2:
        return None
    _, final_image, holes = swap_and_inpaint(image, masks_data)
    scored = score_config(image, pred_boxes, gt_boxes, final_image, holes)
    det_metrics = scored.get("det_metrics") or {}
    return {
        "win_stride": f"{cfg['win_stride'][0]}x{cfg['win_stride'][1]}",
        "padding": f"{cfg['padding'][0]}x{cfg['padding'][1]}",
        "scale": cfg["scale"],
        "score_threshold": cfg["score_threshold"],
        "nms_threshold": cfg["nms_threshold"],
        "hit_threshold": cfg["hit_threshold"],
        "use_preprocessing": cfg["use_preprocessing"],
        "shrink_factor": cfg["shrink_factor"],
        "grabcut_iters": cfg["grabcut_iters"],
        "num_boxes": len(pred_boxes),
        "f1": float(det_metrics.get("f1", 0.0)),
        "precision": float(det_metrics.get("precision", 0.0)),
        "recall": float(det_metrics.get("recall", 0.0)),
        "mean_iou_matched": float(det_metrics.get("mean_iou_matched", 0.0)),
        **scored,
    }


def run_experiment(image, gt_boxes, preset_name, preset_cfg, max_configs, rng):
    all_cfgs = list(expand_grid(preset_cfg))
    sampled = maybe_sample(all_cfgs, max_configs=max_configs, rng=rng)
    rows = []
    for i, cfg in enumerate(sampled):
        row = _evaluate_config(image=image, gt_boxes=gt_boxes, cfg=cfg)
        if row is None:
            continue
        row["preset"] = preset_name
        row["config_index"] = i
        rows.append(row)
    return {
        "preset": preset_name,
        "tested": len(sampled),
        "valid": len(rows),
        "rows": rows,
    }


def save_top_results(df, out_dir, top_k=20):
    top = df.sort_values("total_score", ascending=False).head(top_k)
    top.to_csv(out_dir / "top_configs.csv", index=False)
    with open(out_dir / "top_configs.json", "w", encoding="utf-8") as f:
        json.dump(top.to_dict(orient="records"), f, indent=2, ensure_ascii=False)


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_analytics(df, out_dir):
    pd, sns = _require_analysis_libs()
    sns.set_theme(style="whitegrid")
    numeric_cols = [
        "scale",
        "score_threshold",
        "nms_threshold",
        "hit_threshold",
        "shrink_factor",
        "grabcut_iters",
        "num_boxes",
        "det_score",
        "hole_score",
        "continuity_score",
        "texture_score",
        "hole_ratio",
        "boundary_mae",
        "total_score",
    ]
    corr = df[numeric_cols].corr(numeric_only=True)
    fig = plt.figure(figsize=(12, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False, square=False)
    plt.title("Correlation Heatmap: Params vs Metrics")
    _save(fig, out_dir / "01_correlation_heatmap.png")

    fig, ax = plt.subplots(figsize=(10, 6))
    # Add slight jitter on x to separate overlapping points.
    work = df.copy()
    work["scale_jitter"] = work["scale"] + np.random.default_rng(42).normal(0, 0.0018, len(work))
    sns.scatterplot(
        data=work,
        x="scale_jitter",
        y="total_score",
        hue="score_threshold",
        style="use_preprocessing",
        size="num_boxes",
        sizes=(35, 220),
        alpha=0.8,
        palette="viridis",
        ax=ax,
    )
    # Overlay mean trend to summarize dense clouds.
    trend = df.groupby("scale", as_index=False)["total_score"].agg(["mean", "std"]).reset_index()
    ax.errorbar(
        trend["scale"],
        trend["mean"],
        yerr=trend["std"].fillna(0.0),
        color="#d62728",
        linewidth=2,
        marker="o",
        capsize=4,
        label="mean ± std",
    )
    ax.set_title("Total Score vs Scale (jittered points + mean trend)")
    ax.set_xlabel("scale")
    ax.set_ylabel("total_score")
    ax.legend(loc="best", fontsize=8)
    _save(fig, out_dir / "02_total_vs_scale.png")

    fig = plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x="nms_threshold",
        y="det_score",
        hue="score_threshold",
        palette="viridis",
        alpha=0.8,
    )
    plt.title("Detection Score vs NMS Threshold")
    _save(fig, out_dir / "03_det_vs_nms.png")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(data=df, x="win_stride", y="total_score", inner=None, color="#c7dcef", cut=0, ax=ax)
    sns.boxplot(
        data=df,
        x="win_stride",
        y="total_score",
        width=0.3,
        showcaps=True,
        boxprops={"facecolor": "white", "zorder": 3},
        ax=ax,
    )
    sns.stripplot(data=df, x="win_stride", y="total_score", color="#1f77b4", alpha=0.55, size=5, jitter=0.12, ax=ax)
    medians = df.groupby("win_stride")["total_score"].median().reset_index()
    for i, row in medians.iterrows():
        ax.text(i, row["total_score"] + 0.005, f"median={row['total_score']:.3f}", ha="center", fontsize=9)
    ax.set_title("Total Score by Win Stride (distribution + individual trials)")
    ax.set_xlabel("win_stride")
    ax.set_ylabel("total_score")
    _save(fig, out_dir / "04_box_win_stride.png")

    fig = plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="grabcut_iters", y="continuity_score")
    plt.title("Continuity Score by GrabCut Iterations")
    _save(fig, out_dir / "05_box_grabcut_continuity.png")

    by_preset = (
        df.groupby("preset")[["total_score", "det_score", "hole_score", "continuity_score", "texture_score"]]
        .mean()
        .reset_index()
        .sort_values("total_score", ascending=False)
    )
    fig = plt.figure(figsize=(10, 6))
    sns.barplot(data=by_preset, x="preset", y="total_score")
    plt.title("Average Total Score by Preset")
    _save(fig, out_dir / "06_avg_score_by_preset.png")

    top10 = df.sort_values("total_score", ascending=False).head(10).copy()
    top10["label"] = top10.apply(
        lambda r: f"{r['preset']} | s={r['scale']:.2f} nms={r['nms_threshold']:.2f} gc={int(r['grabcut_iters'])}",
        axis=1,
    )
    fig = plt.figure(figsize=(12, 6))
    sns.barplot(data=top10, x="total_score", y="label", orient="h")
    plt.title("Top 10 Configs by Total Score")
    _save(fig, out_dir / "07_top10_configs.png")

    score_bins = pd.cut(df["score_threshold"], bins=np.linspace(0.0, 0.5, 6), include_lowest=True)
    trend = df.groupby(score_bins)["det_score"].mean().reset_index()
    fig = plt.figure(figsize=(10, 6))
    sns.lineplot(data=trend, x=trend["score_threshold"].astype(str), y="det_score", marker="o")
    plt.xticks(rotation=25)
    plt.title("Detection Score Trend by Score Threshold Bins")
    plt.xlabel("Score Threshold Bin")
    _save(fig, out_dir / "08_trend_score_threshold_bins.png")

    pair_cols = ["scale", "nms_threshold", "score_threshold", "shrink_factor", "total_score", "det_score", "continuity_score"]
    pair_df = df[pair_cols].copy()
    pairplot = sns.pairplot(pair_df, corner=True, diag_kind="kde")
    pairplot.figure.suptitle("Pairwise Dependencies (overview)", y=1.02)
    pairplot.savefig(out_dir / "09_pairplot_dependencies.png", dpi=170, bbox_inches="tight")
    plt.close(pairplot.figure)

    # Split the dense pairplot into focused small charts for readability.
    focused_pairs = [
        ("scale", "total_score"),
        ("nms_threshold", "det_score"),
        ("score_threshold", "det_score"),
        ("shrink_factor", "continuity_score"),
        ("shrink_factor", "total_score"),
        ("score_threshold", "continuity_score"),
    ]
    for idx, (x_col, y_col) in enumerate(focused_pairs, start=1):
        fig = plt.figure(figsize=(6, 4.2))
        sns.scatterplot(
            data=df,
            x=x_col,
            y=y_col,
            hue="use_preprocessing",
            size="grabcut_iters",
            sizes=(40, 180),
            alpha=0.82,
        )
        sns.regplot(data=df, x=x_col, y=y_col, scatter=False, color="#d62728", line_kws={"linewidth": 2})
        plt.title(f"{y_col} vs {x_col}")
        _save(fig, out_dir / f"09_{idx:02d}_{y_col}_vs_{x_col}.png")

    fig = plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x="hole_ratio",
        y="continuity_score",
        hue="preset",
        size="texture_score",
        sizes=(20, 200),
        alpha=0.8,
    )
    plt.title("Hole Ratio vs Continuity (size=Texture Score)")
    _save(fig, out_dir / "10_hole_ratio_vs_continuity.png")

    # Base detection metrics dependency plots (requested for interpretability).
    base_metric_cols = ["f1", "precision", "recall", "mean_iou_matched"]
    available_base = [c for c in base_metric_cols if c in df.columns]
    if available_base:
        # 11: compact heatmap of parameter -> base metrics correlations.
        corr_cols = ["scale", "score_threshold", "nms_threshold", "hit_threshold", "shrink_factor"] + available_base
        corr_df = df[corr_cols].corr(numeric_only=True).loc[available_base, ["scale", "score_threshold", "nms_threshold", "hit_threshold", "shrink_factor"]]
        fig = plt.figure(figsize=(8.5, 4.8))
        sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="coolwarm", center=0)
        plt.title("Base Metrics vs Parameters (Correlation)")
        _save(fig, out_dir / "11_base_metrics_param_correlation.png")

        # 12: F1 and precision vs score threshold.
        if {"score_threshold", "f1", "precision"} <= set(df.columns):
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
            sns.scatterplot(data=df, x="score_threshold", y="f1", hue="nms_threshold", alpha=0.82, ax=axes[0])
            sns.regplot(data=df, x="score_threshold", y="f1", scatter=False, color="#d62728", line_kws={"linewidth": 2}, ax=axes[0])
            axes[0].set_title("F1 vs score_threshold")
            sns.scatterplot(data=df, x="score_threshold", y="precision", hue="nms_threshold", alpha=0.82, ax=axes[1])
            sns.regplot(data=df, x="score_threshold", y="precision", scatter=False, color="#d62728", line_kws={"linewidth": 2}, ax=axes[1])
            axes[1].set_title("Precision vs score_threshold")
            _save(fig, out_dir / "12_f1_precision_vs_score_threshold.png")

        # 13: Recall and mean IoU vs NMS.
        if {"nms_threshold", "recall", "mean_iou_matched"} <= set(df.columns):
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
            sns.scatterplot(data=df, x="nms_threshold", y="recall", hue="score_threshold", alpha=0.82, ax=axes[0])
            sns.regplot(data=df, x="nms_threshold", y="recall", scatter=False, color="#d62728", line_kws={"linewidth": 2}, ax=axes[0])
            axes[0].set_title("Recall vs nms_threshold")
            sns.scatterplot(data=df, x="nms_threshold", y="mean_iou_matched", hue="score_threshold", alpha=0.82, ax=axes[1])
            sns.regplot(data=df, x="nms_threshold", y="mean_iou_matched", scatter=False, color="#d62728", line_kws={"linewidth": 2}, ax=axes[1])
            axes[1].set_title("Mean IoU vs nms_threshold")
            _save(fig, out_dir / "13_recall_iou_vs_nms_threshold.png")

        # 14: Boxplots for base metrics by categorical parameters.
        if "use_preprocessing" in df.columns:
            melted = df.melt(
                id_vars="use_preprocessing",
                value_vars=[m for m in ["f1", "precision", "recall", "mean_iou_matched"] if m in df.columns],
                var_name="metric",
                value_name="value",
            )
            fig = plt.figure(figsize=(11, 5.2))
            sns.boxplot(data=melted, x="metric", y="value", hue="use_preprocessing")
            plt.title("Base Metrics by use_preprocessing")
            _save(fig, out_dir / "14_base_metrics_by_preprocessing.png")


def _default_base_config(preset_cfg):
    return {
        "win_stride": preset_cfg["win_stride"][0],
        "padding": preset_cfg["padding"][0],
        "scale": preset_cfg["scale"][0],
        "score_threshold": preset_cfg["score_threshold"][0],
        "nms_threshold": preset_cfg["nms_threshold"][0],
        "hit_threshold": preset_cfg["hit_threshold"][0],
        "use_preprocessing": preset_cfg["use_preprocessing"][0],
        "shrink_factor": preset_cfg["shrink_factor"][0],
        "grabcut_iters": preset_cfg["grabcut_iters"][0],
    }


def _parse_override_value(param_name, value):
    if param_name in ("win_stride", "padding"):
        parts = [p.strip() for p in value.split(",")]
        if len(parts) != 2:
            raise ValueError(f"{param_name} override must be like '8,8'")
        return (int(parts[0]), int(parts[1]))
    if param_name == "use_preprocessing":
        return value.lower() in ("1", "true", "yes", "y")
    if param_name == "grabcut_iters":
        return int(value)
    return float(value)


def _apply_base_overrides(base_cfg, overrides_raw):
    cfg = dict(base_cfg)
    for chunk in [c.strip() for c in overrides_raw.split(";") if c.strip()]:
        if "=" not in chunk:
            raise ValueError(f"Invalid override '{chunk}'. Expected key=value")
        k, v = [x.strip() for x in chunk.split("=", 1)]
        if k not in cfg:
            raise ValueError(f"Unknown base override key: {k}")
        cfg[k] = _parse_override_value(k, v)
    return cfg


def _to_scalar(v):
    if isinstance(v, tuple):
        return f"{v[0]}x{v[1]}"
    return v


def run_sweep_mode(image, gt_boxes, preset_name, preset_cfg, var1, var2, base_cfg):
    if var1 not in preset_cfg:
        raise ValueError(f"Unknown var1 '{var1}'.")
    if var2 and var2 not in preset_cfg:
        raise ValueError(f"Unknown var2 '{var2}'.")
    values1 = preset_cfg[var1]
    values2 = preset_cfg[var2] if var2 else [None]

    rows = []
    for i, v1 in enumerate(values1):
        for j, v2 in enumerate(values2):
            cfg = dict(base_cfg)
            cfg[var1] = v1
            if var2:
                cfg[var2] = v2
            row = _evaluate_config(image=image, gt_boxes=gt_boxes, cfg=cfg)
            if row is None:
                continue
            row.update(
                {
                    "preset": preset_name,
                    "mode": "sweep",
                    "var1": var1,
                    "var1_value": _to_scalar(v1),
                    "var2": var2 if var2 else "",
                    "var2_value": _to_scalar(v2) if var2 else "",
                    "sweep_i": i,
                    "sweep_j": j,
                }
            )
            rows.append(row)
    return rows


def save_rows_csv(rows, out_path):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_sweep(rows, out_dir, metric, var1, var2):
    pd, sns = _require_analysis_libs()
    sns.set_theme(style="whitegrid")
    df = pd.DataFrame(rows)
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found in sweep results.")

    if not var2:
        fig = plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x="var1_value", y=metric, marker="o")
        plt.title(f"Sweep: {metric} vs {var1}")
        plt.xlabel(var1)
        plt.ylabel(metric)
        _save(fig, out_dir / f"sweep_{metric}_vs_{var1}.png")
        return

    pivot = df.pivot_table(index="var1_value", columns="var2_value", values=metric, aggfunc="mean")
    fig = plt.figure(figsize=(10, 7))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis")
    plt.title(f"Sweep Heatmap: {metric} ({var1} x {var2})")
    plt.xlabel(var2)
    plt.ylabel(var1)
    _save(fig, out_dir / f"sweep_heatmap_{metric}_{var1}_x_{var2}.png")


def estimate_runtime(total_configs, sec_per_config):
    total_sec = total_configs * sec_per_config
    return {
        "total_sec": round(total_sec, 1),
        "total_min": round(total_sec / 60.0, 2),
        "total_hours": round(total_sec / 3600.0, 2),
    }


def probe_runtime(image, gt_boxes, probe_cfgs):
    t0 = time.time()
    ok = 0
    for cfg in probe_cfgs:
        row = _evaluate_config(image=image, gt_boxes=gt_boxes, cfg=cfg)
        if row is not None:
            ok += 1
    elapsed = time.time() - t0
    avg = elapsed / max(1, len(probe_cfgs))
    return {"elapsed": elapsed, "avg_sec_per_config": avg, "valid_probe_runs": ok}


def main():
    parser = argparse.ArgumentParser(
        description="Run multiple tuning presets and build rich analytics plots."
    )
    parser.add_argument("--image", default="test_image.jpg", help="Input image path.")
    parser.add_argument("--gt", default="", help='Optional GT boxes: "x,y,w,h;x,y,w,h"')
    parser.add_argument("--presets-file", default="experiment_presets.json", help="JSON with preset definitions.")
    parser.add_argument("--preset", default="all", help="Preset name or 'all'.")
    parser.add_argument("--max-configs", type=int, default=120, help="Sample limit per preset. 0 means all.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--out-dir", default="analytics_output", help="Output directory.")
    parser.add_argument("--mode", default="full", choices=["full", "sweep"], help="Run full random search or sweep mode.")
    parser.add_argument("--var1", default="scale", help="Primary parameter to vary in sweep mode.")
    parser.add_argument("--var2", default="", help="Secondary parameter to vary in sweep mode.")
    parser.add_argument("--metric", default="total_score", help="Metric for sweep plots (e.g., det_score, continuity_score).")
    parser.add_argument("--base-overrides", default="", help="Base config overrides in sweep mode: key=value;key=value")
    parser.add_argument("--sec-per-config", type=float, default=0.0, help="Manual runtime estimate (seconds per config).")
    parser.add_argument("--probe-runs", type=int, default=3, help="Probe runs for runtime estimate.")
    parser.add_argument("--estimate-only", action="store_true", help="Only estimate runtime and exit.")
    parser.add_argument(
        "--auto-gt-from-base",
        action="store_true",
        help="Build pseudo-GT from base detection boxes (useful for single-image sensitivity demos).",
    )
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {args.image}")
    gt_boxes = parse_gt_boxes(args.gt)

    presets = load_presets(args.presets_file if Path(args.presets_file).exists() else "")
    selected = presets
    if args.preset != "all":
        if args.preset not in presets:
            raise ValueError(f"Unknown preset '{args.preset}'. Available: {', '.join(presets.keys())}")
        selected = {args.preset: presets[args.preset]}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    if args.mode == "sweep":
        if args.preset == "all":
            raise ValueError("Sweep mode requires a specific --preset (not 'all').")
        preset_cfg = selected[args.preset]
        base_cfg = _default_base_config(preset_cfg)
        if args.base_overrides:
            base_cfg = _apply_base_overrides(base_cfg, args.base_overrides)
        if args.auto_gt_from_base and not gt_boxes:
            base_row = _evaluate_config(image=image, gt_boxes=[], cfg=base_cfg)
            if base_row is None:
                raise RuntimeError("Cannot auto-build GT: base config did not return 2 valid detections.")
            # Re-run detection once to extract exact base boxes for pseudo-GT.
            pred_boxes = detect_humans(
                image,
                win_stride=base_cfg["win_stride"],
                padding=base_cfg["padding"],
                scale=base_cfg["scale"],
                score_threshold=base_cfg["score_threshold"],
                nms_threshold=base_cfg["nms_threshold"],
                hit_threshold=base_cfg["hit_threshold"],
                use_preprocessing=base_cfg["use_preprocessing"],
                shrink_factor=base_cfg["shrink_factor"],
            )
            gt_boxes = [tuple(map(int, b)) for b in pred_boxes[:2]]
            print(f"Auto pseudo-GT from base config: {gt_boxes}")
        total_cfgs = len(preset_cfg[args.var1]) * (len(preset_cfg[args.var2]) if args.var2 else 1)
        probe_cfgs = []
        for _ in range(min(args.probe_runs, total_cfgs)):
            cfg = dict(base_cfg)
            cfg[args.var1] = preset_cfg[args.var1][rng.integers(0, len(preset_cfg[args.var1]))]
            if args.var2:
                cfg[args.var2] = preset_cfg[args.var2][rng.integers(0, len(preset_cfg[args.var2]))]
            probe_cfgs.append(cfg)
        sec_per_cfg = args.sec_per_config
        if sec_per_cfg <= 0 and probe_cfgs:
            probe = probe_runtime(image=image, gt_boxes=gt_boxes, probe_cfgs=probe_cfgs)
            sec_per_cfg = probe["avg_sec_per_config"]
            print(
                f"Probe runtime: {probe['elapsed']:.1f}s for {len(probe_cfgs)} configs, "
                f"avg={probe['avg_sec_per_config']:.2f}s/config, valid={probe['valid_probe_runs']}"
            )
        if sec_per_cfg > 0:
            eta = estimate_runtime(total_configs=total_cfgs, sec_per_config=sec_per_cfg)
            print(
                f"Estimated runtime for sweep ({total_cfgs} configs): "
                f"{eta['total_min']} min ({eta['total_hours']} h)"
            )
        if args.estimate_only:
            return

        rows = run_sweep_mode(
            image=image,
            gt_boxes=gt_boxes,
            preset_name=args.preset,
            preset_cfg=preset_cfg,
            var1=args.var1,
            var2=args.var2 if args.var2 else None,
            base_cfg=base_cfg,
        )
        if not rows:
            print("No valid runs found in sweep mode.")
            return
        save_rows_csv(rows, out_dir / "sweep_results.csv")
        with open(out_dir / "sweep_results.json", "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)
        plot_sweep(rows, out_dir=out_dir, metric=args.metric, var1=args.var1, var2=args.var2 if args.var2 else None)
        print(f"Sweep finished. Saved to: {out_dir}")
        return

    full_config_count = 0
    for name, cfg in selected.items():
        combos = list(expand_grid(cfg))
        full_config_count += min(len(combos), args.max_configs) if args.max_configs > 0 else len(combos)
    if args.sec_per_config > 0:
        eta = estimate_runtime(total_configs=full_config_count, sec_per_config=args.sec_per_config)
        print(
            f"Estimated runtime for full mode ({full_config_count} configs): "
            f"{eta['total_min']} min ({eta['total_hours']} h)"
        )
        if args.estimate_only:
            return

    reports = []
    all_rows = []
    for name, cfg in selected.items():
        rep = run_experiment(
            image=image,
            gt_boxes=gt_boxes,
            preset_name=name,
            preset_cfg=cfg,
            max_configs=args.max_configs,
            rng=rng,
        )
        reports.append({"preset": rep["preset"], "tested": rep["tested"], "valid": rep["valid"]})
        all_rows.extend(rep["rows"])
        print(f"[{name}] tested={rep['tested']} valid={rep['valid']}")

    if not all_rows:
        print("No valid runs found. Try lower thresholds or another image.")
        return

    pd, _ = _require_analysis_libs()
    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "all_results.csv", index=False)
    with open(out_dir / "run_report.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "image": args.image,
                "gt_provided": bool(gt_boxes),
                "reports": reports,
                "rows": len(all_rows),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    save_top_results(df, out_dir=out_dir, top_k=20)
    plot_analytics(df, out_dir=out_dir)
    print(f"Saved analytics to: {out_dir}")


if __name__ == "__main__":
    main()
