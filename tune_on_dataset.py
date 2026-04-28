import argparse
import configparser
import json
import re
import urllib.request
import zipfile
import os
import time
from pathlib import Path

import cv2
import numpy as np
import optuna

from detection import preprocess_for_hog, shrink_boxes
from metrics import evaluate_detections

DATASET_URL = "https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip"
DATASET_ZIP = "PennFudanPed.zip"
DATASET_DIR = "PennFudanPed"


def download_dataset():
    if os.path.isdir(DATASET_DIR):
        return
    if not os.path.isfile(DATASET_ZIP):
        urllib.request.urlretrieve(
            DATASET_URL, DATASET_ZIP,
            reporthook=lambda n, bs, ts: print(
                f"\r  {min(100, n*bs*100//max(ts,1))}%", end="", flush=True
            ),
        )
        print()
    with zipfile.ZipFile(DATASET_ZIP, "r") as zf:
        zf.extractall(".")


def parse_annotation(ann_path):
    pattern = re.compile(
        r'bounding box for object\s+\d+.*?:\s*\((\d+),\s*(\d+)\)\s*-\s*\((\d+),\s*(\d+)\)',
        re.IGNORECASE,
    )
    boxes = []
    for line in Path(ann_path).read_text(encoding="utf-8").splitlines():
        m = pattern.search(line)
        if m:
            x1, y1, x2, y2 = [int(m.group(i)) - 1 for i in range(1, 5)]
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            w, h = x2 - x1, y2 - y1
            if w > 10 and h > 10:
                boxes.append((x1, y1, w, h))
    return boxes


def load_dataset(max_images=170):
    ann_dir = Path(DATASET_DIR) / "Annotation"
    img_dir = Path(DATASET_DIR) / "PNGImages"
    entries = []
    for ann_path in sorted(ann_dir.glob("*.txt"))[:max_images]:
        img_path = img_dir / (ann_path.stem + ".png")
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        gt = parse_annotation(str(ann_path))
        if img is not None and gt:
            entries.append({"image": img, "gt_boxes": gt})
    return entries


def run_optuna_search(entries, n_trials=50, iou_threshold=0.5, results_path="dataset_tuning_results_optuna.json"):
    all_results = []
    t_start = time.time()

    def objective(trial):
        params = {
            "win_stride": trial.suggest_categorical("win_stride", ["4,4", "8,8"]),
            "padding": trial.suggest_categorical("padding", ["8,8", "16,16", "32,32"]),
            "scale": trial.suggest_float("scale", 1.01, 1.15, step=0.01),
            "hit_threshold": trial.suggest_float("hit_threshold", 0.0, 0.8, step=0.1),
            "score_threshold": trial.suggest_float("score_threshold", 0.0, 1.0, step=0.1),
            "nms_threshold": trial.suggest_float("nms_threshold", 0.1, 0.8, step=0.05),
            "use_preprocessing": trial.suggest_categorical("use_preprocessing", [True, False]),
            "shrink_factor": trial.suggest_float("shrink_factor", 0.0, 0.25, step=0.02),
        }

        ws = tuple(int(x) for x in params["win_stride"].split(","))
        pd = tuple(int(x) for x in params["padding"].split(","))

        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        tp = fp = fn = 0
        ious = []

        for entry in entries:
            img = entry["image"]
            if params["use_preprocessing"]:
                img = preprocess_for_hog(img)

            raw_boxes, weights = hog.detectMultiScale(
                img, winStride=ws, padding=pd,
                scale=params["scale"], hitThreshold=params["hit_threshold"],
            )

            if len(raw_boxes) > 0:
                weights = weights.reshape(-1)
                keep = weights > params["score_threshold"]
                fb, fw = raw_boxes[keep], weights[keep]
                fx = fb.astype(np.float32)

                if len(fb) > 0:
                    idx = cv2.dnn.NMSBoxes(fx.tolist(), fw.tolist(), 0.0, params["nms_threshold"])
                    idx = np.array(idx).reshape(-1) if idx is not None and len(idx) else []
                    pred = fb[idx].tolist() if len(idx) else []
                else:
                    pred = []
            else:
                pred = []

            if params["shrink_factor"] > 0 and pred:
                pred = shrink_boxes(pred, factor=params["shrink_factor"])

            det = evaluate_detections(pred, entry["gt_boxes"], iou_threshold)
            tp += det["tp"]
            fp += det["fp"]
            fn += det["fn"]
            if det["mean_iou_matched"] > 0:
                ious.append(det["mean_iou_matched"])

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        iou = float(np.mean(ious)) if ious else 0.0

        details = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "mean_iou": round(iou, 4),
            "tp": tp, "fp": fp, "fn": fn
        }

        trial.set_user_attr("params", params)
        trial.set_user_attr("details", details)

        log_data = {
            "trial": trial.number,
            "win_stride": list(ws),
            "padding": list(pd),
            "scale": params["scale"],
            "hit_threshold": params["hit_threshold"],
            "score_threshold": params["score_threshold"],
            "nms_threshold": params["nms_threshold"],
            "use_preprocessing": params["use_preprocessing"],
            "shrink_factor": round(params["shrink_factor"], 3),
            **details
        }
        all_results.append(log_data)

        # Multi-objective: maximize F1 and Mean IoU
        return f1, iou

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(directions=["maximize", "maximize"], sampler=sampler)

    print(f"Optuna Multi-Objective Optimization: {n_trials} trials")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    elapsed = time.time() - t_start

    pareto_trials = study.best_trials
    if not pareto_trials:
        print("No valid results found!")
        return

    pareto_results = []
    for t in pareto_trials:
        pareto_results.append({
            "trial_id": t.number,
            "scores": {
                "f1": round(t.values[0], 4),
                "mean_iou": round(t.values[1], 4),
            },
            "params": t.user_attrs.get("params", {}),
            "details": t.user_attrs.get("details", {})
        })

    # Pick the most balanced option from Pareto front
    best_balanced_trial = max(pareto_trials, key=lambda t: sum(t.values))
    best_params = best_balanced_trial.user_attrs["params"]
    best_details = best_balanced_trial.user_attrs["details"]

    payload = {
        "method": "Multi-Objective Bayesian Optimization (MOTPE)",
        "trials": n_trials,
        "elapsed_sec": round(elapsed, 1),
        "pareto_front_size": len(pareto_trials),
        "pareto_front": sorted(pareto_results, key=lambda x: sum(x["scores"].values()), reverse=True),
        "all_trials": all_results
    }
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    # Auto-update config_pennfudan.cfg with best balanced variant
    ws = tuple(int(x) for x in best_params["win_stride"].split(","))
    pd = tuple(int(x) for x in best_params["padding"].split(","))
    
    cfg = configparser.ConfigParser()
    # Read existing if available to preserve input/output paths
    if os.path.exists("config_pennfudan.cfg"):
        cfg.read("config_pennfudan.cfg")
    else:
        cfg["pipeline"] = {"input": "PennFudanPed/PNGImages/FudanPed00001.png", "output": "result_pennfudan.jpg"}
        cfg["masks"] = {"grabcut_iters": "5"}
        
    if "detection" not in cfg:
        cfg["detection"] = {}
        
    cfg["detection"]["win_stride_w"] = str(ws[0])
    cfg["detection"]["win_stride_h"] = str(ws[1])
    cfg["detection"]["padding_w"] = str(pd[0])
    cfg["detection"]["padding_h"] = str(pd[1])
    cfg["detection"]["scale"] = str(round(best_params["scale"], 2))
    cfg["detection"]["score_threshold"] = str(round(best_params["score_threshold"], 2))
    cfg["detection"]["nms_threshold"] = str(round(best_params["nms_threshold"], 2))
    cfg["detection"]["hit_threshold"] = str(round(best_params["hit_threshold"], 2))
    cfg["detection"]["use_preprocessing"] = str(best_params["use_preprocessing"]).lower()
    cfg["detection"]["shrink_factor"] = str(round(best_params["shrink_factor"], 2))

    with open("config_pennfudan.cfg", "w", encoding="utf-8") as f:
        cfg.write(f)

    print(f"\n{'='*60}")
    print(f"FOUND PARETO FRONT: {len(pareto_trials)} trade-off solutions")
    print(f"{'='*60}")
    print("Selected most balanced variant for config_pennfudan.cfg:")
    for k, v in best_params.items():
        print(f"  {k:<18} = {v}")
    
    print(f"\nIts metrics:")
    print(f"  F1-Score        = {best_balanced_trial.values[0]:.4f}")
    print(f"  Mean IoU        = {best_balanced_trial.values[1]:.4f}")
    print(f"  TP/FP/FN        = {best_details['tp']}/{best_details['fp']}/{best_details['fn']}")
    
    print(f"\nDone in {elapsed:.0f}s ({n_trials} trials)")
    print(f"config_pennfudan.cfg updated.")
    print(f"All Pareto front solutions saved to {results_path}")


def main():
    parser = argparse.ArgumentParser(description="Find best HOG+SVM params on Penn-Fudan dataset using Optuna.")
    parser.add_argument("--max", type=int, default=30, help="Max images to use (default: 30 for speed).")
    parser.add_argument("--trials", type=int, default=50, help="Number of Optuna trials (default: 50).")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU match threshold (default: 0.5).")
    args = parser.parse_args()

    download_dataset()

    print(f"\nLoading dataset (up to {args.max} images)...")
    entries = load_dataset(max_images=args.max)
    if not entries:
        print("No images loaded. Check dataset directory.")
        return
    print(f"Loaded {len(entries)} images.\n")

    run_optuna_search(entries, n_trials=args.trials, iou_threshold=args.iou)


if __name__ == "__main__":
    main()
