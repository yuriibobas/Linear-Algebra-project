"""
smart_tune.py – 2-Phase Optuna Pipeline Tuner
==============================================

Phase 1 – HOG detection parameters (~30 seconds)
  Runs HOG+SVM ONLY (no GrabCut, no inpainting).
  Single objective: maximize det_score (F1 + IoU).
  Params: win_stride, padding, scale, hit_threshold, score_threshold,
          nms_threshold, use_preprocessing, shrink_factor.

Phase 2 – GrabCut + Morphology parameters (~5-8 minutes)
  Runs the FULL pipeline with HOG params fixed from Phase 1.
  3 objectives: mask_quality, continuity, texture.
  Params: grabcut_iters, morph_kernel_size, morph_open_iters, morph_close_iters.

Total runtime: ~6-10 min depending on --trials1 / --trials2.
"""

import argparse
import configparser
import json
import time

import cv2
import numpy as np
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

from detection import detect_humans, get_binary_masks
from inpainting import inpaint_neighbor_averaging
from metrics import evaluate_detections
from transformation import apply_translation


# ─────────────────────────── helpers ────────────────────────────

def parse_gt_boxes(gt_raw):
    if not gt_raw:
        return []
    boxes = []
    for chunk in [c.strip() for c in gt_raw.split(";") if c.strip()]:
        parts = [v.strip() for v in chunk.split(",")]
        if len(parts) != 4:
            raise ValueError(f"Invalid GT box: '{chunk}'")
        boxes.append(tuple(int(v) for v in parts))
    return boxes


def detect_only(image, params):
    """Run HOG detection only — no GrabCut, no inpainting. Very fast."""
    return detect_humans(
        image,
        win_stride=params["win_stride"],
        padding=params["padding"],
        scale=params["scale"],
        score_threshold=params["score_threshold"],
        nms_threshold=params["nms_threshold"],
        hit_threshold=params["hit_threshold"],
        use_preprocessing=params["use_preprocessing"],
        shrink_factor=params["shrink_factor"],
    )


def run_pipeline(image, hog_params, grabcut_params):
    """Run the full pipeline with split HOG and GrabCut param dicts."""
    pred_boxes = detect_only(image, hog_params)

    if len(pred_boxes) < 2:
        return None

    masks_data = get_binary_masks(
        image,
        pred_boxes,
        top_k=2,
        grabcut_iters=grabcut_params["grabcut_iters"],
        morph_kernel_size=grabcut_params["morph_kernel_size"],
        morph_open_iters=grabcut_params["morph_open_iters"],
        morph_close_iters=grabcut_params["morph_close_iters"],
    )
    if len(masks_data) < 2:
        return None

    mask_A, rect_A = masks_data[0]
    mask_B, rect_B = masks_data[1]

    cA_x = rect_A[0] + rect_A[2] // 2
    cA_y = rect_A[1] + rect_A[3] // 2
    cB_x = rect_B[0] + rect_B[2] // 2
    cB_y = rect_B[1] + rect_B[3] // 2

    tA_x, tA_y = cB_x - cA_x, cB_y - cA_y
    tB_x, tB_y = cA_x - cB_x, cA_y - cB_y

    translated_A, translated_mask_A = apply_translation(image, mask_A, tA_x, tA_y)
    translated_B, translated_mask_B = apply_translation(image, mask_B, tB_x, tB_y)

    combined_original = cv2.bitwise_or(mask_A, mask_B)
    dilate_kernel = np.ones((7, 7), np.uint8)
    inpaint_mask = cv2.dilate(combined_original, dilate_kernel, iterations=2)

    clean_bg = image.copy()
    clean_bg[inpaint_mask > 0] = [0, 0, 0]
    clean_bg = inpaint_neighbor_averaging(clean_bg, inpaint_mask)

    composite = clean_bg.copy()
    idx_A = translated_mask_A > 0
    composite[idx_A] = translated_A[idx_A]
    idx_B = translated_mask_B > 0
    composite[idx_B] = translated_B[idx_B]

    combined_placed = cv2.bitwise_or(translated_mask_A, translated_mask_B)
    erode_k = np.ones((5, 5), np.uint8)
    inner = cv2.erode(combined_placed, erode_k, iterations=2)
    alpha = cv2.GaussianBlur(combined_placed.astype(np.float32) * 255, (15, 15), 0) / 255.0
    alpha = np.clip(alpha, 0, 1)
    alpha[inner > 0] = 1.0

    final = (alpha[:, :, None] * composite.astype(np.float64)
             + (1.0 - alpha[:, :, None]) * clean_bg.astype(np.float64))
    final = np.clip(final, 0, 255).astype(np.uint8)

    return {
        "final": final,
        "inpaint_mask": inpaint_mask,
        "pred_boxes": pred_boxes,
        "masks_data": masks_data,
    }


# ─────────────────────── scoring functions ──────────────────────

def score_detection(pred_boxes, gt_boxes):
    """Phase 1 score: weighted F1 + IoU."""
    if gt_boxes:
        det = evaluate_detections(pred_boxes, gt_boxes, iou_threshold=0.5)
        return 0.55 * det["f1"] + 0.45 * det["mean_iou_matched"]
    return 1.0 if len(pred_boxes) >= 2 else 0.0


def compute_mask_quality(masks_data):
    """Replace hole_score: IoU-F1 between GrabCut mask and its HOG bounding box.

    High score → mask tightly fits the person box (not tiny, not spilling outside).
    This avoids the Phase 1 conflict of 'punish large masks' vs 'good coverage'.
    """
    scores = []
    for mask, (x, y, w, h) in masks_data:
        mask_area = int(np.sum(mask > 0))
        if mask_area == 0:
            scores.append(0.0)
            continue
        box_canvas = np.zeros_like(mask)
        box_canvas[max(0, y):y + h, max(0, x):x + w] = 1
        box_area = int(np.sum(box_canvas))
        overlap = int(np.sum((mask > 0) & (box_canvas > 0)))
        precision = overlap / mask_area          # fraction of mask inside box
        recall = overlap / max(box_area, 1)      # fraction of box covered by mask
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        scores.append(f1)
    return float(np.mean(scores)) if scores else 0.0


def compute_boundary_continuity(final, inpaint_mask):
    """Improved continuity: compare inner boundary pixels vs outer boundary pixels.

    Uses per-ring comparison instead of whole-mask mean, so local seam
    artifacts are penalized even when the global average looks acceptable.
    """
    k3 = np.ones((3, 3), np.uint8)
    k5 = np.ones((5, 5), np.uint8)

    eroded = cv2.erode(inpaint_mask, k3, iterations=1)
    ring_in = inpaint_mask - eroded          # pixels just inside the inpainted region
    ring_out = cv2.dilate(inpaint_mask, k3, iterations=1) - inpaint_mask  # just outside

    if not (np.any(ring_in > 0) and np.any(ring_out > 0)):
        return 0.0

    in_vals  = final[ring_in  > 0].astype(np.float32)   # shape (N, 3)
    out_vals = final[ring_out > 0].astype(np.float32)   # shape (M, 3)

    # Mean absolute channel difference between the two border rings
    bmae = float(np.abs(in_vals.mean(axis=0) - out_vals.mean(axis=0)).mean())
    # Also penalise standard-deviation mismatch (texture discontinuity)
    bstd = float(np.abs(in_vals.std(axis=0) - out_vals.std(axis=0)).mean())

    combined = bmae + 0.5 * bstd
    return float(np.exp(-combined / 25.0))


def compute_texture_score(final, inpaint_mask):
    """Ratio of Laplacian variance inside the inpainted hole vs its neighbourhood.

    Score near 1.0 → the hole has similar texture energy to surroundings.
    Clipped to [0, 1] to avoid outliers in homogeneous regions.
    """
    gray = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    k5 = np.ones((5, 5), np.uint8)
    ring5 = cv2.dilate(inpaint_mask, k5, iterations=2) - inpaint_mask
    hole_var = float(np.var(lap[inpaint_mask > 0])) if np.any(inpaint_mask > 0) else 0.0
    ring_var = float(np.var(lap[ring5       > 0])) if np.any(ring5       > 0) else 1.0
    return float(np.clip(hole_var / max(ring_var, 1e-6), 0.0, 1.0))


# ──────────────────────────── Phase 1 ───────────────────────────

def _hog_param_suggest(trial):
    raw = {
        "win_stride": trial.suggest_categorical("win_stride", ["4,4", "8,8"]),
        "padding":    trial.suggest_categorical("padding",    ["8,8", "16,16", "32,32"]),
        "scale":      trial.suggest_float("scale", 1.01, 1.10, step=0.01),
        "hit_threshold":   trial.suggest_float("hit_threshold",   0.0, 0.6, step=0.1),
        "score_threshold": trial.suggest_float("score_threshold", 0.0, 0.5, step=0.05),
        "nms_threshold":   trial.suggest_float("nms_threshold",   0.20, 0.65, step=0.05),
        "use_preprocessing": trial.suggest_categorical("use_preprocessing", [True, False]),
        "shrink_factor": trial.suggest_float("shrink_factor", 0.0, 0.20, step=0.02),
    }
    raw["win_stride"] = tuple(int(v) for v in raw["win_stride"].split(","))
    raw["padding"]    = tuple(int(v) for v in raw["padding"].split(","))
    return raw


def tune_phase1(image, gt_boxes, n_trials, n_startup):
    """Single-objective HOG tuning. Very fast: no GrabCut, no inpainting."""
    print(f"\n{'='*60}")
    print(f"PHASE 1 — HOG Detection Tuning ({n_trials} trials)")
    print(f"{'='*60}")
    t0 = time.time()
    all_results = []

    def objective(trial):
        params = _hog_param_suggest(trial)
        pred_boxes = detect_only(image, params)
        score = score_detection(pred_boxes, gt_boxes)
        trial.set_user_attr("params", params)
        trial.set_user_attr("n_boxes", len(pred_boxes))
        all_results.append({
            "trial": trial.number,
            "det_score": round(score, 4),
            "n_boxes": len(pred_boxes),
            **{k: (list(v) if isinstance(v, tuple) else v) for k, v in params.items()},
        })
        return score

    sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=n_startup)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_trial = study.best_trial
    best_params = best_trial.user_attrs["params"]
    elapsed = time.time() - t0

    print(f"\nPhase 1 best det_score = {study.best_value:.4f}  (in {elapsed:.0f}s)")
    print(f"  n_boxes detected = {best_trial.user_attrs['n_boxes']}")
    for k, v in best_params.items():
        print(f"  {k:<22} = {v}")

    return best_params, all_results


# ──────────────────────────── Phase 2 ───────────────────────────

def tune_phase2(image, hog_params, n_trials, n_startup):
    """3-objective GrabCut + Morphology tuning with fixed HOG params."""
    print(f"\n{'='*60}")
    print(f"PHASE 2 — GrabCut + Morphology Tuning ({n_trials} trials)")
    print(f"{'='*60}")
    t0 = time.time()
    all_results = []

    def objective(trial):
        grabcut_params = {
            "grabcut_iters":   trial.suggest_int("grabcut_iters",   2, 10),
            "morph_kernel_size": trial.suggest_categorical("morph_kernel_size", [3, 5, 7]),
            "morph_open_iters":  trial.suggest_int("morph_open_iters",  1, 3),
            "morph_close_iters": trial.suggest_int("morph_close_iters", 1, 3),
        }

        try:
            result = run_pipeline(image, hog_params, grabcut_params)
        except Exception:
            return 0.0, 0.0, 0.0

        if result is None:
            return 0.0, 0.0, 0.0

        mask_quality = compute_mask_quality(result["masks_data"])
        continuity   = compute_boundary_continuity(result["final"], result["inpaint_mask"])
        texture      = compute_texture_score(result["final"], result["inpaint_mask"])

        trial.set_user_attr("grabcut_params", grabcut_params)
        trial.set_user_attr("scores", {
            "mask_quality": round(mask_quality, 4),
            "continuity":   round(continuity, 4),
            "texture":      round(texture, 4),
        })
        all_results.append({
            "trial": trial.number,
            **grabcut_params,
            "mask_quality": round(mask_quality, 4),
            "continuity":   round(continuity, 4),
            "texture":      round(texture, 4),
        })
        return mask_quality, continuity, texture

    sampler = optuna.samplers.TPESampler(seed=7, n_startup_trials=n_startup)
    study = optuna.create_study(
        directions=["maximize", "maximize", "maximize"],
        sampler=sampler,
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    elapsed = time.time() - t0

    pareto = study.best_trials
    # Pick most balanced: max sum of 3 objectives
    best_t = max(pareto, key=lambda t: sum(t.values))
    best_grabcut = best_t.user_attrs["grabcut_params"]
    scores = best_t.user_attrs["scores"]

    print(f"\nPhase 2 Pareto front size = {len(pareto)}  (in {elapsed:.0f}s)")
    print("Best balanced GrabCut params:")
    for k, v in best_grabcut.items():
        print(f"  {k:<22} = {v}")
    print(f"  mask_quality = {scores['mask_quality']:.4f}")
    print(f"  continuity   = {scores['continuity']:.4f}")
    print(f"  texture      = {scores['texture']:.4f}")

    return best_grabcut, all_results


# ─────────────────────────── main ───────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="2-Phase Smart Pipeline Tuner (HOG -> GrabCut+Morph)."
    )
    parser.add_argument("--image",  default="test_image.jpg")
    parser.add_argument("--gt",     default="", help='GT boxes: "x,y,w,h;x,y,w,h"')
    parser.add_argument("--trials1", type=int, default=25,
                        help="Phase 1 (HOG) trials (default: 25, ~30 sec)")
    parser.add_argument("--trials2", type=int, default=35,
                        help="Phase 2 (GrabCut) trials (default: 35, ~5-8 min)")
    parser.add_argument("--phase",  choices=["1", "2", "both"], default="both",
                        help="Which phase to run (default: both)")
    parser.add_argument("--out",    default="smart_tune_results.json")
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Cannot read: {args.image}")
    gt_boxes = parse_gt_boxes(args.gt)

    n_startup1 = max(5, args.trials1 // 4)
    n_startup2 = max(5, args.trials2 // 5)

    t_total = time.time()

    # ── Phase 1 ──
    if args.phase in ("1", "both"):
        best_hog, p1_results = tune_phase1(image, gt_boxes, args.trials1, n_startup1)
    else:
        # Load HOG params from existing config
        cfg = configparser.ConfigParser()
        cfg.read("config.cfg")
        best_hog = {
            "win_stride": (int(cfg["detection"].get("win_stride_w", 8)),
                           int(cfg["detection"].get("win_stride_h", 8))),
            "padding":    (int(cfg["detection"].get("padding_w", 8)),
                           int(cfg["detection"].get("padding_h", 8))),
            "scale":        float(cfg["detection"].get("scale", 1.05)),
            "hit_threshold":   float(cfg["detection"].get("hit_threshold", 0.0)),
            "score_threshold": float(cfg["detection"].get("score_threshold", 0.0)),
            "nms_threshold":   float(cfg["detection"].get("nms_threshold", 0.25)),
            "use_preprocessing": cfg["detection"].get("use_preprocessing", "true") == "true",
            "shrink_factor":   float(cfg["detection"].get("shrink_factor", 0.0)),
        }
        p1_results = []
        print(f"\nPhase 1 skipped — using HOG params from config.cfg")

    # ── Phase 2 ──
    if args.phase in ("2", "both"):
        best_grabcut, p2_results = tune_phase2(image, best_hog, args.trials2, n_startup2)
    else:
        best_grabcut = {"grabcut_iters": 5, "morph_kernel_size": 5,
                        "morph_open_iters": 2, "morph_close_iters": 2}
        p2_results = []

    total_elapsed = time.time() - t_total

    # ── Save best result image ──
    result = run_pipeline(image, best_hog, best_grabcut)
    if result is not None:
        cv2.imwrite("smart_tune_best.jpg", result["final"])
        print(f"\nBest result image saved -> smart_tune_best.jpg")

    # ── Save JSON ──
    payload = {
        "method": "2-Phase Optuna (HOG -> GrabCut+Morph)",
        "total_elapsed_sec": round(total_elapsed, 1),
        "best_hog_params": {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in best_hog.items()
        },
        "best_grabcut_params": best_grabcut,
        "phase1_trials": p1_results,
        "phase2_trials": p2_results,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"Full results saved -> {args.out}")

    # ── Auto-update config.cfg ──
    ws = best_hog["win_stride"]
    pd = best_hog["padding"]
    cfg = configparser.ConfigParser()
    cfg["pipeline"] = {"input": args.image, "output": "result_image.jpg"}
    cfg["detection"] = {
        "win_stride_w": str(ws[0]),
        "win_stride_h": str(ws[1]),
        "padding_w":    str(pd[0]),
        "padding_h":    str(pd[1]),
        "scale":        str(best_hog["scale"]),
        "score_threshold": str(best_hog["score_threshold"]),
        "nms_threshold":   str(best_hog["nms_threshold"]),
        "hit_threshold":   str(best_hog["hit_threshold"]),
        "use_preprocessing": str(best_hog["use_preprocessing"]).lower(),
        "shrink_factor":   str(best_hog["shrink_factor"]),
    }
    cfg["masks"] = {
        "grabcut_iters":    str(best_grabcut["grabcut_iters"]),
        "morph_kernel_size": str(best_grabcut["morph_kernel_size"]),
        "morph_open_iters":  str(best_grabcut["morph_open_iters"]),
        "morph_close_iters": str(best_grabcut["morph_close_iters"]),
    }
    with open("config.cfg", "w", encoding="utf-8") as f:
        cfg.write(f)

    print(f"\n{'='*60}")
    print(f"DONE in {total_elapsed:.0f}s total")
    print(f"config.cfg updated with best combined params.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()