import argparse
import configparser
import json
import time

import cv2
import numpy as np
import optuna

from detection import detect_humans, get_binary_masks
from inpainting import inpaint_neighbor_averaging
from metrics import evaluate_detections
from transformation import apply_translation


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


def run_pipeline(image, params):
    """Run the full pipeline with given params."""
    pred_boxes = detect_humans(
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

    if len(pred_boxes) < 2:
        return None

    masks_data = get_binary_masks(
        image, pred_boxes, top_k=2, grabcut_iters=params["grabcut_iters"]
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

    # Background-first inpainting
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

    # Feather blending
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
    }


def score_result(image, result, gt_boxes):
    """Compute visual quality metrics."""
    pred_boxes = result["pred_boxes"]
    final = result["final"]
    inpaint_mask = result["inpaint_mask"]

    # Detection quality
    if gt_boxes:
        det = evaluate_detections(pred_boxes, gt_boxes, iou_threshold=0.5)
        det_score = 0.55 * det["f1"] + 0.45 * det["mean_iou_matched"]
    else:
        det = None
        det_score = 1.0 if len(pred_boxes) >= 2 else 0.0

    # Hole ratio
    h, w = image.shape[:2]
    hole_ratio = float(np.sum(inpaint_mask > 0)) / (h * w)
    hole_score = float(np.clip(1.0 - hole_ratio * 5.0, 0.0, 1.0))

    # Boundary continuity
    kernel = np.ones((3, 3), np.uint8)
    ring = cv2.dilate(inpaint_mask, kernel, iterations=1) - inpaint_mask
    if np.any(ring > 0) and np.any(inpaint_mask > 0):
        bmae = float(np.abs(
            final[ring > 0].mean(axis=0) - final[inpaint_mask > 0].mean(axis=0)
        ).mean())
    else:
        bmae = 255.0
    continuity_score = float(np.exp(-bmae / 28.0))

    # Texture ratio
    gray = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    hole_var = float(np.var(lap[inpaint_mask > 0])) if np.any(inpaint_mask > 0) else 0.0
    ring5 = cv2.dilate(inpaint_mask, np.ones((5, 5), np.uint8), iterations=1) - inpaint_mask
    ring_var = float(np.var(lap[ring5 > 0])) if np.any(ring5 > 0) else 1.0
    tex_score = float(np.clip(hole_var / max(ring_var, 1e-6), 0.0, 1.0))

    # Рахуємо total лише як довідковий показник (Optuna його не використовує)
    total_heuristic = 0.40 * det_score + 0.25 * hole_score + 0.22 * continuity_score + 0.13 * tex_score

    return total_heuristic, {
        "det_score": det_score, "hole_score": hole_score,
        "continuity": continuity_score, "texture": tex_score,
        "hole_ratio": hole_ratio, "boundary_mae": bmae,
        "det_metrics": det,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Objective Smart pipeline tuner (Optuna Pareto Front)."
    )
    parser.add_argument("--image", default="test_image.jpg")
    parser.add_argument("--gt", default="", help='GT boxes: "x,y,w,h;x,y,w,h"')
    parser.add_argument("--trials", type=int, default=60, help="Number of Optuna trials.")
    parser.add_argument("--out", default="smart_tune_results.json")
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Cannot read: {args.image}")
    gt_boxes = parse_gt_boxes(args.gt)

    all_results = []
    t_start = time.time()

    def objective(trial):
        params = {
            "win_stride": trial.suggest_categorical("win_stride", ["4,4", "8,8"]),
            "padding":    trial.suggest_categorical("padding", ["8,8", "16,16", "32,32"]),
            "scale":      trial.suggest_float("scale", 1.02, 1.10, step=0.01),
            "hit_threshold":   trial.suggest_float("hit_threshold", 0.0, 0.8, step=0.1),
            "score_threshold": trial.suggest_float("score_threshold", 0.0, 0.5, step=0.1),
            "nms_threshold":   trial.suggest_float("nms_threshold", 0.2, 0.65, step=0.05),
            "use_preprocessing": trial.suggest_categorical("use_preprocessing", [True, False]),
            "shrink_factor":   trial.suggest_float("shrink_factor", 0.0, 0.25, step=0.02),
            "grabcut_iters":   trial.suggest_int("grabcut_iters", 3, 12),
        }

        ws = tuple(int(x) for x in params["win_stride"].split(","))
        pd = tuple(int(x) for x in params["padding"].split(","))
        params["win_stride"] = ws
        params["padding"] = pd

        try:
            result = run_pipeline(image, params)
        except Exception:
            # 4 нулі для 4 напрямків оптимізації
            return 0.0, 0.0, 0.0, 0.0

        if result is None:
            return 0.0, 0.0, 0.0, 0.0

        total_h, details = score_result(image, result, gt_boxes)

        # Зберігаємо параметри у trial для подальшого аналізу
        trial.set_user_attr("params", params)
        trial.set_user_attr("details", details)
        trial.set_user_attr("total_heuristic", total_h)

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
            "grabcut_iters": params["grabcut_iters"],
            **{k: round(v, 4) if isinstance(v, float) else v for k, v in details.items()}
        }
        all_results.append(log_data)

        # Повертаємо 4 значення! Optuna максимізуватиме їх усі одночасно.
        return details["det_score"], details["hole_score"], details["continuity"], details["texture"]

    # --- Run Optuna with MULTI-OBJECTIVE ---
    sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=15)
    # Задаємо 4 напрямки оптимізації (maximize для кожного з 4 метрик)
    study = optuna.create_study(
        directions=["maximize", "maximize", "maximize", "maximize"], 
        sampler=sampler
    )

    print(f"Optuna Multi-Objective Optimization: {args.trials} trials")
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)
    elapsed = time.time() - t_start

    # --- Аналіз Фронту Парето ---
    pareto_trials = study.best_trials
    
    if not pareto_trials:
        print("Не знайдено жодного валідного результату!")
        return

    # Формуємо список найкращих рішень з фронту Парето
    pareto_results = []
    for t in pareto_trials:
        p_data = t.user_attrs.get("params", {})
        d_data = t.user_attrs.get("details", {})
        pareto_results.append({
            "trial_id": t.number,
            "scores": {
                "det_score": round(t.values[0], 4),
                "hole_score": round(t.values[1], 4),
                "continuity": round(t.values[2], 4),
                "texture": round(t.values[3], 4),
            },
            "heuristic_total": round(t.user_attrs.get("total_heuristic", 0.0), 4),
            "params": p_data
        })

    # Щоб оновити config і зберегти картинку, нам треба вибрати ОДИН найкращий з Парето.
    # Найпростіше - взяти той, де сума 4 метрик найбільша.
    best_balanced_trial = max(pareto_trials, key=lambda t: sum(t.values))
    best_params = best_balanced_trial.user_attrs["params"]
    best_details = best_balanced_trial.user_attrs["details"]

    # Перемальовуємо найкращий збалансований варіант, щоб зберегти його
    best_result_img = run_pipeline(image, best_params)
    if best_result_img:
        cv2.imwrite("smart_tune_best_balanced.jpg", best_result_img["final"])

    # Збереження JSON
    payload = {
        "method": "Multi-Objective Bayesian Optimization (MOTPE)",
        "trials": args.trials,
        "elapsed_sec": round(elapsed, 1),
        "pareto_front_size": len(pareto_trials),
        "pareto_front": sorted(pareto_results, key=lambda x: x["heuristic_total"], reverse=True),
        "all_trials": all_results
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    # Авто-оновлення config.cfg збалансованим варіантом
    ws = best_params.get("win_stride", (8, 8))
    pd = best_params.get("padding", (8, 8))
    cfg = configparser.ConfigParser()
    cfg["pipeline"] = {"input": args.image, "output": "result_image.jpg"}
    cfg["detection"] = {
        "win_stride_w": str(ws[0]),
        "win_stride_h": str(ws[1]),
        "padding_w": str(pd[0]),
        "padding_h": str(pd[1]),
        "scale": str(best_params["scale"]),
        "score_threshold": str(best_params["score_threshold"]),
        "nms_threshold": str(best_params["nms_threshold"]),
        "hit_threshold": str(best_params["hit_threshold"]),
        "use_preprocessing": str(best_params["use_preprocessing"]).lower(),
        "shrink_factor": str(best_params["shrink_factor"]),
    }
    cfg["masks"] = {"grabcut_iters": str(best_params["grabcut_iters"])}
    with open("config.cfg", "w", encoding="utf-8") as f:
        cfg.write(f)

    print(f"\n{'='*60}")
    print(f"ЗНАЙДЕНО ФРОНТ ПАРЕТО: {len(pareto_trials)} компромісних рішень")
    print(f"{'='*60}")
    print("Вибрано найбільш збалансований варіант для config.cfg:")
    print(f"  win_stride      = {best_params['win_stride']}")
    print(f"  padding         = {best_params['padding']}")
    print(f"  scale           = {best_params['scale']}")
    print(f"  hit_threshold   = {best_params['hit_threshold']}")
    print(f"  score_threshold = {best_params['score_threshold']}")
    print(f"  nms_threshold   = {best_params['nms_threshold']}")
    print(f"  grabcut_iters   = {best_params['grabcut_iters']}")
    
    print(f"\nЙого показники:")
    print(f"  det_score       = {best_balanced_trial.values[0]:.4f}")
    print(f"  hole_score      = {best_balanced_trial.values[1]:.4f}")
    print(f"  continuity      = {best_balanced_trial.values[2]:.4f}")
    print(f"  texture         = {best_balanced_trial.values[3]:.4f}")
    
    print(f"\nDone in {elapsed:.0f}s ({args.trials} trials)")
    print(f"Файл config.cfg оновлено збалансованим варіантом.")
    print(f"Всі рішення Фронту Парето збережено у {args.out}")

if __name__ == "__main__":
    main()