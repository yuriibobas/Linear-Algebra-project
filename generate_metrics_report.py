import argparse
import os
import time
from itertools import product
import cv2
import numpy as np

from detection import detect_humans, get_binary_masks
from inpainting import inpaint_neighbor_averaging
from metrics import calculate_mse, evaluate_detections


def parse_gt_boxes(gt_raw):
    if not gt_raw:
        return []
    boxes = []
    chunks = [c.strip() for c in gt_raw.split(";") if c.strip()]
    for chunk in chunks:
        parts = [v.strip() for v in chunk.split(",")]
        if len(parts) != 4:
            raise ValueError(
                f"Invalid GT box format: '{chunk}'. Expected exactly 4 values: x,y,w,h"
            )
        if any(not p.lstrip("-").isdigit() for p in parts):
            raise ValueError(
                "GT boxes must contain integer coordinates only. "
                'Example: --gt "120,45,80,210;330,60,95,230"'
            )
        vals = [int(v) for v in parts]
        boxes.append(tuple(vals))
    return boxes


def random_rect_mask(h, w, min_size=0.08, max_size=0.2, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    rh = int(rng.uniform(min_size, max_size) * h)
    rw = int(rng.uniform(min_size, max_size) * w)
    y = int(rng.uniform(0, max(1, h - rh)))
    x = int(rng.uniform(0, max(1, w - rw)))
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y : y + rh, x : x + rw] = 1
    return mask


def benchmark_inpainting_mse(image, num_trials=5, seed=42):
    h, w = image.shape[:2]
    rng = np.random.default_rng(seed)
    region_mse = []
    full_mse = []
    timings = []

    for _ in range(num_trials):
        mask = random_rect_mask(h, w, rng=rng)
        corrupted = image.copy()
        corrupted[mask > 0] = [0, 0, 0]

        t0 = time.time()
        restored = inpaint_neighbor_averaging(corrupted, mask)
        timings.append(time.time() - t0)

        region_mse.append(calculate_mse(restored[mask > 0], image[mask > 0]))
        full_mse.append(calculate_mse(restored, image))

    return {
        "region_mse_mean": float(np.mean(region_mse)),
        "region_mse_std": float(np.std(region_mse)),
        "full_mse_mean": float(np.mean(full_mse)),
        "full_mse_std": float(np.std(full_mse)),
        "avg_runtime_sec": float(np.mean(timings)),
    }


def tune_detection_params(image, gt_boxes):
    grid = {
        "win_stride": [(8, 8), (4, 4)],
        "padding": [(8, 8), (16, 16)],
        "scale": [1.03, 1.05, 1.08],
        "score_threshold": [0.0, 0.1, 0.2, 0.3],
        "nms_threshold": [0.25, 0.35, 0.45],
        "grabcut_iters": [3, 5, 7],
    }

    best = None
    best_score = (-1.0, -1.0)
    tested = 0

    for values in product(
        grid["win_stride"],
        grid["padding"],
        grid["scale"],
        grid["score_threshold"],
        grid["nms_threshold"],
        grid["grabcut_iters"],
    ):
        win_stride, padding, scale, score_thr, nms_thr, gc_iters = values
        pred_boxes = detect_humans(
            image,
            win_stride=win_stride,
            padding=padding,
            scale=scale,
            score_threshold=score_thr,
            nms_threshold=nms_thr,
        )

        _ = get_binary_masks(image, pred_boxes, top_k=2, grabcut_iters=gc_iters)

        det = evaluate_detections(pred_boxes, gt_boxes, iou_threshold=0.5)
        score = (det["f1"], det["mean_iou_matched"])
        tested += 1

        if score > best_score:
            best_score = score
            best = {
                "win_stride": win_stride,
                "padding": padding,
                "scale": scale,
                "score_threshold": score_thr,
                "nms_threshold": nms_thr,
                "grabcut_iters": gc_iters,
                "metrics": det,
            }

    best["tested_configs"] = tested
    return best


def build_report(image_path, result_path, gt_boxes_raw, trials, tune):
    original = cv2.imread(image_path)
    if original is None:
        raise FileNotFoundError(f"Cannot read input image: {image_path}")

    result = cv2.imread(result_path) if result_path else None
    if result_path and result is None:
        raise FileNotFoundError(f"Cannot read result image: {result_path}")

    gt_boxes = parse_gt_boxes(gt_boxes_raw)
    best_tuned = None

    if tune and gt_boxes:
        best_tuned = tune_detection_params(original, gt_boxes)
        pred_boxes = detect_humans(
            original,
            win_stride=best_tuned["win_stride"],
            padding=best_tuned["padding"],
            scale=best_tuned["scale"],
            score_threshold=best_tuned["score_threshold"],
            nms_threshold=best_tuned["nms_threshold"],
        )
        masks_data = (
            get_binary_masks(
                original,
                pred_boxes,
                top_k=2,
                grabcut_iters=best_tuned["grabcut_iters"],
            )
            if len(pred_boxes)
            else []
        )
    else:
        pred_boxes = detect_humans(original)
        masks_data = get_binary_masks(original, pred_boxes) if len(pred_boxes) else []

    lines = []
    lines.append("# Metrics Report")
    lines.append("")
    lines.append(f"- Input image: `{os.path.basename(image_path)}`")
    lines.append(
        f"- Result image: `{os.path.basename(result_path)}`"
        if result_path
        else "- Result image: `N/A`"
    )
    lines.append(f"- Humans detected (HOG+SVM): **{len(pred_boxes)}**")
    lines.append(f"- Masks extracted (GrabCut): **{len(masks_data)}**")
    lines.append("")

    if gt_boxes:
        det = evaluate_detections(pred_boxes, gt_boxes, iou_threshold=0.5)
        lines.append("## Detection Metrics (vs Ground Truth)")
        lines.append(f"- TP: {det['tp']}, FP: {det['fp']}, FN: {det['fn']}")
        lines.append(f"- Precision: {det['precision']:.4f}")
        lines.append(f"- Recall: {det['recall']:.4f}")
        lines.append(f"- F1-score: {det['f1']:.4f}")
        lines.append(f"- Mean IoU (matched): {det['mean_iou_matched']:.4f}")
        lines.append("")
        if best_tuned is not None:
            lines.append("## Tuned Coefficients (Best by F1, then IoU)")
            lines.append(f"- Tested configurations: {best_tuned['tested_configs']}")
            lines.append(f"- winStride: {best_tuned['win_stride']}")
            lines.append(f"- padding: {best_tuned['padding']}")
            lines.append(f"- scale: {best_tuned['scale']}")
            lines.append(f"- score_threshold: {best_tuned['score_threshold']}")
            lines.append(f"- nms_threshold: {best_tuned['nms_threshold']}")
            lines.append(f"- grabcut_iters: {best_tuned['grabcut_iters']}")
            lines.append("")
    else:
        lines.append("## Detection Metrics (vs Ground Truth)")
        lines.append(
            "- Ground-truth boxes were not provided, so Precision/Recall/F1/IoU are `N/A`."
        )
        lines.append(
            '- Provide GT using `--gt "x,y,w,h;x,y,w,h"` to enable full detection metrics.'
        )
        lines.append("")

    lines.append("## Inpainting Quality (Self-supervised MSE)")
    bench = benchmark_inpainting_mse(original, num_trials=trials, seed=42)
    lines.append(
        f"- Region MSE mean: {bench['region_mse_mean']:.4f} +/- {bench['region_mse_std']:.4f}"
    )
    lines.append(
        f"- Full-image MSE mean: {bench['full_mse_mean']:.4f} +/- {bench['full_mse_std']:.4f}"
    )
    lines.append(f"- Average inpainting runtime: {bench['avg_runtime_sec']:.4f} sec")
    lines.append("")

    if result is not None and result.shape == original.shape:
        result_mse = calculate_mse(original, result)
        lines.append("## Swap Pipeline Output")
        lines.append(f"- MSE(original vs final result): {result_mse:.4f}")
        lines.append(
            "- Note: this value is expected to be high because object positions are intentionally changed."
        )
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate metrics report for the lab pipeline.")
    parser.add_argument("--image", default="test_image.jpg", help="Path to input image.")
    parser.add_argument("--result", default="result_image.jpg", help="Path to final result image.")
    parser.add_argument("--gt", default="", help='GT boxes: "x,y,w,h;x,y,w,h"')
    parser.add_argument("--trials", type=int, default=5, help="Number of inpainting benchmark trials.")
    parser.add_argument("--out", default="metrics_report.md", help="Output report file.")
    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="Disable coefficient tuning (enabled by default when GT is provided).",
    )
    args = parser.parse_args()

    try:
        report = build_report(args.image, args.result, args.gt, args.trials, tune=not args.no_tune)
    except ValueError as e:
        print(f"Input error: {e}")
        return
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(report + "\n")

    print(f"Saved metrics report to {args.out}")


if __name__ == "__main__":
    main()
