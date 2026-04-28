import argparse
import json
from itertools import product
import cv2
import numpy as np
from detection import detect_humans, get_binary_masks
from inpainting import inpaint_neighbor_averaging
from metrics import evaluate_detections
from transformation import apply_translation

def parse_gt_boxes(gt_raw):
    if not gt_raw:
        return []
    boxes = []
    for chunk in [c.strip() for c in gt_raw.split(';') if c.strip()]:
        parts = [v.strip() for v in chunk.split(',')]
        if len(parts) != 4:
            raise ValueError(f"Invalid GT box format: '{chunk}'. Expected x,y,w,h")
        vals = [int(v) for v in parts]
        boxes.append(tuple(vals))
    return boxes

def swap_and_inpaint(image, masks_data):
    (mask_a, rect_a), (mask_b, rect_b) = masks_data[:2]
    cax = rect_a[0] + rect_a[2] // 2
    cay = rect_a[1] + rect_a[3] // 2
    cbx = rect_b[0] + rect_b[2] // 2
    cby = rect_b[1] + rect_b[3] // 2
    ta_x, ta_y = (cbx - cax, cby - cay)
    tb_x, tb_y = (cax - cbx, cay - cby)
    translated_a, translated_mask_a = apply_translation(image, mask_a, ta_x, ta_y)
    translated_b, translated_mask_b = apply_translation(image, mask_b, tb_x, tb_y)
    composite = image.copy()
    combined_original = cv2.bitwise_or(mask_a, mask_b)
    composite[combined_original > 0] = [0, 0, 0]
    idx_a = translated_mask_a > 0
    composite[idx_a] = translated_a[idx_a]
    idx_b = translated_mask_b > 0
    composite[idx_b] = translated_b[idx_b]
    holes = combined_original.copy()
    holes[translated_mask_a > 0] = 0
    holes[translated_mask_b > 0] = 0
    final = inpaint_neighbor_averaging(composite, holes)
    return (composite, final, holes)

def boundary_mae(image, hole_mask):
    kernel = np.ones((3, 3), np.uint8)
    ring = cv2.dilate(hole_mask.astype(np.uint8), kernel, iterations=1) - hole_mask.astype(np.uint8)
    ring_idx = ring > 0
    hole_idx = hole_mask > 0
    if not np.any(ring_idx) or not np.any(hole_idx):
        return 255.0
    ring_mean = image[ring_idx].mean(axis=0)
    hole_mean = image[hole_idx].mean(axis=0)
    return float(np.abs(ring_mean - hole_mean).mean())

def hole_texture_ratio(image, hole_mask):
    hole_idx = hole_mask > 0
    if not np.any(hole_idx):
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    hole_var = float(np.var(lap[hole_idx]))
    kernel = np.ones((5, 5), np.uint8)
    ring = cv2.dilate(hole_mask.astype(np.uint8), kernel, iterations=1) - hole_mask.astype(np.uint8)
    ring_idx = ring > 0
    if not np.any(ring_idx):
        return 0.0
    ring_var = float(np.var(lap[ring_idx]))
    if ring_var <= 1e-06:
        return 0.0
    return float(np.clip(hole_var / ring_var, 0.0, 1.0))

def score_config(image, pred_boxes, gt_boxes, final_image, holes):
    if gt_boxes:
        det = evaluate_detections(pred_boxes, gt_boxes, iou_threshold=0.5)
        det_score = 0.55 * det['f1'] + 0.45 * det['mean_iou_matched']
    else:
        det = None
        det_score = 1.0 if len(pred_boxes) >= 2 else 0.0
    hole_ratio = float(np.mean(holes > 0))
    hole_score = float(np.clip(1.0 - hole_ratio * 6.0, 0.0, 1.0))
    bmae = boundary_mae(final_image, holes)
    continuity_score = float(np.exp(-bmae / 28.0))
    texture_score = hole_texture_ratio(final_image, holes)
    total = 0.42 * det_score + 0.23 * hole_score + 0.22 * continuity_score + 0.13 * texture_score
    return {'total_score': float(total), 'det_score': float(det_score), 'hole_score': float(hole_score), 'continuity_score': float(continuity_score), 'texture_score': float(texture_score), 'hole_ratio': hole_ratio, 'boundary_mae': float(bmae), 'det_metrics': det}

def build_grid():
    return {'win_stride': [(8, 8), (4, 4)], 'padding': [(8, 8), (16, 16)], 'scale': [1.03, 1.05], 'score_threshold': [0.0, 0.1, 0.2], 'nms_threshold': [0.25, 0.35], 'hit_threshold': [0.0, 0.2], 'use_preprocessing': [True], 'shrink_factor': [0.0, 0.08], 'grabcut_iters': [5, 7]}

def tune_full_pipeline(image, gt_boxes=None, max_configs=0, seed=42):
    grid = build_grid()
    combos = list(product(grid['win_stride'], grid['padding'], grid['scale'], grid['score_threshold'], grid['nms_threshold'], grid['hit_threshold'], grid['use_preprocessing'], grid['shrink_factor'], grid['grabcut_iters']))
    if max_configs > 0 and max_configs < len(combos):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(combos), size=max_configs, replace=False)
        combos = [combos[i] for i in idx]
    results = []
    best = None
    best_score = -1.0
    for i, values in enumerate(combos):
        win_stride, padding, scale, score_thr, nms_thr, hit_thr, use_preprocessing, shrink_factor, grabcut_iters = values
        pred_boxes = detect_humans(image, win_stride=win_stride, padding=padding, scale=scale, score_threshold=score_thr, nms_threshold=nms_thr, hit_threshold=hit_thr, use_preprocessing=use_preprocessing, shrink_factor=shrink_factor)
        if len(pred_boxes) < 2:
            continue
        masks_data = get_binary_masks(image, pred_boxes, top_k=2, grabcut_iters=grabcut_iters)
        if len(masks_data) < 2:
            continue
        _, final_image, holes = swap_and_inpaint(image, masks_data)
        scored = score_config(image, pred_boxes, gt_boxes or [], final_image, holes)
        row = {'win_stride': win_stride, 'padding': padding, 'scale': scale, 'score_threshold': score_thr, 'nms_threshold': nms_thr, 'hit_threshold': hit_thr, 'use_preprocessing': use_preprocessing, 'shrink_factor': shrink_factor, 'grabcut_iters': grabcut_iters, 'num_boxes': len(pred_boxes), **scored}
        results.append(row)
        if row['total_score'] > best_score:
            best_score = row['total_score']
            best = row
            cv2.imwrite('best_pipeline_result.jpg', final_image)
            print(f"[{i + 1}/{len(combos)}] New best total={best_score:.4f} (det={row['det_score']:.4f}, hole={row['hole_score']:.4f}, cont={row['continuity_score']:.4f}, tex={row['texture_score']:.4f})")
        if (i + 1) % 25 == 0 or i + 1 == len(combos):
            print(f'Progress: {i + 1}/{len(combos)} tested, valid={len(results)}')
    results.sort(key=lambda r: r['total_score'], reverse=True)
    return (best, results[:20], len(combos), len(results))

def main():
    parser = argparse.ArgumentParser(description='Tune full detect->mask->swap->inpaint pipeline by visual proxy score.')
    parser.add_argument('--image', default='test_image.jpg', help='Input image path.')
    parser.add_argument('--gt', default='', help='Optional GT boxes: "x,y,w,h;x,y,w,h"')
    parser.add_argument('--max-configs', type=int, default=0, help='If >0, randomly sample this many configs from the full grid.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling.')
    parser.add_argument('--out', default='full_pipeline_tuning_results.json', help='Output JSON path.')
    args = parser.parse_args()
    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f'Cannot read image: {args.image}')
    gt_boxes = parse_gt_boxes(args.gt)
    best, top20, tested, valid = tune_full_pipeline(image, gt_boxes=gt_boxes, max_configs=args.max_configs, seed=args.seed)
    if best is None:
        print('No valid configuration found (need at least 2 detected humans per config).')
        return
    payload = {'tested_configs': tested, 'valid_configs': valid, 'best': best, 'top20': top20}
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    print('\nBest configuration:')
    print(json.dumps(best, indent=2))
    print(f'\nSaved results to {args.out}')
    print('Saved best visual output to best_pipeline_result.jpg')
if __name__ == '__main__':
    main()
