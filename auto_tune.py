import argparse
import configparser
import json
import time
from itertools import product
import cv2
import numpy as np
from detection import detect_humans, get_binary_masks
from inpainting import inpaint_neighbor_averaging
from metrics import evaluate_detections
from transformation import apply_translation
PARAM_GRID = {'win_stride': [(8, 8), (4, 4)], 'padding': [(8, 8), (16, 16)], 'scale': [1.03, 1.05, 1.08], 'hit_threshold': [0.0, 0.2, 0.5], 'score_threshold': [0.0, 0.1, 0.3], 'nms_threshold': [0.25, 0.35, 0.5], 'use_preprocessing': [True, False], 'shrink_factor': [0.0, 0.08, 0.12, 0.15, 0.2], 'grabcut_iters': [5, 7, 10]}

def parse_gt_boxes(gt_raw):
    if not gt_raw:
        return []
    boxes = []
    for chunk in [c.strip() for c in gt_raw.split(';') if c.strip()]:
        parts = [v.strip() for v in chunk.split(',')]
        if len(parts) != 4:
            raise ValueError(f"Invalid GT box: '{chunk}'")
        boxes.append(tuple((int(v) for v in parts)))
    return boxes

def run_pipeline(image, params):
    pred_boxes = detect_humans(image, win_stride=params['win_stride'], padding=params['padding'], scale=params['scale'], score_threshold=params['score_threshold'], nms_threshold=params['nms_threshold'], hit_threshold=params['hit_threshold'], use_preprocessing=params['use_preprocessing'], shrink_factor=params['shrink_factor'])
    if len(pred_boxes) < 2:
        return None
    masks_data = get_binary_masks(image, pred_boxes, top_k=2, grabcut_iters=params['grabcut_iters'])
    if len(masks_data) < 2:
        return None
    mask_A, rect_A = masks_data[0]
    mask_B, rect_B = masks_data[1]
    cA_x = rect_A[0] + rect_A[2] // 2
    cA_y = rect_A[1] + rect_A[3] // 2
    cB_x = rect_B[0] + rect_B[2] // 2
    cB_y = rect_B[1] + rect_B[3] // 2
    tA_x, tA_y = (cB_x - cA_x, cB_y - cA_y)
    tB_x, tB_y = (cA_x - cB_x, cA_y - cB_y)
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
    final = alpha[:, :, None] * composite.astype(np.float64) + (1.0 - alpha[:, :, None]) * clean_bg.astype(np.float64)
    final = np.clip(final, 0, 255).astype(np.uint8)
    return {'final': final, 'clean_bg': clean_bg, 'inpaint_mask': inpaint_mask, 'pred_boxes': pred_boxes, 'masks_data': masks_data, 'combined_original': combined_original}

def boundary_mae(image, hole_mask):
    kernel = np.ones((3, 3), np.uint8)
    ring = cv2.dilate(hole_mask.astype(np.uint8), kernel, iterations=1) - hole_mask.astype(np.uint8)
    if not np.any(ring > 0) or not np.any(hole_mask > 0):
        return 255.0
    ring_mean = image[ring > 0].mean(axis=0)
    hole_mean = image[hole_mask > 0].mean(axis=0)
    return float(np.abs(ring_mean - hole_mean).mean())

def texture_ratio(image, hole_mask):
    if not np.any(hole_mask > 0):
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    hole_var = float(np.var(lap[hole_mask > 0]))
    kernel = np.ones((5, 5), np.uint8)
    ring = cv2.dilate(hole_mask.astype(np.uint8), kernel, iterations=1) - hole_mask.astype(np.uint8)
    if not np.any(ring > 0):
        return 0.0
    ring_var = float(np.var(lap[ring > 0]))
    return float(np.clip(hole_var / max(ring_var, 1e-06), 0.0, 1.0))

def score_result(image, result, gt_boxes):
    pred_boxes = result['pred_boxes']
    final = result['final']
    inpaint_mask = result['inpaint_mask']
    if gt_boxes:
        det = evaluate_detections(pred_boxes, gt_boxes, iou_threshold=0.5)
        det_score = 0.55 * det['f1'] + 0.45 * det['mean_iou_matched']
    else:
        det = None
        det_score = 1.0 if len(pred_boxes) >= 2 else 0.0
    h, w = image.shape[:2]
    hole_ratio = float(np.sum(inpaint_mask > 0)) / (h * w)
    hole_score = float(np.clip(1.0 - hole_ratio * 5.0, 0.0, 1.0))
    bmae = boundary_mae(final, inpaint_mask)
    continuity_score = float(np.exp(-bmae / 28.0))
    tex_score = texture_ratio(final, inpaint_mask)
    total = 0.4 * det_score + 0.25 * hole_score + 0.22 * continuity_score + 0.13 * tex_score
    return {'total': round(total, 5), 'det_score': round(det_score, 4), 'hole_score': round(hole_score, 4), 'continuity': round(continuity_score, 4), 'texture': round(tex_score, 4), 'hole_ratio': round(hole_ratio, 4), 'boundary_mae': round(bmae, 2), 'det_metrics': det}

def main():
    parser = argparse.ArgumentParser(description='Auto-tune the full pipeline: find best params by exhaustive search.')
    parser.add_argument('--image', default='test_image.jpg', help='Input image.')
    parser.add_argument('--gt', default='', help='GT boxes: "x,y,w,h;x,y,w,h"')
    parser.add_argument('--out', default='auto_tune_results.json', help='Output JSON.')
    parser.add_argument('--max-configs', type=int, default=0, help='If >0, randomly sample this many configs instead of full grid.')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f'Cannot read: {args.image}')
    gt_boxes = parse_gt_boxes(args.gt)
    keys = list(PARAM_GRID.keys())
    all_values = [PARAM_GRID[k] for k in keys]
    combos = [dict(zip(keys, vals)) for vals in product(*all_values)]
    if args.max_configs > 0 and args.max_configs < len(combos):
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(combos), size=args.max_configs, replace=False)
        combos = [combos[i] for i in idx]
    print(f'Testing {len(combos)} parameter combinations...')
    print(f'Image: {args.image} ({image.shape[1]}x{image.shape[0]})')
    if gt_boxes:
        print(f'GT boxes: {gt_boxes}')
    print()
    best = None
    best_score = -1.0
    results = []
    skipped = 0
    t_start = time.time()
    for i, params in enumerate(combos):
        try:
            result = run_pipeline(image, params)
        except Exception as e:
            skipped += 1
            continue
        if result is None:
            skipped += 1
            continue
        scored = score_result(image, result, gt_boxes)
        row = {**params, **scored}
        for k in ('win_stride', 'padding'):
            if isinstance(row[k], tuple):
                row[k] = list(row[k])
        results.append(row)
        if scored['total'] > best_score:
            best_score = scored['total']
            best = row
            cv2.imwrite('auto_tune_best.jpg', result['final'])
            print(f"  [{i + 1}/{len(combos)}] ** NEW BEST total={scored['total']:.4f} (det={scored['det_score']:.3f} hole={scored['hole_score']:.3f} cont={scored['continuity']:.3f} tex={scored['texture']:.3f})")
            print(f"    stride={params['win_stride']} pad={params['padding']} scale={params['scale']} hit={params['hit_threshold']} score={params['score_threshold']} nms={params['nms_threshold']} prep={params['use_preprocessing']} shrink={params['shrink_factor']} gc={params['grabcut_iters']}")
        if (i + 1) % 20 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (len(combos) - i - 1) / rate
            print(f'  [{i + 1}/{len(combos)}] valid={len(results)} skipped={skipped} best={best_score:.4f} elapsed={elapsed:.0f}s ETA={eta:.0f}s')
    elapsed = time.time() - t_start
    print(f'\nDone in {elapsed:.1f}s. Tested={len(combos)}, valid={len(results)}, skipped={skipped}')
    if best is None:
        print('No valid configuration found!')
        return
    results.sort(key=lambda r: r['total'], reverse=True)
    payload = {'tested': len(combos), 'valid': len(results), 'best': best, 'top20': results[:20]}
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    ws = best.get('win_stride', [8, 8])
    pd = best.get('padding', [8, 8])
    cfg = configparser.ConfigParser()
    cfg['pipeline'] = {'input': args.image, 'output': 'result_image.jpg'}
    cfg['detection'] = {'win_stride_w': str(ws[0] if isinstance(ws, list) else ws[0]), 'win_stride_h': str(ws[1] if isinstance(ws, list) else ws[1]), 'padding_w': str(pd[0] if isinstance(pd, list) else pd[0]), 'padding_h': str(pd[1] if isinstance(pd, list) else pd[1]), 'scale': str(best['scale']), 'score_threshold': str(best['score_threshold']), 'nms_threshold': str(best['nms_threshold']), 'hit_threshold': str(best['hit_threshold']), 'use_preprocessing': str(best['use_preprocessing']).lower(), 'shrink_factor': str(best['shrink_factor'])}
    cfg['masks'] = {'grabcut_iters': str(best['grabcut_iters'])}
    with open('config.cfg', 'w', encoding='utf-8') as f:
        cfg.write(f)
    print(f"\n{'=' * 60}")
    print(f'BEST PARAMETERS (total_score = {best_score:.4f})')
    print(f"{'=' * 60}")
    print(f"  win_stride      = {best['win_stride']}")
    print(f"  padding         = {best['padding']}")
    print(f"  scale           = {best['scale']}")
    print(f"  hit_threshold   = {best['hit_threshold']}")
    print(f"  score_threshold = {best['score_threshold']}")
    print(f"  nms_threshold   = {best['nms_threshold']}")
    print(f"  use_preprocess  = {best['use_preprocessing']}")
    print(f"  shrink_factor   = {best['shrink_factor']}")
    print(f"  grabcut_iters   = {best['grabcut_iters']}")
    print(f"\n  det_score       = {best['det_score']}")
    print(f"  hole_score      = {best['hole_score']}")
    print(f"  continuity      = {best['continuity']}")
    print(f"  texture         = {best['texture']}")
    print(f"  hole_ratio      = {best['hole_ratio']}")
    print(f"  boundary_mae    = {best['boundary_mae']}")
    if best.get('det_metrics'):
        d = best['det_metrics']
        print(f"  TP/FP/FN        = {d['tp']}/{d['fp']}/{d['fn']}")
        print(f"  F1              = {d['f1']:.4f}")
        print(f"  IoU             = {d['mean_iou_matched']:.4f}")
    print(f'\nconfig.cfg updated automatically.')
    print(f'Best result saved to auto_tune_best.jpg')
    print(f'Full results in {args.out}')
if __name__ == '__main__':
    main()
