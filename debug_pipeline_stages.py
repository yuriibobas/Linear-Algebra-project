import argparse
import configparser
import os
import cv2
import numpy as np
from detection import detect_humans, get_binary_masks, preprocess_for_hog
from inpainting import inpaint_neighbor_averaging
from transformation import apply_translation

def load_cfg(path):
    cfg = configparser.ConfigParser()
    cfg.read(path, encoding='utf-8')
    return cfg

def _get(cfg, section, key, fallback):
    if cfg.has_option(section, key):
        if isinstance(fallback, bool):
            return cfg.getboolean(section, key)
        if isinstance(fallback, int):
            return cfg.getint(section, key)
        if isinstance(fallback, float):
            return cfg.getfloat(section, key)
        return cfg.get(section, key)
    return fallback

def draw_boxes(image, boxes, color=(0, 255, 0), thickness=2):
    vis = image.copy()
    for i, (x, y, w, h) in enumerate(boxes):
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, thickness)
        cv2.putText(vis, f'#{i} ({x},{y},{w},{h})', (x, max(16, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return vis

def overlay_mask(image, mask, color=(0, 0, 255), alpha=0.45):
    vis = image.copy().astype(np.float32)
    layer = np.zeros_like(vis)
    layer[:, :] = color
    idx = mask > 0
    vis[idx] = vis[idx] * (1.0 - alpha) + layer[idx] * alpha
    return np.clip(vis, 0, 255).astype(np.uint8)

def ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

def main():
    parser = argparse.ArgumentParser(description='Save debug images for each pipeline stage.')
    parser.add_argument('--config', default='config.cfg', help='Config file path.')
    parser.add_argument('--input', default='', help='Override input image path.')
    parser.add_argument('--outdir', default='debug_stages', help='Output directory for debug images.')
    args = parser.parse_args()
    cfg = load_cfg(args.config)
    input_path = args.input or _get(cfg, 'pipeline', 'input', 'test_image.jpg')
    ensure_dir(args.outdir)
    image = cv2.imread(input_path)
    if image is None:
        raise FileNotFoundError(f'Cannot read input image: {input_path}')
    win_stride = (_get(cfg, 'detection', 'win_stride_w', 8), _get(cfg, 'detection', 'win_stride_h', 8))
    padding = (_get(cfg, 'detection', 'padding_w', 8), _get(cfg, 'detection', 'padding_h', 8))
    scale = _get(cfg, 'detection', 'scale', 1.05)
    score_threshold = _get(cfg, 'detection', 'score_threshold', 0.0)
    nms_threshold = _get(cfg, 'detection', 'nms_threshold', 0.25)
    hit_threshold = _get(cfg, 'detection', 'hit_threshold', 0.0)
    use_preprocessing = _get(cfg, 'detection', 'use_preprocessing', True)
    shrink_factor = _get(cfg, 'detection', 'shrink_factor', 0.0)
    grabcut_iters = _get(cfg, 'masks', 'grabcut_iters', 5)
    cv2.imwrite(os.path.join(args.outdir, '00_original.jpg'), image)
    if use_preprocessing:
        pre = preprocess_for_hog(image)
    else:
        pre = image.copy()
    cv2.imwrite(os.path.join(args.outdir, '01_hog_input.jpg'), pre)
    boxes = detect_humans(image, win_stride=win_stride, padding=padding, scale=scale, score_threshold=score_threshold, nms_threshold=nms_threshold, hit_threshold=hit_threshold, use_preprocessing=use_preprocessing, shrink_factor=shrink_factor)
    cv2.imwrite(os.path.join(args.outdir, '02_detected_boxes.jpg'), draw_boxes(image, boxes))
    if len(boxes) < 2:
        print('Detected fewer than 2 people; stopped after detection debug.')
        return
    masks_data = get_binary_masks(image, boxes, top_k=2, grabcut_iters=grabcut_iters)
    if len(masks_data) < 2:
        print('Failed to extract at least 2 masks; stopped after mask debug.')
        return
    (mask_a, rect_a), (mask_b, rect_b) = (masks_data[0], masks_data[1])
    cv2.imwrite(os.path.join(args.outdir, '03_mask_a.jpg'), (mask_a * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(args.outdir, '04_mask_b.jpg'), (mask_b * 255).astype(np.uint8))
    overlay_a = overlay_mask(image, mask_a, color=(0, 0, 255), alpha=0.45)
    overlay_ab = overlay_mask(overlay_a, mask_b, color=(0, 255, 0), alpha=0.45)
    cv2.imwrite(os.path.join(args.outdir, '05_masks_overlay.jpg'), overlay_ab)
    cax = rect_a[0] + rect_a[2] // 2
    cay = rect_a[1] + rect_a[3] // 2
    cbx = rect_b[0] + rect_b[2] // 2
    cby = rect_b[1] + rect_b[3] // 2
    ta_x, ta_y = (cbx - cax, cby - cay)
    tb_x, tb_y = (cax - cbx, cay - cby)
    trans_a, trans_mask_a = apply_translation(image, mask_a, ta_x, ta_y)
    trans_b, trans_mask_b = apply_translation(image, mask_b, tb_x, tb_y)
    cv2.imwrite(os.path.join(args.outdir, '06_translated_person_a.jpg'), trans_a)
    cv2.imwrite(os.path.join(args.outdir, '07_translated_person_b.jpg'), trans_b)
    cv2.imwrite(os.path.join(args.outdir, '08_translated_mask_a.jpg'), (trans_mask_a * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(args.outdir, '09_translated_mask_b.jpg'), (trans_mask_b * 255).astype(np.uint8))
    combined_original_masks = cv2.bitwise_or(mask_a, mask_b)
    inpaint_mask = cv2.dilate(combined_original_masks, np.ones((7, 7), np.uint8), iterations=2)
    cv2.imwrite(os.path.join(args.outdir, '10_inpaint_mask.jpg'), (inpaint_mask * 255).astype(np.uint8))
    clean_bg_input = image.copy()
    clean_bg_input[inpaint_mask > 0] = [0, 0, 0]
    cv2.imwrite(os.path.join(args.outdir, '11_clean_bg_input.jpg'), clean_bg_input)
    clean_bg = inpaint_neighbor_averaging(clean_bg_input, inpaint_mask)
    cv2.imwrite(os.path.join(args.outdir, '12_clean_bg_inpainted.jpg'), clean_bg)
    composite = clean_bg.copy()
    composite[trans_mask_a > 0] = trans_a[trans_mask_a > 0]
    composite[trans_mask_b > 0] = trans_b[trans_mask_b > 0]
    cv2.imwrite(os.path.join(args.outdir, '13_composite_no_blend.jpg'), composite)
    combined_placed = cv2.bitwise_or(trans_mask_a, trans_mask_b)
    inner_mask = cv2.erode(combined_placed, np.ones((5, 5), np.uint8), iterations=2)
    alpha = cv2.GaussianBlur(combined_placed.astype(np.float32) * 255, (15, 15), 0) / 255.0
    alpha = np.clip(alpha, 0, 1)
    alpha[inner_mask > 0] = 1.0
    alpha_vis = (alpha * 255.0).astype(np.uint8)
    cv2.imwrite(os.path.join(args.outdir, '14_alpha_mask.jpg'), alpha_vis)
    final = alpha[:, :, None] * composite.astype(np.float64) + (1.0 - alpha[:, :, None]) * clean_bg.astype(np.float64)
    final = np.clip(final, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(args.outdir, '15_final_blended.jpg'), final)
    with open(os.path.join(args.outdir, 'debug_summary.txt'), 'w', encoding='utf-8') as f:
        f.write('Debug summary for pipeline stages\n')
        f.write(f'Input: {input_path}\n')
        f.write(f'Detected boxes: {len(boxes)}\n')
        for i, b in enumerate(boxes):
            f.write(f'  box_{i}: {b}\n')
        f.write(f'Centers: A=({cax},{cay}), B=({cbx},{cby})\n')
        f.write(f'Translations: A=({ta_x},{ta_y}), B=({tb_x},{tb_y})\n')
        f.write(f'Inpaint pixels: {int(np.sum(inpaint_mask > 0))}\n')
        f.write(f'Mask A pixels: {int(np.sum(mask_a > 0))}\n')
        f.write(f'Mask B pixels: {int(np.sum(mask_b > 0))}\n')
    print(f'Saved stage-by-stage debug images to: {args.outdir}')
if __name__ == '__main__':
    main()
