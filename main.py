import argparse
import configparser
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from detection import detect_humans, get_binary_masks
from transformation import apply_translation
from inpainting import inpaint_neighbor_averaging

def load_config(cfg_path):
    cfg = configparser.ConfigParser()
    cfg.read(cfg_path, encoding='utf-8')
    return cfg

def _get(cfg, section, key, fallback):
    if cfg.has_option(section, key):
        val = cfg.get(section, key)
        if isinstance(fallback, bool):
            return cfg.getboolean(section, key)
        if isinstance(fallback, float):
            return float(val)
        if isinstance(fallback, int):
            return int(val)
        return val
    return fallback

def process_image(image_path, output_path='result_image.jpg', win_stride=(8, 8), padding=(8, 8), scale=1.05, score_threshold=0.0, nms_threshold=0.25, hit_threshold=0.0, use_preprocessing=True, shrink_factor=0.0, grabcut_iters=5, morph_kernel_size=5, morph_open_iters=2, morph_close_iters=2):
    I = cv2.imread(image_path)
    if I is None:
        print(f'Error: Could not load image {image_path}')
        return
    print('Detecting humans using HOG+SVM...')
    boxes = detect_humans(I, win_stride=win_stride, padding=padding, scale=scale, score_threshold=score_threshold, nms_threshold=nms_threshold, hit_threshold=hit_threshold, use_preprocessing=use_preprocessing, shrink_factor=shrink_factor)
    if len(boxes) < 2:
        print('Need at least 2 humans in the image to swap them.')
        return
    print(f'Detected {len(boxes)} humans. Using the two largest.')
    masks_data = get_binary_masks(I, boxes, grabcut_iters=grabcut_iters, morph_kernel_size=morph_kernel_size, morph_open_iters=morph_open_iters, morph_close_iters=morph_close_iters)
    mask_A, rect_A = masks_data[0]
    mask_B, rect_B = masks_data[1]
    cA_x = rect_A[0] + rect_A[2] // 2
    cA_y = rect_A[1] + rect_A[3] // 2
    cB_x = rect_B[0] + rect_B[2] // 2
    cB_y = rect_B[1] + rect_B[3] // 2
    tA_x, tA_y = (cB_x - cA_x, cB_y - cA_y)
    tB_x, tB_y = (cA_x - cB_x, cA_y - cB_y)
    print(f'Swapping Figure A (center {cA_x},{cA_y}) and Figure B (center {cB_x},{cB_y})')
    translated_person_A, translated_mask_A = apply_translation(I, mask_A, tA_x, tA_y)
    translated_person_B, translated_mask_B = apply_translation(I, mask_B, tB_x, tB_y)
    combined_original_masks = cv2.bitwise_or(mask_A, mask_B)
    dilate_kernel = np.ones((7, 7), np.uint8)
    inpaint_mask = cv2.dilate(combined_original_masks, dilate_kernel, iterations=2)
    clean_bg = I.copy()
    clean_bg[inpaint_mask > 0] = [0, 0, 0]
    print(f'Inpainting {np.sum(inpaint_mask > 0)} background pixels...')
    clean_bg = inpaint_neighbor_averaging(clean_bg, inpaint_mask)
    Inew = clean_bg.copy()
    target_A_idx = translated_mask_A > 0
    Inew[target_A_idx] = translated_person_A[target_A_idx]
    target_B_idx = translated_mask_B > 0
    Inew[target_B_idx] = translated_person_B[target_B_idx]
    combined_placed = cv2.bitwise_or(translated_mask_A, translated_mask_B)
    erode_k = np.ones((5, 5), np.uint8)
    inner_mask = cv2.erode(combined_placed, erode_k, iterations=2)
    alpha = cv2.GaussianBlur(combined_placed.astype(np.float32) * 255, (15, 15), 0) / 255.0
    alpha = np.clip(alpha, 0, 1)
    alpha[inner_mask > 0] = 1.0
    final_image = alpha[:, :, None] * Inew.astype(np.float64) + (1.0 - alpha[:, :, None]) * clean_bg.astype(np.float64)
    final_image = np.clip(final_image, 0, 255).astype(np.uint8)
    cv2.imwrite('pre_inpainting_' + os.path.basename(output_path), Inew)
    cv2.imwrite(output_path, final_image)
    print(f'Saved successful result to {output_path}')
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title('Original')
    plt.imshow(cv2.cvtColor(I, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.title('After Transformation')
    plt.imshow(cv2.cvtColor(Inew, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.title('Final Inpainted')
    plt.imshow(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('plot_' + os.path.basename(output_path))
    print('Saved plot visualization.')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Swap two people in an image using HOG+SVM, GrabCut, and Laplace inpainting.')
    parser.add_argument('--config', default='', metavar='FILE', help='Path to a .cfg config file (e.g. config.cfg). CLI arguments override values from the file.')
    parser.add_argument('--input', default='', help='Input image path.')
    parser.add_argument('--output', default='', help='Output image path.')
    parser.add_argument('--win-stride-w', type=int, default=0)
    parser.add_argument('--win-stride-h', type=int, default=0)
    parser.add_argument('--padding-w', type=int, default=0)
    parser.add_argument('--padding-h', type=int, default=0)
    parser.add_argument('--scale', type=float, default=0.0)
    parser.add_argument('--score-threshold', type=float, default=-1.0)
    parser.add_argument('--nms-threshold', type=float, default=0.0)
    parser.add_argument('--hit-threshold', type=float, default=-1.0)
    parser.add_argument('--shrink-factor', type=float, default=-1.0)
    parser.add_argument('--no-preprocess', action='store_true')
    parser.add_argument('--grabcut-iters', type=int, default=0)
    args = parser.parse_args()
    cfg = load_config(args.config) if args.config else configparser.ConfigParser()
    input_path = args.input or _get(cfg, 'pipeline', 'input', 'test_image.jpg')
    output_path = args.output or _get(cfg, 'pipeline', 'output', 'result_image.jpg')
    win_stride_w = args.win_stride_w or _get(cfg, 'detection', 'win_stride_w', 8)
    win_stride_h = args.win_stride_h or _get(cfg, 'detection', 'win_stride_h', 8)
    padding_w = args.padding_w or _get(cfg, 'detection', 'padding_w', 8)
    padding_h = args.padding_h or _get(cfg, 'detection', 'padding_h', 8)
    scale = args.scale or _get(cfg, 'detection', 'scale', 1.05)
    nms_thr = args.nms_threshold or _get(cfg, 'detection', 'nms_threshold', 0.25)
    grabcut_iters = args.grabcut_iters or _get(cfg, 'masks', 'grabcut_iters', 5)
    morph_kernel_size = _get(cfg, 'masks', 'morph_kernel_size', 5)
    morph_open_iters = _get(cfg, 'masks', 'morph_open_iters', 2)
    morph_close_iters = _get(cfg, 'masks', 'morph_close_iters', 2)
    if args.score_threshold >= 0.0:
        score_thr = args.score_threshold
    else:
        score_thr = _get(cfg, 'detection', 'score_threshold', 0.0)
    if args.hit_threshold >= 0.0:
        hit_thr = args.hit_threshold
    else:
        hit_thr = _get(cfg, 'detection', 'hit_threshold', 0.0)
    if args.shrink_factor >= 0.0:
        shrink = args.shrink_factor
    else:
        shrink = _get(cfg, 'detection', 'shrink_factor', 0.0)
    use_preprocess = not args.no_preprocess and _get(cfg, 'detection', 'use_preprocessing', True)
    if not os.path.exists(input_path):
        print(f"Image '{input_path}' not found. Place an image with at least two people there.")
    else:
        process_image(input_path, output_path, win_stride=(win_stride_w, win_stride_h), padding=(padding_w, padding_h), scale=scale, score_threshold=score_thr, nms_threshold=nms_thr, hit_threshold=hit_thr, use_preprocessing=use_preprocess, shrink_factor=shrink, grabcut_iters=grabcut_iters, morph_kernel_size=morph_kernel_size, morph_open_iters=morph_open_iters, morph_close_iters=morph_close_iters)
