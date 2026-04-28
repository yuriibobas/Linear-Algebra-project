import argparse
import configparser
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

from detection import detect_humans, get_binary_masks
from inpainting import inpaint_neighbor_averaging
from transformation import apply_translation


def load_config(cfg_path):
    cfg = configparser.ConfigParser()
    cfg.read(cfg_path, encoding="utf-8")
    return cfg


def cfg_get(cfg, section, key, fallback):
    if not cfg.has_option(section, key):
        return fallback
    raw = cfg.get(section, key)
    if isinstance(fallback, bool):
        return cfg.getboolean(section, key)
    if isinstance(fallback, float):
        return float(raw)
    if isinstance(fallback, int):
        return int(raw)
    return raw


def process_image(
    image_path,
    output_path="result_image.jpg",
    win_stride=(8, 8),
    padding=(8, 8),
    scale=1.05,
    score_threshold=0.0,
    nms_threshold=0.25,
    hit_threshold=0.0,
    use_preprocessing=True,
    shrink_factor=0.0,
    grabcut_iters=5,
    morph_kernel_size=5,
    morph_open_iters=2,
    morph_close_iters=2,
):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return

    print("Detecting humans using HOG+SVM...")
    boxes = detect_humans(
        image,
        win_stride=win_stride,
        padding=padding,
        scale=scale,
        score_threshold=score_threshold,
        nms_threshold=nms_threshold,
        hit_threshold=hit_threshold,
        use_preprocessing=use_preprocessing,
        shrink_factor=shrink_factor,
    )
    if len(boxes) < 2:
        print("Need at least 2 humans in the image to swap them.")
        return

    print(f"Detected {len(boxes)} humans. Using the two largest.")
    masks_data = get_binary_masks(
        image,
        boxes,
        grabcut_iters=grabcut_iters,
        morph_kernel_size=morph_kernel_size,
        morph_open_iters=morph_open_iters,
        morph_close_iters=morph_close_iters,
    )
    mask_a, rect_a = masks_data[0]
    mask_b, rect_b = masks_data[1]

    center_a = (rect_a[0] + rect_a[2] // 2, rect_a[1] + rect_a[3] // 2)
    center_b = (rect_b[0] + rect_b[2] // 2, rect_b[1] + rect_b[3] // 2)
    trans_a = (center_b[0] - center_a[0], center_b[1] - center_a[1])
    trans_b = (center_a[0] - center_b[0], center_a[1] - center_b[1])

    print(f"Swapping Figure A (center {center_a[0]},{center_a[1]}) and Figure B (center {center_b[0]},{center_b[1]})")
    translated_person_a, translated_mask_a = apply_translation(image, mask_a, *trans_a)
    translated_person_b, translated_mask_b = apply_translation(image, mask_b, *trans_b)

    combined_original_masks = cv2.bitwise_or(mask_a, mask_b)
    inpaint_mask = cv2.dilate(
        combined_original_masks,
        np.ones((7, 7), np.uint8),
        iterations=2,
    )

    clean_bg = image.copy()
    clean_bg[inpaint_mask > 0] = [0, 0, 0]
    print(f"Inpainting {np.sum(inpaint_mask > 0)} background pixels...")
    clean_bg = inpaint_neighbor_averaging(clean_bg, inpaint_mask)

    transformed = clean_bg.copy()
    transformed[translated_mask_a > 0] = translated_person_a[translated_mask_a > 0]
    transformed[translated_mask_b > 0] = translated_person_b[translated_mask_b > 0]

    combined_placed = cv2.bitwise_or(translated_mask_a, translated_mask_b)
    inner_mask = cv2.erode(combined_placed, np.ones((5, 5), np.uint8), iterations=2)
    alpha = cv2.GaussianBlur(combined_placed.astype(np.float32) * 255, (15, 15), 0) / 255.0
    alpha = np.clip(alpha, 0, 1)
    alpha[inner_mask > 0] = 1.0

    final_image = (
        alpha[:, :, None] * transformed.astype(np.float64)
        + (1.0 - alpha[:, :, None]) * clean_bg.astype(np.float64)
    )
    final_image = np.clip(final_image, 0, 255).astype(np.uint8)

    cv2.imwrite("pre_inpainting_" + os.path.basename(output_path), transformed)
    cv2.imwrite(output_path, final_image)
    print(f"Saved successful result to {output_path}")

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("After Transformation")
    plt.imshow(cv2.cvtColor(transformed, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Final Inpainted")
    plt.imshow(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("plot_" + os.path.basename(output_path))
    print("Saved plot visualization.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Swap two people in an image using HOG+SVM, GrabCut, and Laplace inpainting."
    )
    parser.add_argument(
        "--config",
        default="",
        metavar="FILE",
        help="Path to a .cfg config file (e.g. config.cfg). CLI arguments override values from the file.",
    )
    parser.add_argument("--input", default="", help="Input image path.")
    parser.add_argument("--output", default="", help="Output image path.")
    parser.add_argument("--win-stride-w", type=int, default=0)
    parser.add_argument("--win-stride-h", type=int, default=0)
    parser.add_argument("--padding-w", type=int, default=0)
    parser.add_argument("--padding-h", type=int, default=0)
    parser.add_argument("--scale", type=float, default=0.0)
    parser.add_argument("--score-threshold", type=float, default=-1.0)
    parser.add_argument("--nms-threshold", type=float, default=0.0)
    parser.add_argument("--hit-threshold", type=float, default=-1.0)
    parser.add_argument("--shrink-factor", type=float, default=-1.0)
    parser.add_argument("--no-preprocess", action="store_true")
    parser.add_argument("--grabcut-iters", type=int, default=0)
    return parser.parse_args()


def resolve_runtime_params(args, cfg):
    input_path = args.input or cfg_get(cfg, "pipeline", "input", "test_image.jpg")
    output_path = args.output or cfg_get(cfg, "pipeline", "output", "result_image.jpg")

    win_stride_w = args.win_stride_w or cfg_get(cfg, "detection", "win_stride_w", 8)
    win_stride_h = args.win_stride_h or cfg_get(cfg, "detection", "win_stride_h", 8)
    padding_w = args.padding_w or cfg_get(cfg, "detection", "padding_w", 8)
    padding_h = args.padding_h or cfg_get(cfg, "detection", "padding_h", 8)

    scale = args.scale or cfg_get(cfg, "detection", "scale", 1.05)
    nms_threshold = args.nms_threshold or cfg_get(cfg, "detection", "nms_threshold", 0.25)
    grabcut_iters = args.grabcut_iters or cfg_get(cfg, "masks", "grabcut_iters", 5)

    score_threshold = (
        args.score_threshold
        if args.score_threshold >= 0.0
        else cfg_get(cfg, "detection", "score_threshold", 0.0)
    )
    hit_threshold = (
        args.hit_threshold
        if args.hit_threshold >= 0.0
        else cfg_get(cfg, "detection", "hit_threshold", 0.0)
    )
    shrink_factor = (
        args.shrink_factor
        if args.shrink_factor >= 0.0
        else cfg_get(cfg, "detection", "shrink_factor", 0.0)
    )
    use_preprocess = not args.no_preprocess and cfg_get(
        cfg, "detection", "use_preprocessing", True
    )

    return {
        "input_path": input_path,
        "output_path": output_path,
        "win_stride": (win_stride_w, win_stride_h),
        "padding": (padding_w, padding_h),
        "scale": scale,
        "score_threshold": score_threshold,
        "nms_threshold": nms_threshold,
        "hit_threshold": hit_threshold,
        "use_preprocessing": use_preprocess,
        "shrink_factor": shrink_factor,
        "grabcut_iters": grabcut_iters,
        "morph_kernel_size": cfg_get(cfg, "masks", "morph_kernel_size", 5),
        "morph_open_iters": cfg_get(cfg, "masks", "morph_open_iters", 2),
        "morph_close_iters": cfg_get(cfg, "masks", "morph_close_iters", 2),
    }


def main():
    args = parse_args()
    cfg = load_config(args.config) if args.config else configparser.ConfigParser()
    params = resolve_runtime_params(args, cfg)

    input_path = params.pop("input_path")
    output_path = params.pop("output_path")
    if not os.path.exists(input_path):
        print(f"Image '{input_path}' not found. Place an image with at least two people there.")
        return

    process_image(input_path, output_path, **params)


if __name__ == "__main__":
    main()
