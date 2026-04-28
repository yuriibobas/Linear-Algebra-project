import argparse
import configparser
from pathlib import Path

import cv2
import numpy as np

from detection import detect_humans, get_binary_masks, preprocess_for_hog
from inpainting import inpaint_neighbor_averaging
from transformation import apply_translation


def load_cfg(path):
    cfg = configparser.ConfigParser()
    cfg.read(path, encoding="utf-8")
    return cfg


def cfg_get(cfg, section, key, fallback):
    if not cfg.has_option(section, key):
        return fallback
    if isinstance(fallback, bool):
        return cfg.getboolean(section, key)
    if isinstance(fallback, int):
        return cfg.getint(section, key)
    if isinstance(fallback, float):
        return cfg.getfloat(section, key)
    return cfg.get(section, key)


def save_image(outdir, name, image):
    cv2.imwrite(str(outdir / name), image)


def draw_boxes(image, boxes, color=(0, 255, 0), thickness=2):
    vis = image.copy()
    for i, (x, y, w, h) in enumerate(boxes):
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, thickness)
        cv2.putText(
            vis,
            f"#{i} ({x},{y},{w},{h})",
            (x, max(16, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return vis


def overlay_mask(image, mask, color=(0, 0, 255), alpha=0.45):
    vis = image.copy().astype(np.float32)
    layer = np.zeros_like(vis)
    layer[:, :] = color
    mask_idx = mask > 0
    vis[mask_idx] = vis[mask_idx] * (1.0 - alpha) + layer[mask_idx] * alpha
    return np.clip(vis, 0, 255).astype(np.uint8)


def get_detection_params(cfg):
    return {
        "win_stride": (
            cfg_get(cfg, "detection", "win_stride_w", 8),
            cfg_get(cfg, "detection", "win_stride_h", 8),
        ),
        "padding": (
            cfg_get(cfg, "detection", "padding_w", 8),
            cfg_get(cfg, "detection", "padding_h", 8),
        ),
        "scale": cfg_get(cfg, "detection", "scale", 1.05),
        "score_threshold": cfg_get(cfg, "detection", "score_threshold", 0.0),
        "nms_threshold": cfg_get(cfg, "detection", "nms_threshold", 0.25),
        "hit_threshold": cfg_get(cfg, "detection", "hit_threshold", 0.0),
        "use_preprocessing": cfg_get(cfg, "detection", "use_preprocessing", True),
        "shrink_factor": cfg_get(cfg, "detection", "shrink_factor", 0.0),
    }


def write_summary(
    outdir, input_path, boxes, center_a, center_b, trans_a, trans_b, inpaint_mask, mask_a, mask_b
):
    with (outdir / "debug_summary.txt").open("w", encoding="utf-8") as f:
        f.write("Debug summary for pipeline stages\n")
        f.write(f"Input: {input_path}\n")
        f.write(f"Detected boxes: {len(boxes)}\n")
        for i, box in enumerate(boxes):
            f.write(f"  box_{i}: {box}\n")
        f.write(f"Centers: A={center_a}, B={center_b}\n")
        f.write(f"Translations: A={trans_a}, B={trans_b}\n")
        f.write(f"Inpaint pixels: {int(np.sum(inpaint_mask > 0))}\n")
        f.write(f"Mask A pixels: {int(np.sum(mask_a > 0))}\n")
        f.write(f"Mask B pixels: {int(np.sum(mask_b > 0))}\n")


def main():
    parser = argparse.ArgumentParser(description="Save debug images for each pipeline stage.")
    parser.add_argument("--config", default="config.cfg", help="Config file path.")
    parser.add_argument("--input", default="", help="Override input image path.")
    parser.add_argument(
        "--outdir", default="debug_stages", help="Output directory for debug images."
    )
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    input_path = args.input or cfg_get(cfg, "pipeline", "input", "test_image.jpg")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(input_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read input image: {input_path}")

    detection_params = get_detection_params(cfg)
    grabcut_iters = cfg_get(cfg, "masks", "grabcut_iters", 5)

    save_image(outdir, "00_original.jpg", image)
    hog_input = (
        preprocess_for_hog(image)
        if detection_params["use_preprocessing"]
        else image.copy()
    )
    save_image(outdir, "01_hog_input.jpg", hog_input)

    boxes = detect_humans(image, **detection_params)
    save_image(outdir, "02_detected_boxes.jpg", draw_boxes(image, boxes))
    if len(boxes) < 2:
        print("Detected fewer than 2 people; stopped after detection debug.")
        return

    masks_data = get_binary_masks(image, boxes, top_k=2, grabcut_iters=grabcut_iters)
    if len(masks_data) < 2:
        print("Failed to extract at least 2 masks; stopped after mask debug.")
        return

    (mask_a, rect_a), (mask_b, rect_b) = masks_data[:2]
    save_image(outdir, "03_mask_a.jpg", (mask_a * 255).astype(np.uint8))
    save_image(outdir, "04_mask_b.jpg", (mask_b * 255).astype(np.uint8))

    mask_overlay = overlay_mask(
        overlay_mask(image, mask_a, color=(0, 0, 255)),
        mask_b,
        color=(0, 255, 0),
    )
    save_image(outdir, "05_masks_overlay.jpg", mask_overlay)

    center_a = (rect_a[0] + rect_a[2] // 2, rect_a[1] + rect_a[3] // 2)
    center_b = (rect_b[0] + rect_b[2] // 2, rect_b[1] + rect_b[3] // 2)
    trans_vec_a = (center_b[0] - center_a[0], center_b[1] - center_a[1])
    trans_vec_b = (center_a[0] - center_b[0], center_a[1] - center_b[1])

    translated_a, translated_mask_a = apply_translation(image, mask_a, *trans_vec_a)
    translated_b, translated_mask_b = apply_translation(image, mask_b, *trans_vec_b)
    save_image(outdir, "06_translated_person_a.jpg", translated_a)
    save_image(outdir, "07_translated_person_b.jpg", translated_b)
    save_image(outdir, "08_translated_mask_a.jpg", (translated_mask_a * 255).astype(np.uint8))
    save_image(outdir, "09_translated_mask_b.jpg", (translated_mask_b * 255).astype(np.uint8))

    original_masks = cv2.bitwise_or(mask_a, mask_b)
    inpaint_mask = cv2.dilate(original_masks, np.ones((7, 7), np.uint8), iterations=2)
    save_image(outdir, "10_inpaint_mask.jpg", (inpaint_mask * 255).astype(np.uint8))

    clean_bg_input = image.copy()
    clean_bg_input[inpaint_mask > 0] = [0, 0, 0]
    save_image(outdir, "11_clean_bg_input.jpg", clean_bg_input)

    clean_bg = inpaint_neighbor_averaging(clean_bg_input, inpaint_mask)
    save_image(outdir, "12_clean_bg_inpainted.jpg", clean_bg)

    composite = clean_bg.copy()
    composite[translated_mask_a > 0] = translated_a[translated_mask_a > 0]
    composite[translated_mask_b > 0] = translated_b[translated_mask_b > 0]
    save_image(outdir, "13_composite_no_blend.jpg", composite)

    combined_placed = cv2.bitwise_or(translated_mask_a, translated_mask_b)
    inner_mask = cv2.erode(combined_placed, np.ones((5, 5), np.uint8), iterations=2)
    alpha = cv2.GaussianBlur(combined_placed.astype(np.float32) * 255, (15, 15), 0) / 255.0
    alpha = np.clip(alpha, 0, 1)
    alpha[inner_mask > 0] = 1.0

    save_image(outdir, "14_alpha_mask.jpg", (alpha * 255.0).astype(np.uint8))

    final = (
        alpha[:, :, None] * composite.astype(np.float64)
        + (1.0 - alpha[:, :, None]) * clean_bg.astype(np.float64)
    )
    final = np.clip(final, 0, 255).astype(np.uint8)
    save_image(outdir, "15_final_blended.jpg", final)

    write_summary(
        outdir,
        input_path,
        boxes,
        center_a,
        center_b,
        trans_vec_a,
        trans_vec_b,
        inpaint_mask,
        mask_a,
        mask_b,
    )

    print(f"Saved stage-by-stage debug images to: {outdir}")


if __name__ == "__main__":
    main()
