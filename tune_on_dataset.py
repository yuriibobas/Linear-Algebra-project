import argparse
import json
import re
import urllib.request
import zipfile
import os
from itertools import product
from pathlib import Path

import cv2
import numpy as np

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


HOG_PARAMS = {
    "win_stride":    [(8, 8), (4, 4)],
    "padding":       [(8, 8), (16, 16), (32, 32)],
    "scale":         [1.01, 1.02, 1.03, 1.05],
    "hit_threshold": [0.0, 0.2, 0.5, 0.8],
}
FILTER_PARAMS = {
    "score_threshold": [0.0, 0.3, 0.5, 0.7, 1.0],
    "nms_threshold":   [0.2, 0.3, 0.4, 0.5, 0.65],
}
PREPROCESS_OPTIONS = [True, False]
SHRINK_FACTORS = [0.0, 0.08, 0.12]


def grid_search(entries, iou_threshold=0.5):
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    hog_combos = list(product(
        HOG_PARAMS["win_stride"],
        HOG_PARAMS["padding"],
        HOG_PARAMS["scale"],
        HOG_PARAMS["hit_threshold"],
        PREPROCESS_OPTIONS,
    ))
    total_hog = len(hog_combos) * len(entries)

    print(f"Step 1/2 — HOG cache ({len(entries)} images × {len(hog_combos)} combos = {total_hog} runs)...")
    cache = {}
    done = 0
    for win_stride, padding, scale, hit_thr, preproc in hog_combos:
        key = (win_stride, padding, scale, hit_thr, preproc)
        cache[key] = []
        for entry in entries:
            img = entry["image"]
            if preproc:
                img = preprocess_for_hog(img)

            raw_boxes, weights = hog.detectMultiScale(
                img, winStride=win_stride, padding=padding,
                scale=scale, hitThreshold=hit_thr,
            )
            if len(raw_boxes) > 0:
                weights = weights.reshape(-1)
                xywh = raw_boxes.astype(np.float32)
            else:
                weights = np.array([], dtype=np.float32)
                xywh = np.zeros((0, 4), dtype=np.float32)
            cache[key].append((raw_boxes, weights, xywh))
            done += 1
            if done % 50 == 0 or done == total_hog:
                print(f"\r  {done}/{total_hog}", end="", flush=True)
    print()

    filter_combos = list(product(
        FILTER_PARAMS["score_threshold"],
        FILTER_PARAMS["nms_threshold"],
        SHRINK_FACTORS,
    ))
    all_combos = list(product(hog_combos, filter_combos))
    print(f"Step 2/2 — Evaluating {len(all_combos)} combinations...")

    best, best_score = None, (-1.0, -1.0)
    top_results = []

    for i, ((win_stride, padding, scale, hit_thr, preproc),
            (score_thr, nms_thr, shrink)) in enumerate(all_combos):
        key = (win_stride, padding, scale, hit_thr, preproc)
        tp = fp = fn = 0
        ious = []

        for j, entry in enumerate(entries):
            raw_boxes, weights, xywh = cache[key][j]
            if len(weights) > 0:
                keep = weights > score_thr
                fb, fw, fx = raw_boxes[keep], weights[keep], xywh[keep]
                if len(fb) > 0:
                    idx = cv2.dnn.NMSBoxes(fx.tolist(), fw.tolist(), 0.0, nms_thr)
                    idx = np.array(idx).reshape(-1) if idx is not None and len(idx) else []
                    pred = fb[idx].tolist() if len(idx) else []
                else:
                    pred = []
            else:
                pred = []

            if shrink > 0 and pred:
                pred = shrink_boxes(pred, factor=shrink)

            det = evaluate_detections(pred, entry["gt_boxes"], iou_threshold)
            tp += det["tp"]; fp += det["fp"]; fn += det["fn"]
            if det["mean_iou_matched"] > 0:
                ious.append(det["mean_iou_matched"])

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2*p*r/(p+r) if (p+r) > 0 else 0.0
        iou = float(np.mean(ious)) if ious else 0.0

        combo_result = dict(
            win_stride=win_stride, padding=padding, scale=scale,
            hit_threshold=hit_thr, preprocess=preproc,
            score_threshold=score_thr, nms_threshold=nms_thr,
            shrink_factor=shrink,
            precision=round(p, 4), recall=round(r, 4),
            f1=round(f1, 4), mean_iou=round(iou, 4),
            tp=tp, fp=fp, fn=fn,
        )

        if (f1, iou) > best_score:
            best_score = (f1, iou)
            best = combo_result

        top_results.append(combo_result)

        if (i + 1) % 200 == 0 or (i + 1) == len(all_combos):
            print(f"\r  {i+1}/{len(all_combos)}  (best F1 so far: {best_score[0]:.4f})", end="", flush=True)
    print()

    top_results.sort(key=lambda r: (r["f1"], r["mean_iou"]), reverse=True)
    top20 = top_results[:20]

    return best, top20


def main():
    parser = argparse.ArgumentParser(description="Find best HOG+SVM params on Penn-Fudan dataset.")
    parser.add_argument("--max", type=int, default=170, help="Max images to use (default: 170).")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU match threshold (default: 0.5).")
    args = parser.parse_args()

    download_dataset()

    print(f"\nLoading dataset (up to {args.max} images)...")
    entries = load_dataset(max_images=args.max)
    if not entries:
        print("No images loaded. Check dataset directory.")
        return
    print(f"Loaded {len(entries)} images.\n")

    best, top20 = grid_search(entries, iou_threshold=args.iou)

    print(f"""
Best parameters (F1={best['f1']:.4f}, IoU={best['mean_iou']:.4f}):

  win_stride      = {best['win_stride']}
  padding         = {best['padding']}
  scale           = {best['scale']}
  hit_threshold   = {best['hit_threshold']}
  preprocess      = {best['preprocess']}
  score_threshold = {best['score_threshold']}
  nms_threshold   = {best['nms_threshold']}
  shrink_factor   = {best['shrink_factor']}

  Precision : {best['precision']:.4f}
  Recall    : {best['recall']:.4f}
  F1        : {best['f1']:.4f}
  Mean IoU  : {best['mean_iou']:.4f}
  TP/FP/FN  : {best['tp']}/{best['fp']}/{best['fn']}
""")

    results_path = "dataset_tuning_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({"best": best, "top20": top20}, f, indent=2)
    print(f"Full results saved to {results_path}")


if __name__ == "__main__":
    main()
