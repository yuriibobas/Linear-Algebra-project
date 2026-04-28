# Image Composition and Spatial Rearrangement

This project detects people in an image, swaps their spatial positions, and reconstructs the background using linear-algebra-based inpainting.

Core pipeline:
- human detection (`HOG + linear SVM`)
- person mask extraction (`GrabCut + morphology`)
- geometric transformation (translation / optional scaling)
- background restoration (discrete Laplace equation on sparse linear system)
- blending and visualization

---

## 1) What the project does

Input: one RGB image with at least two people.  
Output: image where two detected people are swapped and background holes are filled.

Practical modes:
- CLI pipeline: `main.py`
- Interactive GUI mode: `gui.py`
- Detection-focused tuning on dataset: `tune_on_dataset.py`
- Full pipeline tuning on one image: `auto_tune.py`, `tune_full_pipeline.py`, `smart_tune.py`
- Evaluation/reporting: `generate_metrics_report.py`

---

## 2) Step-by-step algorithm

1. Load image `I` and read parameters.
2. Detect person bounding boxes `B_i` using HOG+SVM with NMS.
3. Keep top boxes, run GrabCut in each rectangle to get binary masks `M_i`.
4. Compute figure centers and translation vectors.
5. Warp isolated person pixels into new locations.
6. Build background hole mask and restore missing pixels via Laplace inpainting.
7. Composite translated persons on restored background.
8. Apply edge feathering (alpha blending) for smoother seams.
9. Save outputs and diagnostic plots.

---

## 3) Mathematical foundation

### 3.1 Detection and filtering

- HOG descriptor converts local gradients into orientation histograms.
- Linear SVM outputs confidence score for candidate windows.
- Post-filtering:
  - threshold by score
  - NMS to suppress overlapping duplicates

### 3.2 Geometric transformation

For translation:
\[
T(tx,ty)=
\begin{bmatrix}
1 & 0 & tx \\
0 & 1 & ty \\
0 & 0 & 1
\end{bmatrix}
\]

For scale in homogeneous coordinates:
\[
S(s)=
\begin{bmatrix}
s & 0 & 0 \\
0 & s & 0 \\
0 & 0 & 1
\end{bmatrix}
\]

Composite transform in GUI mode:
\[
T_{total} = T_{translate}\, T_{back}\, S\, T_{to\_origin}
\]

### 3.3 Laplace inpainting (linear algebra core)

For unknown pixel \(X_i\) inside hole \(\Omega\), discrete harmonic constraint:
\[
4X_i - X_{top} - X_{bottom} - X_{left} - X_{right} = 0
\]

This yields sparse linear system:
\[
A X = B
\]

- \(A\): sparse coefficient matrix (5-point stencil)
- \(X\): unknown intensities in hole
- \(B\): known neighbor contributions

Solver:
- Conjugate Gradient (`cg`) by default
- fallback to direct sparse solver (`spsolve`) if needed

Current implementation uses multi-scale strategy for large holes:
- solve at lower resolution first
- upsample coarse estimate
- refine at full resolution

---

## 4) File-by-file function reference

## `detection.py`

- `preprocess_for_hog(image)`
  - CLAHE on L-channel in LAB to stabilize contrast before HOG.
- `shrink_boxes(boxes, factor)`
  - Shrinks predicted boxes inward to better match figure silhouettes.
- `detect_humans(...)`
  - Runs HOG+SVM, score thresholding, NMS, optional shrinking.
- `get_binary_masks(...)`
  - Runs GrabCut per box, morphology open/close, erosion, connected-component cleanup to keep main person blob.

## `transformation.py`

- `create_translation_matrix(tx, ty)`
  - Builds 3x3 translation matrix.
- `create_scale_matrix(s)`
  - Builds 3x3 isotropic scaling matrix.
- `extract_features(image, binary_mask)`
  - Applies mask (Hadamard-like selection in image space).
- `apply_translation(image, binary_mask, tx, ty)`
  - Moves masked person with `warpAffine`.
- `apply_transformation(image, binary_mask, tx, ty, s, cx, cy)`
  - Combined scale + translation around person center.

## `inpainting.py`

- `_inpaint_single_scale(image, mask)`
  - Builds sparse Laplace system and solves each color channel.
- `inpaint_neighbor_averaging(image, mask)`
  - Wrapper with multi-scale path for large missing regions.

## `metrics.py`

- `calculate_iou(boxA, boxB)`
  - Intersection-over-Union for boxes.
- `calculate_mse(imageA, imageB)`
  - Pixel-wise mean squared error.
- `compute_classification_metrics(tp, fp, fn)`
  - Precision / Recall / F1.
- `evaluate_detections(pred_boxes, gt_boxes, iou_threshold)`
  - One-to-one greedy matching, returns TP/FP/FN + metrics.

## `main.py`

- `load_config(cfg_path)`
  - Reads pipeline config file.
- `_get(cfg, section, key, fallback)`
  - Typed config accessor.
- `process_image(...)`
  - Full end-to-end non-interactive pipeline:
    - detect
    - mask
    - swap
    - background-first inpainting
    - feather blending
    - save result and plot

## `gui.py`

Class `GeometricApp`:
- `setup_ui()` creates controls/canvas.
- `load_image()` loads and resizes image.
- `detect()` runs detect+mask and prepares draggable sprites.
- `render_frame()` composites transformed figures in real time.
- mouse handlers: `on_press`, `on_drag`, `on_release`, `on_scroll`.
- `finalize_inpaint()` fills holes and saves `gui_result.jpg`.

## `generate_metrics_report.py`

- `parse_gt_boxes(gt_raw)` validates GT format.
- `random_rect_mask(...)` creates synthetic missing regions.
- `benchmark_inpainting_mse(...)` evaluates inpainting quality statistically.
- `tune_detection_params(image, gt_boxes)` grid-searches detection params for one image.
- `build_report(...)` composes markdown report.
- `main()` CLI entry point.

## `tune_on_dataset.py`

- `download_dataset()` downloads Penn-Fudan.
- `parse_annotation(path)` parses GT boxes from annotations.
- `load_dataset(max_images)` loads images + GT.
- `grid_search(entries, iou_threshold)`:
  - caches HOG outputs
  - evaluates parameter combinations
  - ranks by `(F1, mean IoU)`
- `main()` dataset tuning runner.

## `tune_full_pipeline.py`

- `parse_gt_boxes(gt_raw)`
- `swap_and_inpaint(image, masks_data)`
- `boundary_mae(image, hole_mask)` seam continuity proxy
- `hole_texture_ratio(image, hole_mask)` texture preservation proxy
- `score_config(...)` weighted scalar objective
- `build_grid()` parameter space
- `tune_full_pipeline(...)` exhaustive/randomized search over full pipeline
- `main()` CLI entry point

## `auto_tune.py`

- `run_pipeline(...)` full pipeline execution
- `boundary_mae(...)`, `texture_ratio(...)`, `score_result(...)` scalar scoring
- exhaustive/random search over `PARAM_GRID`
- updates `config.cfg` with best found setup

## `smart_tune.py`

- `run_pipeline(...)` full pipeline with background-first compositing and feather blending
- `score_result(...)` computes:
  - `det_score`
  - `hole_score`
  - `continuity`
  - `texture`
- `objective(trial)` Optuna multi-objective target
- MOTPE optimization and Pareto-front extraction
- writes `smart_tune_results.json`
- selects balanced solution from Pareto front
- updates `config.cfg` and saves `smart_tune_best_balanced.jpg`

## Debug scripts

- `debug_boxes.py`: visual check for detected boxes + masks.
- `debug_masks_detail.py`: per-mask diagnostics and connected components.

---

## 5) How to run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run main pipeline:

```bash
python main.py --config config.cfg
```

Or direct CLI parameters:

```bash
python main.py --input test_image.jpg --output result_image.jpg
```

Run GUI:

```bash
python gui.py
```

Generate metrics report:

```bash
python generate_metrics_report.py --image test_image.jpg --result result_image.jpg --gt "564,110,439,877;46,65,452,903"
```

Tune on dataset (detection-level):

```bash
python tune_on_dataset.py --max 40
```

Tune full pipeline for one image (multi-objective):

```bash
python smart_tune.py --image test_image.jpg --gt "564,110,439,877;46,65,452,903" --trials 60 --out smart_tune_results.json
```

---

## 6) Analysis from `smart_tune_results.json`

Observed run summary:
- method: `Multi-Objective Bayesian Optimization (MOTPE)`
- requested trials: `60`
- wall time: `1044.5 sec`
- logged valid trials in `all_trials`: `32`
- Pareto front size: `9`

Aggregate behavior (from logged trials):
- mean `det_score`: `0.8855`
- mean `continuity`: `0.5290`
- mean `texture`: `0.8271`
- mean `hole_ratio`: `0.3329`
- trials with `det_score = 0`: `1`

Top Pareto entry (`trial_id=18`):
- `det_score=0.9750`, `continuity=0.9204`, `texture=0.7979`
- heuristic total: `0.6962`
- params:
  - `win_stride=(4,4)`
  - `padding=(8,8)`
  - `scale=1.08`
  - `hit_threshold=0.6`
  - `score_threshold=0.4`
  - `nms_threshold=0.55`
  - `use_preprocessing=False`
  - `shrink_factor=0.04`
  - `grabcut_iters=12`

Important interpretation:
- most high-quality trials had strong detection (`F1 ~ 1.0` inside `det_score`) but still large inpaint area (`hole_ratio` often high).
- because `hole_score = clip(1 - 5*hole_ratio, 0, 1)`, many trials collapse to `hole_score=0` when holes are large. This is visible in the JSON (`31/32` trials with `hole_score=0`).
- therefore, optimization is effectively dominated by detection/continuity/texture in this run.

Practical recommendation:
- rebalance objective by reducing hole penalty slope, e.g. `1 - 2.5*hole_ratio`, or use smooth exponential penalty.
- optionally constrain transformations to reduce hole size if visual realism is the primary target.
- keep Pareto selection (not single scalar only) to preserve trade-off visibility.

---

## 7) What is trained vs tuned

- No custom detector is trained in this project.
- OpenCV uses a pretrained default HOG+SVM person detector.
- Project performs parameter tuning (hyperparameter search), not model retraining.

---

## 8) Known limitations

- HOG+SVM is less robust than modern deep detectors under occlusion and unusual poses.
- GrabCut quality depends on bounding-box quality.
- Laplace inpainting tends to oversmooth complex textures in very large holes.
- Best parameters are image-dependent; dataset-optimal and image-optimal settings may differ.

---

## 9) Main outputs

- `result_image.jpg` — final pipeline result
- `pre_inpainting_*.jpg` — intermediate composite
- `plot_*.jpg` — visualization panels
- `metrics_report*.md` — numeric report
- `smart_tune_results.json` — full optimization log + Pareto front
- `smart_tune_best_balanced.jpg` / `auto_tune_best.jpg` — best tuned visual outputs
