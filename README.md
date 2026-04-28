# Linear Algebra Project

Image Composition and Spatial Rearrangement using Geometric Transformations

## Authors

Artem Onyshchuk, Yurii Bobas, Ivan Polianskyi


## Videos

Yurii Bobas - https://youtu.be/Xv2bpqCH97E?si=QyJRaIczXYbWnLUD

Ivan Poliansky  - https://youtu.be/WqXQksyIAvI

Artem Onyshchuk - https://www.youtube.com/watch?v=wBmxTD88TiI


## Short description

Image composition: human detection, geometric transformations (translation/scaling), and hole filling with inpainting.


## Files

- `main.py` - command-line script for processing a single image (automatic pipeline).
- `gui.py` - interactive `tkinter` interface for dragging/scaling detected objects.
- `detection.py` - human detection (`HOG+SVM`) and binary mask extraction (`GrabCut`).
- `transformation.py` - translation/scale matrices and affine transformation helpers.
- `inpainting.py` - fills missing regions using neighbor averaging via a sparse linear system.
- `metrics.py` - helper metrics (`IoU`, `MSE`, `precision/recall/F1`).
- `requirements.txt` - Python dependencies.
- `.gitignore` - files and folders ignored by Git.
- `tuning/` - all tuning and analytics scripts.
- `docs/interim_report.tex` - interim project report.
- `artifacts/` - generated images, tuning results, and presentation plots.

## Quick Start

1. Install dependencies:
   `pip install -r requirements.txt`
2. Run:
   - command-line mode: `python main.py`
   - GUI mode: `python gui.py`

## Tuning Scripts

- `python tuning/auto_tune.py` - brute-force tuning for full pipeline.
- `python tuning/smart_tune.py` - 2-phase Optuna tuning (HOG then masks).
- `python tuning/tune_full_pipeline.py` - lightweight full-pipeline grid tuning.
- `python tuning/tune_on_dataset.py` - dataset-based HOG tuning with GT boxes.
- `python tuning/generate_presentation_plots.py` - build analytics charts for slides.
