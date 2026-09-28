# Post-degradation preprocessing evaluation

This experiment applies frozen preprocessing after each existing synthetic degradation and before frozen YOLOv8n 1280px inference:

1. Original test image.
2. One existing degradation at its fixed severity.
3. Either gamma 0.9 or CLAHE on LAB luminance with clip limit 2 and an 8×8 grid.
4. YOLO evaluation.

Gamma and CLAHE are independent runs and are never combined in this experiment. The source degraded images, split, labels, class definitions, checkpoint, and validator settings remain unchanged. Each result is compared with the existing unprocessed score for that exact degradation and severity, rather than with clean baseline performance.

Run all 60 post-degradation evaluations:

```bash
./.venv/bin/python scripts/run_degraded_preprocessing_eval.py
```

Results are written to `runs/degraded_preprocessing_1280/summary.csv` and `summary.md`; every scenario also has a manifest, metrics, copied labels, and generated preprocessed images under `datasets/`. The 30 original degradation suites are read from `runs/robustness_1280/datasets/` and are never altered.

Generate the report-ready three-way comparison table with:

```bash
./.venv/bin/python scripts/write_degraded_preprocessing_report.py
```

It writes `docs/degraded_preprocessing_results.md`, with precision, recall, mAP50, mAP50–95, and the mAP50–95 delta for unprocessed, gamma, and CLAHE results at every degradation and severity. It can be run while evaluation is active; unfinished cells remain marked as `—`.

For a resumable partial run, use `--only gamma_0.9` or `--only clahe_clip_2_grid_8`; use `--degradation underexposure` to restrict to a named degradation. Existing metrics are reused. The complete CPU evaluation can take roughly an hour because it performs 60 full 300-image YOLO validations.
