# CLAHE on brightness

This experiment applies OpenCV CLAHE only to the L (luminance) channel of LAB images. The A and B channels are retained before conversion back to BGR; color-space conversion can introduce quantization differences. CLAHE enhances local contrast rather than imposing a uniform brightness gain. It is evaluated independently of gamma correction, using the frozen YOLOv8n 1280px checkpoint and unchanged source images, class definitions, split, and labels.

Run the validation sweep:

```bash
./.venv/bin/python scripts/run_clahe_optimization.py
```

The default candidates use clip limits 1, 2, 3, and 4 with an 8×8 tile grid, plus an unchanged baseline. Grid dimensions count CLAHE regions, not pixels. The split loader is reused from the gamma runner, while the CLAHE transform and evaluation loop are separate. Generated suites, manifests, per-class AP, exact evaluation settings, and comparison tables are under `runs/optimization_1280_clahe/val/`. Original images, labels, and checkpoint are not modified. No gamma correction is applied.

Select the highest validation mAP50–95 among the CLAHE candidates, considering the unchanged baseline as well. Freeze the chosen clip limit and grid before a test comparison:

```bash
./.venv/bin/python scripts/run_clahe_optimization.py --split test --clip-limits 2 --grid-size 8
```

The completed sweep selected clip limit 2 as the best CLAHE candidate, though it still trails the unchanged validation baseline. Test results are saved under `runs/optimization_1280_clahe/test/`. Precision and recall use the standard Ultralytics F1-selected confidence operating point, not the manual-inspection threshold of 0.25. Preprocessing time includes LAB conversions and CLAHE but excludes disk reads and writes. Improvements on this split do not establish real-weather robustness or statistical significance.

## Validation results

All candidates use the same 120 validation images and 171 annotated signs, with an 8×8 CLAHE grid:

| Pipeline | Clip limit | Precision | Recall | mAP50 | mAP50–95 | Delta mAP50–95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Unchanged baseline | — | 0.4480 | 0.6026 | 0.5986 | 0.5250 | 0.0000 |
| CLAHE | 1 | 0.4446 | 0.5910 | 0.5669 | 0.4970 | -0.0280 |
| CLAHE | 2 | 0.4499 | 0.5996 | 0.5843 | 0.5166 | -0.0084 |
| CLAHE | 3 | 0.5610 | 0.5132 | 0.5787 | 0.5096 | -0.0154 |
| CLAHE | 4 | 0.4540 | 0.5823 | 0.5707 | 0.5020 | -0.0230 |

CLAHE did not improve validation mAP50–95 for any tested clip limit. Clip limit 2 is the best CLAHE candidate, approximately 0.84 percentage points below baseline, and was frozen before testing to provide a method comparison. It is not selected as the recommended preprocessing pipeline. The separate gamma 0.9 experiment achieved validation mAP50–95 of 0.5276, above both baseline and all tested CLAHE candidates. These measurements do not establish the cause of CLAHE's decrease; amplifying irrelevant local contrast or noise is a hypothesis requiring further inspection.

## Frozen-parameter clean-test comparison

The test comparison uses the same 300 images and 361 annotated signs, with clip limit 2 and the 8×8 grid fixed from the validation comparison. The clean baseline reproduces the recorded metrics. Gamma results are included from the separate completed experiment; gamma and CLAHE were not combined.

| Pipeline | Precision | Recall | mAP50 | mAP50–95 | Delta mAP50–95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Unchanged baseline | 0.6919 | 0.4611 | 0.5579 | 0.4921 | 0.0000 |
| Gamma 0.9 | 0.6781 | 0.4760 | 0.5624 | 0.4993 | +0.0073 |
| CLAHE, clip limit 2, grid 8×8 | 0.5448 | 0.5554 | 0.5808 | 0.5188 | +0.0267 |

CLAHE improves test mAP50–95 by approximately 2.67 percentage points and mAP50 by approximately 2.30 percentage points. Reported recall increases, with a substantial decrease in precision at the evaluator's F1-selected operating point. CLAHE preprocessing costs approximately 13.6 ms per image, including LAB conversions and excluding disk reads and writes; gamma correction costs approximately 1 ms per image in its separate run.

Validation and test results disagree: this CLAHE configuration decreases validation mAP50–95 but increases test mAP50–95. Report both rather than asserting a consistent improvement. Gamma 0.9 remains the choice supported by validation selection among these methods. Selecting CLAHE as the final pipeline because it wins on test would be an exploratory test-informed decision and must be disclosed. Possible differences in lighting or class composition between splits require further analysis; they are not established explanations. The baseline model itself was already chosen using a test-based resolution comparison. Neither preprocessing method has yet been evaluated across the degraded test suites.
