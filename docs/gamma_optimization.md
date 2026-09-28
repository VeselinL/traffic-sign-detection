# Gamma brightening experiment

This preprocessing experiment is separate from the robustness pipeline. It uses the frozen `models/yolov8n_baseline_1280/best.pt` checkpoint and preserves the source images, split, class definitions, and labels. Gamma values below one brighten using `255 * (I / 255) ** gamma`; gamma 1.0 is the unchanged baseline. Each candidate is applied independently to the original image and saved as lossless PNG.

Run the validation comparison:

```bash
./.venv/bin/python scripts/run_gamma_optimization.py
```

The default sweep is 0.6, 0.7, 0.8, and 0.9. All candidates and the baseline use the same 1280px Ultralytics evaluation settings. Results, per-class AP, manifests, and generated suites are saved under `runs/optimization_1280_gamma/val/`. The comparison table is available as both `summary.csv` and `summary.md`. Gamma timing excludes disk reads and writes.

Select a gamma using validation mAP50–95, considering recall and per-class changes as well. Then freeze it and evaluate against the clean test baseline. The completed validation sweep selected 0.9:

```bash
./.venv/bin/python scripts/run_gamma_optimization.py --split test --gammas 0.9
```

Test results are saved separately under `runs/optimization_1280_gamma/test/`. The test baseline must reproduce the recorded clean mAP before correction is evaluated. Do not select gamma by repeatedly sweeping test scores.

## Validation results

The completed comparison uses 120 validation images and 171 annotated signs:

| Gamma | Precision | Recall | mAP50 | mAP50–95 | Delta mAP50–95 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1.0 (baseline) | 0.4480 | 0.6026 | 0.5986 | 0.5250 | 0.0000 |
| 0.6 | 0.4694 | 0.6384 | 0.5940 | 0.5241 | -0.0009 |
| 0.7 | 0.4691 | 0.6418 | 0.5935 | 0.5229 | -0.0021 |
| 0.8 | 0.4614 | 0.6490 | 0.5974 | 0.5246 | -0.0004 |
| 0.9 | 0.4522 | 0.6083 | 0.6007 | 0.5276 | +0.0026 |

Gamma 0.9 was selected by validation mAP50–95, before evaluating corrected test images. Its gain is small: approximately 0.26 percentage points of validation mAP50–95. Gamma 0.8 has the highest reported recall but slightly lower mAP50–95 than baseline. Precision and recall are the standard Ultralytics summary values at its F1-selected confidence operating point, not counts at the manual-inspection threshold of 0.25. These results do not establish statistical significance or prove which individual failures were corrected.

## Frozen-parameter clean-test comparison

The selected gamma was evaluated on the same 300 test images and 361 annotated signs. The unchanged test baseline reproduces the previously recorded metrics.

| Pipeline | Precision | Recall | mAP50 | mAP50–95 | Delta mAP50–95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Unchanged baseline | 0.6919 | 0.4611 | 0.5579 | 0.4921 | 0.0000 |
| Gamma 0.9 | 0.6781 | 0.4760 | 0.5624 | 0.4993 | +0.0073 |

Gamma 0.9 improves clean-test mAP50–95 by approximately 0.73 percentage points and mAP50 by approximately 0.46 percentage points. Reported recall increases while precision decreases. Gamma correction itself takes approximately 1 ms per image on this CPU, excluding image decoding and output. This is a measured improvement on the current split, not proof of generalization or degraded-weather robustness. Gamma correction has not yet been evaluated across the degraded test suites. The existing exploratory test-based choice of the baseline model remains a limitation.
