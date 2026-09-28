# Frozen-model robustness evaluation

Run from the repository root with the project environment:

```bash
./.venv/bin/python scripts/run_robustness_eval.py
```

The runner uses `models/yolov8n_baseline_1280/best.pt` without training. It reads the class names and test split from `data/GTSDB/dataset/data.yaml`. That YAML lists PPM images; Ultralytics 8.4.116 cannot read PPM, so the runner verifies that `test_png.txt` has the same 300 ordered image IDs and evaluates the existing lossless PNG copies. The 300 PNGs were checked against their PPM counterparts pixel for pixel. Labels are copied unchanged into each output suite. Original images and labels are never rewritten.

The first evaluation is clean. It must reproduce the recorded mAP50 of 0.5579 and mAP50–95 of 0.4921 within 0.005 or the run stops. Every degradation then starts from the original clean PNG, never from another degraded result. The image dimensions and bounding-box coordinates stay unchanged.

| Degradation | Mild | Medium | Severe |
| --- | ---: | ---: | ---: |
| Underexposure, gamma | 1.5 | 2.0 | 3.0 |
| Overexposure, gain | 1.25 | 1.5 | 2.0 |
| Low contrast, factor | 0.75 | 0.50 | 0.25 |
| Gaussian noise, pixel sigma | 8 | 16 | 25 |
| Defocus blur, disk radius | 3 | 6 | 10 |
| Motion blur, kernel length | 7 | 15 | 25 |
| Resolution loss, scale | 0.75 | 0.50 | 0.25 |
| JPEG, quality | 70 | 40 | 15 |
| Fog, coefficient / alpha | 0.2 / 0.08 | 0.4 / 0.10 | 0.6 / 0.12 |
| Rain, type / blur / brightness | drizzle / 3 / 0.90 | heavy / 5 / 0.80 | torrential / 7 / 0.70 |

The eight single-factor transforms use OpenCV and NumPy. Resolution loss uses `INTER_AREA` when shrinking and `INTER_LINEAR` when restoring the original size. Motion-blur angle and Gaussian noise are seeded for each image. JPEG uses one encode/decode pass; the result is saved as PNG. Fog and rain use Albumentations with `p=1`; the default seed is 42 and each image receives a deterministic seed derived from that value, its ID, degradation, and severity.

Rain combines streaks, blur, and darkening, so compare it as a compound scenario rather than attributing its result to one factor. Fog and rain are synthetic approximations and do not establish real-weather performance.

Generated suites, manifests, per-scenario metrics, and `summary.csv` are under `runs/robustness_1280/`. The summary records precision, recall, mAP50, mAP50–95, and absolute mAP drops against clean. The baseline checkpoint was selected after a resolution comparison that used this test set, so report these scores as exploratory robustness measurements.

To verify only the clean score, use `--clean-only`. To run one degradation after the clean check, use `--only fog` (or another degradation name). `--seed`, `--batch`, `--device`, and `--output` set run options. Existing completed scenario metrics are reused when their seed matches.
