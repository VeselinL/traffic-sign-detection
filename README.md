# Traffic Sign Detection & Classification with Robustness Analysis

![Banner](banner/banner.png)

An AI course project using **YOLOv8n** and the **German Traffic Sign Detection Benchmark (GTSDB)** to detect and classify traffic signs in driving scenes. The project compares input resolutions, analyzes failures, evaluates synthetic robustness, and tests inexpensive gamma and CLAHE preprocessing without retraining the selected detector.

**Status:** resolution comparison, clean-image preprocessing, 30 robustness scenarios, and all 60 separate post-degradation preprocessing evaluations are complete. Architectural improvements and embedded deployment remain future work.

## Dataset and Evaluation

- **900 scenes**, 1360×800 pixels, **1,213 annotated signs**, and **43 fine-grained classes**.
- Fixed split: **480 training images / 681 signs**, **120 validation images / 171 signs**, and **300 test images / 361 signs**.
- The first 600 scenes use a seeded multilabel-stratified training/validation split (seed 42); test scenes are IDs 00600–00899.
- Robustness and preprocessing use frozen `models/yolov8n_baseline_1280/best.pt` weights and preserve original images, class definitions, splits, dimensions, and labels.
- Comparable CPU evaluations use batch size 8, confidence threshold 0.001, NMS IoU 0.7, and no test-time augmentation. Summary precision and recall use the evaluator's F1-selected operating point, not a shared fixed confidence threshold.

Small signs and class imbalance are central challenges: a median sign is only 38×37 pixels in the original scene. Downscaling removes details needed to distinguish digits and visually similar classes.

## Clean-Test Results

All methods use the same 300-image test split. Metrics are on a 0–1 scale. Timings are per image on an AMD Ryzen 5 5600 CPU, batch size 8.

| Method | Precision | Recall | mAP50 | mAP50–95 | Correction (ms) | Inference (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLOv8n 640px | 0.4858 | 0.3233 | 0.3534 | 0.2909 | — | 29.0 |
| YOLOv8n 960px | 0.6009 | 0.4013 | 0.4519 | 0.3938 | — | 72.3 |
| YOLOv8n 1280px | 0.6919 | 0.4611 | 0.5579 | 0.4921 | 0.0 | 138.4 |
| 1280px + gamma 0.9 | 0.6781 | 0.4760 | 0.5624 | 0.4993 | 1.0 | 138.1 |
| 1280px + CLAHE | 0.5448 | 0.5554 | 0.5808 | 0.5188 | 13.6 | 137.1 |

Increasing resolution from 640px to 1280px improves test mAP50–95 by **20.12 percentage points**, with roughly 4.77× the detector inference cost. Qualitative comparisons show additional small-sign detections and improved speed-limit classification, but misses, wrong classes, and false positives remain.

Gamma 0.9 improves clean-test mAP50–95 by **0.73 percentage points** and is supported by validation selection. CLAHE improves it by **2.67 percentage points**, but decreases validation mAP50–95, so its larger test gain is not evidence of consistent improvement. CLAHE operates on the LAB luminance channel with clip limit 2 and an **8×8 grid of regions**, not pixels. Gamma and CLAHE are tested separately, never combined.

Correction times exclude disk I/O and are additional to detector preprocessing, inference, and postprocessing. These are not end-to-end camera or Raspberry Pi latency measurements.

## Robustness Results

Eight independent OpenCV/NumPy corruptions and two separate Albumentations weather simulations are applied at three fixed severities. Each corruption starts from the original image; operations are not stacked. Random operations use recorded seeds, and derived images are saved as lossless PNGs.

| Scenario | Mild mAP50–95 | Medium mAP50–95 | Severe mAP50–95 |
| --- | ---: | ---: | ---: |
| Underexposure | 0.4697 | 0.4230 | 0.3405 |
| Overexposure | 0.4974 | 0.4829 | 0.4740 |
| Low contrast | 0.5005 | 0.4869 | 0.4367 |
| Gaussian noise | 0.4482 | 0.3437 | 0.2428 |
| Defocus blur | 0.4442 | 0.2484 | 0.0792 |
| Motion blur | 0.3565 | 0.1588 | 0.0332 |
| Resolution loss | 0.5012 | 0.4937 | 0.3557 |
| JPEG compression | 0.4753 | 0.4027 | 0.2584 |
| Synthetic fog | 0.1585 | 0.1472 | 0.1373 |
| Synthetic rain (compound) | 0.4106 | 0.3004 | 0.2853 |

Severe motion blur and defocus produce the largest losses; synthetic fog is damaging at every tested severity. Mild overexposure, low contrast, and resolution loss slightly exceed clean performance on this split, so a nominal degradation does not necessarily lower every measured score.

Fog and rain are **synthetic approximations, not proof of real-weather performance**. Rain combines streaks, blur, and darkening and is reported separately from single-factor corruptions.

### Preprocessing After Degradation

Frozen gamma 0.9 and CLAHE configurations are applied independently **after** each degradation, yielding 60 additional evaluations. Each result is compared with its matching uncorrected degradation.

| Correction | Evaluations | Improved scenarios | Worsened scenarios | Mean signed Δ mAP50–95 |
| --- | ---: | ---: | ---: | ---: |
| Gamma 0.9 | 30 | 15 | 15 | +0.0009 |
| CLAHE, clip 2 / grid 8×8 | 30 | 14 | 16 | −0.0025 |

Neither correction universally improves robustness. Brightening helps some underexposed scenes; local contrast enhancement can amplify noise or background texture. These averages describe the selected synthetic scenarios, not a real driving-condition distribution.

## Repository Structure

```text
├── src/
│   ├── data/                 # Annotation conversion and fixed dataset splits
│   ├── degradation/          # Corruptions and synthetic weather transforms
│   ├── optimization/         # Gamma, CLAHE, and shared dataset verification
│   ├── train.py              # Original training experiment
│   └── evaluate.py           # Original evaluation experiment
├── scripts/
│   ├── run_robustness_eval.py
│   ├── run_degraded_preprocessing_eval.py
│   └── run_clahe_optimization.py
├── utils/                    # Lossless PPM-to-PNG conversion
├── notebooks/                # Dataset inspection and prediction visualization
├── docs/                     # Protocols, dataset notes, and detailed results
├── plots/                    # Dataset visualizations
├── reports/latex/report.pdf   # English project report
├── VALIDATION.md              # Qualitative validation failure analysis
└── requirements.txt
```

Datasets, weights, evaluation outputs, temporary previews, and local working notes are excluded from Git. Only the English PDF is retained from `reports/`; only the three listed entrypoints are retained from `scripts/`.

## Setup and Local Inputs

Create a Python environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The recorded evaluation environment uses Python 3.12.3. On Windows, activate with `.venv\Scripts\activate`.

Obtain GTSDB from the [official benchmark website](https://benchmark.ini.rub.de/gtsdb_news.html). **Dataset files and trained checkpoints are not distributed in this repository.** Evaluation requires:

- `data/GTSDB/dataset/data.yaml`, preserving the original 43-class ordering and `train.txt`, `val.txt`, and `test.txt` split references.
- Original PPM scenes in `data/GTSDB/dataset/images/` and corresponding YOLO labels in `data/GTSDB/dataset/labels/`.
- Matching PNG copies and `train_png.txt`, `val_png.txt`, and `test_png.txt` manifests.
- The frozen checkpoint at `models/yolov8n_baseline_1280/best.pt`.

For an already prepared PPM dataset, create lossless PNG copies and matching manifests:

```bash
python utils/convert_ppm_to_png.py
```

See [dataset notes](docs/dataset_notes.md) for annotation format and statistics. Evaluation inputs resolve relative to the repository root. The clean preprocessing comparisons verify split IDs and pixel equivalence; a different split or checkpoint cannot reproduce the recorded results.

## Run Evaluations

Verify the clean baseline, then generate and evaluate the 30 degraded test suites:

```bash
python scripts/run_robustness_eval.py --clean-only
python scripts/run_robustness_eval.py
```

The runner stops before degradation evaluation if clean mAP differs from the recorded reference beyond its configured tolerance. Outputs are saved under `runs/robustness_1280/`.

Compare CLAHE parameters on validation, then evaluate the frozen comparison configuration on test:

```bash
python scripts/run_clahe_optimization.py
python scripts/run_clahe_optimization.py --split test --clip-limits 2 --grid-size 8
```

Outputs are under `runs/optimization_1280_clahe/`. All tested CLAHE settings decrease validation mAP50–95; clip limit 2 is retained for comparison, not as a validation-supported improvement.

After the full robustness run, evaluate separate gamma and CLAHE corrections on all saved degraded suites:

```bash
python scripts/run_degraded_preprocessing_eval.py
```

Outputs are under `runs/degraded_preprocessing_1280/`. Use `--help` for runner-specific output, batch-size, device, and scenario-selection options.

## Documentation

- [English project report](reports/latex/report.pdf)
- [Validation failure analysis](VALIDATION.md)
- [Robustness protocol](docs/robustness_evaluation.md) and [results](docs/robustness_results.md)
- [Gamma optimization](docs/gamma_optimization.md) and [CLAHE optimization](docs/clahe_optimization.md)
- [Post-degradation preprocessing protocol](docs/degraded_preprocessing_evaluation.md) and [full metric comparison](docs/degraded_preprocessing_results.md)

## Limitations and Future Plans

The test split informed resolution selection and was reused for exploratory comparisons, so results are not estimates from an untouched final holdout. The dataset is small and imbalanced; no confidence intervals or significance tests were computed. Simulated weather does not establish road-deployment reliability.

- **SAHI tiling:** test overlapping image tiles for small signs, selecting tile size, overlap, and merging rules on validation. Measure missed signs, duplicate detections, false positives, and full inference cost.
- **Bigger models:** compare YOLOv8s or larger detectors on identical splits, with an explicit accuracy–latency budget.
- **Specialized classifier:** train a CNN on cropped signs to distinguish digits and similar classes. Separate crops by source scene and evaluate detector-produced crops, retaining localization failures.
- **Balanced data and robustness training:** expand rare classes and difficult scenes; compare class-aware sampling and realistic corruption augmentation while monitoring clean accuracy.
- **Embedded deployment (Raspberry Pi):** export and benchmark with a camera, potentially using a Hailo AI accelerator. Measure complete capture/preprocessing/inference/postprocessing latency, FPS, memory, and accuracy after export or quantization. No embedded performance is claimed yet.
- **Stronger evaluation:** reserve a new independent holdout, quantify uncertainty, and evaluate real adverse-weather data before deployment claims.

## Acknowledgments

- GTSDB: Institut für Neuroinformatik, Ruhr-Universität Bochum.
- Detection framework: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics).
- Image processing and synthetic weather: OpenCV, NumPy, and Albumentations.
