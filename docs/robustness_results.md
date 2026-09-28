# Frozen YOLOv8n 1280px robustness results

The clean test rerun reproduced the recorded baseline on the same 300 images and 361 annotated signs: precision 0.6919, recall 0.4611, mAP50 0.5579, and mAP50–95 0.4921. The checkpoint was not retrained. Each degraded suite contains the same test images and labels, with one degradation applied to each original image.

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

The lowest mAP50–95 is severe motion blur (0.0332), followed by severe defocus blur (0.0792). Fog is low at every severity, starting at 0.1585 for mild. Small improvements over clean at mild overexposure, mild low contrast, and mild resolution loss are measured outcomes on this fixed test split; they do not establish that those changes improve general performance.

The full precision, recall, mAP50, mAP50–95, and clean-relative drops are in `runs/robustness_1280/summary.csv`. Each scenario's exact settings, seed, image list, labels, and metrics are under `runs/robustness_1280/datasets/`. Fog and rain are Albumentations simulations, not evidence of performance in real weather. Rain combines streaks, blur, and darkening and should be interpreted separately from the single-factor corruptions.

The baseline checkpoint was chosen after comparing input resolutions on this test set, so these are exploratory robustness results rather than an untouched final test estimate. Failure-case inspection and preprocessing experiments remain subsequent stages.
