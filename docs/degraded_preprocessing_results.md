# Post-degradation preprocessing results

The frozen YOLOv8n 1280px model is evaluated on the same 300-image test split for every row. Each source image first receives the named synthetic degradation, then either no correction, gamma 0.9, or CLAHE on LAB luminance (clip limit 2, grid 8×8). Gamma and CLAHE are separate pipelines and are never combined.

The table reports precision, recall, mAP50, and mAP50–95. Deltas are for mAP50–95 and are relative to the matching unprocessed degradation, not clean performance. At generation time, 60 of 60 post-degradation evaluations were complete; unfinished cells are shown as —.

| Degradation | Severity | Base P | Base R | Base mAP50 | Base mAP50–95 | Gamma P | Gamma R | Gamma mAP50 | Gamma mAP50–95 | Δ gamma | CLAHE P | CLAHE R | CLAHE mAP50 | CLAHE mAP50–95 | Δ CLAHE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| underexposure | mild | 0.5556 | 0.4922 | 0.5298 | 0.4697 | 0.5844 | 0.5092 | 0.5330 | 0.4708 | +0.0011 | 0.5852 | 0.4802 | 0.5206 | 0.4602 | -0.0096 |
| underexposure | medium | 0.5266 | 0.4592 | 0.4793 | 0.4230 | 0.5260 | 0.5389 | 0.4959 | 0.4361 | +0.0131 | 0.5169 | 0.5170 | 0.4872 | 0.4249 | +0.0018 |
| underexposure | severe | 0.5663 | 0.3726 | 0.3969 | 0.3405 | 0.5645 | 0.3922 | 0.4192 | 0.3619 | +0.0214 | 0.6340 | 0.3729 | 0.4293 | 0.3621 | +0.0216 |
| overexposure | mild | 0.5903 | 0.5177 | 0.5615 | 0.4974 | 0.5374 | 0.6053 | 0.5587 | 0.4968 | -0.0006 | 0.6695 | 0.4868 | 0.5612 | 0.5021 | +0.0047 |
| overexposure | medium | 0.6040 | 0.5106 | 0.5480 | 0.4829 | 0.6183 | 0.5062 | 0.5482 | 0.4831 | +0.0002 | 0.6065 | 0.4841 | 0.5706 | 0.5100 | +0.0271 |
| overexposure | severe | 0.6505 | 0.4521 | 0.5419 | 0.4740 | 0.7018 | 0.4087 | 0.5431 | 0.4761 | +0.0021 | 0.5826 | 0.5240 | 0.5602 | 0.4927 | +0.0187 |
| low_contrast | mild | 0.6400 | 0.5392 | 0.5599 | 0.5005 | 0.6425 | 0.5375 | 0.5585 | 0.4990 | -0.0015 | 0.6130 | 0.5208 | 0.5744 | 0.5126 | +0.0121 |
| low_contrast | medium | 0.5848 | 0.5343 | 0.5419 | 0.4869 | 0.5804 | 0.5285 | 0.5451 | 0.4893 | +0.0024 | 0.6484 | 0.5186 | 0.5616 | 0.5027 | +0.0158 |
| low_contrast | severe | 0.6988 | 0.3997 | 0.4888 | 0.4367 | 0.6463 | 0.4152 | 0.4857 | 0.4316 | -0.0051 | 0.6039 | 0.5169 | 0.5293 | 0.4709 | +0.0342 |
| gaussian_noise | mild | 0.6042 | 0.4196 | 0.4968 | 0.4482 | 0.5745 | 0.4444 | 0.4962 | 0.4473 | -0.0009 | 0.5171 | 0.4924 | 0.4786 | 0.4292 | -0.0189 |
| gaussian_noise | medium | 0.4866 | 0.4330 | 0.3970 | 0.3437 | 0.4807 | 0.4303 | 0.3870 | 0.3342 | -0.0095 | 0.4825 | 0.3862 | 0.3554 | 0.3038 | -0.0399 |
| gaussian_noise | severe | 0.4474 | 0.3084 | 0.2813 | 0.2428 | 0.4381 | 0.3038 | 0.2750 | 0.2382 | -0.0045 | 0.4114 | 0.2862 | 0.2487 | 0.2130 | -0.0298 |
| defocus_blur | mild | 0.5879 | 0.4680 | 0.5153 | 0.4442 | 0.5821 | 0.4841 | 0.5132 | 0.4430 | -0.0012 | 0.6349 | 0.4243 | 0.5078 | 0.4319 | -0.0123 |
| defocus_blur | medium | 0.4976 | 0.3280 | 0.3347 | 0.2484 | 0.4974 | 0.3346 | 0.3399 | 0.2521 | +0.0037 | 0.6135 | 0.2772 | 0.3248 | 0.2324 | -0.0160 |
| defocus_blur | severe | 0.3196 | 0.1437 | 0.1455 | 0.0792 | 0.3687 | 0.1285 | 0.1484 | 0.0812 | +0.0020 | 0.5507 | 0.1636 | 0.1814 | 0.0980 | +0.0187 |
| motion_blur | mild | 0.5177 | 0.4012 | 0.4161 | 0.3565 | 0.5677 | 0.3851 | 0.4222 | 0.3603 | +0.0038 | 0.4714 | 0.4131 | 0.4073 | 0.3480 | -0.0086 |
| motion_blur | medium | 0.4286 | 0.2085 | 0.2207 | 0.1588 | 0.4437 | 0.2036 | 0.2177 | 0.1554 | -0.0034 | 0.5206 | 0.1890 | 0.2057 | 0.1456 | -0.0132 |
| motion_blur | severe | 0.0698 | 0.1194 | 0.0620 | 0.0332 | 0.0695 | 0.1213 | 0.0618 | 0.0334 | +0.0002 | 0.0657 | 0.0984 | 0.0550 | 0.0275 | -0.0057 |
| resolution_loss | mild | 0.5663 | 0.5229 | 0.5612 | 0.5012 | 0.5667 | 0.5140 | 0.5621 | 0.5003 | -0.0009 | 0.6684 | 0.4508 | 0.5831 | 0.5231 | +0.0218 |
| resolution_loss | medium | 0.6124 | 0.4804 | 0.5581 | 0.4937 | 0.6514 | 0.4750 | 0.5734 | 0.5071 | +0.0134 | 0.6196 | 0.4868 | 0.5710 | 0.5030 | +0.0092 |
| resolution_loss | severe | 0.5190 | 0.3991 | 0.4196 | 0.3557 | 0.5507 | 0.4182 | 0.4205 | 0.3552 | -0.0006 | 0.6867 | 0.3427 | 0.4134 | 0.3438 | -0.0120 |
| jpeg_compression | mild | 0.5969 | 0.5210 | 0.5385 | 0.4753 | 0.5895 | 0.5280 | 0.5438 | 0.4784 | +0.0031 | 0.6130 | 0.4939 | 0.5428 | 0.4774 | +0.0020 |
| jpeg_compression | medium | 0.6282 | 0.3810 | 0.4665 | 0.4027 | 0.6177 | 0.3958 | 0.4573 | 0.3948 | -0.0079 | 0.6415 | 0.3827 | 0.4653 | 0.4032 | +0.0005 |
| jpeg_compression | severe | 0.5853 | 0.2836 | 0.3132 | 0.2584 | 0.5882 | 0.2932 | 0.3111 | 0.2572 | -0.0013 | 0.5210 | 0.3203 | 0.2992 | 0.2436 | -0.0149 |
| fog | mild | 0.3963 | 0.2290 | 0.2273 | 0.1585 | 0.4017 | 0.2297 | 0.2280 | 0.1572 | -0.0013 | 0.4677 | 0.2343 | 0.2329 | 0.1572 | -0.0014 |
| fog | medium | 0.4528 | 0.2165 | 0.2101 | 0.1472 | 0.4029 | 0.2231 | 0.2107 | 0.1477 | +0.0005 | 0.4865 | 0.2116 | 0.2280 | 0.1539 | +0.0067 |
| fog | severe | 0.3956 | 0.1962 | 0.1876 | 0.1373 | 0.4028 | 0.1830 | 0.1877 | 0.1376 | +0.0003 | 0.4615 | 0.1860 | 0.1819 | 0.1300 | -0.0073 |
| rain | mild | 0.5437 | 0.4734 | 0.4689 | 0.4106 | 0.6671 | 0.4374 | 0.4701 | 0.4106 | +0.0000 | 0.5753 | 0.4669 | 0.4670 | 0.4070 | -0.0036 |
| rain | medium | 0.4939 | 0.3536 | 0.3473 | 0.3004 | 0.4575 | 0.3579 | 0.3445 | 0.2984 | -0.0020 | 0.3918 | 0.3386 | 0.3116 | 0.2636 | -0.0368 |
| rain | severe | 0.4666 | 0.3664 | 0.3311 | 0.2853 | 0.4588 | 0.3686 | 0.3296 | 0.2844 | -0.0008 | 0.4385 | 0.3129 | 0.2876 | 0.2442 | -0.0411 |

The detailed precision, recall, mAP50, preprocessing time, and per-class AP are in `runs/degraded_preprocessing_1280/summary.csv` and the scenario-level `metrics.json` files.

Interpret this table together with the clean-test comparison: gamma 0.9 was selected by validation mAP50–95, while CLAHE clip limit 2 improved this project’s clean test but decreased validation mAP50–95. The post-degradation results are exploratory because the same fixed test split is used for analysis.
