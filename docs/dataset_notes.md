# GTSDB Dataset Notes

Working notes for dataset inspection, annotation decisions, preprocessing hypotheses, and observations that are not yet ready for the main README or final report.

## Dataset Location and Structure

- **Local dataset mount:** `/data`
- **Source archive/version:** GTSDB / FullIJCNN2013 dataset
- **Full-scene images:** approximately 900
- **Image resolution:** 1360 × 800 pixels (width × height)
- **Native annotations:** `gt.txt`
- **Annotation format:** `filename;left;top;right;bottom;class_id`
- **Additional folders:** cropped traffic-sign images grouped by class ID

## Annotation Parsing and Visualization

- `gt.txt` has been parsed into a mapping from image filename to a list of bounding boxes.
- Repeated filenames are expected because each line in `gt.txt` represents one annotated sign. Images containing multiple signs therefore appear on multiple rows.
- Native boxes have been drawn on sample images and visually align with the signs.
- Images absent from `gt.txt` are presumed to contain no annotated signs.
- The current annotation count has been verified in three equivalent ways:
  - total parsed rows in `gt.txt`
  - sum of all class counts
  - sum of the lengths of all per-image bounding-box lists

## Confirmed Dataset Statistics

- **Total annotated signs / bounding boxes:** 1213
- **Number of fine-grained classes:** 43
- **Image area:** 1,088,000 px²

### Bounding-box summary

| Metric | Width | Height | Area |
|---|---:|---:|---:|
| Minimum | 16 px | 16 px | 256 px² |
| 25th percentile | 28 px | 28 px | 784 px² |
| Median / 50th percentile | 38 px | 37 px | 1369 px² |
| Mean | 43.60 px | 42.86 px | 2322.00 px² |
| 90th percentile | 73.8 px | 71 px | 5236.8 px² |
| Maximum | 127 px | 128 px | 15376 px² |

### What each metric means

- **Minimum:** the smallest observed value. It identifies the most extreme small-object cases, but it does not describe what is typical.
- **25th percentile:** 25% of boxes have a value at or below this threshold. For example, 25% of signs are no wider than 28 pixels in the original image.
- **Median / 50th percentile:** half of all boxes are smaller and half are larger. It is more representative than the mean when the distribution contains unusually large boxes.
- **Mean:** the arithmetic average. It is sensitive to large outliers and is therefore higher than the median for this dataset.
- **90th percentile:** 90% of boxes are at or below this value, while the largest 10% exceed it. It describes the upper end without relying only on the single maximum.
- **Maximum:** the largest observed value. Like the minimum, it is useful for identifying extremes but not for describing the typical box.
- **Area:** `width × height` for each individual box. The correct mean area is the mean of all individual areas, not `mean(width) × mean(height)`.

The mean area is substantially larger than the median area (`2322` vs. `1369` px²), indicating a right-skewed distribution: most signs are relatively small, while a smaller number of large signs increase the average.

### Relative size and resizing implications

- Median sign dimensions relative to the original image:
  - width: approximately 2.79% of image width
  - height: approximately 4.63% of image height
  - area: approximately 0.126% of the full image
- Mean sign area: approximately 0.213% of the full image.

If the 1360-pixel image width is resized to 640 pixels, the approximate scale factor is `640 / 1360 ≈ 0.471`:

- median sign: roughly 18 × 17 pixels
- 25th-percentile sign: roughly 13 × 13 pixels
- minimum sign: roughly 7.5 × 7.5 pixels

This confirms that GTSDB is a small-object detection problem. Aggressive input downscaling may reduce training time, but it can also remove useful detail and lower recall for the smallest signs. The initial baseline should therefore use `imgsz=640`; smaller values should only be tested after measuring their effect on small-object performance.

## Class Distribution

The 43-class distribution is strongly imbalanced. Class counts range from 2 to 88 examples, producing an imbalance ratio of 44:1.

Notable frequent classes:

- Class 38 — keep right: 88
- Class 12 — priority road: 85
- Class 13 — give way: 83
- Class 2 — speed limit 50: 81
- Class 10 — no overtaking for trucks: 80
- Class 1 — speed limit 30: 79

Rare classes:

- Class 19 — bend left: 2
- Class 31 — animals: 2
- Class 37 — go left or straight: 2
- Class 27 — pedestrian crossing: 3
- Class 0 — speed limit 20: 4

A class share computed as `class_count / 1213` is a **proportion** between 0 and 1. Multiply by 100 to express it as a percentage.

### Superclass totals

- **Prohibitory:** 557 signs, approximately 45.9%
- **Other:** 274 signs, approximately 22.6%
- **Danger:** 219 signs, approximately 18.1%
- **Mandatory:** 163 signs, approximately 13.4%

### Interpretation

- The class imbalance is severe enough that overall mAP alone may hide poor performance on rare classes.
- Per-class precision, recall, and AP will therefore be necessary during evaluation.
- Classes with only 2–5 examples cannot support strong or stable conclusions from a single split.
- The four-superclass grouping is considerably better populated, but still imbalanced.
- No final taxonomy decision has been made. The current plan is to preserve the 43-class problem for the primary baseline and potentially report superclass-level results as an auxiliary analysis.

## Initial Visual Observations

Several distinct image-quality failure modes have been observed:

- severe global overexposure
- severe underexposure
- motion-blurred signs
- local sun glare or reflection affecting the sign while the rest of the scene remains usable
- locally dark signs in otherwise acceptable scenes

These should be treated separately because they remove or distort information in different ways:

- global overexposure may clip large image regions to white
- underexposure reduces visible contrast and detail
- local glare can destroy sign information without affecting the full scene
- motion blur smears edges and internal symbols

A single global enhancement method is unlikely to solve all of these cases.

## Planned Image-quality Metrics

The following statistics will be calculated for both full images and annotated sign crops.

- **Mean luminance:** average brightness of the pixels. High values suggest bright images; low values suggest dark images. It can be misleading when only one image region is extremely bright or dark.
- **Median luminance:** the middle luminance value. It is less affected than the mean by isolated highlights or shadows.
- **Luminance standard deviation:** measures brightness variation. Low values indicate a flat, low-contrast image; higher values indicate stronger intensity variation.
- **Near-black pixel fraction:** proportion of pixels below a chosen low-intensity threshold. It helps identify heavy shadowing, underexposure, or clipped dark regions.
- **Near-white pixel fraction:** proportion of pixels above a chosen high-intensity threshold. It helps identify overexposure, glare, and clipped highlights.

Comparing these metrics at image level and sign-crop level will distinguish:

- globally bright image, normally exposed sign
- normally exposed image, overexposed sign
- globally dark image
- locally dark sign
- acceptable average brightness but low local contrast

## Preprocessing Hypothesis: Exposure Normalization

Potential experiment: apply deterministic exposure or contrast correction before model training and inference, then compare it against the untouched-image baseline.

Candidate methods:

- global gamma correction
- histogram equalization on the luminance channel
- CLAHE on the luminance channel
- percentile-based contrast stretching
- tone mapping, if justified by the exposure analysis

Important experimental rules:

- The untouched dataset remains the baseline.
- Any preprocessing method must be evaluated as a separate controlled condition.
- The same preprocessing must be applied consistently to train, validation, and test images.
- Saturated or motion-blurred information cannot necessarily be recovered by contrast enhancement.
- Preprocessing should not be adopted solely because individual examples look better; it must improve measured performance.

## Remaining Phase 1 Work

### Annotation integrity

- [ ] Count total images, annotated images, and unannotated images
- [ ] Count images with one sign and multiple signs
- [ ] Record the maximum signs in one image
- [ ] Verify every annotated filename exists and every image loads
- [ ] Validate that `right > left` and `bottom > top`
- [ ] Validate that no coordinate is negative or outside image boundaries

### Box geometry and visualization

- [x] Compute minimum, mean, 25th percentile, median, 90th percentile, and maximum for width, height, and area
- [ ] Plot width distribution
- [ ] Plot height distribution
- [ ] Plot relative-area distribution
- [ ] Inspect examples from the smallest-box tail of the distribution

### Image-quality analysis

For full images and annotated sign crops:

- [ ] mean luminance
- [ ] median luminance
- [ ] luminance standard deviation
- [ ] fraction of nearly black pixels
- [ ] fraction of nearly white pixels

Create a fixed qualitative set containing:

- [ ] random annotated images
- [ ] smallest-sign images
- [ ] multi-sign images
- [ ] overexposed images
- [ ] underexposed images
- [ ] local glare cases
- [ ] dark-sign cases
- [ ] motion-blur cases

### Decisions still required

- [ ] final 43-class vs. superclass evaluation strategy
- [ ] final train/validation/test strategy
- [ ] confirm `imgsz=640` as the baseline input resolution

## Current Decision

No preprocessing method will silently replace the raw images. The first trained model will use the untouched dataset so that all later image-processing and robustness experiments have a valid baseline.
