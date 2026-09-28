## YOLOv8n 960px baseline validation failures:

| Image       | Error                       | Likely condition                                                                                                                                        |
|-------------|-----------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|
| `00003.png` | wrong class                 | very small sign                                                                                                                                         |
| `00006.png` | wrong class                 | bad lighting - too much light entering into the sensor                                                                                                  |
| `00009.png` | wrong class                 | bad lighting - reflection off the sign                                                                                                                  |
| `00053.png` | false positive              | cross-class duplicate - visually similar classes                                                                                                        |
| `00067.png` | wrong class                 | small sign - very confident that no overtaking was speed limit 50                                                                                       |
| `00101.png` | wrong class                 | motion-induced distortion - deformed pixels                                                                                                             |
| `00113.png` | wrong class                 | motion-induced distortion - visually similar classes(overtaking vs overtaking trucks)                                                                   |
| `00146.png` | false negative              | 3 FNs: one has a patch of corrupted pixels, other two are very small and reflecting light - cannot be ID'd by a human on full sized image               |
| `00147.png` | wrong class                 | small sign                                                                                                                                              |
| `00149.png` | wrong class                 | don't know- it's big, clear and easily readable(predicted 50 instead of 70)                                                                             |
| `00156.png` | wrong class                 | small sign, but very confident that no overtaking was speed limit 30                                                                                    |
| `00159.png` | wrong class                 | predicted 2 x keep left as keep right, not enough training data?                                                                                        |
| `00172.png` | false negative              | 2 false negatives, one blends with its background, both small?                                                                                          |
| `00173.png` | wrong class                 | clear sign, predicted that limit 30 is limit 50, maybe too small?                                                                                       |
| `00185.png` | wrong class                 | motion-induced distortion, probably deformed pixels because of it, but it's close and pretty clear though                                               |
| `00186.png` | wrong class, false negative | 2 x wc, 1 x fn: very small signs, but priority road is distinct from its background                                                                      |
| `00202.png` | wrong class                 | double prediction on stop sign, one correct one is no entry. Possibly because of it not being lit up enough.                                            |
| `00203.png` | wrong class                 | small sign?                                                                                                                                             |
| `00204.png` | wrong class, false negative | deformed pixels because of motion-induced distortion, but still quite clear                                                                             |
| `00221.png` | wrong class                 | not enough training data                                                                                                                                |
| `00223.png` | wrong class                 | don't know, pretty clear that it's limit 30(predicted 50 with 0.94 conf)                                                                                |
| `00224.png` | wrong class                 | pretty clear, not enough training data?                                                                                                                 |
| `00237.png` | wrong class                 | again, double prediction on stop sign, predicted stop and no entry                                                                                      |
| `00254.png` | false negative              | small sign?                                                                                                                                             |
| `00272.png` | wrong class                 | predicted well on the opposite side of road but not on the slightly shadowed side                                                                       |
| `00286.png` | false negative              | didn't detect stop sign, small sign?                                                                                                                    |
| `00293.png` | wrong class                 | predicted correctly with 0.92 confidence on one side and 0.92 for other sign, even though the fn sign looks clearer                                     |
| `00295.png` | false positive, wrong class | double prediction on restriction ends for speed limit 80(predicted correctly and predicted speed limit 80), predicted a random yellow sign as limit 100 |
| `00299.png` | false negative              | dirty sign, but still very distinguishable                                                                                                              |
| `00305.png` | wrong class                 | 2 x wrong class, not enough data?                                                                                                                       |
| `00307.png` | wrong class                 | not enough data? again predicted school crossing for uneven road - visually similar?                                                                    |
| `00320.png` | false negative              | 2 x FN, one blends in the bg, one can be seen, but overall lighting is pretty dim                                                                       |
| `00338.png` | wrong class, false negative | wrong class + fn: motion-induced distortion, small signs                                                                                                |
| `00343.png` | wrong class                 | 2 x wrong class: very dim and small                                                                                                                     |
| `00346.png` | wrong class                 | motion-induced distortion, small sign, dim                                                                                                              |
| `00352.png` | wrong class                 | small, dim                                                                                                                                              |
| `00360.png` | wrong class, false positive | double prediction for 120: 120 and 80; predicted 70 for motorway number, predicted no overtaking for 120. very dim, speed warp and small                |
| `00372.png` | wrong class                 | dim, small                                                                                                                                              |
| `00375.png` | false negative              | extremely dim, blends in background, small                                                                                                              |
| `00390.png` | wrong class                 | dim, small                                                                                                                                              |
| `00414.png` | wrong class                 | dim, small                                                                                                                                              |
| `00418.png` | wrong class                 | double prediction wrong class: predicted snow and bend right for uneven road; predicted 60 for 120. small signs, speed warp                             |
| `00423.png` | wrong class                 | predicted well on the dimmer side, predicted snow on the other(both signs are construction)                                                             |
| `00432.png` | wrong class                 | 2 x wc: very dim and very small                                                                                                                         |
| `00436.png` | false positive              | predicted speed limit 100 for the opposite direction sign                                                                                               |
| `00440.png` | wrong class                 | quite clear, maybe small?                                                                                                                               |
| `00448.png` | false negative              | too small, blends in?                                                                                                                                   |
| `00458.png` | false positive              | predicted bike path(not in training set) as go right                                                                                                    |
| `00467.png` | wrong class                 | very dim, predicted 50 for 60                                                                                                                           |
| `00470.png` | false negative              | 2 x fn: give way is angled, speed limit 30 is lit up too much                                                                                           |
| `00478.png` | wrong class                 | very dim, predicted 60 for 80                                                                                                                           |
| `00482.png` | false negative              | 2 x fn: angled no entry, angled give way, both pretty small                                                                                             |
| `00490.png` | false negative              | dim and small, not enough data? (roundabout)                                                                                                            |
| `00494.png` | wrong class                 | dim, but close to camera(predicted 50 for 30(0.28 conf))                                                                                                |
| `00507.png` | false negative              | too small: go right                                                                                                                                     |
| `00521.png` | wrong class                 | 2 x wc: small and dim, predicted 60 and 50 for 100                                                                                                      |
| `00544.png` | false positive              | detected reflection in window                                                                                                                           |
| `00551.png` | false negative              | motion-induced distortion,a bit dim, not enough data(no trucks)                                                                                         |
| `00554.png`  | wrong class                 | dim, small                                                                                                                                              |

## Manual comparison: YOLOv8n 960px versus 1280px

The following observations compare the saved clean-validation predictions at the respective input resolutions, using a displayed confidence threshold of 0.25. They were recorded during manual inspection, primarily revisiting the 960px failures listed above. This is a qualitative comparison of selected images, not an exhaustive error audit or a quantitative estimate of improvement. Images without a documented change were not necessarily checked for every possible new error.

“Not detected” below means no corresponding detection was visible above the inspection threshold. It does not establish that the model produced no lower-confidence prediction. The reference classes and error descriptions reflect the manual observations; ambiguous instances should be checked against the ground-truth labels before calculating error counts.

| Image | Reference / condition | 960px observation | 1280px observation | Comparison |
| --- | --- | --- | --- | --- |
| `00003.png` | Bend sign | Correctly classified as bend. | Classified as pedestrian crossing. | Classification regression. |
| `00053.png` | Previously reported false positive | Produced a false positive. | That false positive was absent. | False positive removed. |
| `00067.png` | No overtaking (cars) | Classified as speed limit 50. | Classified as no overtaking (trucks). | Both wrong; the 1280px prediction confuses related overtaking classes. |
| `00113.png` | No overtaking (trucks) | Classified as no overtaking, confidence 0.36. | Correctly classified as no overtaking (trucks), confidence 0.95. | Classification corrected, with higher reported confidence. |
| `00146.png` | Three signs; corrupted pixels | No signs detected. | Two signs detected; one correctly classified. | Partial recovery; one sign remained undetected and one detection was misclassified. |
| `00147.png` | Bend right | Classified as cycles crossing. | Not detected. | Wrong-class detection changed to a miss. |
| `00156.png` | No overtaking | Classified as speed limit 30. | Also classified as speed limit 30, with lower confidence. | Classification error persisted. |
| `00159.png` | Keep-left and keep-right signs | Predicted keep right for all observed signs. | Also predicted keep right for all observed signs. | Keep-left confusion persisted. The reported dataset imbalance is 6 keep-left versus 88 keep-right instances; training-split counts were not verified here. |
| `00172.png` | Small signs with similar backgrounds | No signs detected. | Signs remained undetected; a suspected give-way false positive appeared on a tree/background region. | Misses persisted, with a suspected new false positive. |
| `00173.png` | Speed limit 30 | Classified as speed limit 50, confidence 0.94. | Correctly classified as speed limit 30, confidence 0.57. | Classification corrected despite lower confidence. |
| `00185.png` | Speed limit 50 | Classified as speed limit 70. | Correctly classified as speed limit 50. | Classification corrected. |
| `00186.png` | Two speed-limit-70 signs and a priority-road sign | Produced two speed-limit-50 predictions. | Correctly classified priority road; classified one speed-limit sign as 50 and missed the other. | Partial improvement; speed-limit errors remained. Candidate for tiled inference. |
| `00202.png` | Two stop signs | Produced a duplicate prediction on one sign. | Correctly detected and classified both stop signs without the reported duplicate. | Duplicate removed and both signs correctly recognized. |
| `00223.png` | Speed limit 30 | Classified as speed limit 50. | Correctly classified as speed limit 30. | Classification corrected. |
| `00224.png` | Three signs | Two correct predictions; third sign misclassified. | Two correct predictions; third sign not detected. | Two correct predictions retained; wrong-class detection changed to a miss. |
| `00241.png` | Speed limit 50 | Classified as speed limit 80. | Correctly classified as speed limit 50. | Classification corrected. |
| `00272.png` | Two signs | Predictions differed between the two sign locations. | The assignments were reported as switched relative to 960px. | Changed predictions; an overall improvement was not established. |
| `00286.png` | Previously missed sign | Not detected. | Detected. | Detection recovered; classification correctness was not explicitly recorded. |
| `00293.png` | Two signs | Correctly recognized one sign. | Correctly recognized both signs. | Additional correct recognition. |
| `00295.png` | Two signs | Struggled with the predictions, as recorded above. | Correctly recognized both signs. | Qualitative improvement. |
| `00307.png` | Four signs | Three correct predictions; fourth sign misclassified. | Three correct predictions; fourth sign not detected. | Wrong-class detection changed to a miss. |
| `00320.png` | Two signs | Previously recorded misses. | Detected one sign but misclassified it; missed the other. | Partial localization recovery without correct classification. |
| `00343.png` | Two very dim signs | Previously recorded wrong-class predictions. | One correctly classified; the other misclassified with very high confidence. | Partial improvement; confident error remained. |
| `00352.png` | Small, dim sign | Failed. | Also failed. | No documented improvement. |
| `00360.png` | Multiple signs and a waypoint/route marker | Previously recorded wrong classes and false positives. | One sign correctly classified; waypoint/route marker still predicted as speed limit 70. | Partial improvement; background/other-sign confusion persisted. |
| `00372.png` | Speed limit 70 | Classified as speed limit 50. | Correctly classified as speed limit 70. | Classification corrected. |
| `00375.png` | Very dim sign | Not detected. | Detected and correctly classified. | Correct recognition recovered. |
| `00390.png` | Two signs | Previously recorded classification errors. | One correct prediction; the other remained incorrect. | Partial improvement. |
| `00418.png` | Speed-limit sign and a second sign | Previously recorded classification errors. | Speed-limit sign correctly classified; second sign not detected. | Partial improvement; another sign remained missed. |
| `00423.png` | Signs in the scene | At least one correct prediction was recorded above. | No signs detected above confidence 0.25. | Detection regression at the inspection threshold. |
| `00448.png` | Small sign | Not detected. | Correctly recognized. | Correct recognition recovered. |
| `00470.png` | Three signs, including small signs | Previously recorded missed signs. | All three detected and correctly classified. | Strong qualitative example for the resolution comparison. |
| `00482.png` | Multiple signs, including angled give way | Previously recorded misses. | An additional sign detected and correctly classified; angled give-way sign remained undetected. | Partial recovery; angled sign still missed. |
| `00507.png` | Small keep-right sign, per the latest inspection | Not detected. | Detected and correctly classified as keep right. | Correct recognition recovered. The earlier table calls this sign “go right”; confirm its class against the label before using it as a class-specific example. |
| `00521.png` | Two signs | Previously recorded wrong classes. | Both signs correctly classified, but a duplicate detection introduced a false positive. | Classification improved, with an additional false positive. |
| `00544.png` | Construction sign, per the latest inspection | Correctly classified as construction. | Classified as priority at next intersection. | Classification regression. The earlier table also records a reflection-related false positive; that separate observation is retained above. |
| `00551.png` | Previously missed sign | Not detected. | Detected and correctly classified. | Correct recognition recovered. |
| `00554.png` | Dim, small sign | Misclassified. | Not detected. | Wrong-class detection changed to a miss. |

### Summary and interpretation

The selected comparisons suggest improved recognition of several speed-limit signs at 1280px, notably `00173`, `00185`, `00223`, `00241`, and `00372`. Several signs previously described as small or dim were also recovered, including `00375`, `00448`, `00470`, `00507`, and `00551`. Image `00470` is a useful presentation example because all three signs were detected and correctly classified at 1280px, whereas the 960px inspection recorded misses.

The improvement is not universal. Images `00003` and `00544` show classification regressions, `00423` shows a detection regression at the inspection threshold, and `00521` introduces a duplicate false positive. Confusion between similar classes persists, including overtaking variants, keep-left/keep-right signs, and some speed limits. The reported class imbalance in `00159` is a plausible contributing factor, not a demonstrated cause.

Several wrong-class detections at 960px became misses at 1280px. This changes the error type; it does not by itself establish improvement or greater safety. A missed sign can also be consequential, and the saved predictions do not establish whether the change was caused by lower confidence or prediction suppression. Both precision and recall must be considered. Higher confidence also does not guarantee correctness: `00173` was corrected with lower confidence, while `00343` retained a very confident error.

### Next experiments

1. Evaluate the frozen 1280px checkpoint on the complete clean validation split to establish the quantitative comparison baseline.
2. Test gentle gamma brightening (`gamma=0.8`) and CLAHE on LAB luminance (`clipLimit=2.0`, `tileGridSize=(8, 8)`) independently on the same validation images. Compare mAP50, mAP50–95, recall, and inference/preprocessing time using identical evaluation settings.
3. Inspect a small set of representative cases after each experiment: `00173` for speed-limit confusion, `00186` for small-sign detection, `00343` for dim lighting and confident errors, and `00470` for successful small-sign recognition. Record regressions as well as recoveries.
4. Select preprocessing settings using validation results, freeze them, and evaluate on the clean test split plus the relevant degraded suites. If time permits, test one SAHI configuration; otherwise present tiled inference as future work, with `00186` as a motivating example.
