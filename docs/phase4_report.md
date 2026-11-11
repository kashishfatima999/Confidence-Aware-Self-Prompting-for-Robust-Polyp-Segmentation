# F26-13 — Phase 4 report: Evaluation and analysis

**Project:** Confidence-Aware Self-Prompting for Robust Polyp Segmentation Under Image and Domain Shifts · Track A
**Inputs:** the Phase 3 per-image results (`runs/*/*/per_image.csv`, backed up in Drive `MyDrive/F26-13/runs_phase3/`). Nothing was re-run and no Phase 3 number changed.
**Tables and figures:** `docs/tables/phase4/`, `docs/figures/phase4_*.png`, rebuilt by `notebooks/06_phase4_analysis.ipynb` (CPU only) from `src/eval/compare.py`, `src/eval/error_analysis.py` and `src/eval/bias_check.py`; summary of the subgroup intervals: `docs/figures/phase4_bias_summary.png`. Decisions: DL-39 to DL-43.

Runs, as in Phase 3: **B** baseline (YOLO boxes ≥ 0.5 → SAM 2), **T** threshold only (boxes ≥ 0.25), **C** CASP v1 (our method), **C-all** (every box refined), **C-noiter** (no self-prompting step), **O** oracle (ground-truth boxes).

---

## 1. Comparative evaluation (Member 3)

All differences are paired: the same images are scored by two runs and we bootstrap the mean of the per-image differences (2,000 resamples, seed 42, `metrics.bootstrap_ci`). "Better / worse / tie" counts images whose Dice changed by more than 0.0001.

### 1.1 Kvasir-SEG test (200 images)

| Run | Dice [95% CI] | IoU | Pixel prec. | Pixel recall | Pixel acc. | Area err. (pp) | Missed | Box P / R / F1 |
|---|---|---|---|---|---|---|---|---|
| B | 0.831 [0.792, 0.871] | 0.780 | 0.873 | 0.810 | 0.960 | 3.62 | 18 | 0.877 / 0.865 / 0.871 |
| T | 0.893 [0.871, 0.913] | 0.832 | 0.929 | 0.890 | 0.966 | 2.93 | 1 | 0.807 / 0.947 / 0.871 |
| C | 0.891 [0.868, 0.912] | 0.831 | 0.926 | 0.892 | 0.963 | 3.24 | 1 | 0.802 / 0.942 / 0.867 |
| C-all | 0.885 [0.860, 0.907] | 0.824 | 0.916 | 0.891 | 0.958 | 3.50 | 1 | 0.802 / 0.942 / 0.867 |
| C-noiter | 0.892 [0.869, 0.913] | 0.832 | 0.927 | 0.893 | 0.964 | 3.18 | 1 | 0.807 / 0.947 / 0.871 |
| O | 0.952 [0.943, 0.958] | 0.912 | 0.979 | 0.931 | 0.985 | 1.16 | 0 | 1.000 / 1.000 / 1.000 |

### 1.2 CVC-ClinicDB (612 images, never used for training or tuning)

| Run | Dice [95% CI] | IoU | Pixel prec. | Pixel recall | Pixel acc. | Area err. (pp) | Missed | Box P / R / F1 |
|---|---|---|---|---|---|---|---|---|
| B | 0.677 [0.648, 0.710] | 0.628 | 0.712 | 0.671 | 0.962 | 3.60 | 144 | 0.890 / 0.686 / 0.774 |
| T | 0.759 [0.736, 0.785] | 0.698 | 0.800 | 0.764 | 0.966 | 3.09 | 75 | 0.820 / 0.780 / 0.799 |
| C | 0.757 [0.734, 0.783] | 0.695 | 0.795 | 0.765 | 0.965 | 3.14 | 75 | 0.815 / 0.776 / 0.795 |
| C-all | 0.757 [0.734, 0.784] | 0.696 | 0.792 | 0.768 | 0.965 | 3.19 | 75 | 0.808 / 0.769 / 0.788 |
| C-noiter | 0.757 [0.734, 0.783] | 0.695 | 0.794 | 0.765 | 0.965 | 3.13 | 75 | 0.813 / 0.774 / 0.793 |
| O | 0.945 [0.942, 0.948] | 0.899 | 0.966 | 0.928 | 0.990 | 0.65 | 0 | 1.000 / 1.000 / 1.000 |

Dice, IoU and their CIs are the Phase 3 values; the pixel metrics, area error and box F1 are new (definitions in §2 and DL-41).

### 1.3 Paired differences in Dice

| Comparison | Kvasir-SEG test | CVC-ClinicDB |
|---|---|---|
| C − B (our method vs baseline) | **+0.061** [+0.032, +0.090], 27 / 8 / 165 | **+0.080** [+0.060, +0.100], 100 / 28 / 484 |
| T − B (lower threshold alone) | +0.062 [+0.033, +0.092], 22 / 4 / 174 | +0.082 [+0.062, +0.102], 93 / 10 / 509 |
| C − T (refinement on top of T) | −0.002 [−0.006, +0.002], 13 / 14 / 173 | −0.002 [−0.004, +0.000], 39 / 60 / 513 |
| C-noiter − T | −0.001 [−0.005, +0.003], 13 / 10 / 177 | −0.002 [−0.004, −0.000], 35 / 38 / 539 |
| C-all − T | −0.008 [−0.021, +0.001], 26 / 54 / 120 | −0.002 [−0.005, +0.001], 103 / 122 / 387 |
| O − T (what a perfect detector would still add) | +0.059 [+0.039, +0.080], 145 / 39 / 16 | +0.186 [+0.160, +0.210], 466 / 121 / 25 |

### 1.4 What the comparison says

- **C is better than B on every metric that measures the polyp.** On ClinicDB, Dice, IoU, pixel precision, pixel recall and area error all improve and every 95% CI excludes zero (pixel recall +0.094 [+0.072, +0.115], area error −0.45 pp [−0.72, −0.19]). On Kvasir-SEG the same holds for Dice, IoU, pixel precision and pixel recall; the area-error and pixel-accuracy CIs include zero there.
- **The gain is recall.** Pixel recall rises by about twice as much as pixel precision on Kvasir-SEG (+0.082 vs +0.053), and box recall goes from 0.865 to 0.942. Box precision falls (0.877 → 0.802), so box F1 does not move on Kvasir-SEG (0.871 → 0.867) and rises only slightly on ClinicDB (0.774 → 0.795).
- **The checking steps add nothing measurable.** Every C variant is within 0.002 Dice of T except C-all on Kvasir-SEG (−0.008), and removing self-prompting (C-noiter) changes nothing measurable. On ClinicDB, C is slightly *less precise* than T (pixel precision −0.005 [−0.010, −0.001]), a small but real cost of re-prompting boxes that T would have used as they are.
- **The ranking holds under image shift.** C beats B in all nine corrupted settings, with CIs above zero in all nine (from +0.019 at noise s3 to +0.082 at blur s3), and box precision drops in eight of them (`docs/tables/phase4/metrics.csv`).
- **Pixel accuracy cannot separate the runs.** B, which misses 18 Kvasir images and 144 ClinicDB images completely, still scores 0.960 and 0.962, because the polyp is a small part of each image. This is why Dice and IoU are the headline metrics.

---

## 2. Evaluation metrics (Member 2)

### 2.1 Segmentation Metrics

Segmentation performance is evaluated using Dice coefficient, Intersection over Union (IoU), pixel precision, pixel recall, and pixel accuracy. These metrics are calculated for each image, with dataset-level mean values also reported.

- **Dice coefficient (F1-score):** Measures the overlap between the predicted polyp mask and the ground-truth mask. It is calculated as 2·TP / (2·TP + FP + FN). Higher values indicate better segmentation.

- **Intersection over Union (IoU):** Measures the ratio of the intersection to the union of the predicted and ground-truth masks: TP / (TP + FP + FN).

- **Pixel Precision:** Measures the proportion of predicted polyp pixels that are correctly classified: TP / (TP + FP).

- **Pixel Recall:** Measures the proportion of ground-truth polyp pixels correctly detected: TP / (TP + FN).

- **Pixel Accuracy:** Measures the proportion of all pixels classified correctly: (TP + TN) / (TP + TN + FP + FN).

Dice and IoU are the primary segmentation metrics because they focus on overlap between predicted and ground-truth polyp regions. Pixel accuracy is reported but should not be interpreted as a reliable indicator of segmentation quality. Endoscopic images contain large background regions, so a model may achieve high pixel accuracy while failing to detect a polyp.

An empty prediction against a non-empty ground-truth mask receives a Dice and IoU score of zero. Images where both masks are empty receive a score of one, since neither contains a polyp to detect.

### 2.2 Detection Metrics

Detection performance is evaluated separately using precision, recall, and F1-score. Predicted bounding boxes are matched to ground-truth boxes using a greedy one-to-one matching procedure. A match is accepted when box IoU is at least 0.5. Unmatched predictions count as false positives, while unmatched ground-truth boxes count as false negatives.

### 2.3 Mask-Area Error

Instead of using RMSE as a measure of segmentation quality, the evaluation reports absolute mask-area error in percentage points. This is the absolute difference between the percentage of image area occupied by the predicted polyp mask and the percentage occupied by the ground-truth mask.

This provides an interpretable measure of how much the predicted polyp area differs from the annotated area. However, it does not measure spatial overlap, so it is interpreted alongside Dice and IoU.

### 2.4 Per-Image and Pooled Evaluation

The evaluation reports both the mean of per-image Dice and IoU scores and pooled Dice and IoU. Per-image averages give each image equal weight, while pooled metrics calculate overlap after accumulating pixel counts across the dataset. Reporting both provides a view of typical image-level performance and overall pixel-level performance.

The evaluation also calculates 95% percentile-bootstrap confidence intervals for mean per-image Dice and IoU using 1,000 resamples and a fixed random seed of 42.

---

## 3. Error analysis (Member 3)

### 3.1 Failure taxonomy

Every image of every run gets exactly one label (DL-41), checked in this order: **missed** (no prediction), **false positive** (a prediction with Dice 0), **extra box** (at least one correct box and at least one false box), **good** (Dice ≥ 0.7), **under-segmentation** (Dice < 0.7, predicted area < true area), **over-segmentation** (Dice < 0.7, predicted area > true area). The rules use only columns already in `per_image.csv`, so the labels are reproducible and need no masks.

| Dataset | Run | Good | Extra box | Under-seg. | Over-seg. | False pos. | Missed |
|---|---|---|---|---|---|---|---|
| Kvasir-SEG test | B | 159 | 15 | 6 | 2 | 0 | 18 |
| | T | 160 | 29 | 7 | 3 | 0 | 1 |
| | C | 160 | 28 | 7 | 4 | 0 | 1 |
| | O | 199 | 0 | 1 | 0 | 0 | 0 |
| CVC-ClinicDB | B | 409 | 22 | 26 | 9 | 2 | 144 |
| | T | 426 | 59 | 34 | 16 | 2 | 75 |
| | C | 424 | 58 | 34 | 19 | 2 | 75 |
| | O | 610 | 0 | 0 | 2 | 0 | 0 |

Full table with C-all, C-noiter and the nine corrupted settings: `docs/tables/phase4/errors_counts.csv`; Figure: `docs/figures/phase4_error_taxonomy.png`.

**Our method changes the kind of error, more than the number of errors.** On Kvasir-SEG, C turns 17 of B's 18 misses into masks, but it nearly doubles the extra-box cases (15 → 28), so the count of images that are not "good" only falls from 41 to 40. The same trade appears on ClinicDB (misses 144 → 75, extra boxes 22 → 58). Extra-box images are still mostly usable, with mean Dice 0.81 (Kvasir-SEG) and 0.79 (ClinicDB) under C, against 0.94 for the "good" images, but each one contains a false alarm. With ground-truth boxes (O) almost every image is "good", so under- and over-segmentation by SAM 2 itself is rare: 1 of 200 and 2 of 612 images.

### 3.2 Where C beats B and where it loses

Figure `docs/figures/phase4_c_vs_b.png` plots every image's Dice under B against C.

- **Kvasir-SEG: 27 images better, 8 worse.** 17 of the 27 wins are images B missed completely (the vertical line at B = 0). The 8 losses are small next to the wins: together they cost 0.86 Dice points, the wins add 13.0. Five of the losses keep more false boxes than B. The two largest (0.97 → 0.66 and 0.97 → 0.69) add a second box of confidence 0.41–0.42 next to the correct one, and SAM 2 scores that wrong mask 0.95 and 0.92, almost as high as the real polyp (0.97). Two more losses are single boxes with confidence just above 0.5 (0.503 and 0.505): B sends them to SAM 2 unchanged, C treats them as uncertain and re-prompts, and the mask ends slightly worse (−0.04 and −0.02). The last changes by 0.002 (`docs/tables/phase4/errors_kvasir_c_loses.csv`).
- **ClinicDB: 100 better, 28 worse.** 67 of the wins are images B missed (C gives 69 of B's missed images a mask, mean Dice 0.70; two of those masks miss the polyp). Of the 28 losses, 13 end as extra-box images and 10 are still "good" but slightly worse.

### 3.3 The 75 ClinicDB images no run except the oracle segments

These are the same 75 images for T, C, C-all and C-noiter (`docs/tables/phase4/errors_clinicdb_never_found.csv`, Figure `docs/figures/phase4_clinicdb_misses.png`).

- **The detector is the cause, not SAM 2.** In 19 images YOLO proposes no box at all, even at confidence 0.05; in the other 56 every box is below `t_low` = 0.25 (the highest is 0.248). Given the ground-truth box, SAM 2 segments these images with mean Dice 0.92 (lowest 0.59).
- **They are not only tiny polyps.** Their median polyp area is 5.3% of the image against 7.0% for the images C segments, but the spread is large: missed polyps go up to 48% of the image, and the scatter plot shows big polyps at confidence 0 next to small polyps found at high confidence. Five of the 75 have more than one polyp.
- **A lower `t_low` is not a free fix.** 56 of the images sit between 0.05 and 0.25, but on the validation split lowering `t_low` from 0.25 to 0.15 added 7 false boxes on 100 images and lowered Dice (Phase 3, §4). Above 0.25 the picture is the opposite: the 69 ClinicDB images whose highest confidence lies between 0.25 and 0.5 are exactly the images T and C rescue over B.

### 3.4 What the failures have in common

1. **Almost every failure of C starts at the detector.** Misses are polyps YOLO never boxed or scored below 0.25; most of the extra boxes C adds are boxes YOLO scored between 0.25 and 0.5 on something that is not a polyp. SAM 2's own mistakes (under- or over-segmentation inside a box) are 11 of 200 Kvasir-SEG images and 53 of 612 ClinicDB images under C, and almost vanish with the oracle boxes (1 and 2).
2. **SAM 2's predicted IoU cannot tell a polyp from a wrong box.** The acceptance rule never rejected anything (Phase 3: every uncertain box scored at least 0.78), and in the worst Kvasir-SEG losses the wrong box scored 0.92–0.95 next to 0.97 for the polyp. A filter has to use information SAM 2's score does not have.
3. **The domain shift lowers confidence before it removes boxes.** The ClinicDB scatter and the corrupted settings show the same pattern: under shift, true polyps slide from above 0.5 into the 0.25–0.5 band, which is where keeping uncertain boxes pays off, until the detector proposes nothing at all (19 ClinicDB images; 165 of 200 images at the strongest noise level).

---

## 4. Bias and subgroup analysis (Member 2)

### 4.1 Analysis Method

The subgroup analysis evaluates whether segmentation performance varies across datasets and image characteristics. The factors examined are dataset, polyp size, brightness, contrast, image width, image height, polyp count, and corruption type and severity. Continuous characteristics are divided into low, medium, and high tertiles using the baseline dataset distributions. The same baseline-derived group boundaries are applied to the corresponding baseline, threshold-only, and CASP results.

For each subgroup, the analysis reports the number of images (n), mean per-image Dice for baseline (B), threshold-only (T), and CASP (C), miss rate, false-positive boxes per image, and the paired difference in Dice between CASP and baseline (C − B). The 95% confidence intervals are calculated using 2,000 paired bootstrap resamples (seed 42). Subgroup means are computed from the per-image values, so a few overall means differ from §1 in the third decimal (for example T on CVC-ClinicDB: 0.760 here, 0.759 in §1, which keeps the Phase 3 value). A confidence interval that includes zero does not provide clear evidence of an improvement in that subgroup. Corrupted-image evaluations are reported separately because threshold-only results were not generated for those runs.

### 4.2 Dataset

CASP improves mean Dice on both evaluation datasets. The improvement is larger on CVC-ClinicDB (+0.0802) than on Kvasir-SEG (+0.0607). Baseline performance is also lower on ClinicDB, with a higher miss rate of 23.5%, compared with 9.0% on Kvasir. These differences indicate that performance depends partly on the evaluation dataset and its image characteristics. The results should therefore not be interpreted as evidence that performance will transfer equally to every dataset.

| Dataset | n | Dice B/T/C | Miss rate B/T/C (%) | FP boxes/image B/T/C | C − B [95% CI] |
|---|---:|---|---|---|---|
| CVC-ClinicDB | 612 | 0.677 / 0.760 / 0.757 | 23.5 / 12.3 / 12.3 | 0.090 / 0.181 / 0.186 | +0.0802 [0.0604, 0.1001] |
| Kvasir-SEG | 200 | 0.831 / 0.893 / 0.891 | 9.0 / 0.5 / 0.5 | 0.125 / 0.235 / 0.240 | +0.0607 [0.0318, 0.0902] |

### 4.3 Polyp Size

CASP improves Dice across all polyp-size groups in both clean datasets. On ClinicDB, the largest gains occur in the low- and high-size groups, while the medium-size group has a smaller improvement. Kvasir shows a similar pattern: the high-size group has the largest gain, while the medium-size group's confidence interval includes zero. The results suggest that performance differs across polyp-size groups, although the subgroup analysis alone cannot establish that size is the cause of these differences.

| Dataset | Size group | n | Dice B/T/C | Miss rate B/T/C (%) | FP boxes/image B/T/C | C − B [95% CI] |
|---|---|---:|---|---|---|---|
| ClinicDB | Low | 204 | 0.614 / 0.718 / 0.714 | 29.9 / 15.7 / 15.7 | 0.059 / 0.118 / 0.123 | +0.1000 [0.0636, 0.1382] |
| ClinicDB | Medium | 204 | 0.782 / 0.825 / 0.823 | 14.2 / 8.8 / 8.8 | 0.074 / 0.127 / 0.132 | +0.0409 [0.0170, 0.0685] |
| ClinicDB | High | 204 | 0.635 / 0.735 / 0.735 | 26.5 / 12.3 / 12.3 | 0.137 / 0.299 / 0.304 | +0.0996 [0.0665, 0.1375] |
| Kvasir | Low | 67 | 0.811 / 0.872 / 0.872 | 10.4 / 1.5 / 1.5 | 0.045 / 0.090 / 0.090 | +0.0611 [0.0192, 0.1122] |
| Kvasir | Medium | 66 | 0.863 / 0.901 / 0.899 | 6.1 / 0.0 / 0.0 | 0.167 / 0.273 / 0.273 | +0.0366 [-0.0063, 0.0906] |
| Kvasir | High | 67 | 0.819 / 0.906 / 0.903 | 10.4 / 0.0 / 0.0 | 0.164 / 0.343 / 0.358 | +0.0842 [0.0279, 0.1505] |

### 4.4 Brightness

CASP improves Dice across all brightness groups in both clean datasets. On ClinicDB, the low- and medium-brightness groups show larger gains than the high-brightness group. On Kvasir, the largest gain occurs in the low-brightness group, while the high-brightness group's confidence interval includes zero. The different patterns across datasets suggest that brightness-related performance should be evaluated separately for each dataset rather than assuming one brightness level is consistently more difficult.

| Dataset | Brightness group | n | Dice B/T/C | Miss rate B/T/C (%) | FP boxes/image B/T/C | C − B [95% CI] |
|---|---|---:|---|---|---|---|
| ClinicDB | Low | 204 | 0.609 / 0.711 / 0.709 | 28.9 / 14.7 / 14.7 | 0.127 / 0.260 / 0.265 | +0.0996 [0.0622, 0.1391] |
| ClinicDB | Medium | 204 | 0.635 / 0.741 / 0.738 | 27.5 / 14.2 / 14.2 | 0.078 / 0.142 / 0.137 | +0.1028 [0.0687, 0.1381] |
| ClinicDB | High | 204 | 0.787 / 0.826 / 0.825 | 14.2 / 7.8 / 7.8 | 0.064 / 0.142 / 0.157 | +0.0381 [0.0151, 0.0643] |
| Kvasir | Low | 67 | 0.820 / 0.908 / 0.909 | 11.9 / 1.5 / 1.5 | 0.030 / 0.060 / 0.060 | +0.0897 [0.0283, 0.1574] |
| Kvasir | Medium | 66 | 0.835 / 0.895 / 0.894 | 9.1 / 0.0 / 0.0 | 0.091 / 0.167 / 0.167 | +0.0586 [0.0180, 0.1123] |
| Kvasir | High | 67 | 0.838 / 0.876 / 0.871 | 6.0 / 0.0 / 0.0 | 0.254 / 0.478 / 0.493 | +0.0338 [-0.0059, 0.0810] |

### 4.5 Contrast

CASP improves all three contrast groups on ClinicDB. The medium-contrast group has the largest observed gain. On Kvasir, the high-contrast group has the largest gain, while the medium-contrast group has the smallest gain and a confidence interval that includes zero. This difference between datasets indicates that contrast-related performance is not uniform. The relatively high false-positive box rates in some groups also show that reducing missed detections can come with additional false-positive predictions.

| Dataset | Contrast group | n | Dice B/T/C | Miss rate B/T/C (%) | FP boxes/image B/T/C | C − B [95% CI] |
|---|---|---:|---|---|---|---|
| ClinicDB | Low | 204 | 0.614 / 0.696 / 0.693 | 28.9 / 18.1 / 18.1 | 0.064 / 0.152 / 0.157 | +0.0786 [0.0459, 0.1165] |
| ClinicDB | Medium | 204 | 0.714 / 0.804 / 0.801 | 20.6 / 8.8 / 8.8 | 0.093 / 0.176 / 0.176 | +0.0871 [0.0540, 0.1227] |
| ClinicDB | High | 204 | 0.703 / 0.779 / 0.778 | 21.1 / 9.8 / 9.8 | 0.113 / 0.216 / 0.225 | +0.0747 [0.0426, 0.1103] |
| Kvasir | Low | 67 | 0.871 / 0.920 / 0.921 | 6.0 / 0.0 / 0.0 | 0.090 / 0.179 / 0.179 | +0.0507 [0.0103, 0.1034] |
| Kvasir | Medium | 66 | 0.858 / 0.884 / 0.882 | 4.5 / 0.0 / 0.0 | 0.212 / 0.364 / 0.364 | +0.0232 [-0.0143, 0.0723] |
| Kvasir | High | 67 | 0.764 / 0.875 / 0.871 | 16.4 / 1.5 / 1.5 | 0.075 / 0.164 / 0.179 | +0.1078 [0.0508, 0.1758] |

### 4.6 Image Resolution: Width

ClinicDB images have a fixed width, so they form a single group and cannot support a comparison across width levels. Kvasir images are divided into low-, medium-, and high-width groups. CASP improves Dice in all three groups, with the largest observed gain in the high-width group. These results describe the tested images and do not establish that image width itself causes the performance differences.

| Dataset | Width group | n | Dice B/T/C | Miss rate B/T/C (%) | FP boxes/image B/T/C | C − B [95% CI] |
|---|---|---:|---|---|---|---|
| ClinicDB | All | 612 | 0.677 / 0.760 / 0.757 | 23.5 / 12.3 / 12.3 | 0.090 / 0.181 / 0.186 | +0.0802 [0.0604, 0.1001] |
| Kvasir | Low | 67 | 0.830 / 0.902 / 0.897 | 9.0 / 0.0 / 0.0 | 0.090 / 0.134 / 0.149 | +0.0672 [0.0214, 0.1218] |
| Kvasir | Medium | 83 | 0.854 / 0.896 / 0.896 | 6.0 / 0.0 / 0.0 | 0.169 / 0.289 / 0.289 | +0.0417 [0.0029, 0.0890] |
| Kvasir | High | 50 | 0.793 / 0.876 / 0.877 | 14.0 / 2.0 / 2.0 | 0.100 / 0.280 / 0.280 | +0.0837 [0.0227, 0.1606] |

### 4.7 Image Resolution: Height

As with width, ClinicDB has a fixed image height and therefore only has an overall group. Kvasir results show positive CASP improvements across all three height groups. The largest gain is in the high-height group, followed by the medium-height group. The low-height group has a smaller improvement. These findings should be interpreted as subgroup associations rather than evidence that image height independently determines segmentation quality.

| Dataset | Height group | n | Dice B/T/C | Miss rate B/T/C (%) | FP boxes/image B/T/C | C − B [95% CI] |
|---|---|---:|---|---|---|---|
| ClinicDB | All | 612 | 0.677 / 0.760 / 0.757 | 23.5 / 12.3 / 12.3 | 0.090 / 0.181 / 0.186 | +0.0802 [0.0604, 0.1001] |
| Kvasir | Low | 72 | 0.813 / 0.854 / 0.851 | 8.3 / 1.4 / 1.4 | 0.208 / 0.319 / 0.333 | +0.0378 [0.0028, 0.0816] |
| Kvasir | Medium | 75 | 0.854 / 0.926 / 0.924 | 8.0 / 0.0 / 0.0 | 0.067 / 0.133 / 0.133 | +0.0699 [0.0169, 0.1256] |
| Kvasir | High | 53 | 0.821 / 0.900 / 0.900 | 11.3 / 0.0 / 0.0 | 0.094 / 0.264 / 0.264 | +0.0790 [0.0193, 0.1523] |

### 4.8 Polyp Count

CASP improves Dice for both single- and multiple-polyp images. On ClinicDB, multiple-polyp images have substantially lower Dice than single-polyp images for all three methods. This indicates that images containing multiple polyps are a more challenging subgroup in this dataset. However, the multiple-polyp group contains only 30 images on ClinicDB and six on Kvasir. The Kvasir confidence interval is particularly wide and includes zero, so its multiple-polyp result is inconclusive. Larger and more balanced samples are needed before drawing strong conclusions about this subgroup.

| Dataset | Polyp count | n | Dice B/T/C | Miss rate B/T/C (%) | FP boxes/image B/T/C | C − B [95% CI] |
|---|---|---:|---|---|---|---|
| ClinicDB | Single | 582 | 0.689 / 0.771 / 0.770 | 23.2 / 12.0 / 12.0 | 0.093 / 0.187 / 0.192 | +0.0811 [0.0615, 0.1016] |
| ClinicDB | Multiple | 30 | 0.452 / 0.527 / 0.513 | 30.0 / 16.7 / 16.7 | 0.033 / 0.067 / 0.067 | +0.0612 [0.0117, 0.1188] |
| Kvasir | Single | 194 | 0.831 / 0.894 / 0.893 | 9.3 / 0.5 / 0.5 | 0.129 / 0.242 / 0.247 | +0.0615 [0.0331, 0.0923] |
| Kvasir | Multiple | 6 | 0.816 / 0.854 / 0.854 | 0.0 / 0.0 / 0.0 | 0.000 / 0.000 / 0.000 | +0.0377 [0.0000, 0.1130] |

### 4.9 Corruption Type and Severity

CASP produces higher mean Dice than baseline across all nine tested corruption conditions. The size of the improvement varies by corruption type and severity. However, improvement over baseline does not mean that the model is robust under every condition. Severe noise (noise_s3) is a major failure case: baseline Dice is 0.0155 and CASP Dice is only 0.0346, while miss rates remain 98.0% and 96.0%, respectively. This demonstrates that CASP does not recover reliable segmentation under the most severe noise condition. Threshold-only results are unavailable for these corrupted runs because that experiment was not generated for the corrupted datasets.

| Corruption | n | Dice B/C | Miss rate B/C (%) | FP boxes/image B/C | C − B [95% CI] |
|---|---:|---|---|---|---|
| Blur, severity 1 | 200 | 0.842 / 0.870 | 7.5 / 2.5 | 0.115 / 0.225 | +0.0280 [0.0059, 0.0539] |
| Blur, severity 2 | 200 | 0.811 / 0.846 | 10.5 / 4.0 | 0.090 / 0.235 | +0.0353 [0.0103, 0.0631] |
| Blur, severity 3 | 200 | 0.724 / 0.806 | 18.5 / 5.5 | 0.105 / 0.270 | +0.0821 [0.0452, 0.1204] |
| Brightness/contrast, severity 1 | 200 | 0.837 / 0.883 | 8.5 / 1.5 | 0.100 / 0.230 | +0.0461 [0.0149, 0.0778] |
| Brightness/contrast, severity 2 | 200 | 0.834 / 0.885 | 8.0 / 1.5 | 0.110 / 0.220 | +0.0503 [0.0187, 0.0847] |
| Brightness/contrast, severity 3 | 200 | 0.799 / 0.838 | 12.5 / 6.0 | 0.085 / 0.205 | +0.0397 [0.0136, 0.0690] |
| Noise, severity 1 | 200 | 0.800 / 0.875 | 10.0 / 1.0 | 0.125 / 0.270 | +0.0749 [0.0436, 0.1087] |
| Noise, severity 2 | 200 | 0.687 / 0.761 | 20.0 / 11.5 | 0.155 / 0.300 | +0.0738 [0.0441, 0.1066] |
| Noise, severity 3 | 200 | 0.015 / 0.035 | 98.0 / 96.0 | 0.005 / 0.005 | +0.0192 [0.0047, 0.0388] |

### 4.10 Pipeline Biases and Limitations

The evaluation reveals several limitations that may affect how fairly the pipeline performs across different image groups. First, the two datasets have different baseline performance, which suggests that dataset-specific characteristics influence the results. Second, some subgroups are much smaller than others. In particular, multiple-polyp images are underrepresented, making conclusions about that group less reliable. Third, the fixed image dimensions in ClinicDB prevent meaningful within-dataset comparisons by width and height. Fourth, the corruption experiment shows that severe noise can cause near-total segmentation failure even when CASP improves the average Dice score. Finally, CASP often increases false-positive boxes per image compared with baseline. This is an important trade-off: lower miss rates and higher Dice do not necessarily mean fewer false detections.

The subgroup analysis is descriptive and does not isolate the independent effect of each image characteristic. Brightness, contrast, resolution, polyp size, and dataset may be related to one another. The tertile boundaries are also dataset-specific, so a low-brightness group in one dataset does not necessarily represent the same absolute brightness range in another. The results should therefore be interpreted as evidence of performance variation across the evaluated groups, not as proof that a particular characteristic causes the variation.

### 4.11 Recommendations for a Fairer and More Robust Pipeline

A fairer pipeline should be trained and evaluated using images from multiple datasets rather than relying heavily on one dataset's characteristics. Training data should include a more balanced representation of small, medium, and large polyps, single- and multiple-polyp images, and different brightness, contrast, and resolution conditions. Severe image corruptions should also be represented during training and testing so that robustness limitations can be identified before deployment.

The pipeline should be evaluated separately on each dataset and subgroup, with confidence intervals and sample counts reported alongside average performance. Thresholds should be selected using validation data that represents the intended deployment domain, rather than assuming that one threshold will work equally well across all datasets. Any dataset-specific threshold adjustment should be documented and evaluated on held-out data to avoid overfitting. Finally, false-positive detections and missed polyps should both be considered when selecting operating thresholds. These changes would help prevent improvements in average Dice from hiding poor performance on smaller or more difficult subgroups.

---

## 5. Key insights and next steps (Member 3)

1. **The method works for the reason we did not expect.** CASP v1 is better than the baseline on both test sets and all nine corruptions, with CIs above zero, but all of the gain comes from keeping boxes with confidence 0.25–0.5. The refinement steps (four prompts, self-prompting, acceptance) are neutral to slightly harmful. We report it as it is.
2. **It is a recall method with a precision cost.** It removes most missed polyps (18 → 1, 144 → 75) and in exchange keeps more false boxes. In screening, a missed polyp is usually the worse error, but the false alarms are real and box F1 shows the trade-off more honestly than Dice.
3. **The bottleneck is the detector under domain shift.** SAM 2 with correct boxes reaches about 0.95 Dice on both hospitals' images. Every remaining failure in §3 starts with a box that YOLO did not propose, scored too low, or proposed on the wrong structure.
4. **Pixel accuracy and RMSE are the wrong metrics here.** Accuracy stays at 0.96 for a pipeline that misses a quarter of ClinicDB; per-image Dice that counts misses as 0, plus box precision, is what separates the runs.

**Next (Phases 5–6, analysis and documentation only):** use the existing `detections.jsonl` logs to test, on CPU, whether a rule that combines YOLO confidence with agreement between the candidate masks could have removed the extra boxes without losing the rescued polyps (the DL-38 follow-up); write up the theoretical reason SAM 2's score fails as a filter; and record the deferred items (second training seed, per-dataset thresholds).

---

## 6. Challenges faced (Member 3)

- **New metrics without re-running anything.** Pixel precision, recall and accuracy need the overlap between the predicted and true masks, which `per_image.csv` does not store. Because Dice = 2·overlap / (predicted + true area), the overlap follows exactly from the Dice and the two area columns already saved, so all 8,472 image–run pairs were re-scored without masks or a GPU (DL-41).
- **Keeping Phase 3 numbers fixed.** Recomputing Dice from the rounded per-image values moves some means by 0.001, so the tables take Dice, IoU and their CIs from the Phase 3 results file and add only the new columns.
- **Paired intervals.** The Phase 3 paired CIs were first computed by hand once; Phase 4 computes all 162 paired comparisons with one script, so every interval in this report can be reproduced.
- **CI dependencies.** The analysis uses pandas, which CI did not install; it was added to `requirements.txt` (2.3.3, the version series Colab uses) and the 12 new tests run in CI with the rest.
