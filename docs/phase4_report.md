# F26-13 — Phase 4 report: Evaluation and analysis

**Project:** Confidence-Aware Self-Prompting for Robust Polyp Segmentation Under Image and Domain Shifts · Track A
**Inputs:** the Phase 3 per-image results (`runs/*/*/per_image.csv`, backed up in Drive `MyDrive/F26-13/runs_phase3/`). Nothing was re-run and no Phase 3 number changed.
**Tables and figures:** `docs/tables/phase4/`, `docs/figures/phase4_*.png`, rebuilt by `notebooks/06_phase4_analysis.ipynb` (CPU only) from `src/eval/compare.py` and `src/eval/error_analysis.py`. Decisions: DL-39, DL-40, DL-41.

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

## 2. Metrics (Member 2)

*To be written by Khushbakht Sohail (task 2.4).*

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

## 4. Bias check (Member 2)

*To be written by Khushbakht Sohail (task 2.4), from `src/eval/bias_check.py`.*

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
