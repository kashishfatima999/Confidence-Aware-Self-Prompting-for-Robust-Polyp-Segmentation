# F26-13 — Phase 3 Report: Implementation & Execution

**Project:** Confidence-Aware Self-Prompting for Robust Polyp Segmentation Under Image and Domain Shifts
**Group:** F26-13 — Kashish Fatima (24L-2605), Khushbakht Sohail (24L-2604), Zahra Saeed (24L-2512). AI2002, BDS-5A, FAST-NUCES Lahore. Track A.
**State:** all runs done (2026-10-15/16, Colab T4, commit `2b9c761`); text in draft, sections owned as in `project-division/Project-Division-Phase-3.md`.

Every number below is read from `docs/tables/phase3/results.csv` (built by `src/eval/summarize.py` from the `summary.json` of each run; the local rebuild is byte-identical to the Colab one), from `docs/tables/phase3/casp_val_grid.csv`, from `docs/tables/phase3/yolo/summary.json`, or from the per-image files under `runs/` (git-ignored, backed up in Drive `MyDrive/F26-13/runs_phase3/`). Decisions: DL-30 to DL-38 in `docs/decision_log.md`. The executed notebook is `notebooks/04_phase3_colab.ipynb`.

## 0. Summary

- The YOLOv8-m → SAM 2 baseline was trained and reproduced on our split: Kvasir-SEG test pooled Dice **0.878** / IoU **0.782** (paper: 0.866 / 0.764); per image, with misses counted as zero, **0.831** / **0.780**.
- CASP v1 raises per-image Dice from **0.831 to 0.891** on Kvasir-SEG test and from **0.677 to 0.757** on CVC-ClinicDB (cross-dataset, never trained on), and is 0.02 to 0.08 above the baseline on all nine corrupted settings.
- The ablations show **where the gain comes from**: lowering the detector threshold from 0.5 to the chosen `t_low` = 0.25 (run **T**) gives the same gain (0.893 / 0.759). Prompt refinement, the self-prompting iteration and the acceptance rule are, in v1, neutral: the acceptance rule never rejected a single detection because SAM 2's predicted IoU was ≥ 0.78 for every uncertain box, true or false.
- The oracle (ground-truth boxes → SAM 2) reaches **0.952** / **0.945**. The whole remaining gap, including the domain-shift gap on ClinicDB, is the detector: SAM 2 segments ClinicDB polyps as well as Kvasir ones once it is given the right box.
- The price of the recall gain is false positives: box precision drops from 0.877 to 0.802 (Kvasir test) and from 0.890 to 0.815 (ClinicDB). Phase 4 starts from here (DL-38).

## 1. Setup (Member 1)

- Data: Kvasir-SEG 700 / 100 / 200 (DL-02), CVC-ClinicDB 612 images as external test only (DL-03), corrupted Kvasir test set, 9 settings × 200 images (DL-20). Details and every count: `docs/data_report.md`.
- Hardware and software: Google Colab, Tesla T4 (15 GB); ultralytics 8.4.174, SAM 2 at commit `2b90b9f`, torch 2.11.0+cu130, numpy 2.1.3, Pillow 11.3.0 (`requirements-colab.txt`); seed 42 everywhere. 65 unit tests pass on Colab and locally.
- Evaluation (DL-32): per-image Dice and IoU averaged over images, with a bootstrap 95% CI (1,000 resamples); dataset-pooled Dice and IoU (the paper's definition: all pixels of the dataset at once); images with no prediction count as empty (Dice 0, DL-18); box precision and recall at IoU ≥ 0.5 against boxes derived from the ground-truth masks. "Missed" below = images with an empty prediction.
- Execution record: all Colab sessions were run by Member 1 on 2026-10-15 (training, detection caches) and 2026-10-16 (validation grid, the six experiments, corrupted set), because the schedule slipped; the division of the *writing* stays as planned (`project-division/Project-Division-Phase-3.md` §7). Every run is reproducible from `runs/detections/*.json` + one config file + one command. Problems met on Colab are C7 and C8 in `docs/setup_guide.md`.

## 2. Detector (Member 2)

Recipe (DL-31, `configs/yolo_train.yaml`, exact arguments in `docs/tables/phase3/yolo/args.yaml`): YOLOv8-m from COCO weights, 50 epochs, 640 px, batch 16, seed 42, deterministic, Ultralytics default optimiser ("auto") and augmentation, best checkpoint by val mAP50-95. Training took 25 minutes on the T4.

**Table 2.1 — detector quality** (`docs/tables/phase3/yolo/summary.json`, Ultralytics `val()` at its default confidence)

| Split | mAP50 | mAP50-95 | Precision | Recall |
|---|---|---|---|---|
| val (100) | 0.953 | 0.750 | 0.960 | 0.869 |
| test (200) | 0.943 | 0.747 | 0.867 | 0.908 |

Figure 2.1 (`docs/figures/phase3_yolo_training_curves.png`): all three training losses fall monotonically; val mAP50 passes 0.9 by epoch 20 and plateaus at 0.95. The validation classification loss spikes at epoch 2 (27,400, a warm-up instability of the single-class head) and is stable from epoch 4 on; it has no effect on the final model.

**Figure 2.2 — recall vs confidence threshold on val** (`docs/figures/phase3_recall_vs_threshold.png`, from the detection cache). Polyp recall at IoU ≥ 0.5 is 0.888 at threshold 0.5, 0.935 at 0.25 and 0.953 at 0.05; the number of false-positive boxes on the 100 images goes from 5 to 12 to 55 over the same range. The curve is flat between 0.10 and 0.35 and falls steeply above 0.6. This is the headroom the method works with.

**Detection cache** (DL-33, `runs/detections/`, backed up in Drive): candidates at confidence ≥ 0.05 are 156 boxes on val (100 images), 310 on test (200) and 865 on ClinicDB (612). Images where no prompt refinement can help because the detector proposes **nothing** even at 0.05: 0 on Kvasir test, 19 on ClinicDB (3.1%). Images whose only candidates are below the chosen `t_low` = 0.25: 1 on Kvasir test, 56 on ClinicDB.

## 3. Baseline reproduction (Member 2)

B = YOLO boxes with confidence ≥ 0.5 → SAM 2 (`sam2_hiera_large.pt`, frozen, box prompt, `multimask_output=False`), union of the masks per image, no box = empty mask (`configs/experiments/baseline.yaml`).

**Table 3.1 — baseline against the paper**

| Dataset | Ours, pooled Dice / IoU | Ours, per-image Dice / IoU (misses = 0) | Missed images | Paper (Mansoori et al.) Dice / IoU |
|---|---|---|---|---|
| Kvasir-SEG test (200) | 0.878 / 0.782 | 0.831 / 0.780 | 18 / 200 | 0.866 / 0.764 (80/20 split, in-domain) |
| CVC-ClinicDB (612) | 0.755 / 0.606 | 0.677 / 0.628 | 144 / 612 | 0.951 / 0.909 (trained on ClinicDB, in-domain) |

The Kvasir numbers reproduce the paper within 1–2 points on its own metric. The ClinicDB numbers are not comparable to the paper's: the paper trains and tests on ClinicDB, we test a Kvasir-trained detector on it (domain shift, the point of the project), and the baseline misses 24% of ClinicDB images outright. Other differences: our split (700/100/200, seed 42) vs the paper's 80/20; 640 px vs the paper's stated 680 (DL-31); batch 16 on a T4 vs 64 on an A100.

**Oracle O** (ground-truth boxes → the same SAM 2): per-image Dice 0.952 on Kvasir test and 0.945 on ClinicDB, no misses. The gap O − B is 0.12 on Kvasir and 0.27 on ClinicDB: that is what the detector costs, and it is almost all of the domain-shift drop.

## 4. Method: Confidence-Aware Self-Prompting (Member 3)

### 4.1 The problem with the baseline

The baseline works in two steps. YOLO draws boxes around things it thinks are polyps, and gives each box a confidence score between 0 and 1. Then SAM 2 colours in the polyp inside each box. The baseline only keeps boxes with a confidence of **0.5 or more**.

That one cut-off causes trouble. If YOLO finds a real polyp but is only 30% sure, the box is thrown away and the image gets no mask at all. This happened to 18 of the 200 Kvasir test images and 144 of the 612 ClinicDB images (Table 3.1). Meanwhile, a box at 51% goes through without any check, even though YOLO was barely more sure about it.

### 4.2 Our idea: treat sure and unsure boxes differently

Instead of one cut-off, we use two: `t_low` and `t_high`. Every box falls into one of three groups (code: `refine_detection` in `src/pipeline/refine.py`, decision DL-34):

| YOLO confidence | Group | What we do |
|---|---|---|
| `t_high` or above | **Confident** | Send the box to SAM 2 as it is (same as the baseline). |
| between `t_low` and `t_high` | **Uncertain** | Keep the box, but double-check it (the four steps below). |
| below `t_low` | **Dropped** | Ignore it. |

For an **uncertain** box we do four things:

1. **Ask SAM 2 in four ways.** We give SAM 2 the box, the box grown by 10% on every side, the box grown by 20% on every side, and the box with a dot at its centre. The idea is that a slightly wrong box might still work if we try a few versions of it.
2. **Keep the best answer.** With every mask, SAM 2 also gives a score saying how good it thinks its mask is. We keep the mask with the highest score.
3. **Ask again with the mask (self-prompting).** We draw a tight box around the chosen mask, mark its centre, and give these back to SAM 2. If the new mask scores at least as well, we keep it.
4. **Final check.** We keep the mask only if SAM 2's score is at least `s_min` and the mask fills at least 10% of its box. Otherwise we throw the box away. This step was meant to catch wrong boxes.

At the end, all kept masks in an image are combined into one prediction. If we set `t_low` = `t_high` = 0.5 and switch off the final check, we get exactly the baseline. So any difference between the baseline (B) and our method (C) comes only from these new rules.

### 4.3 Choosing the three thresholds

We picked `t_low`, `t_high` and `s_min` using only the 100 **validation** images, never the test images. We tried every combination of `t_low` ∈ {0.15, 0.25, 0.35}, `t_high` ∈ {0.6, 0.7, 0.8} and `s_min` ∈ {0.6, 0.7, 0.8}, which is 27 runs, and kept the one with the highest average Dice. This was quick because YOLO's boxes were saved once and reused (DL-33).

**Table 4.1 — validation grid** (27 runs on `kvasir_val`, `docs/tables/phase3/casp_val_grid.csv`; baseline on the same 100 images: Dice 0.866, 4 missed, 5 false-positive boxes). Top six rows:

| t_low | t_high | s_min | mean Dice | mean IoU | missed | FP boxes | prompts / image |
|---|---|---|---|---|---|---|---|
| **0.25** | **0.6** | **0.6** | **0.8985** | 0.840 | 0 | 12 | 1.79 |
| 0.35 | 0.6 | 0.6 | 0.8975 | 0.838 | 0 | 11 | 1.74 |
| 0.25 | 0.7 | 0.6 | 0.8955 | 0.835 | 0 | 12 | 2.15 |
| 0.35 | 0.7 | 0.6 | 0.8944 | 0.834 | 0 | 11 | 2.10 |
| 0.25 | 0.8 | 0.6 | 0.8935 | 0.834 | 0 | 11 | 2.87 |
| 0.35 | 0.8 | 0.6 | 0.8924 | 0.832 | 0 | 10 | 2.82 |

What we learned from the grid:

- **`t_low` matters most.** At 0.15, too many wrong boxes get in (18–19 instead of 10–12), and 0.15 always does worse than 0.25 or 0.35 when the other two thresholds are kept the same. 0.25 and 0.35 are almost the same (at most 0.002 apart).
- **A lower `t_high` is always better.** A lower `t_high` means fewer boxes get double-checked. That was an early hint that the double-checking is not what helps.
- **`s_min` hardly matters.** 0.7 and 0.8 give the same result, and both lose one image compared to 0.6. The number of wrong boxes never changes with `s_min`, so the final check is not catching wrong boxes.

**Chosen:** `t_low` = 0.25, `t_high` = 0.6, `s_min` = 0.6. Validation Dice 0.8985, compared with 0.866 for the baseline. These values were fixed in the config files before we looked at any test result.

### 4.4 What happened on the test images

**How the test detections were handled** (from `per_image.csv` and `detections.jsonl` of run C):

| Dataset | candidates ≥ 0.05 | confident (≥ 0.6) | uncertain (0.25–0.6) | dropped (< 0.25) | rejected by the acceptance rule | kept |
|---|---|---|---|---|---|---|
| Kvasir test | 310 | 183 | 60 | 67 | **0** | 243 |
| ClinicDB | 865 | 435 | 180 | 250 | **0** | 615 |

- **The four prompts did get used.** For uncertain boxes, SAM 2 liked the plain box best in 40% (Kvasir) / 36% (ClinicDB) of cases, the box with a centre dot in 25% / 26%, and a bigger box in 22% / 21%. Self-prompting gave a better-scored mask in 43% / 37% of cases.
- **But the final check rejected nothing.** Look at the zeros in the "rejected" column. SAM 2's score for uncertain boxes was never below 0.784 (usually around 0.95 on Kvasir and 0.92 on ClinicDB), so a cut-off of 0.6 never removed anything, and neither did the 10% rule.

**Why?** SAM 2's score tells us how *neat* the mask is, not whether it is *a polyp*. If YOLO puts a box around a fold of tissue or a light reflection, SAM 2 still draws a clean shape inside it and gives itself a high score. So in this first version every box above 0.25 ends up in the output, and the method behaves almost exactly like "just lower the threshold to 0.25". Section 5 tests this directly.

## 5. Results (Member 1 assembles, all write)

**Table 5.1 — the six experiments on the two test sets** (`docs/tables/phase3/results.md`; Dice and IoU per image, misses = 0; "pooled" = the paper's definition)

| Run | Dataset | Dice [95% CI] | IoU | pooled Dice / IoU | Missed | Box prec. / recall | Prompts / img |
|---|---|---|---|---|---|---|---|
| B baseline | Kvasir test | 0.831 [0.792, 0.871] | 0.780 | 0.878 / 0.782 | 18 | 0.877 / 0.865 | 1.0 |
| T threshold-only | Kvasir test | 0.893 [0.871, 0.913] | 0.832 | 0.901 / 0.820 | 1 | 0.807 / 0.947 | 1.2 |
| C CASP | Kvasir test | 0.891 [0.868, 0.912] | 0.831 | 0.894 / 0.808 | 1 | 0.802 / 0.942 | 2.4 |
| C-all (no gating) | Kvasir test | 0.885 [0.860, 0.907] | 0.824 | 0.879 / 0.784 | 1 | 0.802 / 0.942 | 6.1 |
| C-noiter | Kvasir test | 0.892 [0.869, 0.913] | 0.832 | 0.895 / 0.811 | 1 | 0.807 / 0.947 | 2.1 |
| O oracle | Kvasir test | 0.952 [0.943, 0.958] | 0.912 | 0.951 / 0.906 | 0 | 1.000 / 1.000 | 1.0 |
| B baseline | ClinicDB | 0.677 [0.648, 0.710] | 0.628 | 0.755 / 0.606 | 144 | 0.890 / 0.686 | 0.8 |
| T threshold-only | ClinicDB | 0.759 [0.736, 0.785] | 0.698 | 0.799 / 0.665 | 75 | 0.820 / 0.780 | 1.0 |
| C CASP | ClinicDB | 0.757 [0.734, 0.783] | 0.695 | 0.795 / 0.660 | 75 | 0.815 / 0.775 | 2.2 |
| C-all (no gating) | ClinicDB | 0.757 [0.734, 0.784] | 0.696 | 0.794 / 0.658 | 75 | 0.808 / 0.769 | 5.0 |
| C-noiter | ClinicDB | 0.757 [0.734, 0.783] | 0.695 | 0.795 / 0.660 | 75 | 0.813 / 0.774 | 1.9 |
| O oracle | ClinicDB | 0.945 [0.942, 0.948] | 0.899 | 0.945 / 0.896 | 0 | 1.000 / 1.000 | 1.1 |

**Paired differences** (same images, mean of per-image Dice differences, bootstrap 95% CI, 2,000 resamples; wins / losses / ties at 10⁻⁴):

| Comparison | Kvasir test (200) | ClinicDB (612) |
|---|---|---|
| C − B (the hypothesis) | **+0.061** [+0.032, +0.090], 27 / 8 / 165 | **+0.080** [+0.062, +0.101], 100 / 28 / 484 |
| T − B (recall alone) | +0.063 [+0.033, +0.097], 22 / 4 / 174 | +0.082 [+0.063, +0.103], 93 / 10 / 509 |
| C − T (refinement + acceptance on top of recall) | −0.002 [−0.006, +0.003], 13 / 14 / 173 | −0.002 [−0.004, −0.000], 39 / 60 / 513 |
| C − C-noiter (self-prompting iteration) | −0.001 [−0.002, +0.000] | −0.000 [−0.001, +0.001] |
| C-all − C (gating) | −0.006 [−0.017, +0.001], 16 / 40 / 144 | +0.000 [−0.002, +0.002] |

**Reading the ablations (Member 3).** An ablation is a run where we remove or change one part of our method to see how much that part helps. Here is what each comparison tells us, in plain words.

1. **T vs B: the gain comes from keeping more boxes.** T is the baseline with the cut-off lowered from 0.5 to 0.25, and nothing else. It improves Dice by +0.063 on Kvasir and +0.082 on ClinicDB, which is the whole improvement our method gets. The extra boxes rescue 17 of the 18 Kvasir images the baseline missed (average Dice 0.73 on them) and 69 of the 144 ClinicDB ones (average 0.70). No image that the baseline got right was lost. **In short: our method helps because it stops throwing away polyps that YOLO was unsure about.**

2. **C vs T: the double-checking does not help (yet).** Our full method is 0.002 Dice *below* T on both datasets. It gets worse on slightly more images than it gets better (13 better / 14 worse on Kvasir, 39 / 60 on ClinicDB). On the images that actually had an uncertain box (45 on Kvasir, 154 on ClinicDB), it is 0.008 below T. Giving SAM 2 the plain box already works as well as trying four prompts and asking again, and the final check never removes anything (§4.4).

3. **C vs C-noiter: self-prompting makes no real difference.** Removing the "ask again" step changes Dice by less than 0.001. It does change the mask in 37–43% of uncertain cases, but those masks were already good.

4. **C-all vs C: double-checking sure boxes slightly hurts.** C-all double-checks *every* box, not just the uncertain ones. On Kvasir it is 0.006 worse (40 images worse, 16 better) and uses 6.1 prompts per image instead of 2.4. On ClinicDB there is no difference. A box YOLO is sure about is already a good prompt, and trying alternatives more often swaps it for a worse one than a better one. So only checking the uncertain boxes is the right choice, even if it matters little.

5. **C vs O: what is still missing is the detector's fault.** O (oracle) gives SAM 2 the perfect boxes from the ground truth. It is 0.06 Dice above our method on Kvasir and 0.19 on ClinicDB. With a perfect box, SAM 2 scores about 0.95 on *both* datasets, so SAM 2 is not the weak link. YOLO is. On ClinicDB, 75 images still get no prediction: in 19 of them YOLO finds nothing at all, and in the other 56 it only finds boxes below 0.25. Fixing these needs a better detector, or a lower `t_low` plus a filter that can actually remove wrong boxes.

6. **The price: more wrong boxes.** Lowering the cut-off lets in wrong boxes too. Wrong boxes kept go from 25 to 48 on Kvasir and from 55 to 114 on ClinicDB. Box precision (the share of kept boxes that are real polyps) falls from 0.877 to 0.802 and from 0.890 to 0.815. Dice barely shows this, because a small extra blob next to a correct mask costs little. For a doctor, though, a false alarm matters, and this is exactly what the final check was supposed to prevent. The worst examples in Figure 5.1 are all of this kind.

**Bottom line.** Our method beats the baseline, but because of its lower threshold, not its double-checking. The next step (Phase 4, DL-38) is to build a check that can actually tell a polyp from a wrong box, so that we keep the rescued polyps without the extra false alarms.

**Polyp size** (per-image Dice by terciles of ground-truth area, from `per_image.csv`; the Phase 4 bias check starts here):

| Dataset | Tercile (GT area) | B | C | O | B missed → C missed |
|---|---|---|---|---|---|
| Kvasir test | small (< 9.1%) | 0.811 | 0.872 | 0.951 | 7 → 1 |
| Kvasir test | medium (9.1–18.5%) | 0.863 | 0.899 | 0.958 | 4 → 0 |
| Kvasir test | large (> 18.5%) | 0.819 | 0.903 | 0.948 | 7 → 0 |
| ClinicDB | small (< 4.4%) | 0.614 | 0.714 | 0.939 | 61 → 32 |
| ClinicDB | medium (4.4–10.2%) | 0.782 | 0.823 | 0.952 | 29 → 18 |
| ClinicDB | large (> 10.2%) | 0.635 | 0.735 | 0.945 | 54 → 25 |

Small ClinicDB polyps are the hardest case for the detector (30% of them missed by the baseline) and SAM 2 handles them as well as the others once prompted; the large-polyp ClinicDB misses are mostly frames where the polyp fills much of the 384 × 288 image and the Kvasir-trained detector does not fire.

**Table 5.2 — image shift: B and C on the nine corrupted settings** (Kvasir test, 200 images each; `docs/figures/phase3_corrupted_dice.png`)

| Setting | B Dice | B missed | C Dice | C missed | C − B | Images with no candidate at 0.05 |
|---|---|---|---|---|---|---|
| clean | 0.831 | 18 | 0.891 | 1 | +0.061 | 0 |
| blur s1 | 0.842 | 15 | 0.870 | 5 | +0.028 | 0 |
| blur s2 | 0.811 | 21 | 0.846 | 8 | +0.035 | 0 |
| blur s3 | 0.724 | 37 | 0.806 | 11 | +0.082 | 1 |
| brightness/contrast s1 | 0.837 | 17 | 0.883 | 3 | +0.046 | 0 |
| brightness/contrast s2 | 0.834 | 16 | 0.885 | 3 | +0.050 | 0 |
| brightness/contrast s3 | 0.799 | 25 | 0.838 | 12 | +0.040 | 6 |
| noise s1 | 0.800 | 20 | 0.875 | 2 | +0.075 | 0 |
| noise s2 | 0.687 | 40 | 0.761 | 23 | +0.074 | 8 |
| noise s3 | 0.016 | 196 | 0.035 | 192 | +0.019 | 165 |

CASP is above the baseline in every setting, and the gain grows with the severity of the shift up to the point where the detector stops proposing anything (noise s3: 165 of 200 images have no candidate at all, both pipelines collapse). The robustness claim of the title holds in the same form as the clean result: under shift, the detector's confidence drops before its boxes disappear, and a lower threshold recovers the polyps. Because the chosen thresholds were tuned on clean validation images, this is a genuine out-of-distribution test of the setting.

**Figure 5.1** (`docs/figures/phase3_examples_kvasir_test.png`): the five Kvasir test images where C gains most over B and the five where it loses most, each as ground truth / B / C overlay with Dice. All five best cases are recovered misses (B: no detection, C: Dice 0.96–0.99). All five worst cases are a second, low-confidence box that C admits and B did not (Dice 0.97 → 0.66 in the worst): the false-positive cost of item 6 made visible, and the case the acceptance rule should have caught.

## 6. Challenges and limitations (all)

- **Colab** (`docs/setup_guide.md` C7, C8): the free tier dropped sessions during the silent 50-minute grid cell and two hours of runs were lost; the driver now saves every run to Drive as it finishes and restores it on re-run, so a disconnect costs one run. The first notebook version also hid all output.
- **Thresholds on 100 images.** The grid differences are ≤ 0.006 Dice between neighbouring settings on 100 validation images; `t_high` = 0.6 and `s_min` = 0.6 both sit on the edge of the grid. The chosen setting generalised to the test sets and to all nine corruptions, but a wider grid is cheap and due.
- **The acceptance rule is inert.** SAM 2's predicted IoU is high for false positives too (it rates its own mask, not whether the region is a polyp). `s_min` cannot do the job it was designed for; see DL-38.
- **Single seed, single detector.** One YOLOv8-m training run; no variance estimate across seeds. The bootstrap CIs cover image sampling only.
- **Per-image Dice with misses = 0** rewards recall and under-weights false positives; the pooled numbers and the box precision column are reported for that reason.
- **Execution by one member.** Sessions were run by Member 1 (record in the division file §7); Members 2 and 3 own the analysis and text of §2–4 and the Phase 4 follow-ups.

## 7. Next (Phase 4)

1. **A false-positive filter that works** (DL-38): candidates are the YOLO confidence × SAM score product, the agreement between the four candidate masks (a true polyp gives the same mask whatever the prompt; a spurious box does not), and mask-area / box-fill consistency. Evaluate on `kvasir_val` by box precision at fixed recall, then once on the test sets.
2. **Wider validation grid**: `t_low` ∈ {0.15 … 0.45}, `t_high` ∈ {0.4, 0.5, 0.6}, the new filter's threshold; still val only, cheap runs from the cache.
3. **Error and bias analysis** from the per-image CSVs: Dice vs polyp size, brightness, contrast and dataset; the 75 ClinicDB images with no prediction; the 10 Kvasir images where C loses.
4. **Second training seed** if GPU time allows, to put a variance on Table 2.1 and Table 5.1.
5. Long GPU jobs move to a workspace that runs without the browser (Kaggle "Save and Run All"), see C8.
6. **Evidence still to add:** an example figure for CVC-ClinicDB like Figure 5.1 (Member 2; the masks are saved, the images were not local when Figure 5.1 was made), and the epoch-by-epoch console log of the next training run (Member 2; the Phase 3 run's log was not saved, its per-epoch metrics are in `docs/tables/phase3/yolo/results.csv`). Owners and dates for every open item: `project-division/Project-Division-Phase-3.md` §7, "Missing or open items".
