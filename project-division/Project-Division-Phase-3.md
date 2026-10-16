# F26-13 — Project Division for Phase 3 (Implementation & Execution)

**Project:** Confidence-Aware Self-Prompting for Robust Polyp Segmentation Under Image and Domain Shifts
**Track:** A (Research & Development)
**Phase 3 in the guidelines:** "Choose appropriate algorithms based on the problem. Use standard frameworks such as PyTorch, TensorFlow, or Scikit-Learn."
**Rubric weight:** Technical Implementation = 30% (Track A)
**Time:** one week, Monday to Sunday. Decisions for this phase are DL-30 to DL-37 in `docs/decision_log.md`.

| Member | Name | Owns in Phase 3 | Share |
|---|---|---|---|
| Member 1 | Kashish Fatima (24L-2605), lead | Evaluation harness, experiment configs, results tables, report assembly, reviews | ~25% (lighter after Phase 2) |
| Member 2 | Khushbakht Sohail (24L-2604) | The detector: YOLO training, detection cache, baseline / threshold / oracle runs | ~35% |
| Member 3 | Zahra Saeed (24L-2512) | The segmenter and the method: SAM 2 wrapper, CASP refinement, ablations, validation grid | ~40% |

---

## 1. What Phase 3 must produce

1. A YOLOv8-m detector trained on **our** train split with a written, fixed recipe (DL-31), with val/test mAP recorded.
2. The baseline YOLO → SAM 2 pipeline (B) reproduced on our split, evaluated the honest way (misses count, DL-18).
3. Our method, Confidence-Aware Self-Prompting v1 (C), with its three thresholds chosen on the **validation split only** (DL-34).
4. The four control experiments (T, C-all, C-noiter, O) that show where the gain comes from (DL-35).
5. Per-image results for every run on Kvasir-SEG test and CVC-ClinicDB (corrupted set if time), so Phase 4 can do error analysis and the bias check from files.
6. A Phase 3 report (`docs/phase3_report.md` → `docs/submissions/phase3/F26-13.pdf`) and a Colab notebook that reproduces everything.

Phase 3 does **not** include the full error/ablation analysis, the bias check, or the final LaTeX report. Those are Phases 4–6.

---

## 2. The method in one paragraph (DL-34)

The detector runs once per dataset at a very low threshold and every candidate box is cached (DL-33). Each detection is gated by its YOLO confidence `c`: `c ≥ t_high` → the box goes to SAM 2 unchanged (as the baseline); `t_low ≤ c < t_high` → *uncertain*: four candidate prompts are tried (the box, the box enlarged by 10% and 20%, the box plus a positive point at its centre), SAM 2's own predicted IoU picks the best, one self-prompting iteration re-prompts SAM 2 with the tight box and centroid of that mask, and the result is accepted only if SAM 2's predicted IoU ≥ `s_min` and the mask fills ≥ 10% of its box; `c < t_low` → dropped. The baseline is the special case `t_low = t_high = 0.5` with no acceptance rule. Code: `src/pipeline/refine.py`.

---

## 3. Code and files

| Path | What | Written by | Status on Monday |
|---|---|---|---|
| `src/eval/metrics.py`, `tests/test_metrics.py` | Dice/IoU per image + pooled, box matching, bootstrap CI (DL-32) | Member 1 | done, 11 tests |
| `src/pipeline/refine.py`, `tests/test_refine.py` | CASP geometry and logic with a mocked SAM 2 (DL-34) | Member 1 (skeleton), Member 3 (owner) | done, 14 tests |
| `src/pipeline/run_experiment.py`, `tests/test_run_experiment.py` | config + dataset + cache → per_image.csv, detections.jsonl, summary.json | Member 1 (skeleton), Member 3 (owner) | done, 6 tests |
| `src/pipeline/sam2_prompter.py` | SAM 2 wrapper: embed once, prompt many, returns mask + predicted IoU | Member 1 (skeleton), Member 3 (owner) | untested on GPU |
| `src/pipeline/detect.py` | YOLO → `runs/detections/<dataset>.json` (DL-33) | Member 1 (skeleton), Member 2 (owner) | untested on GPU |
| `src/train_yolo.py`, `configs/yolo_train.yaml` | training recipe and script (DL-31) | Member 1 (skeleton), Member 2 (owner) | untested on GPU |
| `configs/experiments/*.yaml` | B, T, C, C-all, C-noiter, O (DL-35) | Member 1 | done |
| `src/eval/summarize.py` | all `summary.json` → `docs/tables/phase3/results.{csv,md}` | Member 1 | done |
| `notebooks/04_phase3_colab.ipynb` | Colab driver: install → data → train → detect → experiments → summarize | Member 1 (skeleton), Member 3 (owner) | to run |
| `docs/phase3_report.md` | the Phase 3 report, written from the tables | all (sections below) | skeleton |

"Owner" means: runs it on Colab, fixes what breaks, extends it, and can explain every line of it at the evaluation.

---

## 4. Task division

### Member 1 — Kashish Fatima: harness, configs, tables, report

| # | Task | Output |
|---|---|---|
| 1.1 | Housekeeping: merge the last Phase 2 commit, branch `kashish-phase3` from `main`, this file, DL-30..37. | Branch + docs |
| 1.2 | Metrics module and tests; experiment configs; `summarize.py`. | `src/eval/`, `configs/experiments/` |
| 1.3 | Review every PR from Members 2 and 3 (CI green, decision-log row present, numbers come from files). | Reviews |
| 1.4 | Build the results tables with `summarize.py`; 5 best / 5 worst images per run for the report. | `docs/tables/phase3/`, figures |
| 1.5 | Assemble `docs/phase3_report.md` and the submission package (`docs/submissions/phase3/`). | PDF + notebook |

### Member 2 — Khushbakht Sohail: the detector

| # | Task | Output |
|---|---|---|
| 2.1 | Train YOLOv8-m with `src/train_yolo.py` (recipe in `configs/yolo_train.yaml`, do not change it without a DL entry). Save `best.pt` to Drive `MyDrive/F26-13/weights/`. | `runs/yolo/.../summary.json`, curves |
| 2.2 | Record val and test mAP50 / mAP50-95 / precision / recall; plot recall vs confidence threshold on val (from the cache). | Training section of the report |
| 2.3 | Build the detection cache with `detect.py` for `kvasir_val`, `kvasir_test`, `clinicdb` (then the 9 corrupted settings if time). Save the JSON files to Drive. | `runs/detections/*.json` |
| 2.4 | Run **B** (baseline), **T** (threshold-only, with the `t_low` Member 3 chooses) and **O** (oracle) on `kvasir_test` and `clinicdb`. | `runs/{baseline,threshold_only,oracle}/` |
| 2.5 | Write the "Detector" and "Baseline reproduction" sections of the report; compare B with the paper's 0.764 / 0.866 and explain the difference (DL-18). | Report sections |

### Member 3 — Zahra Saeed: the segmenter and the method

| # | Task | Output |
|---|---|---|
| 3.1 | Get `sam2_prompter.py` running on Colab; run `run_experiment.py --config configs/experiments/baseline.yaml --dataset kvasir_val --limit 5` end to end. | Working pipeline |
| 3.2 | Validation grid for CASP on `kvasir_val` only: `t_low ∈ {0.15, 0.25, 0.35}`, `t_high ∈ {0.6, 0.7, 0.8}`, `s_min ∈ {0.6, 0.7, 0.8}` (27 runs, cheap from the cache). Pick by mean Dice; write the chosen values into `casp.yaml`, `casp_all.yaml`, `casp_noiter.yaml`, `threshold_only.yaml` and DL-34. | Grid table + DL-34 |
| 3.3 | Run **C** on `kvasir_test` and `clinicdb` (then `kvasir_corrupted/*` if time, with B). | `runs/casp/` |
| 3.4 | Run the ablations **C-all** and **C-noiter** on `kvasir_test` and `clinicdb`. | `runs/casp_all/`, `runs/casp_noiter/` |
| 3.5 | Keep the Colab notebook runnable top to bottom; `--save-masks` for C and B on `kvasir_test` so the report can show examples. | Notebook + masks on Drive |
| 3.6 | Write the "Method" and "Ablations" sections of the report. | Report sections |

### Dependencies

- Everything needs Member 2's detection cache (2.3) → target **Tuesday evening**. Until then Member 3 works with `--limit 5` on a cache from the baseline's shipped weights (`external/YOLO_SAM2/YOLO_Checkpoints/Kvasir_yolov8m.pt`, smoke-test use only, DL-17).
- Member 2's T run (2.4) needs Member 3's chosen `t_low` (3.2) → target **Wednesday evening**.
- Member 1's tables (1.4) need all `summary.json` files → **Saturday**.

---

## 5. Day-by-day

| Day | Member 1 | Member 2 | Member 3 |
|---|---|---|---|
| Mon | 1.1, 1.2 (done before the week starts) | 2.1 start training | 3.1 |
| Tue | 1.3 | 2.1 done, 2.2, 2.3 | 3.1 done, notebook runs B on 5 images |
| Wed | 1.3 | 2.4: B and O | 3.2 grid → thresholds → DL-34 |
| Thu | 1.3, report skeleton | 2.4: T | 3.3: C on test + ClinicDB |
| Fri | 1.4 first tables | 2.3 corrupted cache (if time) | 3.4: C-all, C-noiter |
| Sat | 1.4, 1.5 text | 2.5 | 3.6; 3.3 corrupted (if time) |
| Sun | 1.5 PDF + notebook, PR, submission | review | review |

---

## 6. Rules for this phase

**Do**
- Choose every threshold on `kvasir_val`; touch `kvasir_test` and `clinicdb` once per experiment, at the end.
- One config file per experiment; one command per run; the output folder name is the experiment name.
- Save `runs/detections/*.json`, `runs/yolo/.../best.pt` and every `summary.json` to Drive `MyDrive/F26-13/`; commit only `docs/tables/phase3/`, small figures and the report.
- Add a `docs/decision_log.md` entry (or change-log row) in the same PR as any parameter change.

**Don't**
- Don't change `configs/yolo_train.yaml` after training has started; a second recipe is a second experiment with its own name.
- Don't use the baseline's shipped weights for any reported number (DL-17).
- Don't tune anything on the test split, and don't look at test results before the grid is finished.
- Don't commit `runs/`, weights or masks.

---

## 7. Execution record (filled 2026-10-17)

What actually happened, kept next to the plan so the evaluation can compare the two. The plan above is unchanged.

| Task | Planned | Done by | When | Evidence |
|---|---|---|---|---|
| 1.1, 1.2 harness, configs, DL-30..37, skeletons | M1 | M1 | 2026-10-11 | PR #8 |
| 2.1 YOLO training | M2 | **M1** | 2026-10-15 | `docs/tables/phase3/yolo/`, `docs/figures/phase3_yolo_training_curves.png` |
| 2.2 recall vs threshold | M2 | **M1** | 2026-10-16 | `docs/figures/phase3_recall_vs_threshold.png` |
| 2.3 detection caches (val, test, ClinicDB, 9 corrupted) | M2 | **M1** | 2026-10-15/16 | Drive `detections/`, `runs/detections/` |
| 2.4 runs B, T, O | M2 | **M1** | 2026-10-16 | `docs/tables/phase3/results.csv` |
| 3.1 pipeline on Colab | M3 | **M1** | 2026-10-16 | executed notebook |
| 3.2 validation grid, thresholds, DL-34 | M3 | **M1** | 2026-10-16 | `casp_val_grid.csv`, DL-34, PR #9 |
| 3.3, 3.4 runs C, C-all, C-noiter; B + C on corrupted | M3 | **M1** | 2026-10-16 | `results.csv` rows |
| 3.5 notebook runnable, masks saved | M3 | **M1** | 2026-10-16 | `notebooks/04_phase3_colab.ipynb`, Drive `runs_phase3/` |
| 1.4 tables and example figures | M1 | M1 | 2026-10-17 | `docs/tables/phase3/`, `phase3_examples_kvasir_test.png`, `phase3_corrupted_dice.png` |
| 1.5 report draft | M1 | M1 | 2026-10-17 | `docs/phase3_report.md` |
| 2.5 Detector + Baseline sections (text, analysis) | M2 | open | — | `docs/phase3_report.md` §2–3 |
| 3.6 Method + Ablations sections (text, analysis) | M3 | open | — | `docs/phase3_report.md` §4, §5 ablations |
| Submission package (`docs/submissions/phase3/`) | M1 | open | — | — |

Why the Colab work was run by Member 1: the schedule slipped in the first days of the week and the cached-detection design made every run a single command, so the lead ran the sessions on 15–16 October to have the numbers in time; two Colab problems (C7, C8 in `docs/setup_guide.md`) cost most of a day. Members 2 and 3 keep ownership of their code files and write their report sections from the committed tables and the per-image files; the Phase 4 items in DL-38 are theirs to run.

### Missing or open items (logged 2026-10-17, so nothing is forgotten)

| # | Item | Why it is missing | Owner | Due |
|---|---|---|---|---|
| O1 | Example figure for CVC-ClinicDB (5 best / 5 worst images, GT vs B vs C), like `docs/figures/phase3_examples_kvasir_test.png`. | The ClinicDB images are not on the lead's laptop; the B and C masks are in `runs/` and Drive `runs_phase3/`. Needs the ClinicDB raw data prepared locally (`src/data/prepare_clinicdb.py`) or a Colab cell. | Member 2 (ClinicDB pipeline owner) | Phase 4, with the error analysis |
| O2 | Console log of the YOLO training run (epoch-by-epoch output). | Training ran on 2026-10-15 in a Colab session whose notebook was not downloaded; the facts are in `docs/tables/phase3/yolo/results.csv` (per-epoch metrics), `args.yaml` and `summary.json`, so no number is missing. Not recoverable. | Member 2: save the full log when the second seed is trained (Phase 4 item 4) | Phase 4 |
| O3 | Report sections 2–3 (Detector, Baseline) in Member 2's words and sections 4–5 ablation text in Member 3's words; the draft is Member 1's. | Written by the lead from the tables so the numbers were in on time. | Member 2 (§2–3), Member 3 (§4, §5 ablations) | before the Phase 3 PDF is built |
| O4 | `docs/submissions/phase3/F26-13.pdf` and `F26-13.ipynb`; `docs/requirements/phase3_requirements.md` checklist. | Needs the TA's Phase 3 submission text (page limit, file names). | Member 1 | Phase 3 deadline |
| O5 | Kaggle (or other background-execution) version of the Colab driver for long runs, see setup guide C8. | Free Colab dropped the session repeatedly on 2026-10-16. | Member 3 (notebook owner), Member 1 helps | before the first Phase 4 GPU job |
| O6 | Second training seed; variance on Tables 2.1 and 5.1. | GPU time. | Member 2 | Phase 4, if time |
| O7 | Phase 4 method work from DL-38: a false-positive filter that works, wider validation grid, box precision in every table. | Phase 3 finding. | Member 3 (filter, grid), Member 1 (tables) | Phase 4 |
| O8 | Error and bias analysis from the per-image CSVs (`runs/*/*/per_image.csv`): Dice vs polyp size, brightness, contrast, dataset; the 75 ClinicDB images with no prediction; the 10 Kvasir images where C loses. | Phase 4 scope by plan. | all three, Member 1 assembles | Phase 4 |
| O9 | Local `PyYAML` install so the one skipped test (`test_yaml_experiment_configs_parse`) runs on the lead's laptop too; CI already runs it. | ~1 MB install not yet approved. | Member 1 | any time |
