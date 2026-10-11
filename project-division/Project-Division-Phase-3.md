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
