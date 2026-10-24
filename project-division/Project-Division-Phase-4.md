# F26-13 — Project Division for Phase 4 (Evaluation & Bias Check)

**Project:** Confidence-Aware Self-Prompting for Robust Polyp Segmentation Under Image and Domain Shifts
**Track:** A (Research & Development)
**Phase 4 in the guidelines:** "Evaluate using relevant metrics (Accuracy, F1, RMSE, etc.) and perform a bias check."
**Time:** one week, Monday 26 October to Sunday 1 November 2026. Decisions for this phase are DL-39 and DL-40 in `docs/decision_log.md`; the method follow-ups come from DL-38.
**After this phase:** Phases 5 and 6 are reports only (technical report with error analysis and theoretical insights; LaTeX report with architecture diagrams, challenges and the Track B replication section). No new GPU work is planned after Phase 4.

| Member | Name | Owns in Phase 4 | Share |
|---|---|---|---|
| Member 1 | Kashish Fatima (24L-2605), lead | Planning, decision log, pull-request review, final table assembly. **No code, no Colab or Kaggle sessions.** | ~15% (ran every Phase 3 session, see Phase 3 division §7) |
| Member 2 | Khushbakht Sohail (24L-2604) | Metrics the TA named, the bias check (tables, figures, text), the CVC-ClinicDB example figure | ~40% |
| Member 3 | Zahra Saeed (24L-2512) | Comparative evaluation with confidence intervals, error analysis and failure taxonomy, submission package | ~45% |

---

## 1. What Phase 4 must produce

1. **Metrics.** Every run reported with the metrics the TA named that make sense for segmentation and detection: per-image pixel precision, recall, F1 (= Dice), IoU and pixel accuracy; detection-level precision, recall and F1 of boxes at IoU ≥ 0.5. RMSE is not meaningful for binary masks and is replaced by the mask-area error in percentage points of the image (DL-40). Pixel accuracy is reported but flagged as uninformative (polyps cover ~9–15% of an image, so an empty mask already scores ~0.9).
2. **Comparative evaluation.** B, T, C, C-all, C-noiter and O on Kvasir-SEG test, CVC-ClinicDB and the nine corrupted settings, with paired bootstrap 95% CIs for every difference against B and against T. Most numbers exist from Phase 3; this phase adds the new metrics and one clean table per dataset.
3. **Bias check** (DL-40). For each factor below: Dice, miss rate and false-positive boxes per subgroup for B and C, and the C − B gap with a CI, so the reader sees both *where the pipeline is weak* and *whether the method helps everyone equally*. Factors: polyp size (terciles of `gt_area_percent`), image brightness and contrast (terciles, from `docs/tables/dataset_statistics.csv`), image resolution (Kvasir-SEG has 332 × 487 to 1920 × 1072), single vs multiple polyps (`n_gt_polyps`), dataset (Kvasir-SEG vs ClinicDB: domain shift), corruption type and severity. Plus a short section on the pipeline's own biases: the detector saw only Kvasir-SEG training images; `t_low`, `t_high`, `s_min` were tuned on 100 clean Kvasir-SEG images; Kvasir-SEG and ClinicDB come from two hospitals and ClinicDB has smaller polyps and lower resolution.
4. **Error analysis.** One label per image and run from a fixed taxonomy: *missed* (no prediction, GT present), *false positive* (prediction, no GT overlap), *under-segmentation* (Dice < 0.7 with pred area < GT area), *over-segmentation* (Dice < 0.7 with pred area > GT area), *extra box* (a correct polyp plus a false box), *good* (Dice ≥ 0.7, no false box). Counts per run and dataset; figures for the 75 ClinicDB images no run except O predicts and the 10 Kvasir-SEG images where C loses to B; the CVC-ClinicDB example figure (open item O1).
5. **Optional, GPU, only if the above is done by Friday:** the DL-38 method work (new false-positive filter, wider validation grid, val split only) and a second training seed. Both run on Kaggle with `notebooks/05_kaggle_driver.ipynb` (open item O5: its first run is the smoke run with `SMOKE = True`).
6. A Phase 4 report (`docs/phase4_report.md` → `docs/submissions/phase4/F26-13.pdf`) and a notebook that rebuilds every table and figure from the per-image files (`docs/submissions/phase4/F26-13.ipynb`).

Phase 4 does **not** change any Phase 3 number. The per-image files in `runs/` and Drive `MyDrive/F26-13/runs_phase3/` are the inputs; nothing is re-run unless it is an optional item 5 job, and those write to new experiment names.

---

## 2. Workspace decision (DL-39)

Colab stays the workspace. Items 1–4 and 6 read 30 `per_image.csv` files (about 2,000 rows in total), the `detections.jsonl` files and the saved masks; they run in minutes on CPU, so a laptop or a Colab CPU runtime is enough and nothing can be lost to a dropped session. Kaggle is used only for item 5, because those jobs take one to three hours and must survive a closed browser (setup guide C8, C9). Phases 5 and 6 need no GPU at all, so there is no reason to move the project.

---

## 3. Code and files

| Path | What | Written by | Status on Monday |
|---|---|---|---|
| `src/eval/metrics.py`, `tests/test_metrics.py` | add pixel precision / recall / accuracy and mask-area error to `per_image_scores`; detection F1 to `precision_recall` | Member 2 | to do (small) |
| `src/eval/bias_check.py`, `tests/test_bias_check.py` | join `per_image.csv` with `dataset_statistics.csv` on the image id; subgroup tables per factor with CIs; `docs/tables/phase4/bias_*.csv` + `.md` | Member 2 | to do |
| `src/eval/error_analysis.py`, `tests/test_error_analysis.py` | taxonomy label per image and run; counts table; lists of the worst images per run; `docs/tables/phase4/errors_*.csv` | Member 3 | to do |
| `src/eval/compare.py` | paired differences with `metrics.bootstrap_ci` for every pair of runs and dataset; `docs/tables/phase4/comparison.{csv,md}` | Member 3 | to do (Phase 3 did this ad hoc in the notebook) |
| `notebooks/06_phase4_analysis.ipynb` | runs the three scripts on `runs/` (or Drive), draws every figure, no GPU | Member 3 (owner), Member 2 (bias cells) | to do |
| `docs/figures/phase4_*.png` | bias plots (Dice by subgroup, B vs C), error taxonomy bars, failure examples, ClinicDB examples | Members 2 and 3 | to do |
| `docs/phase4_report.md` | the Phase 4 report, written from the tables | Members 2 and 3 (sections below) | to do |
| `docs/submissions/phase4/`, `docs/requirements/phase4_requirements.md` | package in the Phase 3 format; TA text and checklist | Member 3 | to do, after the TA's submission text arrives |
| `runs/`, Drive `runs_phase3/` | the inputs: 30 per-image files, detection logs, masks | — | exist (Phase 3) |

Rules carried over: every number in a table comes from a script that reads files, never typed; every parameter choice gets a decision-log row in the same pull request; `runs/`, weights and masks are not committed.

---

## 4. Task division

### Member 1 — Kashish Fatima: planning and review only

| # | Task | Output |
|---|---|---|
| 1.1 | Branch `kashish-phas4` from `main`; this file; DL-39 and DL-40; change-log row. | Done before the week starts |
| 1.2 | Review every pull request from Members 2 and 3: CI green, decision-log row present, numbers come from files, no Phase 3 number changed. | Reviews |
| 1.3 | On Saturday: run `summarize.py`, `compare.py`, `bias_check.py` and `error_analysis.py` once locally on `runs/` and confirm the committed tables are byte-identical to the Colab ones; update README §8 status and the §6 headline table with the Phase 4 pointers. | README, one check |
| 1.4 | Keep the open-items table (section 7 below) current. | This file |

No GPU session, no new script, no report section for Member 1 this phase.

### Member 2 — Khushbakht Sohail: metrics and the bias check

| # | Task | Output |
|---|---|---|
| 2.1 | Extend `metrics.py` with pixel precision, recall, accuracy and mask-area error (percentage points), and detection F1; tests. `run_experiment.py` already writes `pred_area_percent`, so the new per-image numbers can be computed from the saved masks or from the existing columns without re-running anything. | `src/eval/metrics.py`, tests |
| 2.2 | `bias_check.py`: for each factor in §1 item 3, a table with n, Dice (B, T, C), miss rate, FP boxes per image, C − B gap with 95% CI per subgroup; one figure per factor (Dice by subgroup, B and C side by side). Factors from `per_image.csv` (`gt_area_percent`, `n_gt_polyps`) and `dataset_statistics.csv` (width, height, brightness, contrast; join on image id without extension). | `docs/tables/phase4/bias_*.{csv,md}`, `docs/figures/phase4_bias_*.png` |
| 2.3 | The CVC-ClinicDB example figure (open item O1): 5 best / 5 worst C images, GT vs B vs C, same layout as `docs/figures/phase3_examples_kvasir_test.png`. ClinicDB images come from `src/data/prepare_clinicdb.py` on Colab, masks from Drive `runs_phase3/`. | `docs/figures/phase4_examples_clinicdb.png` |
| 2.4 | Report sections: metrics (what each metric measures, why accuracy and RMSE are not informative here), the bias check (one paragraph and one table per factor), the pipeline's own biases, what a fairer pipeline would need (e.g. training on both datasets, tuning thresholds per dataset). | `docs/phase4_report.md` §2, §4 |
| 2.5 | Optional (item 5): second training seed on Kaggle (seed 43, `configs/yolo_train.yaml` unchanged except `seed`, new run name), save the console log (open item O2), caches and B / C runs under new experiment names; variance row for the main table. | `runs/yolo_seed43/`, tables |

### Member 3 — Zahra Saeed: comparison, error analysis, package

| # | Task | Output |
|---|---|---|
| 3.1 | `compare.py`: paired bootstrap CIs (seed 42, 2,000 resamples, `metrics.bootstrap_ci`) for every run pair against B and against T, per dataset and corrupted setting; win / loss / tie counts; one clean comparison table per dataset with all metrics from 2.1. | `docs/tables/phase4/comparison.{csv,md}` |
| 3.2 | `error_analysis.py`: the taxonomy in §1 item 4 applied to every run and dataset; counts table; per-run lists of the worst 10 images; the 75 ClinicDB no-prediction images (what the detector saw: `max_confidence`, `n_candidates`, polyp size) and the 10 Kvasir images where C < B. | `docs/tables/phase4/errors_*.csv`, `docs/figures/phase4_error_taxonomy.png`, `phase4_failures_*.png` |
| 3.3 | `notebooks/06_phase4_analysis.ipynb`: runs 2.2, 3.1, 3.2 from `runs/` or Drive, rebuilds every Phase 4 table and figure, CPU only. | Notebook, executed |
| 3.4 | Report sections: comparative evaluation (tables, CIs, what the ablations say), error analysis (taxonomy counts, failure figures, what the failures have in common), summary and next steps for Phases 5–6. | `docs/phase4_report.md` §1, §3, §5 |
| 3.5 | Submission package in the Phase 3 format: `report_source.html` → `F26-13.pdf`, `F26-13.ipynb` (executed notebook with title cell and section map), `docs/requirements/phase4_requirements.md`. | `docs/submissions/phase4/` |
| 3.6 | Optional (item 5): first Kaggle smoke run of `05_kaggle_driver.ipynb` (open item O5), then the DL-38 filter and wider grid on `kvasir_val` only; results into a new DL entry. Test sets are touched only if val shows a clear gain, and once. | Kaggle run, `casp_v2.yaml`, DL-41 |

### Dependencies

- 2.1 (new metrics) is needed by 3.1 and 2.2 → target **Monday evening**; until then both scripts work with the existing Dice / IoU / box columns.
- 3.1 and 2.2 tables are needed for the report text → **Thursday**.
- 1.3 needs every table committed → **Saturday**.
- Item 5 jobs must not block anything: start them only if 2.1–2.4 or 3.1–3.4 are done by Friday.

---

## 5. Day-by-day

| Day | Member 1 | Member 2 | Member 3 |
|---|---|---|---|
| Mon | 1.1 done; 1.2 | 2.1 metrics + tests | 3.2 taxonomy script on `runs/` |
| Tue | 1.2 | 2.2 bias script, first tables | 3.1 comparison script and tables |
| Wed | 1.2 | 2.2 figures; 2.3 ClinicDB figure on Colab | 3.2 failure figures (75 ClinicDB, 10 Kvasir) |
| Thu | 1.2 | 2.4 report §2, §4 | 3.3 notebook; 3.4 report §1, §3 |
| Fri | 1.2 | 2.4 done; optional 2.5 start on Kaggle | 3.4 done; optional 3.6 Kaggle smoke run |
| Sat | 1.3 rebuild check, README | review Member 3's text | 3.5 package; review Member 2's text |
| Sun | final review, merge | — | submission |

Estimated effort: Member 2 about 3 to 4 working days (one day scripts, one day figures and the ClinicDB figure, one to two days report text); Member 3 the same (one day scripts, one day figures and notebook, one day text, half a day package). Every mandatory step is CPU-only and under 10 minutes of compute; the optional Kaggle jobs are 1 to 3 hours each of unattended GPU time.

---

## 6. Rules for this phase

**Do**
- Read results from `runs/*/*/per_image.csv`, `detections.jsonl` and `docs/tables/phase3/`; never retype a number.
- Report pixel accuracy, but say in the same sentence why it is uninformative for this task.
- Give every subgroup table its n per subgroup and a CI on the gap; a difference without a CI is not a finding.
- Keep the Phase 3 taxonomy of runs (B, T, C, C-all, C-noiter, O) and names.
- Decision-log row in the same pull request as any new threshold, taxonomy rule or metric definition.

**Don't**
- Don't re-run or change any Phase 3 experiment; new experiments get new names and configs.
- Don't tune anything on the test splits; the optional DL-38 work is validation-only until the very end.
- Don't commit `runs/`, masks or weights.
- Don't start an optional GPU job before the mandatory items are done.

---

## 7. Open items carried from Phase 3 (status at the start of Phase 4)

| # | Item | Owner in Phase 4 | Plan |
|---|---|---|---|
| O1 | CVC-ClinicDB example figure | Member 2 | task 2.3 |
| O2 | Training console log | Member 2 | saved during optional 2.5; otherwise stays open with the per-epoch CSV as the record |
| O5 | First Kaggle run of `05_kaggle_driver.ipynb` | Member 3 | optional 3.6 smoke run; if no GPU job is needed in Phase 4, closed as "not needed" |
| O6 | Second seed | Member 2 | optional 2.5 |
| O7 | DL-38 method work | Member 3 | optional 3.6 |
| O8 | Error and bias analysis | Members 2 and 3 | tasks 2.2, 3.1, 3.2 (mandatory) |
| O9 | Local PyYAML | Member 1 | any time; not needed for Phase 4 |

### Execution record

To be filled on 2026-11-01, in the same form as the Phase 3 division §7, so plan and reality sit next to each other.
