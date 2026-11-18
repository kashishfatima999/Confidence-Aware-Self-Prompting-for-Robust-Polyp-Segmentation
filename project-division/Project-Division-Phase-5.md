# F26-13 — Project Division for Phase 5 (Output Analysis)

**Project:** Confidence-Aware Self-Prompting for Robust Polyp Segmentation Under Image and Domain Shifts
**Track:** A (Research & Development)
**Phase 5 in the guidelines (§3.5):** "Track A & B: A technical report documenting experimental results, error analysis, and theoretical insights."
**TA submission text:** `F26-ID.pdf` (2–4 pages: experimental results, error analysis, theoretical insights; supporting visualisations and tables; demonstrate functionality of the product or reproducibility of the analysis), `F26-ID.ipynb` or `.py`, optional GitHub link, optional hosted web application link.

| Member | Name | Phase 5 work | Share |
|---|---|---|---|
| Member 1 | Kashish Fatima (24L-2605), lead | Results section, decision-log rows, review, final checks, GCR upload | ~15% |
| Member 2 | Khushbakht Sohail (24L-2604) | Hosted demo, error-analysis section and figure, reproducibility section | ~30% |
| Member 3 | Zahra Saeed (24L-2512) | Filter analysis and fallback rule, second detector seed, theoretical insights, analysis notebooks, submission package | ~55% |

---

## 1. Scope

1. The analysis reads the Phase 3 logs (`per_image.csv`, `detections.jsonl`) from Drive `MyDrive/F26-13/runs_phase3/`. One GPU job runs on Colab: YOLOv8-m retrained with seed 43 and runs B, T, C repeated on both test sets, saved to new folders (`runs_phase5_s43/`). No Phase 3 or Phase 4 number changes.
2. **Experimental results** (report §1): B, T, C, O and the fallback rule F on both test sets (Dice with CI, pixel precision / recall, misses, extra-box images, box P / R / F1), paired differences, and the nine corrupted settings.
3. **Error analysis** (report §2): mean Dice = P(image gets a mask) × E[Dice | mask]; the gap to the oracle split into missed images and box quality; how F changes the Phase 4 failure taxonomy; where F loses.
4. **Theoretical insights** (report §3): domain shift lowers detector confidence before it removes detections; SAM 2's predicted IoU measures mask quality, not class, so it cannot filter false boxes; most frames show one polyp, so image context carries the information the per-box scores lack, which gives the fallback rule F.
5. **Box-level check (the DL-38 follow-up, CPU only):** every candidate box labelled true / false against the oracle's ground-truth boxes; YOLO confidence, SAM 2 score, their product and image context compared by AUROC; F evaluated with the existing thresholds 0.25 and 0.5, so nothing is tuned on the test sets.
6. **Second seed (open item O6):** the detector is retrained with seed 43 (same recipe, DL-31) and the headline results (C − B, T − B, F − T, extra-box counts, box F1, band AUROCs) are reported for both seeds. Rule F was designed after looking at the seed-42 errors, so seed 43 is an independent check of it.
7. **Functionality / reproducibility** (report §4): a hosted Gradio demo on Hugging Face Spaces running the project's own code; the analysis notebook, which first reproduces the Phase 4 tables byte for byte; CI tests.
8. Package in the Phase 3–4 format: `docs/submissions/phase5/F26-13.pdf` (from `report_source.html`), `F26-13.ipynb`, `docs/requirements/phase5_requirements.md`.

---

## 2. Code and files

| Path | What | Member |
|---|---|---|
| `src/eval/filter_analysis.py`, `tests/test_filter_analysis.py` | fallback rule F (built from the B / T / C per-image rows), box labelling, AUROC with bootstrap CI, context table | 3 |
| `docs/tables/phase5/` | tables written by the script | 3 |
| `notebooks/07_phase5_analysis.ipynb` | runs the script and draws the box-signal figure, CPU only | 3 |
| `docs/figures/phase5_box_signals.png` | confidence vs SAM score for true and false boxes; context bars | 3 |
| `notebooks/08_phase5_seed43_colab.ipynb` | seed-43 training, detection caches and runs B, T, C on Colab (T4) | 3 |
| `src/eval/seed_compare.py`, `tests/test_seed_compare.py` | seed 42 vs seed 43 on the key results | 3 |
| `docs/tables/phase5/seed_comparison.*` | seed comparison table | 3 |
| `docs/figures/phase5_fallback_taxonomy.png` | failure labels for B / T / C / F | 2 |
| `app/` | Gradio demo and its deploy script | 2 |
| `docs/phase5_report.md`, `docs/submissions/phase5/` | report and submission package | all |
| `docs/requirements/phase5_requirements.md` | TA text and compliance checklist | 3 |
| `docs/decision_log.md` | DL-44 (scope, rule F), DL-45 (box labelling, AUROC), DL-46 (demo), DL-47 (second seed) | 1 |

---

## 3. Tasks

### Member 1 — Kashish Fatima

| # | Task | Output |
|---|---|---|
| 1.1 | DL-44, DL-45, DL-46, DL-47 and the change-log rows | `docs/decision_log.md` |
| 1.2 | Report §1: results tables, paired differences, corrupted settings, from `docs/tables/phase5/fallback_metrics.csv` and `fallback_comparison.csv` | report §1 |
| 1.3 | Review of both pull requests: CI green, decision-log rows present, numbers match the tables, seed-43 runs in their own folders | reviews |
| 1.4 | Final check from a fresh clone (notebook, PDF, Space); README §3 and §8; GCR upload | README, submission |

### Member 2 — Khushbakht Sohail

| # | Task | Output |
|---|---|---|
| 2.1 | Gradio demo: B / T / C / F side by side from one YOLO pass, using `src/pipeline/refine.py` and `sam2_prompter.py` unchanged; deploy script; tests with a mocked SAM 2 | `app/`, `tests/test_app.py` |
| 2.2 | Hugging Face Space (CPU; YOLO weights only on the Space) | Space link |
| 2.3 | Demo tested on Kvasir-SEG, ClinicDB and corrupted images; screenshot for the report | screenshot |
| 2.4 | Failure-taxonomy figure for B / T / C / F from `docs/tables/phase5/fallback_errors.csv` | `docs/figures/phase5_fallback_taxonomy.png` |
| 2.5 | Report §2: P(mask) × E[Dice | mask] table, oracle-gap split, what F fixes and where it loses | report §2 |
| 2.6 | Report §4: demo, how to rerun the analysis, tests | report §4 |

### Member 3 — Zahra Saeed

| # | Task | Output |
|---|---|---|
| 3.1 | `filter_analysis.py` and tests; run on the Drive files after checking that the Phase 4 tables reproduce | script, tables |
| 3.2 | `notebooks/07_phase5_analysis.ipynb`: Phase 4 reproduction check, results, oracle-gap split, box signals, context, F | notebook |
| 3.3 | Box-signal figure | `docs/figures/phase5_box_signals.png` |
| 3.4 | Report §3 (theoretical insights), with references (SAM [Kirillov 2023], dataset shift [Ovadia 2019], cost-sensitive thresholds [Elkan 2001], optimizer's curse [Smith & Winkler 2006]) | report §3 |
| 3.5 | Summary and conclusions; `report_source.html` → `F26-13.pdf`; `F26-13.ipynb`; checklist | `docs/submissions/phase5/`, checklist |
| 3.6 | Second seed: `08_phase5_seed43_colab.ipynb` on Colab (training with seed 43 from a temporary copy of the recipe, full training log kept, caches, runs B, T, C on Kvasir-SEG test and ClinicDB); `seed_compare.py`; seed table and paragraph in §3 | Drive `runs_phase5_s43/`, `docs/tables/phase5/seed_comparison.*` |
| 3.7 | Local PyYAML so the YAML test runs outside CI too; full test suite passes locally | — |

---

## 4. Method rules

- Every number comes from a script that reads the result files.
- Every difference has a CI and every subgroup its n.
- Run names stay B, T, C, C-all, C-noiter, O; the new rule is F (F_T from B + T, F_C from B + C).
- No Phase 3 run is overwritten and no Phase 4 table changes; the seed-43 runs have their own folders.
- Nothing is tuned on the test splits; F uses only the existing 0.25 and 0.5.

---

## 5. Open items carried into Phase 5

| # | Item | Phase 5 outcome |
|---|---|---|
| O1 | CVC-ClinicDB example figure | closed in Phase 4 (DL-43) |
| O2 | YOLO training console log | closed: the seed-42 log is not recoverable and its per-epoch CSV is the record; the seed-43 run keeps its full log (task 3.6) |
| O5 | First Kaggle run | not pursued: the Phase 5 GPU job is short enough for Colab, with every step saved to Drive |
| O6 | Second training seed | Zahra, task 3.6 |
| O7 | DL-38 false-box filter | Zahra: box-level check and fallback rule (task 3.1); the candidate-mask-agreement signal is not run, since the 0.25–0.5 band has too few boxes (39 and 117) to separate it from the SAM 2 score, and is listed as future work |
| O8 | Error and bias analysis | closed in Phase 4 |
| O9 | Local PyYAML | Zahra, task 3.7 |
