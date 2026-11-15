# F26-13 — Project Division for Phase 5 (Output Analysis)

**Project:** Confidence-Aware Self-Prompting for Robust Polyp Segmentation Under Image and Domain Shifts
**Track:** A (Research & Development)
**Phase 5 in the guidelines (§3.5):** "Track A & B: A technical report documenting experimental results, error analysis, and theoretical insights."
**TA submission text:** `F26-ID.pdf` (2–4 pages: experimental results, error analysis, theoretical insights; supporting visualisations and tables; demonstrate functionality of the product or reproducibility of the analysis), `F26-ID.ipynb` or `.py`, optional GitHub link, optional hosted web application link.

| Member | Name | Phase 5 work | Share |
|---|---|---|---|
| Member 1 | Kashish Fatima (24L-2605), lead | Results section, decision-log rows, review, final checks, GCR upload | ~20% |
| Member 2 | Khushbakht Sohail (24L-2604) | Hosted demo, error-analysis section and figure, reproducibility section | ~35% |
| Member 3 | Zahra Saeed (24L-2512) | Filter analysis and fallback rule, theoretical insights, analysis notebook, submission package | ~45% |

---

## 1. Scope

1. No new GPU runs. Everything reads the Phase 3 logs (`per_image.csv`, `detections.jsonl`) from Drive `MyDrive/F26-13/runs_phase3/`. No Phase 3 or Phase 4 number changes.
2. **Experimental results** (report §1): B, T, C, O and the fallback rule F on both test sets (Dice with CI, pixel precision / recall, misses, extra-box images, box P / R / F1), paired differences, and the nine corrupted settings.
3. **Error analysis** (report §2): mean Dice = P(image gets a mask) × E[Dice | mask]; the gap to the oracle split into missed images and box quality; how F changes the Phase 4 failure taxonomy; where F loses.
4. **Theoretical insights** (report §3): domain shift lowers detector confidence before it removes detections; SAM 2's predicted IoU measures mask quality, not class, so it cannot filter false boxes; most frames show one polyp, so image context carries the information the per-box scores lack, which gives the fallback rule F.
5. **Box-level check (the DL-38 follow-up, CPU only):** every candidate box labelled true / false against the oracle's ground-truth boxes; YOLO confidence, SAM 2 score, their product and image context compared by AUROC; F evaluated with the existing thresholds 0.25 and 0.5, so nothing is tuned on the test sets.
6. **Functionality / reproducibility** (report §4): a hosted Gradio demo on Hugging Face Spaces running the project's own code; the analysis notebook, which first reproduces the Phase 4 tables byte for byte; CI tests.
7. Package in the Phase 3–4 format: `docs/submissions/phase5/F26-13.pdf` (from `report_source.html`), `F26-13.ipynb`, `docs/requirements/phase5_requirements.md`.

---

## 2. Code and files

| Path | What | Member |
|---|---|---|
| `src/eval/filter_analysis.py`, `tests/test_filter_analysis.py` | fallback rule F (built from the B / T / C per-image rows), box labelling, AUROC with bootstrap CI, context table | 3 |
| `docs/tables/phase5/` | tables written by the script | 3 |
| `notebooks/07_phase5_analysis.ipynb` | runs the script and draws the box-signal figure, CPU only | 3 |
| `docs/figures/phase5_box_signals.png` | confidence vs SAM score for true and false boxes; context bars | 3 |
| `docs/figures/phase5_fallback_taxonomy.png` | failure labels for B / T / C / F | 2 |
| `app/` | Gradio demo and its deploy script | 2 |
| `docs/phase5_report.md`, `docs/submissions/phase5/` | report and submission package | all |
| `docs/requirements/phase5_requirements.md` | TA text and compliance checklist | 3 |
| `docs/decision_log.md` | DL-44 (scope, rule F), DL-45 (box labelling, AUROC), DL-46 (demo) | 1 |

---

## 3. Tasks

### Member 1 — Kashish Fatima

| # | Task | Output |
|---|---|---|
| 1.1 | DL-44, DL-45, DL-46 and the change-log rows | `docs/decision_log.md` |
| 1.2 | Report §1: results tables, paired differences, corrupted settings, from `docs/tables/phase5/fallback_metrics.csv` and `fallback_comparison.csv` | report §1 |
| 1.3 | Review of both pull requests: CI green, decision-log rows present, numbers match the tables | reviews |
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

---

## 4. Method rules

- Every number comes from a script that reads the result files.
- Every difference has a CI and every subgroup its n.
- Run names stay B, T, C, C-all, C-noiter, O; the new rule is F (F_T from B + T, F_C from B + C).
- No Phase 3 experiment is re-run and no Phase 4 table changes.
- Nothing is tuned on the test splits; F uses only the existing 0.25 and 0.5.
