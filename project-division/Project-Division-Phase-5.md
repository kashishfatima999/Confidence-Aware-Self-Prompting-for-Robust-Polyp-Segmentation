# F26-13 — Project Division for Phase 5 (Output Analysis)

**Project:** Confidence-Aware Self-Prompting for Robust Polyp Segmentation Under Image and Domain Shifts
**Track:** A (Research & Development)
**Phase 5 in the guidelines (§3.5):** "Track A & B: A technical report documenting experimental results, error analysis, and theoretical insights."
**TA submission text:** `F26-ID.pdf` (2–4 pages: experimental results, error analysis, theoretical insights; supporting visualisations and tables; demonstrate functionality of the product or reproducibility of the analysis), `F26-ID.ipynb` or `.py`, optional GitHub link, optional hosted web application link.
**Due:** _(fill in)_. The schedule below counts days back from the due date (D).
**After this phase:** Phase 6 (LaTeX report: methodology and architecture diagrams, challenges faced). Planned separately; Phase 5 text should be written so it can be reused there.

| Member | Name | Owns in Phase 5 | Share |
|---|---|---|---|
| Member 1 | Kashish Fatima (24L-2605), lead | Results section, decision-log rows, PR review, final checks, GCR upload | ~20% |
| Member 2 | Khushbakht Sohail (24L-2604) | Hosted demo (deploy, test, screenshot), error-analysis section, reproducibility section | ~35% |
| Member 3 | Zahra Saeed (24L-2512) | New analysis (filter / fallback rule), theoretical insights, analysis notebook, submission package | ~45% |

---

## 1. What Phase 5 must produce

1. **No new GPU runs.** Everything reads the Phase 3 logs (`per_image.csv`, `detections.jsonl`) from Drive `MyDrive/F26-13/runs_phase3/`. No Phase 3 or Phase 4 number changes.
2. **Experimental results** (report §1): one table per test set with B, T, C, O and the new rule F (Dice with CI, pixel precision / recall, misses, extra-box images, box P / R / F1), paired differences, and a short summary of the nine corrupted settings.
3. **Error analysis** (report §2): the exact split mean Dice = P(image gets a mask) × E[Dice | mask]; the gap to the oracle divided into missed images and box quality; how F changes the Phase 4 failure taxonomy; where F loses.
4. **Theoretical insights** (report §3), each backed by a number from the logs:
   - domain shift lowers detector confidence before it removes detections (why the gate works);
   - SAM 2's predicted IoU measures mask quality, not class, so it cannot filter false boxes (the DL-38 question);
   - context (most frames show one polyp) is the missing information, which gives the fallback rule F.
5. **The DL-38 follow-up, CPU only:** label every candidate box true / false against the oracle's ground-truth boxes; compare YOLO confidence, SAM 2 score, their product and image context by AUROC; evaluate F, fixed in advance with the existing thresholds 0.25 and 0.5 (no test-set tuning).
6. **Functionality / reproducibility** (report §4): a hosted Gradio demo on Hugging Face Spaces that runs the project's own code; the notebook re-runs the Phase 4 comparison and checks it is byte-identical before the new analysis; CI tests.
7. **Package** in the Phase 3–4 format: `docs/submissions/phase5/F26-13.pdf` (from `report_source.html`), `F26-13.ipynb`, `docs/requirements/phase5_requirements.md`.

---

## 2. Code and files

| Path | What | Owner | Status |
|---|---|---|---|
| `src/eval/filter_analysis.py`, `tests/test_filter_analysis.py` | fallback rule F (exact, row by row from B / T / C), box labelling, AUROC with bootstrap CI, context table | Member 3 | done |
| `docs/tables/phase5/` | tables written by the script | Member 3 | done |
| `notebooks/07_phase5_analysis.ipynb` | runs the script, draws the box-signal figure, CPU only | Member 3 | done, executed |
| `docs/figures/phase5_box_signals.png` | confidence vs SAM score, true vs false boxes; context bars | Member 3 | done |
| `docs/figures/phase5_fallback_taxonomy.png` | failure labels for B / T / C / F | Member 2 | to do |
| `app/app.py`, `app/requirements.txt`, `app/space_README.md`, `app/deploy_space.py`, `tests/test_app.py` | Gradio demo and deploy script | Member 2 | to do |
| `docs/phase5_report.md` | the report, written from the tables | all (sections below) | to do |
| `docs/submissions/phase5/`, `docs/requirements/phase5_requirements.md` | package and checklist | Member 3 | source and checklist done; PDF after §1, §2, §4 and the Space link arrive |
| `docs/decision_log.md` | DL-44 (scope, rule F), DL-45 (box labelling, AUROC), DL-46 (demo) + change-log rows | Member 1 | to do |

Rules carried over: every number comes from a script that reads files, never typed; a decision-log row in the same PR as any new rule or definition; `runs/`, masks and weights are never committed (the YOLO weights go only to the Space).

---

## 3. Task division

### Member 1 — Kashish Fatima: results, decisions, review

| # | Task | Output |
|---|---|---|
| 1.1 | Write DL-44, DL-45, DL-46 from Members 2 and 3's notes; change-log row per merged PR. | `docs/decision_log.md` |
| 1.2 | Report §1 (experimental results): the two results tables and the paired differences, from `docs/tables/phase5/fallback_metrics.csv` and `fallback_comparison.csv`; one paragraph on the corrupted settings. | `docs/phase5_report.md` §1 |
| 1.3 | Review both PRs: CI green, DL rows present, numbers match the tables, no Phase 3 / 4 number changed. | Reviews |
| 1.4 | Final check: run the notebook once from a fresh clone; open the PDF and the Space; README §3 and §8. Upload on GCR. | README, submission |

### Member 2 — Khushbakht Sohail: demo, error analysis, reproducibility

| # | Task | Output |
|---|---|---|
| 2.1 | Build the Gradio demo (`app/app.py`): upload a frame, show B / T / C / F side by side from one YOLO pass, importing `src/pipeline/refine.py` and `sam2_prompter.py` unchanged; a deploy script and a few tests with a mocked SAM 2. | `app/`, `tests/test_app.py` |
| 2.2 | Create a Hugging Face account / token and deploy the Space (CPU; YOLO weights uploaded only to the Space). Fix any build error (dependency versions are the usual cause; record what changed). | Live Space link |
| 2.3 | Test the Space on 3–4 images (one Kvasir-SEG, one ClinicDB, one corrupted); take one screenshot for the report. Optionally add 2–3 example images in `app/examples/` (check the dataset licences first). | Screenshot, link |
| 2.4 | Figure: failure labels for B / T / C / F on both test sets, from `docs/tables/phase5/fallback_errors.csv` (same style as `phase4_error_taxonomy.png`). | `docs/figures/phase5_fallback_taxonomy.png` |
| 2.5 | Report §2 (error analysis): the P(mask) × E[Dice | mask] table, the oracle-gap split, what F fixes and where it loses (ClinicDB second polyps / fragments, noise severity 2). | `docs/phase5_report.md` §2 |
| 2.6 | Report §4 (functionality and reproducibility): what the demo shows, how to run the analysis, tests. | `docs/phase5_report.md` §4 |

### Member 3 — Zahra Saeed: analysis, theory, notebook, package

| # | Task | Output |
|---|---|---|
| 3.1 | Finalise `filter_analysis.py` and its tests; re-run on the Drive files; confirm the Phase 4 tables reproduce byte for byte first. | Script, tables |
| 3.2 | `notebooks/07_phase5_analysis.ipynb`: setup, Phase 4 reproduction check, results, oracle-gap split, box signals, context, F (both figures drawn here). | Notebook, executed |
| 3.3 | Figure: band boxes, YOLO confidence vs SAM 2 score, true vs false; share of true boxes with and without a box ≥ 0.5 in the image. | `docs/figures/phase5_box_signals.png` |
| 3.4 | Report §3 (theoretical insights): confidence shift under domain shift and the cost argument for a lower threshold; why SAM 2's IoU head is class-agnostic (AUROC, share of false boxes scoring ≥ 0.8, correlation with confidence); the single-polyp prior → rule F and its limits. With references (SAM [Kirillov 2023], dataset shift [Ovadia 2019], cost-sensitive thresholds [Elkan 2001], optimizer's curse [Smith & Winkler 2006]). | `docs/phase5_report.md` §3 |
| 3.5 | Summary and conclusions; assemble the report; `report_source.html` → `F26-13.pdf` (2–4 pages); `F26-13.ipynb`; checklist. | `docs/submissions/phase5/`, `docs/requirements/phase5_requirements.md` |

### Dependencies

- 3.1 (tables) is needed by 1.2, 2.4, 2.5 → **first**.
- 2.2 (Space link) and 2.3 (screenshot) are needed by 3.5 → do not leave the deployment to the last day; a first Space build takes ~10 minutes, and fixing a failed build can take longer.
- 1.2, 2.5, 2.6, 3.4 text is needed by 3.5.
- 1.4 needs everything merged.

---

## 4. Schedule

| When | Member 1 — Kashish | Member 2 — Khushbakht | Member 3 — Zahra |
|---|---|---|---|
| D−5 | Branch `kashish-phase5`; DL-44 draft | Branch `khushbakht-phase5`; 2.1 build the app | Branch `zahra-phase5`; 3.1 script + tables, push |
| D−4 | 1.2 results section | 2.2 deploy the Space | 3.2 notebook, 3.3 figure |
| D−3 | DL-45, DL-46; review Zahra's PR | 2.3 test + screenshot; 2.4 figure | 3.4 theory section |
| D−2 | review Khushbakht's PR | 2.5, 2.6 report sections | 3.4 finish; review §1, §2 |
| D−1 | 1.4 fresh-clone check | fix anything from review | 3.5 assemble, PDF, notebook, checklist |
| D | upload on GCR | — | — |

Estimated effort: Member 1 ~5 h, Member 2 ~10 h, Member 3 ~10 h. All CPU only.

---

## 5. Rules for this phase

**Do**
- Read results from the Phase 3 files; never retype a number.
- Give every difference a CI and every subgroup its n.
- Keep the run names B, T, C, C-all, C-noiter, O; the new rule is F (F_T from B + T, F_C from B + C).
- Be able to explain every line of the files you own at the evaluation.

**Don't**
- Don't re-run or change any Phase 3 experiment or Phase 4 table.
- Don't tune anything on the test splits (F uses only the existing 0.25 and 0.5).
- Don't commit `runs/`, masks, weights or the Hugging Face token.

---

## 6. Execution record

To be filled at submission, in the same form as the Phase 3 division §7.
