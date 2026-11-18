# Phase 5 requirements and compliance checklist

Group **F26-13** · AI2002 Artificial Intelligence (BDS-5A), FAST-NUCES Lahore · Track A

## 1. Requirement text

### As received from the TA

> **Submission Files (on GCR)**
> - F26-ID.pdf (report / analysis)
> - F26-ID.ipynb or F26-ID.py (code / implementation)
> - You can also share Github repository link
> - Hosted web application link (if hosted)
>
> **Report (2–4 pages)** — Include:
> - A technical report documenting experimental results, error analysis, and theoretical insights
> - Any additional visualizations, tables, or explanations supporting your work
> - Demonstrate functionality of the product or reproducibility of the analysis

### Project guidelines

> **3.5 Phase 5: Output (Analysis)** — Track A & B: A technical report documenting experimental results, error analysis, and theoretical insights.

## 2. Compliance checklist

| # | Requirement | Where it is met | Status |
|---|---|---|---|
| R1 | `F26-13.pdf`, 2–4 pages | `../submissions/phase5/F26-13.pdf`, built from `report_source.html` (5 pages: 1 title page + 4 content pages) | done |
| R2 | `F26-13.ipynb` or `.py` | `../submissions/phase5/F26-13.ipynb`: executed `notebooks/07_phase5_analysis.ipynb` | done |
| R3 | GitHub repository link | https://github.com/kashishfatima999/Confidence-Aware-Self-Prompting-for-Robust-Polyp-Segmentation | done |
| R4 | Hosted web application link | https://huggingface.co/spaces/sugarrmelonnnnn/casp-polyp-segmentation (verified live on ZeroGPU); link on title page and in §4 | done |
| R5 | Experimental results | PDF §1: B, T, C, F, O on both test sets, paired differences, nine corrupted settings; `../tables/phase5/fallback_*.csv` | done |
| R6 | Error analysis | PDF §2: factorisation of mean Dice, oracle-gap split, failure taxonomy with F; numbers in notebook section 2 | done |
| R7 | Theoretical insights | PDF §3: confidence shift under domain shift and cost-based thresholding; why SAM 2's IoU score is class-agnostic (AUROC, Figure 1); single-polyp prior and Bayes → fallback rule | done |
| R8 | Visualisations, tables, explanations | Figure 1 (`../figures/phase5_box_signals.png`), Figure 2 (failure taxonomy), Figure 3 (demo UI), tables in §1–3 | done |
| R9 | Functionality or reproducibility | PDF §4: live demo; the notebook reproduces the Phase 4 tables byte for byte before the new analysis; CI tests | done |
| R10 | No re-tuning on test data | F uses only the existing 0.25 and 0.5 thresholds, fixed in DL-44 before its results | done |
| R11 | One submission by the lead, naming format | Uploaded by the group lead as `F26-13.pdf` and `F26-13.ipynb` with the GitHub and Space links | pending (GCR upload) |
| R12 | Able to explain the work | `../../project-division/Project-Division-Phase-5.md`; reasoning in `../decision_log.md` DL-44 to DL-47 | done |
