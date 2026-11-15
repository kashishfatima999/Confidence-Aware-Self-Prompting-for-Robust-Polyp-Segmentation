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
| R1 | `F26-13.pdf`, 2–4 pages | `../submissions/phase5/F26-13.pdf` from `report_source.html` (A4, Phase 3–4 style) | source ready; PDF after §1, §2, §4 |
| R2 | `F26-13.ipynb` or `.py` | `../submissions/phase5/F26-13.ipynb`: executed `notebooks/07_phase5_analysis.ipynb` with title cell and section map | done |
| R3 | GitHub repository link | https://github.com/kashishfatima999/Confidence-Aware-Self-Prompting-for-Robust-Polyp-Segmentation | done |
| R4 | Hosted web application link | Hugging Face Space from `app/deploy_space.py` (DL-46); link on the title page and in §4 | Member 2 (task 2.2); link goes on the title page | to do |
| R5 | Experimental results | PDF §1 (Member 1): B, T, C, F, O on both test sets, paired differences, nine corrupted settings; tables ready in `../tables/phase5/fallback_*.csv` | to do |
| R6 | Error analysis | PDF §2 (Member 2): factorisation of mean Dice, oracle-gap split, failure taxonomy with F (Figure 2); numbers in notebook section 2 | to do |
| R7 | Theoretical insights | PDF §3 (Member 3): confidence shift under domain shift and cost-based thresholding; why SAM 2's IoU score is class-agnostic (AUROC, Figure 1); single-polyp prior and Bayes → fallback rule | done |
| R8 | Visualisations, tables, explanations | Figure 1 (`../figures/phase5_box_signals.png`, Member 3), Figure 2 (Member 2), tables from §1–2 | Figure 1 done |
| R9 | Functionality or reproducibility | PDF §4 (Member 2): live demo; notebook reproduces the Phase 4 tables byte for byte before the new analysis (done, Member 3); CI tests | notebook done; §4 to do |
| R10 | No re-tuning on test data | F uses only the existing 0.25 and 0.5 thresholds, fixed in DL-44 before its results | done |
| R11 | One submission by the lead, naming format | Kashish Fatima (24L-2605) uploads `F26-13.pdf`, `F26-13.ipynb`, the GitHub link and the Space link | — |
| R12 | Able to explain the work | `../../project-division/Project-Division-Phase-5.md`; reasoning in `../decision_log.md` DL-44 to DL-46 (Member 1) | — |
