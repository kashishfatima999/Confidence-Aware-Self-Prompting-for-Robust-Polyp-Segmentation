# Phase 4 requirements and compliance checklist

Group **F26-13** · AI2002 Artificial Intelligence (BDS-5A), FAST-NUCES Lahore · Track A

## 1. Requirement text (as received from the TA)

> This phase focuses on evaluating your implemented model(s) and performing a detailed analysis of results. Refer to the project document for track-specific details.
>
> **Submission Files (on GCR)**
> - F26-ID.pdf
> - F26-ID.ipynb or F26-ID.py
> - You can also share Github repository link
>
> **Report (2–4 pages)** — Include:
> - Evaluation metrics (e.g., Accuracy, F1-score, RMSE, etc.) with results
> - Comparison of models / approaches (if applicable)
> - Error analysis (where the model performs well/poorly)
> - Bias or fairness check (if relevant)
> - Key insights and observations
> - For RAG based projects: Compare with baseline results (e.g., generation without retrieval). Ensure your model minimizes hallucination using appropriate embedding size and strict system prompts
> - Challenges faced during implementation
>
> Notes: only one submission per group (Lead); follow naming format strictly; no late submissions / plagiarism; you should be able to explain your work. Focus on correctness, understanding, and proper experimentation rather than just results.

## 2. Compliance checklist

| # | Requirement | Where it is met | Status |
|---|---|---|---|
| R1 | `F26-13.pdf`, 2–4 pages | `../submissions/phase4/F26-13.pdf`: title page + 4 pages, A4 (source `report_source.html`) | done |
| R2 | `F26-13.ipynb` or `.py` | `../submissions/phase4/F26-13.ipynb`: executed `notebooks/06_phase4_analysis.ipynb` with title cell; rebuilds every table and figure on CPU | done |
| R3 | GitHub repository link | https://github.com/kashishfatima999/Confidence-Aware-Self-Prompting-for-Robust-Polyp-Segmentation | done |
| R4 | Evaluation metrics with results | PDF §1: Dice (= pixel F1), IoU, pixel precision / recall / accuracy, area error in place of RMSE, box P / R / F1, for all six runs on both test sets; `../tables/phase4/metrics.csv`; DL-40, DL-41, DL-42 | done |
| R5 | Comparison of approaches | PDF §2: B, T, C, C-all, C-noiter, O, paired bootstrap CIs, better / worse / tie counts, nine corrupted settings; `../tables/phase4/comparison.csv` | done |
| R6 | Error analysis | PDF §3: failure taxonomy (Figure 1), Kvasir wins and losses, the 75 ClinicDB images with no mask (Figure 2); `../tables/phase4/errors_*.csv`; DL-41 | done |
| R7 | Bias or fairness check | PDF §4: subgroups by dataset, polyp size, brightness, contrast, width, height, polyp count, corruption (Figure 3); `../tables/phase4/bias_*.csv`; DL-42 | done |
| R8 | Key insights and observations | PDF §5 | done |
| R9 | RAG items | Not applicable: not a RAG project | n/a |
| R10 | Challenges faced | PDF §6 | done |
| R11 | Correctness and proper experimentation | Phase 3 numbers unchanged; every number produced by a script; 84 unit tests in CI; notebook re-run gives byte-identical tables | done |
| R12 | One submission by the lead, naming format | Kashish Fatima (24L-2605) uploads `F26-13.pdf` and `F26-13.ipynb` | — |
| R13 | Able to explain the work | File owners in `../../project-division/Project-Division-Phase-4.md`; reasoning in `../decision_log.md` DL-39 to DL-43 | — |
