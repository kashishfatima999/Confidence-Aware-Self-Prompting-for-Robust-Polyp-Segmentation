# Phase 2 requirements and compliance checklist

Group **F26-13** · AI2002 Artificial Intelligence (BDS-5A), FAST-NUCES Lahore · Track A

This file keeps the course requirements for Phase 2 next to the evidence that each one is met. The general project guidelines are in `Project_Deliverables_and_Guidelines.pdf` in this folder; the approved proposal is in `../proposal/F26-13_Project_Proposal.pdf`.

## 1. Requirement text (as received from the TA)

> This phase focuses on collecting, understanding, and preparing data for your AI project. Ensure your dataset is clean, well-documented, and ready for model implementation.
> Refer to the project document for track-specific details.
>
> **Submission Files (on GCR)**
> - F26-ID.pdf
> - F26-ID.ipynb or F26-ID.py
> - You can also share Github repository link
>
> **Report (2–4 pages)** — Include:
> - Dataset source & description
> - Data collection method
> - Preprocessing steps
> - Brief EDA (key insights)
> - Final dataset summary
>
> **Notes**
> - Only one submission per group (Lead)
> - Follow naming format strictly
> - No late submissions / plagiarism
> - You should be able to explain your work

From the project guidelines PDF, Phase 2 for Track A is: *"Gather data via scraping, APIs, or repositories. Clean and preprocess as required."* The rubric weight for "Data Preprocessing / Setup" is 15%. The general guidelines also require that *any external code, dataset, or model must be properly cited*.

## 2. Compliance checklist

| # | Requirement | Where it is met | Status |
|---|---|---|---|
| R1 | `F26-13.pdf`, 2–4 pages | `../submissions/phase2/F26-13.pdf` (source: `report_source.html`) | done |
| R2 | `F26-13.ipynb` or `F26-13.py` | `../submissions/phase2/F26-13.ipynb` — one Colab notebook that reproduces the whole Phase 2 pipeline | done; run once on Colab before upload so the outputs are filled in |
| R3 | GitHub repository link | https://github.com/kashishfatima999/Confidence-Aware-Self-Prompting-for-Robust-Polyp-Segmentation | done |
| R4 | Report: dataset source & description | PDF §1; `../data_report.md` §1.1, §2.1 | done |
| R5 | Report: data collection method | PDF §2; `../data_report.md` §1.1, §2.1, `../setup_guide.md` §4 | done |
| R6 | Report: preprocessing steps | PDF §3; `../data_report.md` §1.2–1.6, §2.2–2.4, §4; code in `src/data/` | done |
| R7 | Report: brief EDA (key insights) | PDF §4; `../data_report.md` §1.4, §3; figures in `../figures/stats_*.png`, tables in `../tables/` | done |
| R8 | Report: final dataset summary | PDF §5; `../data_report.md` §6 | done |
| R9 | Dataset is clean | Masks binarised, boxes derived, every file verified before output; 34 automated tests; overlay check 20/20 on both datasets (`../setup_guide.md` §8) | done |
| R10 | Dataset is well documented | `../data_report.md` (every number from `verification.json`), `../decision_log.md` (every choice with alternatives and evidence) | done |
| R11 | Dataset is ready for model implementation | YOLO labels + `configs/*.yaml`; smoke test shows YOLO → SAM 2 runs on the processed data on Colab (`../setup_guide.md` §6) | done |
| R12 | External code, datasets and models cited | `README.md` §7, `../data_report.md` citations, `../setup_guide.md` §1 (baseline commit hash, no license → not copied) | done |
| R13 | One submission per group, by the lead | Kashish Fatima (24L-2605) submits | — |
| R14 | Naming format | Files named exactly `F26-13.pdf` and `F26-13.ipynb` | done |
| R15 | Able to explain the work | Each member owns the scripts and sections listed in `../../project-division/Project-Division-Phase-2.md`; reasoning is in `../decision_log.md` | — |

## 3. Known limitations stated in the submission

- Baseline weights were used for the smoke test only; no reported results come from them (DL-17).
- CVC-ClinicDB results in later phases are cross-dataset and not comparable with the paper's in-domain numbers (DL-03).
- The Kvasir-SEG split is random, not stratified; the validation split has slightly larger polyps on average (DL-02).
- Decisions D2 and D5 (ClinicDB test-only, corruption set) are implemented but still marked "to be confirmed by the group / instructor" in the log.
