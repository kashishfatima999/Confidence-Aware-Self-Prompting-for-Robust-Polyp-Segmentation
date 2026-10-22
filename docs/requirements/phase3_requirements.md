# Phase 3 requirements and compliance checklist

Group **F26-13** · AI2002 Artificial Intelligence (BDS-5A), FAST-NUCES Lahore · Track A

This file keeps the course requirements for Phase 3 next to the evidence that each one is met, in the same form as `phase2_requirements.md`. The general project guidelines are in `Project_Deliverables_and_Guidelines.pdf` in this folder.

## 1. Requirement text (as received from the TA)

> **Submission Files (on GCR)**
> - F26-ID.pdf
> - F26-ID.ipynb or F26-ID.py
> - You can also share Github repository link
>
> **Report (2–4 pages)** — Include:
> - Model/algorithm selection with justification
> - Implementation details (libraries, frameworks used)
> - Code workflow / pipeline (data input → model → output)
> - Training process (data split, hyperparameters, epochs, basic setup etc.)
> - Initial results / outputs (to demonstrate working model)
> - For RAG-based projects: Make sure your embedding pipeline is functional and the system prompts are set up properly
> - Challenges faced during implementation

From the project guidelines PDF, Phase 3 for Track A is: *"Choose appropriate algorithms based on the problem. Use standard frameworks such as PyTorch, TensorFlow, or Scikit-Learn."* The rubric weight for "Technical Implementation" is 30%.

## 2. Compliance checklist

| # | Requirement | Where it is met | Status |
|---|---|---|---|
| R1 | `F26-13.pdf`, 2–4 pages | `../submissions/phase3/F26-13.pdf`: title page + 4 pages of report, A4 (source: `report_source.html`) | done |
| R2 | `F26-13.ipynb` or `F26-13.py` | `../submissions/phase3/F26-13.ipynb`, the executed Colab run (`notebooks/04_phase3_colab.ipynb`, all 14 code cells with outputs) plus a title cell with names and roll numbers and section 10 with the report's tables and figures | done |
| R3 | GitHub repository link | https://github.com/kashishfatima999/Confidence-Aware-Self-Prompting-for-Robust-Polyp-Segmentation | done |
| R4 | Model/algorithm selection with justification | PDF §1 (YOLOv8-m, frozen SAM 2, CASP v1, rejected alternatives, control experiments); `../decision_log.md` DL-31, DL-34, DL-35 | done |
| R5 | Implementation details (libraries, frameworks) | PDF §2 (PyTorch, Ultralytics, SAM 2, NumPy/Pillow/OpenCV/pandas, Colab T4, code layout, 65 tests in CI); `requirements-colab.txt` | done |
| R6 | Code workflow / pipeline | PDF §3 (diagram: input → detector + cache → CASP gate → SAM 2 → per-image results); `../decision_log.md` DL-33, DL-36 | done |
| R7 | Training process | PDF §4 (700 / 100 / 200 split, recipe table, 50 epochs, detector mAP, training curves, threshold grid on validation only); `configs/yolo_train.yaml`, `../tables/phase3/yolo/` | done |
| R8 | Initial results / outputs | PDF §5 (six runs on two test sets, baseline vs paper, corrupted sets, Figure 2 example masks); `../tables/phase3/results.csv` | done |
| R9 | RAG embedding pipeline and prompts | Not applicable: not a RAG project | n/a |
| R10 | Challenges faced during implementation | PDF §6; `../setup_guide.md` challenges C1–C9 | done |
| R11 | Standard frameworks (guidelines) | PyTorch + Ultralytics + SAM 2 | done |
| R12 | External code, datasets and models cited | PDF references [1]–[6], IEEE style, numbered in order of first citation; `README.md` §9 | done |
| R13 | One submission per group, by the lead | Kashish Fatima (24L-2605) submits | — |
| R14 | Naming format | Files named exactly `F26-13.pdf` and `F26-13.ipynb` | done |
| R15 | Able to explain the work | Owners of each file in `../../project-division/Project-Division-Phase-3.md` §3 and §7; reasoning in `../decision_log.md` | — |
