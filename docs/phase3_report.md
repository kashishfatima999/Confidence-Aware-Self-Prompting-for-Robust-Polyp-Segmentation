# F26-13 — Phase 3 Report: Implementation & Execution (draft)

**Project:** Confidence-Aware Self-Prompting for Robust Polyp Segmentation Under Image and Domain Shifts

Every number in this report is read from `docs/tables/phase3/results.csv` (built by `src/eval/summarize.py` from the `summary.json` files of the runs) or from `runs/yolo/<name>/summary.json`. Decisions: DL-30 to DL-37 in `docs/decision_log.md`. Sections are owned as listed in `project-division/Project-Division-Phase-3.md`.

## 1. Setup (Member 1)

- Data: Kvasir-SEG 700 / 100 / 200 (DL-02), CVC-ClinicDB 612 external test (DL-03), corrupted Kvasir test set (DL-20). Phase 2 report: `docs/data_report.md`.
- Hardware and software: Colab T4; ultralytics 8.4.174, SAM 2 commit `2b90b9f`, torch 2.11 (`requirements-colab.txt`); seed 42 everywhere.
- Evaluation (DL-32): per-image mean Dice / IoU with bootstrap 95% CI; dataset-pooled Dice / IoU (the paper's definition); images without a prediction count as empty (DL-18); box precision / recall at IoU ≥ 0.5 against mask-derived boxes.

## 2. Detector (Member 2)

- Recipe (DL-31): `configs/yolo_train.yaml` — YOLOv8-m, 50 epochs, 640 px, batch 16, seed 42, Ultralytics defaults otherwise.
- Table 2.1: val and test mAP50, mAP50-95, precision, recall (from `runs/yolo/yolov8m_kvasir_s42/summary.json`).
- Figure 2.1: training curves (`results.png`). Figure 2.2: recall vs confidence threshold on val (from the detection cache).
- Note: how many val/test polyps have **no** candidate box even at confidence 0.05 (these cannot be recovered by any prompt refinement).

## 3. Baseline reproduction (Member 2)

- B = YOLO ≥ 0.5 → SAM 2 (`configs/experiments/baseline.yaml`).
- Table 3.1: B on Kvasir test and CVC-ClinicDB, per-image and pooled, next to the paper's 0.764 / 0.866 (Kvasir) and 0.909 / 0.951 (ClinicDB, in-domain in the paper, cross-dataset here).
- Why ours differs: misses count (DL-18); our split; 640 vs 680; T4 batch 16 vs A100 batch 64.
- Oracle O (GT boxes → SAM 2): the ceiling. Gap O − B = what the detector costs.

## 4. Method: Confidence-Aware Self-Prompting (Member 3)

- Algorithm (DL-34) with the chosen thresholds `t_low`, `t_high`, `s_min` and the validation grid (Table 4.1, 27 runs on `kvasir_val`).
- How many test detections fell in each branch (confident / uncertain / dropped / rejected), from `per_image.csv`.

## 5. Results (Member 1 assembles, all write)

- Table 5.1: B, T, C, C-all, C-noiter, O on Kvasir test and CVC-ClinicDB (`docs/tables/phase3/results.md`).
- Table 5.2 (if run): B and C on the nine corrupted settings.
- Figure 5.1: 5 images where C improved most over B, 5 where it hurt most (`--save-masks` outputs).
- Reading the ablations: T vs B (recall alone), C vs T (acceptance + refinement), C vs C-all (gating), C vs C-noiter (self-prompting), C vs O (what is left).

## 6. Challenges and limitations (all)

- Everything that broke on Colab and how it was fixed (with the C-numbers in `docs/setup_guide.md`).
- Single training seed; thresholds chosen on 100 validation images; corrupted set only if time.

## 7. Next (Phase 4)

- Error analysis by polyp size, brightness and dataset (the bias check), from the per-image CSVs.
- Corrupted-set evaluation if not finished; second seed if time.
