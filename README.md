# Confidence-Aware Self-Prompting for Robust Polyp Segmentation Under Image and Domain Shifts

[![CI](https://github.com/kashishfatima999/Confidence-Aware-Self-Prompting-for-Robust-Polyp-Segmentation/actions/workflows/ci.yml/badge.svg)](https://github.com/kashishfatima999/Confidence-Aware-Self-Prompting-for-Robust-Polyp-Segmentation/actions/workflows/ci.yml)

AI2002 Artificial Intelligence (BDS-5A), FAST-NUCES Lahore, Fall 2026. Group **F26-13**, Track A (Research & Development). Instructor: Dr. Hajra Waheed.

| Member | ID | Phase 2 responsibility |
|---|---|---|
| Kashish Fatima (lead) | 24L-2605 | Repository, Kvasir-SEG pipeline, shared helpers, tests and CI, submission package |
| Khushbakht Sohail | 24L-2604 | CVC-ClinicDB pipeline, dataset statistics, domain-shift comparison |
| Zahra Saeed | 24L-2512 | Baseline code review, Colab environment, weights, smoke test, overlay check, corruption set |

## 1. The idea

Automatic polyp segmentation in colonoscopy images matters because a missed or badly outlined polyp is a missed lesion. The YOLO-SAM2 pipeline (Mansoori et al., 2024) detects a polyp with YOLOv8 and feeds the detected bounding box as a prompt to a frozen SAM 2, which draws the pixel-level mask. The mask is only as good as the box: a hesitant or badly placed detection gives SAM 2 a weak prompt.

**Hypothesis.** Using the YOLO detection confidence to identify uncertain detections, and refining their bounding-box prompt before it reaches SAM 2, improves segmentation accuracy and robustness compared with passing the raw YOLO box.

```
Baseline   YOLOv8 ──box──────────────────────────────▶ SAM 2 ──▶ mask
Proposed   YOLOv8 ──box + confidence──▶ confidence analysis ──▶ prompt refinement ──▶ SAM 2 ──▶ mask
```

Both pipelines are evaluated with Dice and IoU on the same held-out images, then stressed in the two ways the title promises:

- **Domain shift:** train on Kvasir-SEG, test on CVC-ClinicDB, a dataset the model never saw.
- **Image shift:** blur, brightness/contrast changes and noise applied to the Kvasir-SEG test images.

## 2. Datasets

| Dataset | Role | Size | Source |
|---|---|---|---|
| Kvasir-SEG (Jha et al., 2020) | Primary: train / val / test = 700 / 100 / 200 | 1,000 images with masks | https://datasets.simula.no/kvasir-seg/ |
| CVC-ClinicDB (Bernal et al., 2015) | External test set only; never trained on | 612 frames with masks, 384 × 288 | https://polyp.grand-challenge.org/CVCClinicDB/ (Kaggle mirror `balraj98/cvcclinicdb`) |
| Kvasir-SEG corrupted (ours) | Image-shift test set | 9 settings × 200 test images | built by `src/data/make_corruptions.py` |

Both datasets are restricted to research and educational use and are **not** stored in this repository. `docs/data_report.md` records the download source, date, size and SHA-256 (where stable) of every file used, and every number produced by the preprocessing.

## 3. Where to find things

| You want | Look at |
|---|---|
| The Phase 2 report and notebook handed in | `docs/submissions/phase2/F26-13.pdf`, `F26-13.ipynb` |
| The Phase 3 report and notebook for submission | `docs/submissions/phase3/F26-13.pdf` (source `report_source.html`), `F26-13.ipynb` |
| The Phase 3 results and report draft | `docs/phase3_report.md`, `docs/tables/phase3/`, `notebooks/04_phase3_colab.ipynb` (executed) |
| The course requirements and a checklist of how each is met | `docs/requirements/` |
| The approved proposal | `docs/proposal/F26-13_Project_Proposal.pdf` |
| Every number about the data (provenance, verification, split, statistics) | `docs/data_report.md` |
| Why each choice was made, with alternatives and evidence | `docs/decision_log.md` |
| How to set up Colab, the baseline's problems, the smoke test | `docs/setup_guide.md` |
| Figures and tables | `docs/figures/`, `docs/tables/` (Phase 3: `docs/tables/phase3/`) |
| Who does what | `project-division/Project-Division-Phase-2.md` |

## 4. Repository layout

```
.
├── configs/                      YOLO dataset configs (kvasir.yaml, cvc_clinicdb.yaml); set `path` per machine
├── docs/                         all documents: reports, decision log, setup guide, requirements, proposal,
│                                 figures/, tables/, submissions/
├── notebooks/                    02_dataset_statistics.ipynb (EDA), 03_environment_smoke_test.ipynb (Colab)
├── project-division/             the Phase 2 work plan, with the facts checked in the baseline repo
├── splits/                       kvasir_{train,val,test}.txt: the fixed split, one image id per line
├── src/data/
│   ├── common.py                 shared helpers: binarise masks, mask → boxes, box IoU, YOLO labels
│   ├── prepare_kvasir.py         verify → clean → split → convert Kvasir-SEG, writes a report
│   ├── prepare_clinicdb.py       verify → clean → convert CVC-ClinicDB (test only), writes a report
│   └── make_corruptions.py       build the image-shift test set from the Kvasir test split
├── tests/                        34 tests: helpers, split files, synthetic end-to-end runs of all three scripts
├── .github/workflows/ci.yml      runs the tests on every push and pull request
├── requirements.txt              local CPU environment (numpy, Pillow, OpenCV)
├── requirements-colab.txt        model packages for Colab (ultralytics; SAM 2 pinned by commit)
├── data/                         not in git: raw/ (untouched downloads) and processed/
├── external/                     not in git: clone of the baseline repository, for reading only
└── weights/                      not in git: SAM 2 and YOLO checkpoints
```

## 5. Reproduce the data preparation

Needs Python 3.12 and about 500 MB of disk. No GPU.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt      # Windows; use .venv/bin/python elsewhere
.venv/Scripts/python -m unittest discover -s tests -v        # 34 tests, about 10 s, no data needed

# Kvasir-SEG: download https://datasets.simula.no/downloads/kvasir-seg.zip and unzip it into data/raw/kvasir-seg/
.venv/Scripts/python src/data/prepare_kvasir.py

# CVC-ClinicDB: download the Kaggle mirror balraj98/cvcclinicdb and put its PNG/ folder at data/raw/cvc-clinicdb/PNG/
.venv/Scripts/python src/data/prepare_clinicdb.py

# Image-shift test set (needs the processed Kvasir data)
.venv/Scripts/python src/data/make_corruptions.py
```

Each script verifies its input (counts, pairing, sizes, readability, empty masks) and stops before writing anything if a check fails. Outputs go to `data/processed/<dataset>/{images,masks,labels}/<split>/` plus a `reports/verification.json` whose numbers are the ones quoted in `docs/data_report.md`. The split files in `splits/` are reused if present, so every run on every machine produces the same split.

Everything that needs a GPU (statistics on both datasets, YOLO + SAM 2 smoke test, and later training) runs on Google Colab: open `docs/submissions/phase2/F26-13.ipynb` or `notebooks/03_environment_smoke_test.ipynb` with a T4 runtime and *Run all* (about 10 minutes). `docs/setup_guide.md` explains each step.

## 6. Phase 3: baseline and method (runs done, report in draft)

Decisions DL-30 to DL-38. Everything below runs on Colab (`notebooks/04_phase3_colab.ipynb`, executed copy committed); the harness and the method logic are tested without a GPU. Full numbers: `docs/tables/phase3/results.md`; reading: `docs/phase3_report.md`.

| Per-image Dice (misses = 0) | Kvasir-SEG test (200) | CVC-ClinicDB (612, never trained on) |
|---|---|---|
| B: YOLOv8-m ≥ 0.5 → SAM 2 (baseline) | 0.831 (18 missed) | 0.677 (144 missed) |
| T: threshold 0.25, nothing else | 0.893 (1) | 0.759 (75) |
| C: CASP v1 | **0.891** (1) | **0.757** (75) |
| O: ground-truth boxes → SAM 2 | 0.952 (0) | 0.945 (0) |

CASP beats the baseline on both test sets and on all nine corrupted settings; the ablation shows the gain is the confidence gate (T = C), while the refinement and acceptance steps are neutral in v1 (DL-38). The detector is the whole remaining gap.

```bash
python src/train_yolo.py                                                   # YOLOv8-m, recipe in configs/yolo_train.yaml
python src/pipeline/detect.py --weights runs/yolo/yolov8m_kvasir_s42/weights/best.pt --dataset kvasir_test
python src/pipeline/run_experiment.py --config configs/experiments/baseline.yaml --dataset kvasir_test
python src/pipeline/run_experiment.py --config configs/experiments/casp.yaml --dataset kvasir_test --save-masks
python src/eval/summarize.py                                               # -> docs/tables/phase3/results.{csv,md}
```

| Piece | Where |
|---|---|
| Metrics (per-image + pooled Dice/IoU, misses count as 0, bootstrap CI, box P/R) | `src/eval/metrics.py` |
| Detection cache (YOLO once per dataset at conf 0.05) | `src/pipeline/detect.py` |
| SAM 2 wrapper (embed once, prompt many, returns predicted IoU) | `src/pipeline/sam2_prompter.py` |
| **CASP v1**: gate by confidence, candidate prompts, SAM-score selection, self-prompt iteration, accept/reject | `src/pipeline/refine.py` |
| Experiments B, T, C, C-all, C-noiter, O | `configs/experiments/*.yaml`, `src/pipeline/run_experiment.py` |
| Results tables | `src/eval/summarize.py` → `docs/tables/phase3/` |
| Same runs on Kaggle, for long jobs that must survive a closed browser | `notebooks/05_kaggle_driver.ipynb` (setup guide C9) |
| Report | `docs/phase3_report.md` |

## 7. How we keep this reproducible

- **Raw data is never edited.** Everything derived lives in `data/processed/`; processed images are byte-identical copies of the raw ones.
- **One split, committed.** `splits/` is the only split anyone uses; the tests fail if the files change or stop matching seed 42.
- **Every choice is logged.** `docs/decision_log.md` records each decision with the alternatives considered, the reason, the evidence and its status, plus a dated change log. New work adds an entry in the same commit.
- **Numbers come from files, not memory.** The data report quotes `verification.json`; the reports quote the data report.
- **The baseline is cited, not copied.** The baseline repository has no license, so it is cloned outside git for reading and all our code is written from scratch.
- **Tests on every push.** CI compiles the code, runs the helper tests, runs all three preparation scripts on synthetic mini-datasets, and checks the split files were not modified.
- **One branch per member, merged by pull request** into `main` after another member has read it (DL-21).

## 8. Status

| Phase | Content | State |
|---|---|---|
| 1 | Problem definition and proposal (`docs/proposal/`) | done |
| 2 | Data acquisition, cleaning, split, YOLO format, environment, tests, image-shift set | done; submitted (`docs/submissions/phase2/`) |
| 3 | Baseline reproduction, YOLO training, confidence-aware prompt refinement (CASP v1) | runs done (2026-10-16); tables in `docs/tables/phase3/`, report draft `docs/phase3_report.md`, record in `project-division/Project-Division-Phase-3.md` §7 |
| 4–6 | Evaluation on both datasets and under corruptions, error and ablation analysis, LaTeX report | not started |

## 9. References

1. M. Mansoori et al. *Self-Prompting Polyp Segmentation in Colonoscopy using Hybrid YOLO-SAM 2 Model.* arXiv:2409.09484, 2024. Code: https://github.com/sajjad-sh33/YOLO_SAM2 (commit `d86e375`, no license).
2. N. Ravi et al. *SAM 2: Segment Anything in Images and Videos.* arXiv:2408.00714, 2024.
3. D. Jha et al. *Kvasir-SEG: A Segmented Polyp Dataset.* MMM 2020, pp. 451–462. arXiv:1911.07069.
4. J. Bernal et al. *WM-DOVA maps for accurate polyp highlighting in colonoscopy: Validation vs. saliency maps from physicians.* Computerized Medical Imaging and Graphics 43:99–111, 2015. (CVC-ClinicDB)
5. J. Choi et al. *Gaussian YOLOv3: An Accurate and Fast Object Detector Using Localization Uncertainty for Autonomous Driving.* arXiv:1904.04620, 2019.
6. Y. He et al. *Bounding Box Regression with Uncertainty for Accurate Object Detection.* arXiv:1809.08545, 2019.
7. Y. Huang et al. *On the Robustness of Segment Anything.* arXiv:2305.16220, 2023.
8. G. Jocher et al. *Ultralytics YOLOv8.* https://github.com/ultralytics/ultralytics

## License

The code in this repository is released under the MIT License (see `LICENSE`). The datasets and the baseline repository keep their own terms and are not part of this license.
