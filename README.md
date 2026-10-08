# Confidence-Aware Self-Prompting for Robust Polyp Segmentation Under Image and Domain Shifts

[![CI](https://github.com/kashishfatima999/Confidence-Aware-Self-Prompting-for-Robust-Polyp-Segmentation/actions/workflows/ci.yml/badge.svg)](https://github.com/kashishfatima999/Confidence-Aware-Self-Prompting-for-Robust-Polyp-Segmentation/actions/workflows/ci.yml)

AI2002 Artificial Intelligence, FAST-NUCES Lahore, Fall 2026. Group **F26-13**, Track A (Research & Development).

| Member | ID | Phase 2 responsibility |
|---|---|---|
| Kashish Fatima | 24L-2605 | Repository, Kvasir-SEG pipeline, shared helpers, tests and CI |
| Khushbakht Sohail | 24L-2604 | CVC-ClinicDB pipeline, dataset statistics, domain-shift comparison |
| Zahra Saeed | 24L-2512 | Baseline code, environment, weights, smoke tests, corruption set |

## 1. The idea

Automatic polyp segmentation in colonoscopy images matters because a missed or badly outlined polyp is a missed lesion. The YOLO-SAM2 pipeline (Mansoori et al., 2024) detects a polyp with YOLOv8 and feeds the detected bounding box as a prompt to a frozen SAM 2, which draws the pixel-level mask. The mask is only as good as the box: a hesitant or badly placed detection gives SAM 2 a weak prompt.

**Hypothesis.** Using the YOLO detection confidence to identify uncertain detections, and refining their bounding-box prompt before it reaches SAM 2, improves segmentation accuracy and robustness compared with passing the raw YOLO box.

```
Baseline   YOLOv8 ──box──────────────────────────────▶ SAM 2 ──▶ mask
Proposed   YOLOv8 ──box + confidence──▶ confidence analysis ──▶ prompt refinement ──▶ SAM 2 ──▶ mask
```

Both pipelines are evaluated with Dice and IoU on the same held-out images, then stressed in two ways that the title promises:

- **Domain shift:** train on Kvasir-SEG, test on CVC-ClinicDB, a dataset the model never saw.
- **Image shift:** blur, brightness/contrast changes and noise applied to the Kvasir-SEG test images.

## 2. Datasets

| Dataset | Role | Size | Source |
|---|---|---|---|
| Kvasir-SEG (Jha et al., 2020) | Primary: train / val / test = 700 / 100 / 200 | 1,000 images with masks | https://datasets.simula.no/kvasir-seg/ |
| CVC-ClinicDB (Bernal et al., 2015) | External test set only; never trained on | 612 images with masks | https://polyp.grand-challenge.org/CVCClinicDB/ |

Both datasets are restricted to research and educational use and are **not** stored in this repository. `docs/data_report.md` records the download URL, date, byte size and SHA-256 of every file we used, and every number produced by the preprocessing.

## 3. Repository layout

```
.
├── configs/kvasir.yaml           YOLO dataset config (set `path` to your machine, see the comment inside)
├── docs/
│   ├── data_report.md            provenance, verification tables, cleaning, split, box analysis
│   ├── decision_log.md           every decision and judgment call: alternatives, reasons, evidence, change log
│   └── setup_guide.md            environment and baseline notes (Member 3)
├── project-division/             the Phase 2 work plan, with the facts checked in the baseline repo
├── splits/kvasir_{train,val,test}.txt   the fixed split, one image id per line (committed)
├── src/data/
│   ├── common.py                 shared helpers: binarise masks, mask → boxes, box IoU, YOLO labels
│   └── prepare_kvasir.py         verify → clean → split → convert Kvasir-SEG, writes a report
├── tests/                        unit tests for the helpers, split-file checks, synthetic end-to-end run
├── .github/workflows/ci.yml      runs the tests on every push and pull request
├── data/                         not in git: raw/ (untouched downloads) and processed/
├── external/                     not in git: clone of the baseline repository, for reading only
└── weights/                      not in git: SAM 2 and YOLO checkpoints
```

## 4. Reproduce the Kvasir-SEG preparation

Needs Python 3.12 and about 300 MB of disk (environment plus data). No GPU.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt      # Windows; use .venv/bin/python elsewhere

# download https://datasets.simula.no/downloads/kvasir-seg.zip and unzip it so that
# data/raw/kvasir-seg/Kvasir-SEG/{images,masks,kavsir_bboxes.json} exist

.venv/Scripts/python src/data/prepare_kvasir.py
.venv/Scripts/python -m unittest discover -s tests -v
```

The script verifies the download (counts, pairing, sizes, readability, empty masks), binarises the masks to PNG, derives one bounding box per polyp from each mask, copies the images into `data/processed/kvasir/{images,masks,labels}/{train,val,test}/`, and writes `reports/verification.json` and `reports/box_comparison.csv`. The split files in `splits/` are reused if present, so every run on every machine produces the same split.

Training YOLOv8 and running SAM 2 need a GPU and are done on Google Colab or Kaggle in Phase 3; the environment for that is documented in `docs/setup_guide.md`.

## 5. How we keep this reproducible

- **Raw data is never edited.** Everything derived lives in `data/processed/`, and the processed images are byte-identical copies of the raw ones.
- **One split, committed.** `splits/` is the only split anyone uses; the tests fail if the files change or stop matching seed 42.
- **Every choice is logged.** `docs/decision_log.md` records each decision with the alternatives considered, the reason, the evidence and its status, plus a dated change log. New work adds an entry in the same commit.
- **Numbers come from files, not memory.** The data report quotes `verification.json`; the final paper quotes the data report.
- **The baseline is cited, not copied.** The baseline repository has no license, so it is cloned outside git for reading and all our code is written from scratch.
- **Tests on every push.** CI compiles the code, runs the helper tests, runs the pipeline on a synthetic mini-dataset, and checks the split files were not modified.

## 6. Status

| Phase | Content | State |
|---|---|---|
| 1 | Problem definition and proposal (`F26-13(Project Proposal).pdf`) | done |
| 2 | Data acquisition, cleaning, split, YOLO format, environment, tests | Kvasir-SEG done; CVC-ClinicDB and environment in progress |
| 3 | Baseline reproduction, YOLO training, confidence-aware prompt refinement | not started |
| 4–6 | Evaluation on both datasets and under corruptions, error and ablation analysis, LaTeX report | not started |

## 7. References

1. M. Mansoori et al. *Self-Prompting Polyp Segmentation in Colonoscopy using Hybrid YOLO-SAM 2 Model.* arXiv:2409.09484, 2024. Code: https://github.com/sajjad-sh33/YOLO_SAM2
2. N. Ravi et al. *SAM 2: Segment Anything in Images and Videos.* arXiv:2408.00714, 2024.
3. D. Jha et al. *Kvasir-SEG: A Segmented Polyp Dataset.* MMM 2020, pp. 451–462. arXiv:1911.07069.
4. J. Bernal et al. *WM-DOVA maps for accurate polyp highlighting in colonoscopy: Validation vs. saliency maps from physicians.* Computerized Medical Imaging and Graphics 43, 2015. (CVC-ClinicDB)
5. J. Choi et al. *Gaussian YOLOv3: An Accurate and Fast Object Detector Using Localization Uncertainty for Autonomous Driving.* arXiv:1904.04620, 2019.
6. Y. He et al. *Bounding Box Regression with Uncertainty for Accurate Object Detection.* arXiv:1809.08545, 2019.
7. Y. Huang et al. *On the Robustness of Segment Anything.* arXiv:2305.16220, 2023.
8. G. Jocher et al. *Ultralytics YOLOv8.* https://github.com/ultralytics/ultralytics

## License

The code in this repository is released under the MIT License (see `LICENSE`). The datasets and the baseline repository keep their own terms and are not part of this license.
