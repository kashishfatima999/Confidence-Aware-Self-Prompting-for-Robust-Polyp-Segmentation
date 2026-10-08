# F26-13 — Project Division for Phase 2 (Data Acquisition & Setup)

**Project:** Confidence-Aware Self-Prompting for Robust Polyp Segmentation Under Image and Domain Shifts
**Track:** A (Research & Development)
**Phase 2 in the guidelines:** "Gather data via scraping, APIs, or repositories. Clean and preprocess as required."
**Rubric weight:** Data Preprocessing / Setup = 15% (Track A)

| Member | Name | Owns |
|---|---|---|
| Member 1 | Kashish Fatima (24L-2605) | Repo structure + Kvasir-SEG (primary dataset) pipeline |
| Member 2 | Khushbakht Sohail (24L-2604) | CVC-ClinicDB (external dataset) pipeline + dataset statistics |
| Member 3 | Zahra Saeed (24L-2512) | Baseline code + environment setup + sanity checks |

---

## 1. What Phase 2 must produce

By the end of Phase 2 the group should be able to show:

1. Both datasets downloaded from a cited source, with counts verified.
2. Cleaned data: every image has a matching mask, masks are strictly binary, bounding boxes exist for every image.
3. A fixed, saved train / val / test split for Kvasir-SEG (same split for every member, every run).
4. Data converted into the format the baseline needs (YOLO label files + dataset YAML).
5. A working environment where YOLOv8 and SAM 2 load and run on a few images.
6. A short data report (numbers, tables, a few figures) that goes into the final LaTeX report.

Phase 2 does **not** include training YOLO, running the full baseline evaluation, or building the refinement method. Those are Phase 3.

---

## 2. Facts checked in the baseline repo and paper

These were checked directly in `sajjad-sh33/YOLO_SAM2` (`Test.py`, `Creat_YAML.py`, README) and the arXiv paper. They change how Phase 2 should be done.

| # | What was found | Why it matters for us |
|---|---|---|
| 1 | The baseline repo has **no license**. | Do not copy its files into our repo. Clone it separately, cite it, and write our own scripts. |
| 2 | `Creat_YAML.py` uses `!mkdir` (notebook syntax inside a `.py` file) and splits paths on `/`. | It does not run as plain Python and breaks on Windows paths. We write our own preprocessing instead of fixing theirs. |
| 3 | `Creat_YAML.py` splits with unsorted `glob` and no seed (first 800 files = train, rest = valid), and **moves** the original images. | The paper's split cannot be reproduced exactly, and the raw download is destroyed. We need our own seeded split saved to files, and raw data kept untouched. |
| 4 | The repo ships trained YOLO weights (`YOLO_Checkpoints/Kvasir_yolov8m.pt`) but the split used to train them is unknown. | Using these weights on our test split risks data leakage (test images may have been in their training set). For clean results we train YOLO ourselves on our split in Phase 3. Their weights are fine for smoke tests only. |
| 5 | Paper says YOLOv8 **medium**, image size 680. README says `yolov8l.pt`, image size 640, 50 epochs. Shipped weights are `yolov8m`. | Paper and code disagree. Record this now; it goes into the reproducibility section of the report. |
| 6 | `Test.py` calls YOLO with `conf=0.5` hard-coded. | Detections below 0.5 confidence never reach SAM 2. Our "uncertain detections" are exactly the ones this threshold throws away or barely keeps, so this number is central to our research design. |
| 7 | In `Test.py`, an image with zero detections is skipped: nothing is added to intersection or union. | Missed polyps do not lower the reported IoU/Dice. Our evaluation must count them (empty prediction), so our baseline numbers may be lower than the paper's. |
| 8 | Metrics are computed by summing pixels over the whole dataset, not averaging per image. | We should report per-image mean Dice/IoU as well, and say clearly which one each table uses. |
| 9 | Kvasir masks are JPG and are read as `mask / 255` without thresholding. | JPEG artifacts give values like 0.98 that are not counted as a match. Masks must be binarized (threshold at 127) during preprocessing. |
| 10 | SAM 2 is **frozen** (`sam2_hiera_large.pt`); only YOLO is trained. | No SAM training needed. Compute need is one YOLOv8 fine-tune plus SAM 2 inference. |
| 11 | Paper reports Kvasir-SEG mIoU 0.764 / mDice 0.866 and CVC-ClinicDB mIoU 0.909 / mDice 0.951, each with an 80/20 split **of the same dataset**. | The paper's CVC-ClinicDB number is in-domain (trained on ClinicDB). Ours is cross-dataset (trained on Kvasir only), so it will be lower and is not directly comparable. |
| 12 | `Test.py` as published has indentation errors and missing imports (`argparse`, `matplotlib`). | It will not run as-is. Member 3 documents this; the rewrite is Phase 3 work. |

Local machine check (Member 1's laptop): no Python on PATH, no NVIDIA GPU, 13 GB free disk, repo sits inside OneDrive.
**Consequence:** light preprocessing can run locally after installing Python, but YOLO training and SAM 2 must run on Google Colab or Kaggle Notebooks (free GPU).

---

## 3. Decisions the group must agree on before starting

| # | Decision | Recommendation | Reason |
|---|---|---|---|
| D1 | Kvasir-SEG split | **700 train / 100 val / 200 test**, seed 42, saved as text files in `splits/` | Test size of 200 matches the paper's 20%. The val set is needed later to choose the confidence threshold without touching the test set. |
| D2 | CVC-ClinicDB usage | **All 612 images as test only.** No training on it. | It is our external dataset. Its frames come from video sequences, so splitting it randomly would leak near-identical frames anyway. |
| D3 | Where compute runs | Colab or Kaggle Notebooks with GPU | No local GPU. |
| D4 | Where data lives | `data/` folder that is git-ignored; a shared Google Drive folder for the team | Datasets and weights must not be pushed to GitHub (size, and CVC-ClinicDB is research-use only). |
| D5 | "Image shifts" in our title | Small corruption set (blur, brightness, noise) made from the Kvasir test split | The approved title promises image shifts, but the proposal body only describes the dataset shift. This is the cheapest way to cover it. Confirm with the instructor if unsure. |

---

## 4. Repository folder structure

```
Confidence-Aware-Self-Prompting-for-Robust-Polyp-Segmentation/
├── README.md
├── requirements.txt
├── .gitignore                     # ignores data/, .venv/, external/, weights/, runs/
├── F26-13(Project Proposal).pdf
├── Project Deliverables and guidelines.pdf
├── project-division/
│   └── Project-Division-Phase-2.md
├── docs/
│   ├── data_report.md             # Member 2 (stats) + all members add their section
│   └── setup_guide.md             # Member 3
├── data/                          # NOT in git
│   ├── raw/
│   │   ├── kvasir-seg/            # untouched download
│   │   └── cvc-clinicdb/          # untouched download
│   └── processed/
│       ├── kvasir/                # images/, masks/, labels/ per split + reports/
│       ├── cvc-clinicdb/          # images/, masks/, labels/ (test only)
│       └── kvasir-corrupted/      # image-shift test set
├── splits/                        # IN git — small text files
│   ├── kvasir_train.txt
│   ├── kvasir_val.txt
│   └── kvasir_test.txt
├── configs/
│   ├── kvasir.yaml                # YOLO dataset config
│   └── cvc_clinicdb.yaml
├── src/
│   └── data/
│       ├── prepare_kvasir.py      # Member 1
│       ├── prepare_clinicdb.py    # Member 2
│       ├── make_corruptions.py    # Member 3
│       └── common.py              # shared helpers (binarize mask, mask→boxes, YOLO label writer)
├── notebooks/
│   ├── 02_dataset_statistics.ipynb# Member 2
│   └── 03_environment_smoke_test.ipynb  # Member 3
├── external/                      # NOT in git — clone of the baseline repo
└── weights/                       # NOT in git — SAM 2 + YOLO checkpoints
```

---

## 5. Task division

### Member 1 — Kashish Fatima: Repo structure + Kvasir-SEG pipeline

| # | Task | Output |
|---|---|---|
| 1.1 | Create `.gitignore` and `requirements.txt`; folders in section 4 are created as their first file is added. The two PDFs stay in the repo root. | Repo skeleton on the `project-divison` branch |
| 1.2 | Download Kvasir-SEG from Simula (`https://datasets.simula.no/downloads/kvasir-seg.zip`, 46 MB) into `data/raw/kvasir-seg/`. Record source, date, file size. | Raw dataset + entry in data report |
| 1.3 | Verify: 1,000 images, 1,000 masks, names match one-to-one, image and mask sizes match, no unreadable files, no empty masks, bounding-box file covers all images. | Verification report written by `src/data/prepare_kvasir.py` to `data/processed/kvasir/reports/` (`verification.json` + `box_comparison.csv`); replaces the planned notebook |
| 1.4 | Clean: binarize masks (threshold 127) and save as PNG; check each box against its mask (box derived from mask vs box in the provided file) and list mismatches. | `data/processed/kvasir/masks/` + mismatch list |
| 1.5 | Create the seeded 700/100/200 split and save the three text files in `splits/`. | `splits/kvasir_*.txt` (committed) |
| 1.6 | Convert to YOLO format: one `.txt` label per image (`0 x_center y_center width height`, normalised), copy images into split folders (copy, never move), write `configs/kvasir.yaml`. | `data/processed/kvasir/` + `configs/kvasir.yaml` |
| 1.7 | Write `src/data/common.py` helpers so Members 2 and 3 use the same mask and label code. | Shared helpers |
| 1.8 | Write the Kvasir section of `docs/data_report.md`. | Report section |

### Member 2 — Khushbakht Sohail: CVC-ClinicDB pipeline + dataset statistics

| # | Task | Output |
|---|---|---|
| 2.1 | Download CVC-ClinicDB. The official Grand Challenge page needs a login, so use the Kaggle mirror the baseline uses (`balraj98/cvcclinicdb`) and cite the original (Bernal et al., 2015). Record the license: research and education use only. | `data/raw/cvc-clinicdb/` + entry in data report |
| 2.2 | Verify: 612 images, 612 masks, names match, sizes match (expected 384×288), no unreadable files. Use the PNG version. | Verification table |
| 2.3 | Clean: binarize masks; derive bounding boxes from masks using connected components (one box per polyp; some frames have more than one). | `data/processed/cvc-clinicdb/masks/` |
| 2.4 | Convert to YOLO labels using Member 1's helpers; all 612 images go to a single `test` folder; write `configs/cvc_clinicdb.yaml`. | `data/processed/cvc-clinicdb/` + config |
| 2.5 | Statistics for **both** datasets in `02_dataset_statistics.ipynb`: image resolution, polyp area as % of image, polyps per image, box aspect ratio, mean brightness/contrast. | Notebook + 3–4 figures |
| 2.6 | Write a Kvasir vs CVC-ClinicDB comparison table. This is our evidence that a domain shift exists. | Table in `docs/data_report.md` |
| 2.7 | Write the CVC-ClinicDB section of `docs/data_report.md`. | Report section |

### Member 3 — Zahra Saeed: Baseline code + environment + sanity checks

| # | Task | Output |
|---|---|---|
| 3.1 | Clone the baseline into `external/YOLO_SAM2` (git-ignored). Record the commit hash. Read `Test.py` and `Creat_YAML.py` and confirm the findings in section 2. | Commit hash + notes in `docs/setup_guide.md` |
| 3.2 | Build the working environment on Colab or Kaggle: `ultralytics`, PyTorch, SAM 2. Pin the versions in `requirements.txt` (with Member 1). | Notebook that installs everything from scratch |
| 3.3 | Download `sam2_hiera_large.pt` (the repo's `checkpoints/download_ckpts.sh`) and `yolov8m.pt` into `weights/`. | Weights + download instructions |
| 3.4 | Smoke test in `03_environment_smoke_test.ipynb`: run YOLO then SAM 2 on 5 Kvasir images, print the box coordinates and **confidence score** for each detection, and show the mask. Use the repo's shipped Kvasir weights for this test only. | Notebook showing the pipeline runs end to end |
| 3.5 | Visual check of Members 1 and 2's output: draw image + mask + YOLO box overlay for 20 random images per dataset and confirm boxes sit on the polyps. | Overlay figure for the report |
| 3.6 | Write `src/data/make_corruptions.py`: blur, brightness/contrast change, and Gaussian noise at 3 strengths, applied to the Kvasir **test** split only (decision D5). | `data/processed/kvasir-corrupted/` |
| 3.7 | Write `docs/setup_guide.md`: exact steps to recreate the environment, plus the list of problems found in the baseline code. | Setup guide ("challenges faced" material for the final report) |

### Balance check

| | Member 1 | Member 2 | Member 3 |
|---|---|---|---|
| Dataset download + verification | Kvasir (1,000) | ClinicDB (612) | — |
| Cleaning + format conversion | Kvasir + shared helpers | ClinicDB + box derivation | Corruption set |
| Analysis | Box-vs-mask check | Statistics for both datasets | Overlay check for both datasets |
| Setup | Repo skeleton | — | Environment + baseline + weights |
| Writing | Kvasir section | ClinicDB section + comparison | Setup guide |

---

## 6. Order of work

```
Day 1      All three: agree on decisions D1–D5.
           Member 1: task 1.1 (skeleton) and push  →  unblocks everyone.
Days 2–4   Member 1: 1.2–1.5, 1.7      Member 2: 2.1–2.3      Member 3: 3.1–3.3
Days 5–6   Member 1: 1.6               Member 2: 2.4–2.5      Member 3: 3.4, 3.6
Day 7      Member 1: 1.8               Member 2: 2.6–2.7      Member 3: 3.5, 3.7
           All three: read each other's sections; one pull request into main.
```

Dependencies:
- Members 2 and 3 need the skeleton (1.1) and helpers (1.7) from Member 1 first.
- Member 3's overlay check (3.5) and corruption set (3.6) need Member 1's processed Kvasir data and split files.

---

## 7. Do / Don't

**Do**
- Keep `data/raw/` untouched. Always copy, never move.
- Use the split files in `splits/` everywhere. Nobody creates their own split.
- Cite every external source: baseline repo, both datasets, SAM 2, Ultralytics.
- Write down every problem you hit. The final report needs a "challenges faced" section.

**Don't**
- Don't push datasets, weights, or the baseline repo's code to GitHub.
- Don't train YOLO on CVC-ClinicDB.
- Don't use the baseline's shipped Kvasir weights for any reported result.
- Don't apply corruptions to training data; they are for testing only.
- Don't look at test-split results when choosing thresholds later; use the val split.
