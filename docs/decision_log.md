# F26-13 — Decision Log

**Project:** Confidence-Aware Self-Prompting for Robust Polyp Segmentation Under Image and Domain Shifts

Every decision, judgment call and methodological choice in this project is recorded here, with the date, the alternatives that were considered, the reason, and the evidence. The final report's methodology and "challenges faced" sections are written from this file. Nothing in this log is deleted; a reversed decision gets a new entry that points back to the old one.

**How to add an entry.** Copy the template, give it the next `DL-` number, and add a line to the change log at the end. Do this in the same commit as the change it describes.

```
### DL-NN — Short title
- **Date:** YYYY-MM-DD   **Owner:** Member N
- **Decision:** what was decided, in one or two sentences.
- **Alternatives considered:** what else could have been done.
- **Reason / evidence:** why this option, with numbers or file references.
- **Status:** proposed | accepted | implemented | superseded by DL-MM
- **Affects:** files, scripts or report sections.
```

Status meanings: *proposed* = suggested, not yet agreed by the group; *accepted* = agreed; *implemented* = in the code and data; *superseded* = replaced by a later entry.

---

## Part A — Data and preprocessing

### DL-01 — Raw downloads are never modified
- **Date:** 2026-10-06   **Owner:** Member 1
- **Decision:** Every dataset is kept exactly as downloaded in `data/raw/`. All cleaning writes to `data/processed/`. Images are copied into split folders, never moved.
- **Alternatives considered:** Clean in place (what the baseline's `Creat_YAML.py` does: it moves the original images into train/valid folders).
- **Reason / evidence:** In-place cleaning destroys the download, so a mistake cannot be undone and the pipeline cannot be re-run from scratch. Keeping the raw copy also lets us prove the processed data is derived from it (see `docs/data_report.md` §1.6: all 1,000 processed images are byte-identical to the raw files; the zip's SHA-256 is unchanged after processing).
- **Status:** implemented
- **Affects:** `src/data/prepare_kvasir.py`, `docs/data_report.md`

### DL-02 — Kvasir-SEG split: 700 train / 100 val / 200 test, seed 42
- **Date:** proposed 2026-10-06, accepted 2026-10-08   **Owner:** Member 1
- **Decision:** Image ids are sorted alphabetically, shuffled once with `random.Random(42)`, and cut into 700 / 100 / 200. The lists are saved in `splits/kvasir_{train,val,test}.txt` and committed; the script reuses them if they exist.
- **Alternatives considered:**
  1. The paper's 800 / 200 split. Not reproducible: the baseline code takes the first 800 files of an unsorted `glob` with no seed, so its exact membership is unknown.
  2. A stratified split by polyp size or number of polyps. Rejected for now as added complexity; the random split's imbalance is small (see evidence) and is recorded rather than corrected.
  3. No separate validation split. Rejected: the confidence threshold for "uncertain" detections must be chosen on data that is not the test set.
- **Reason / evidence:** 200 test images keep the 20% test fraction of the paper so test-set size is comparable. Sorting before shuffling makes the split depend only on the seed, not on file-system order. Verified on 2026-10-07: no image in two splits, all 1,000 ids covered, regenerating from seed 42 reproduces the saved files exactly. Mean polyp area is 14.72% (train), 18.17% (val), 16.36% (test) of the image: the validation split has somewhat larger polyps on average because the split is not stratified.
- **Status:** accepted by Member 1 on 2026-10-08 and implemented; Members 2 and 3 use the saved files and must not create their own split.
- **Affects:** `splits/`, `src/data/prepare_kvasir.py`, `docs/data_report.md` §1.5

### DL-03 — CVC-ClinicDB is an external test set only
- **Date:** 2026-10-06   **Owner:** Member 2 (decision shared by the group)
- **Decision:** All 612 CVC-ClinicDB images are used for evaluation only. Nothing is trained or tuned on them.
- **Alternatives considered:** Splitting CVC-ClinicDB into train/test as the paper does (80/20 of the same dataset).
- **Reason / evidence:** The project's question is robustness under domain shift, which needs a dataset the model has never seen. CVC-ClinicDB frames come from video sequences, so a random split would also leak near-identical frames between train and test. Consequence: our CVC-ClinicDB numbers are cross-dataset and are not directly comparable to the paper's in-domain mIoU 0.909 / mDice 0.951.
- **Status:** proposed (to be confirmed by Members 2 and 3)
- **Affects:** `src/data/prepare_clinicdb.py`, `configs/cvc_clinicdb.yaml`, evaluation in Phase 3

### DL-04 — Masks are binarised at 127 and saved as PNG
- **Date:** 2026-10-06, verified 2026-10-07   **Owner:** Member 1
- **Decision:** Each raw mask is converted to greyscale, thresholded (value > 127 = polyp) and saved as a single-channel PNG with values 0 and 255 only.
- **Alternatives considered:** Use the JPG masks as they are, as the baseline's `Test.py` does (`mask / 255` with no threshold).
- **Reason / evidence:** The raw masks are RGB JPGs. Every one of the 1,000 masks contains pixels that are not exactly 0 or 255 (0.73% of all mask pixels). Under `mask / 255` these become values like 0.98 that are neither polyp nor background, which silently distorts Dice/IoU. No raw mask pixel lies between 11 and 244, so a threshold of 127 is unambiguous and any threshold between 11 and 244 would give the same result. PNG is lossless, so the cleaned masks cannot drift again.
- **Status:** implemented
- **Affects:** `src/data/common.py` (`MASK_THRESHOLD`, `load_binary_mask`, `save_binary_mask`), `data/processed/kvasir/masks/`

### DL-05 — Specks: kept in the masks, ignored for boxes
- **Date:** 2026-10-06, counts verified 2026-10-07   **Owner:** Member 1
- **Decision:** A "speck" is a connected region (8-connectivity) smaller than 0.05% of the image area (`MIN_COMPONENT_AREA_FRACTION = 0.0005`). Specks stay in the saved masks but are skipped when bounding boxes are derived.
- **Alternatives considered:**
  1. Remove specks from the masks too. Rejected: that changes the ground truth we evaluate against, and the effect of 143 tiny regions on Dice/IoU is negligible.
  2. Keep a box for every region including specks. Rejected: YOLO would be trained to detect compression noise, and the official box file shows exactly this problem (DL-06, group B).
  3. A different size threshold. 0.05% was chosen after inspecting the component sizes: the 143 specks are all far below it and no real polyp region is near it. The threshold is a named constant so it can be changed and the data regenerated.
- **Reason / evidence:** 143 specks were found across the 1,000 masks; none is a polyp.
- **Status:** implemented
- **Affects:** `src/data/common.py` (`mask_to_boxes`), labels, `docs/data_report.md` §1.3

### DL-06 — YOLO boxes are derived from the masks, not taken from the official box file
- **Date:** 2026-10-06, evidence confirmed 2026-10-07   **Owner:** Member 1
- **Decision:** One box per connected polyp region (above the speck size) is computed from the cleaned mask. The official `kavsir_bboxes.json` is kept only for comparison.
- **Alternatives considered:** Use the official boxes as labels, as the baseline does.
- **Reason / evidence:**
  1. The official file is wrong for 18 of 1,000 images. Group A (10 images): the official box starts at pixel (0, 0) and covers 82–100% of the frame while the polyp region covers 46–90%; extent IoU with the mask-derived box is 0.46–0.90. Group B (8 images): the official file has one extra box around a speck. Full list in `docs/data_report.md` §1.4 and `data/processed/kvasir/reports/box_comparison.csv`.
  2. CVC-ClinicDB ships masks only, so deriving boxes from masks is the only way to give both datasets the same box definition.
  3. On the other 982 images the two sources agree (mean extent IoU 0.987 over all 1,000), so nothing is lost.
- **Status:** implemented
- **Affects:** labels in `data/processed/kvasir/labels/`, `docs/data_report.md` §1.3–1.4

### DL-07 — Box coordinate convention
- **Date:** 2026-10-06, measured 2026-10-07   **Owner:** Member 1
- **Decision:** Everywhere in our code a box is `(xmin, ymin, xmax, ymax)` in pixels with `xmax`/`ymax` exclusive, so a full-image box is `(0, 0, width, height)`. YOLO labels are `class x_center y_center width height`, normalised to 0..1, with class 0 = polyp.
- **Alternatives considered:** Inclusive maxima (last pixel index), which is what the official box file appears to use.
- **Reason / evidence:** Exclusive maxima are what `cv2.connectedComponentsWithStats` returns (`x + w`), so no ±1 adjustments are needed. Measured on the 982 agreeing images: the official `xmin`/`ymin` is 1 px smaller than ours in 957 / 926 cases, and the official `xmax`/`ymax` equals ours in 959 cases and is 1 px smaller in 23 (all at the image border). So the official file agrees with ours to within 1 px per side; the difference does not affect our labels because they come from the masks. Round-trip check: converting every label back to pixels matches the mask box within 0.001 px.
- **Status:** implemented
- **Affects:** `src/data/common.py` (`box_to_yolo`, `write_yolo_label`, `box_iou`)

### DL-08 — Criterion for "official box agrees with mask box"
- **Date:** 2026-10-07   **Owner:** Member 1
- **Decision:** For the comparison report only, an image "agrees" when the IoU between the extent of all official boxes and the extent of all mask-derived boxes is ≥ 0.9 **and** the number of boxes is equal.
- **Alternatives considered:** Per-box matching (Hungarian assignment). Rejected as unnecessary: 952 images have one polyp, and the extent comparison already isolates the two failure modes.
- **Reason / evidence:** The threshold 0.9 separates the data cleanly: 982 images have IoU ≥ 0.985, the 10 group-A images have IoU ≤ 0.8952. This is a reporting judgment call; it does not change any label.
- **Status:** implemented
- **Affects:** `src/data/prepare_kvasir.py` (`BOX_AGREEMENT_IOU`), `reports/box_comparison.csv`

### DL-09 — No resizing or augmentation during preprocessing
- **Date:** 2026-10-07   **Owner:** Member 1
- **Decision:** Processed images and masks keep their original resolution (333 distinct sizes, 332–1920 × 352–1072 px). Resizing is left to YOLO (on load) and SAM 2 (internally); evaluation is done at the original mask resolution.
- **Alternatives considered:** Resize everything to a fixed size (e.g. 640×640) in preprocessing.
- **Reason / evidence:** Resizing masks changes the ground truth (interpolation at the boundary) and would make our Dice/IoU not comparable with the dataset's intended evaluation. Augmentation belongs to training (Phase 3) and must never touch the val/test splits.
- **Status:** implemented
- **Affects:** `data/processed/kvasir/`, evaluation code in Phase 3

### DL-10 — Verification report instead of a notebook
- **Date:** 2026-10-06   **Owner:** Member 1
- **Decision:** `prepare_kvasir.py` writes `verification.json` and `box_comparison.csv`; the planned `01_kvasir_checks.ipynb` is dropped.
- **Alternatives considered:** A Jupyter notebook with the same checks.
- **Reason / evidence:** A script re-runs identically on every machine; the JSON is the single source for the numbers in the data report. Notebook outputs go stale and bloat git diffs.
- **Status:** implemented
- **Affects:** `project-division/Project-Division-Phase-2.md` task 1.3, `docs/data_report.md`

### DL-11 — Checks are done before any output is written
- **Date:** 2026-10-07   **Owner:** Member 1
- **Decision:** `verify_raw()` fully decodes every image and mask, checks pairing, sizes and empty masks, and stops with a list of problems before the processing loop starts. The first version only checked image headers.
- **Alternatives considered:** Keep the lighter header check and rely on the processing loop to catch bad files.
- **Reason / evidence:** The processing loop deletes the previous output first, so a failure half-way leaves a partial dataset. Full decoding costs about 30 s for Kvasir-SEG. Result on 2026-10-07: 0 unreadable files, 0 empty masks, both recorded in `verification.json`.
- **Status:** implemented
- **Affects:** `src/data/prepare_kvasir.py`

## Part B — Repository, environment and tooling

### DL-12 — Repository layout and what git ignores
- **Date:** 2026-10-06, implemented 2026-10-07   **Owner:** Member 1
- **Decision:** `data/`, `weights/`, `external/`, `runs/` and `.venv/` are git-ignored, with root-anchored patterns (`/data/` etc.). Small derived files (`splits/`, `configs/`, reports in `docs/`) are committed. The two PDFs stay in the repository root. Empty folders are not created in advance.
- **Alternatives considered:** Un-anchored `data/` pattern (first version). Rejected after it was found to also ignore `src/data/`. PDFs in `docs/` (first plan). Reverted because the group had already committed them at the root.
- **Reason / evidence:** Datasets and weights are too large for GitHub and CVC-ClinicDB is research-use only. The split and config files are what makes results reproducible, so they must be in git. Git does not track empty folders, so creating them adds nothing.
- **Status:** implemented
- **Affects:** `.gitignore`, `project-division/Project-Division-Phase-2.md` §4

### DL-13 — Local environment: Python 3.12 virtual environment on D:, pinned versions
- **Date:** 2026-10-07   **Owner:** Member 1
- **Decision:** Preprocessing runs locally (CPU) in `.venv/` created with Python 3.12.10, with `numpy==2.5.3`, `Pillow==12.3.0`, `opencv-python-headless==5.0.0.93`. Model packages (ultralytics, torch, SAM 2) are installed only on Colab/Kaggle, by Member 3.
- **Alternatives considered:** Install everything locally. Rejected: no NVIDIA GPU on the local machine and the C: drive is nearly full. Conda. Rejected: larger footprint, not needed for three packages.
- **Reason / evidence:** The pinned versions are the ones the data was actually produced with, so anyone can rebuild identical outputs. The environment is 191 MB / 2,497 files and lives inside the project folder on D:.
- **Status:** implemented
- **Affects:** `requirements.txt`, `docs/data_report.md` §1.7

### DL-14 — Download provenance
- **Date:** 2026-10-07   **Owner:** Member 1
- **Decision:** Every download is recorded with URL, date, byte size and SHA-256 in the data report, and the archive is kept next to the extracted files.
- **Alternatives considered:** Record only the URL.
- **Reason / evidence:** The hash proves which exact file the results come from. Kvasir-SEG: 46,227,172 bytes, SHA-256 `03b30e21…3478f7`, downloaded 2026-10-07. Practical note: Git Bash `curl` on Windows rejects the certificate of `datasets.simula.no`; the Windows system `curl.exe` works.
- **Status:** implemented
- **Affects:** `docs/data_report.md` §1.1

### DL-15 — YOLO dataset config uses a path that must be set per machine
- **Date:** 2026-10-07   **Owner:** Member 1
- **Decision:** `configs/kvasir.yaml` has `path: data/processed/kvasir` with a comment that it must be replaced by the absolute location on the training machine.
- **Alternatives considered:** Hard-code a Colab path. Rejected: it would be wrong for everyone else.
- **Reason / evidence:** Ultralytics can resolve a relative `path` against its own datasets directory rather than the repository, which fails with "images not found". Labels are found by replacing `images` with `labels` in the folder path, which is why the processed layout is `images/<split>` and `labels/<split>`.
- **Status:** implemented; not yet loaded by Ultralytics (first test is Member 3's smoke test)
- **Affects:** `configs/kvasir.yaml`

### DL-16 — The baseline repository is cited, not copied
- **Date:** 2026-10-06   **Owner:** Member 3 (decision shared by the group)
- **Decision:** `sajjad-sh33/YOLO_SAM2` is cloned into the git-ignored `external/` folder for reading and smoke tests. None of its files are copied into our repository; all preprocessing, evaluation and refinement code is written by us and the baseline is cited.
- **Alternatives considered:** Fork the repository and fix its scripts.
- **Reason / evidence:** The repository has no license, so redistribution is not permitted. Its scripts also do not run as published (`!mkdir` notebook syntax in a `.py` file, `/`-split paths, indentation errors and missing imports in `Test.py`), so rewriting is cheaper than fixing.
- **Status:** accepted
- **Affects:** `external/`, `docs/setup_guide.md`

### DL-17 — The baseline's shipped YOLO weights are for smoke tests only
- **Date:** 2026-10-06   **Owner:** Member 3 (decision shared by the group)
- **Decision:** `YOLO_Checkpoints/Kvasir_yolov8m.pt` from the baseline may be used to check that the pipeline runs. All reported results use a YOLO model we train on our own train split.
- **Alternatives considered:** Use the shipped weights for the baseline numbers.
- **Reason / evidence:** The split those weights were trained on is unknown, so our test images may be in their training set (data leakage). The paper and the code also disagree on the model (paper: YOLOv8-m, image size 680; README: `yolov8l.pt`, 640, 50 epochs; shipped: yolov8m).
- **Status:** accepted
- **Affects:** Phase 3 training and evaluation

## Part C — Evaluation plan (recorded now, applied in Phases 3–4)

### DL-18 — Missed polyps count against the score
- **Date:** 2026-10-06   **Owner:** group
- **Decision:** An image with no detection is scored as an empty prediction (Dice = IoU = 0 for that image, and it adds to the union in pooled metrics).
- **Alternatives considered:** Skip such images, as the baseline's `Test.py` does (nothing is added to intersection or union).
- **Reason / evidence:** Skipping hides missed polyps, which are exactly the uncertain cases this project is about. Consequence: our baseline numbers may be lower than the paper's and this must be stated next to every table.
- **Status:** accepted
- **Affects:** evaluation code in Phase 3

### DL-19 — Report both per-image mean and dataset-pooled Dice/IoU
- **Date:** 2026-10-06   **Owner:** group
- **Decision:** Every results table states which of the two it uses. The paper pools pixels over the whole dataset; we report that for comparability and the per-image mean because it weights small polyps equally.
- **Alternatives considered:** One metric only.
- **Reason / evidence:** Pooled metrics are dominated by large polyps; the per-image mean is the standard in the polyp-segmentation literature and is more sensitive to the uncertain, small cases.
- **Status:** accepted
- **Affects:** evaluation code and all results tables

### DL-20 — Image-shift test set built from the Kvasir test split
- **Date:** 2026-10-06   **Owner:** Member 3
- **Decision:** Blur, brightness/contrast change and Gaussian noise at three strengths are applied to the 200 test images only, giving the "image shifts" promised in the title.
- **Alternatives considered:** Drop image shifts and cover only the dataset shift (CVC-ClinicDB).
- **Reason / evidence:** The approved title promises both kinds of shift. Corruptions never touch training data, so they measure robustness, not learned invariance. To be confirmed with the instructor if in doubt.
- **Status:** proposed
- **Affects:** `src/data/make_corruptions.py`, `data/processed/kvasir-corrupted/`

## Part D — Working process

### DL-21 — Git workflow
- **Date:** 2026-10-08   **Owner:** Member 1
- **Decision:** All members work on the `project-divison` branch during Phase 2 and push it so the skeleton, helpers and split files are shared; one pull request merges it into `main` at the end of the phase. Every commit is made by a group member by hand, with a short message naming what changed and why. Every change to data, code or decisions gets an entry in this log in the same commit.
- **Alternatives considered:** One branch per member. Rejected for Phase 2 because Members 2 and 3 depend on Member 1's files immediately; separate branches can be used in Phase 3 for the method and the baseline.
- **Reason / evidence:** The division plan's dependency list: Members 2 and 3 need tasks 1.1 and 1.7 before they can start.
- **Status:** accepted
- **Affects:** all commits

---

## Change log

Dated record of what changed and why. One line per change; the `DL-` number links it to the decision.

| Date | Change | Why / decision |
|---|---|---|
| 2026-10-06 | `project-division/Project-Division-Phase-2.md` created: task division, baseline facts, decisions D1–D5. | Phase 2 kickoff. |
| 2026-10-06 | PDFs moved to the repository root (commit `295172a`). | DL-12 |
| 2026-10-07 | `.gitignore` and `requirements.txt` created; division file updated (task 1.3 outputs a report, PDFs in root, `.venv/` ignored). | DL-10, DL-12 |
| 2026-10-07 | `.gitignore` patterns anchored to the root after `data/` was found to ignore `src/data/`. | DL-12 |
| 2026-10-07 | Kvasir-SEG downloaded (46,227,172 bytes, SHA-256 verified) into `data/raw/kvasir-seg/`. | DL-14 |
| 2026-10-07 | `.venv/` created on D: with numpy 2.5.3, Pillow 12.3.0, opencv-python-headless 5.0.0.93. | DL-13 |
| 2026-10-07 | `src/data/common.py` and `src/data/prepare_kvasir.py` added. | DL-04 to DL-08 |
| 2026-10-07 | `verify_raw()` changed to fully decode every file and check empty masks before writing output; counts added to `verification.json`. | DL-11 |
| 2026-10-07 | Read-only verification run passed (1,000 / 1,000 / 1,000; 0 unreadable; 0 empty). | DL-11 |
| 2026-10-07 | Full `prepare_kvasir.py` run: split files, 1,000 image copies, 1,000 PNG masks, 1,000 labels, 2 reports. Independent check of every output passed. | DL-01, DL-02, DL-04 to DL-09 |
| 2026-10-07 | `configs/kvasir.yaml` written; package versions pinned in `requirements.txt`. | DL-13, DL-15 |
| 2026-10-07 | `docs/data_report.md` created with the Kvasir-SEG section. | — |
| 2026-10-07 | `common.py` docstring corrected: the official box file agrees with our convention to within 1 px per side, not exactly. | DL-07 |
| 2026-10-08 | Member 1 committed the ten Phase 2 files on `project-divison` (one file per commit, `e546895`..`339130a`). | DL-21 |
| 2026-10-08 | Split D1 accepted by Member 1. | DL-02 |
| 2026-10-08 | This decision log created; division file updated (machine facts, decision status, link to this log). | DL-21 |
