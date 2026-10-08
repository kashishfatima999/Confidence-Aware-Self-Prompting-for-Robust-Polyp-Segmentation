# F26-13 — Data Report (Phase 2)

**Project:** Confidence-Aware Self-Prompting for Robust Polyp Segmentation Under Image and Domain Shifts

This report records where each dataset came from, how it was checked and cleaned, and the exact numbers produced. Every number in section 1 comes from `data/processed/kvasir/reports/verification.json` and `box_comparison.csv`, written by `src/data/prepare_kvasir.py` on 2026-10-07.

---

## 1. Kvasir-SEG (primary dataset) — Member 1

### 1.1 Source

| | |
|---|---|
| Dataset | Kvasir-SEG: A Segmented Polyp Dataset (Jha et al., 2020) |
| Dataset page | https://datasets.simula.no/kvasir-seg/ |
| Download URL | https://datasets.simula.no/downloads/kvasir-seg.zip |
| Downloaded | 2026-10-07 |
| File size | 46,227,172 bytes |
| SHA-256 | `03b30e21d584e04facf49397a2576738fd626815771afbbf788f74a7153478f7` |
| Terms of use | Research and educational purposes only; commercial use needs written permission (stated on the dataset page) |
| Stored in | `data/raw/kvasir-seg/` (not in git) |

**Citation:** D. Jha, P. H. Smedsrud, M. A. Riegler, P. Halvorsen, T. de Lange, D. Johansen, H. D. Johansen. "Kvasir-SEG: A Segmented Polyp Dataset." *International Conference on Multimedia Modeling (MMM)*, pp. 451–462, Springer, 2020. arXiv:1911.07069.

The zip contains `Kvasir-SEG/images/` (1,000 JPG), `Kvasir-SEG/masks/` (1,000 JPG) and `Kvasir-SEG/kavsir_bboxes.json` (the file name is misspelled in the original download). The raw folder is never modified; all outputs go to `data/processed/kvasir/`.

### 1.2 Verification of the raw download

| Check | Result |
|---|---|
| Images | 1,000 |
| Masks | 1,000 |
| Entries in the official bounding-box file | 1,000 |
| Images without a mask / masks without an image | 0 / 0 |
| Images without a box entry / box entries without an image | 0 / 0 |
| Image size ≠ mask size | 0 |
| Image size ≠ size recorded in the box file | 0 |
| Unreadable files (every image and mask fully decoded) | 0 |
| Empty masks after thresholding | 0 |
| Distinct resolutions | 333 |
| Width range | 332 – 1920 px |
| Height range | 352 – 1072 px |

The four most common resolutions are 622×530 (78 images), 626×547 (66), 626×546 (61) and 622×529 (56). The images are not resized during preprocessing; YOLO and SAM 2 resize on load.

### 1.3 Cleaning

**Masks.** The raw masks are RGB JPG files, so they are not strictly binary: in every one of the 1,000 masks, JPEG compression leaves pixels that are close to, but not exactly, 0 or 255 (0.73% of all mask pixels; for example 250 instead of 255). The baseline code reads masks as `mask / 255` without thresholding, so such a pixel becomes a value like 0.98 that is neither polyp nor background. No mask pixel lies between 11 and 244, so thresholding is unambiguous. We convert each mask to greyscale, threshold at 127 (value > 127 = polyp) and save it as a single-channel PNG with values 0 and 255 only.

**Specks.** After thresholding, the masks contain 143 small isolated regions in addition to the real polyp regions. We define a speck as a connected region smaller than 0.05% of the image area (`MIN_COMPONENT_AREA_FRACTION = 0.0005`, 8-connectivity). Specks are **kept in the saved masks**, so the ground truth used for Dice/IoU is the thresholded original, but they are **ignored when deriving bounding boxes**, so YOLO is not trained on boxes around compression noise.

**Bounding boxes.** We derive the YOLO boxes from the cleaned masks (one box per connected region above the speck size) instead of using the official box file. Reasons:

1. The official file is wrong for 18 of the 1,000 images (table below).
2. CVC-ClinicDB provides masks only, so deriving boxes from masks gives both datasets the same box definition.

Box convention in our code: `(xmin, ymin, xmax, ymax)` in pixels, with `xmax`/`ymax` exclusive.

### 1.4 Official boxes compared with mask-derived boxes

For each image we compare the overall extent of the official boxes with the overall extent of the mask-derived boxes (IoU of the two extents), and the number of boxes. An image "agrees" when the extent IoU is at least 0.9 and the box counts are equal.

| | Images |
|---|---|
| Agree | 982 |
| Extent IoU below 0.9 | 10 |
| Different number of boxes | 8 |
| Mean extent IoU over all 1,000 images | 0.9874 |

On the 982 agreeing images the official box is typically 1 px larger on the left and top than the mask-derived box, so even agreeing images have an extent IoU slightly below 1.

**Group A — official box far larger than the polyp region (10 images).** In all ten, the official box starts at pixel (0, 0) and covers 82–100% of the frame, while the mask region covers less.

| Image id | Split | Size | Official box (% of frame) | Mask-derived box (% of frame) | Extent IoU |
|---|---|---|---|---|---|
| cju2raxlosl630988jdbfy9b0 | train | 557×528 | 100 | 46 | 0.4585 |
| cju323ypb1fbb0988gx5rzudb | val | 604×530 | 82 | 50 | 0.6035 |
| cju7dhpsc2dnn0818025m6857 | train | 554×531 | 100 | 63 | 0.6324 |
| cju7ehljc2or70871261br8ai | val | 569×531 | 99 | 74 | 0.7433 |
| cju3y54kwj3nr0801biidlb4e | val | 620×546 | 100 | 75 | 0.7504 |
| cju2sszfq3uye0878sucelzk2 | train | 622×531 | 99 | 77 | 0.7767 |
| cju30ia8da2bq0799klnehml2 | train | 571×531 | 100 | 79 | 0.7887 |
| cju87li0zn3yb0817kbwgjiz8 | train | 562×449 | 99 | 82 | 0.8218 |
| cju2igw4gvxds0878808qj398 | train | 626×546 | 85 | 75 | 0.8715 |
| cju88l66no10s0850rsda7ej1 | train | 571×530 | 100 | 90 | 0.8952 |

**Group B — official file has one extra box around a speck (8 images).** The extents agree (IoU 0.985–0.998) but the official file lists one more box than there are polyp regions.

| Image id | Split | Official boxes | Mask-derived boxes |
|---|---|---|---|
| cjyzjzssvd8pq0838f4nolj5l | train | 2 | 1 |
| cju7ddtz729960801uazp1knc | train | 2 | 1 |
| cju43in5fm22c08175rxziqrk | train | 2 | 1 |
| cju15jr8jz8sb0855ukmkswkz | train | 2 | 1 |
| cju2uzabhs6er0993x3aaf87p | test | 2 | 1 |
| cju0roawvklrq0799vmjorwfv | test | 3 | 2 |
| cju7ajnbo1gvm098749rdouk0 | test | 2 | 1 |
| ck2bxiswtxuw80838qkisqjwz | test | 2 | 1 |

The full per-image comparison is in `data/processed/kvasir/reports/box_comparison.csv`.

### 1.5 Split

The paper's split cannot be reproduced: the baseline's `Creat_YAML.py` takes the first 800 files of an unsorted `glob` with no seed. We use our own fixed split.

- 700 train / 100 val / 200 test.
- Image ids are sorted, then shuffled with `random.Random(42)`; the first 700 are train, the next 100 val, the last 200 test.
- Saved to `splits/kvasir_train.txt`, `kvasir_val.txt`, `kvasir_test.txt` (one image id per line, committed to git). The script reuses these files if they exist, so every member and every run uses the same split.
- The validation split exists so the confidence threshold can be chosen later without touching the test split.

| | Train | Val | Test | Total |
|---|---|---|---|---|
| Images | 700 | 100 | 200 | 1,000 |
| Boxes (polyp regions) | 749 | 107 | 207 | 1,063 |
| Mean polyp area (% of image) | 14.72 | 18.17 | 16.36 | — |
| Median polyp area (% of image) | 10.82 | 13.87 | 12.65 | — |

The split is random, not stratified by polyp size, so the mean polyp area differs between splits (the validation split has somewhat larger polyps on average).

Polyp regions per image, over all 1,000 images:

| Regions | 1 | 2 | 3 | 4 | 10 |
|---|---|---|---|---|---|
| Images | 952 | 41 | 5 | 1 | 1 |

The image with 10 regions is `cju3uhb79gcgr0871orbrbi3x` (train); the official file also lists 10 boxes for it.

### 1.6 Output format

```
data/processed/kvasir/
├── images/{train,val,test}/<id>.jpg   byte-identical copies of the raw images
├── masks/{train,val,test}/<id>.png    binary masks, values 0 and 255
├── labels/{train,val,test}/<id>.txt   YOLO labels: "0 x_center y_center width height", normalised to 0..1
└── reports/
    ├── verification.json              the numbers in this section
    └── box_comparison.csv             official vs mask-derived boxes, one row per image
```

`configs/kvasir.yaml` is the YOLO dataset config (one class, `polyp`). Its `path` must be set to the absolute location of `data/processed/kvasir` on the training machine.

After the run, the output was checked against the raw data by a separate script:

| Check | Result |
|---|---|
| Split files: no image in two splits, all 1,000 ids covered | passed |
| Split regenerated from seed 42 equals the saved files | passed |
| Image copies identical to the raw files | 1,000 of 1,000 |
| Masks single-channel with only the values 0 and 255 | 1,000 of 1,000 |
| Masks equal to the raw mask thresholded at 127 | 1,000 of 1,000 |
| Label files well-formed, values in (0, 1] | 1,000 of 1,000 |
| Labels converted back to pixels match the mask boxes | within 0.001 px |
| Raw folder unchanged (2,001 files, zip SHA-256 unchanged) | passed |

### 1.7 How to reproduce

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
# download kvasir-seg.zip from the URL in 1.1 and unzip it into data/raw/kvasir-seg/
.venv\Scripts\python src/data/prepare_kvasir.py
```

Run with Python 3.12.10, numpy 2.5.3, Pillow 12.3.0, opencv-python-headless 5.0.0.93. The shared helpers in `src/data/common.py` (mask loading and saving, mask → boxes, box IoU, YOLO label writing) are used by all dataset scripts so both datasets are processed identically.

### 1.8 Problems found (for the "challenges" section of the final report)

- The official bounding-box file is unreliable for 18 images (section 1.4), so it cannot be used as-is for YOLO labels.
- The masks are lossy JPGs, not binary images (0.73% of pixels are not exactly 0 or 255), and contain 143 specks.
- The baseline's preprocessing script moves the raw images and uses an unseeded split, so the paper's split cannot be recovered.
- The dataset server's certificate is rejected by Git Bash `curl` on Windows; the Windows system `curl` works.

---

## 2. CVC-ClinicDB (external dataset) — Member 2

*To be written by Member 2 (tasks 2.1–2.4, 2.7).*

---

## 3. Kvasir-SEG vs CVC-ClinicDB comparison — Member 2

*To be written by Member 2 (tasks 2.5–2.6).*
