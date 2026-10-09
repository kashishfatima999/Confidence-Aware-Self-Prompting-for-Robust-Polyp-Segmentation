# F26-13 — Data Report (Phase 2)

**Project:** Confidence-Aware Self-Prompting for Robust Polyp Segmentation Under Image and Domain Shifts

This report records where each dataset came from, how it was checked and cleaned, and the exact numbers produced. The reasoning behind each choice (alternatives considered and evidence) is in `docs/decision_log.md`; the `DL-` numbers below refer to it.

---

## 1. Kvasir-SEG (primary dataset) — Member 1

### 1.1 Source

|              |                                                                                                                 |
| ------------ | --------------------------------------------------------------------------------------------------------------- |
| Dataset      | Kvasir-SEG: A Segmented Polyp Dataset (Jha et al., 2020)                                                        |
| Dataset page | https://datasets.simula.no/kvasir-seg/                                                                          |
| Download URL | https://datasets.simula.no/downloads/kvasir-seg.zip                                                             |
| Downloaded   | 2026-10-07                                                                                                      |
| File size    | 46,227,172 bytes                                                                                                |
| SHA-256      | `03b30e21d584e04facf49397a2576738fd626815771afbbf788f74a7153478f7`                                              |
| Terms of use | Research and educational purposes only; commercial use needs written permission (as stated on the dataset page) |
| Stored in    | `data/raw/kvasir-seg/` (not in git)                                                                             |

**Citation:** D. Jha, P. H. Smedsrud, M. A. Riegler, P. Halvorsen, T. de Lange, D. Johansen, H. D. Johansen. “Kvasir-SEG: A Segmented Polyp Dataset.” *International Conference on Multimedia Modeling (MMM)*, pp. 451–462, Springer, 2020. arXiv:1911.07069.

The archive contains `Kvasir-SEG/images/` (1,000 JPG images), `Kvasir-SEG/masks/` (1,000 JPG masks), and `Kvasir-SEG/kavsir_bboxes.json` (the filename is misspelled in the original download). The raw folder is never modified; all outputs go to `data/processed/kvasir/`.

### 1.2 Verification of the raw download (DL-11, DL-14)

| Check                                                     |      Result |
| --------------------------------------------------------- | ----------: |
| Images                                                    |       1,000 |
| Masks                                                     |       1,000 |
| Entries in the official bounding-box file                 |       1,000 |
| Images without a mask / masks without an image            |       0 / 0 |
| Images without a box entry / box entries without an image |       0 / 0 |
| Image size different from mask size                       |           0 |
| Image size different from size recorded in the box file   |           0 |
| Unreadable files (every image and mask fully decoded)     |           0 |
| Empty masks after thresholding                            |           0 |
| Distinct resolutions                                      |         333 |
| Width range                                               | 332–1920 px |
| Height range                                              | 352–1072 px |

The four most common resolutions are 622×530 (78 images), 626×547 (66), 626×546 (61), and 622×529 (56). Images are not resized during preprocessing; YOLO and SAM 2 resize them internally or on loading.

### 1.3 Cleaning (DL-04, DL-05, DL-06, DL-07, DL-09)

**Masks.** The raw masks are RGB JPG files, so they are not strictly binary. JPEG compression leaves pixels close to, but not exactly, 0 or 255 in every one of the 1,000 masks. These pixels account for 0.73% of all mask pixels; for example, a foreground pixel may have value 250 instead of 255. The baseline code reads masks as `mask / 255` without thresholding, so such pixels become values like 0.98 rather than unambiguous foreground or background values. No mask pixel lies between 11 and 244, so thresholding is unambiguous. Each mask is converted to greyscale, thresholded at 127 (values greater than 127 represent polyp pixels), and saved as a single-channel PNG with values 0 and 255 only.

**Specks.** After thresholding, the masks contain 143 small isolated regions in addition to the main polyp regions. A speck is defined as a connected region smaller than 0.05% of the image area (`MIN_COMPONENT_AREA_FRACTION = 0.0005`), using 8-connectivity. Specks are **kept in the saved masks**, so the ground truth used for Dice and IoU remains the thresholded mask. They are **ignored when deriving bounding boxes** to avoid generating boxes around tiny isolated regions.

**Bounding boxes.** YOLO boxes are derived from the cleaned masks, with one box per connected component above the minimum area threshold, instead of using the official box file. The reasons are:

1. The official file has discrepancies in 18 of the 1,000 images (see Section 1.4).
2. CVC-ClinicDB provides masks but no equivalent official box file, so deriving boxes from masks gives both datasets the same box definition.

The box convention used in the code is `(xmin, ymin, xmax, ymax)` in pixels, with `xmax` and `ymax` exclusive.

### 1.4 Official boxes compared with mask-derived boxes (DL-06, DL-08)

For each image, the overall extent of the official boxes is compared with the overall extent of the mask-derived boxes using intersection over union (IoU). The number of boxes is also compared. An image is considered to agree when the extent IoU is at least 0.9 and the box counts are equal.

| Comparison                            | Images |
| ------------------------------------- | -----: |
| Agree                                 |    982 |
| Extent IoU below 0.9                  |     10 |
| Different number of boxes             |      8 |
| Mean extent IoU over all 1,000 images | 0.9874 |

On the 982 agreeing images, the official box is typically one pixel larger on the left and top than the mask-derived box. As a result, even agreeing images can have an extent IoU slightly below 1.

**Group A — official box substantially larger than the polyp region (10 images).** In all ten cases, the official box starts at pixel (0, 0) and covers 82–100% of the frame, while the mask region covers less.

| Image ID                  | Split |    Size | Official box (% of frame) | Mask-derived box (% of frame) | Extent IoU |
| ------------------------- | ----- | ------: | ------------------------: | ----------------------------: | ---------: |
| cju2raxlosl630988jdbfy9b0 | train | 557×528 |                       100 |                            46 |     0.4585 |
| cju323ypb1fbb0988gx5rzudb | val   | 604×530 |                        82 |                            50 |     0.6035 |
| cju7dhpsc2dnn0818025m6857 | train | 554×531 |                       100 |                            63 |     0.6324 |
| cju7ehljc2or70871261br8ai | val   | 569×531 |                        99 |                            74 |     0.7433 |
| cju3y54kwj3nr0801biidlb4e | val   | 620×546 |                       100 |                            75 |     0.7504 |
| cju2sszfq3uye0878sucelzk2 | train | 622×531 |                        99 |                            77 |     0.7767 |
| cju30ia8da2bq0799klnehml2 | train | 571×531 |                       100 |                            79 |     0.7887 |
| cju87li0zn3yb0817kbwgjiz8 | train | 562×449 |                        99 |                            82 |     0.8218 |
| cju2igw4gvxds0878808qj398 | train | 626×546 |                        85 |                            75 |     0.8715 |
| cju88l66no10s0850rsda7ej1 | train | 571×530 |                       100 |                            90 |     0.8952 |

**Group B — official file has one extra box around a speck (8 images).** The extents agree (IoU 0.985–0.998), but the official file lists one more box than there are mask-derived polyp components.

| Image ID                  | Split | Official boxes | Mask-derived boxes |
| ------------------------- | ----- | -------------: | -----------------: |
| cjyzjzssvd8pq0838f4nolj5l | train |              2 |                  1 |
| cju7ddtz729960801uazp1knc | train |              2 |                  1 |
| cju43in5fm22c08175rxziqrk | train |              2 |                  1 |
| cju15jr8jz8sb0855ukmkswkz | train |              2 |                  1 |
| cju2uzabhs6er0993x3aaf87p | test  |              2 |                  1 |
| cju0roawvklrq0799vmjorwfv | test  |              3 |                  2 |
| cju7ajnbo1gvm098749rdouk0 | test  |              2 |                  1 |
| ck2bxiswtxuw80838qkisqjwz | test  |              2 |                  1 |

The full per-image comparison is in `data/processed/kvasir/reports/box_comparison.csv`.

### 1.5 Split (DL-02)

The paper's split cannot be reproduced exactly: the baseline's `Creat_YAML.py` takes the first 800 files from an unsorted `glob` with no seed. We therefore use a fixed split.

* 700 train / 100 validation / 200 test images.
* Image IDs are sorted, then shuffled with `random.Random(42)`. The first 700 IDs are assigned to train, the next 100 to validation, and the last 200 to test.
* The IDs are saved in `splits/kvasir_train.txt`, `splits/kvasir_val.txt`, and `splits/kvasir_test.txt`, one image ID per line. These files are committed to git and reused by the script.
* The validation split allows the confidence threshold to be selected later without using the test split.

|                                | Train |   Val |  Test | Total |
| ------------------------------ | ----: | ----: | ----: | ----: |
| Images                         |   700 |   100 |   200 | 1,000 |
| Boxes (polyp regions)          |   749 |   107 |   207 | 1,063 |
| Mean polyp area (% of image)   | 14.72 | 18.17 | 16.36 |     — |
| Median polyp area (% of image) | 10.82 | 13.87 | 12.65 |     — |

The split is random, not stratified by polyp size, so mean polyp area differs between splits. The validation split has somewhat larger polyps on average.

Polyp regions per image across all 1,000 images:

| Regions |   1 |  2 |  3 |  4 | 10 |
| ------- | --: | -: | -: | -: | -: |
| Images  | 952 | 41 |  5 |  1 |  1 |

The image with ten regions is `cju3uhb79gcgr0871orbrbi3x` (train); the official file also lists ten boxes for it.

### 1.6 Output format and verification

```text
data/processed/kvasir/
├── images/{train,val,test}/<id>.jpg
├── masks/{train,val,test}/<id>.png
├── labels/{train,val,test}/<id>.txt
└── reports/
    ├── verification.json
    └── box_comparison.csv
```

Images are byte-identical copies of the raw images. Masks are single-channel PNGs with values 0 and 255. Labels use the YOLO format `0 x_center y_center width height`, normalised to 0–1.

`configs/kvasir.yaml` is the YOLO dataset configuration with one class, `polyp`. Its `path` must be set to the absolute location of `data/processed/kvasir` on the training machine if needed.

After preprocessing, an independent check was performed:

| Check                                                      | Result          |
| ---------------------------------------------------------- | --------------- |
| Split files: no image in two splits, all 1,000 IDs covered | Passed          |
| Split regenerated from seed 42 equals saved files          | Passed          |
| Image copies identical to raw files                        | 1,000 of 1,000  |
| Masks single-channel with only values 0 and 255            | 1,000 of 1,000  |
| Masks equal to raw masks thresholded at 127                | 1,000 of 1,000  |
| Label files well-formed, values in (0, 1]                  | 1,000 of 1,000  |
| Labels converted back to pixels match mask boxes           | Within 0.001 px |
| Raw folder unchanged (2,001 files, zip SHA-256 unchanged)  | Passed          |

### 1.7 How to reproduce

Run these commands from the repository root:

```cmd
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

Download `kvasir-seg.zip` from the URL in Section 1.1 and extract it into `data/raw/kvasir-seg/`. Then run:

```cmd
.venv\Scripts\python src\data\prepare_kvasir.py
```

The recorded environment uses Python 3.12.10, numpy 2.5.3, Pillow 12.3.0, and opencv-python-headless 5.0.0.93. The shared helpers in `src/data/common.py` are used for mask loading and saving, mask-to-box conversion, box IoU, and YOLO label writing.

### 1.8 Problems found

* The official bounding-box file has discrepancies for 18 images (Section 1.4), so it is not used directly for YOLO labels.
* The masks are lossy JPGs rather than binary images; 0.73% of mask pixels are not exactly 0 or 255, and 143 small isolated regions were found.
* The baseline preprocessing script moves raw images and uses an unseeded split, so the paper's exact split cannot be recovered.
* The dataset server's certificate is rejected by Git Bash `curl` on Windows; the Windows system `curl.exe` works.

---

## 2. CVC-ClinicDB (external dataset) — Member 2

### 2.1 Source

|                            |                                                                             |
| -------------------------- | --------------------------------------------------------------------------- |
| Dataset                    | CVC-ClinicDB                                                                |
| Dataset page               | https://www.kaggle.com/datasets/balraj98/cvcclinicdb                        |
| Original dataset reference | Bernal et al. (2015)                                                        |
| Downloaded                 | 2026-10-09                                                                  |
| Source format used         | PNG                                                                         |
| Stored in                  | `data/raw/cvc-clinicdb/` (not in git)                                       |
| Usage terms                | Verify the applicable dataset terms before redistribution or commercial use |

The downloaded dataset contains `PNG/Original/` and `PNG/Ground Truth/`, each with 612 PNG files. The TIF versions were not used in preprocessing. Raw files remain unchanged; processed outputs are written to `data/processed/cvc-clinicdb/`.

**Citation:** Bernal et al. (2015). *Comparative validation of polyp detection methods in video colonoscopy: results from the MICCAI 2015 Endoscopic Vision Challenge.*

### 2.2 Verification

| Check                                                         |       Result |
| ------------------------------------------------------------- | -----------: |
| Images                                                        |          612 |
| Masks                                                         |          612 |
| Images without matching masks / masks without matching images |        0 / 0 |
| Distinct image resolutions                                    |            1 |
| Image and mask resolution                                     | 384 × 288 px |
| Unreadable images                                             |            0 |
| Unreadable masks                                              |            0 |
| Empty masks after thresholding                                |            0 |

All image and mask files were decoded and checked before processing. Filenames matched between the image and ground-truth folders, and every image-mask pair had matching dimensions.

### 2.3 Cleaning and bounding boxes

Masks were converted to greyscale and binarised using the shared threshold of 127: pixel values greater than 127 represent polyp pixels. The binary masks were saved as single-channel PNG files with values 0 and 255.

Bounding boxes were derived from connected components using `src/data/common.py`, applying the same minimum component area threshold of 0.05% of the image area as the Kvasir-SEG pipeline. Boxes were converted to YOLO format, with class 0 representing `polyp`. The original images were copied without resizing or augmentation.

### 2.4 External test-only preparation

All 612 images were placed in the `test` split. ClinicDB was not used for training or confidence-threshold tuning, preserving its role as an external dataset for evaluating robustness under domain shift. This protocol is recorded in DL-03; group approval should be confirmed.

| Output                       |  Count |
| ---------------------------- | -----: |
| Processed images             |    612 |
| Binary masks                 |    612 |
| YOLO label files             |    612 |
| Bounding boxes               |    646 |
| Test images                  |    612 |
| Train images                 |      0 |
| Validation images            |      0 |
| Mean polyp area (% of image) | 9.1657 |

The 646 boxes across 612 images indicate that some images contain multiple connected mask components. These are mask-derived boxes, not independently annotated detection boxes.

Output structure:

```text
data/processed/cvc-clinicdb/
├── images/test/<id>.png
├── masks/test/<id>.png
├── labels/test/<id>.txt
└── reports/verification.json
```

The dataset configuration is `configs/cvc_clinicdb.yaml`. Its `path` may need to be set to the absolute processed-dataset path on the machine running Ultralytics.

### 2.5 Reproducibility and limitations

Run the preprocessing script from the repository root:

```cmd
.venv\Scripts\python src\data\prepare_clinicdb.py
```

The script checks the expected file counts, filename pairing, image readability, matching dimensions, and non-empty masks before creating processed outputs. It then writes the verification summary to `data/processed/cvc-clinicdb/reports/verification.json`.

CVC-ClinicDB is used as an external test set, so its results measure cross-dataset performance and should not be presented as directly equivalent to results from a model evaluated on its own dataset's training/test split.

The downloaded archive's exact byte size, SHA-256, and applicable usage terms should be recorded after checking the local archive and its source terms.

---

## 3. Kvasir-SEG vs CVC-ClinicDB comparison — Member 2

### 3.1 Dataset overview

| Property                         |                  Kvasir-SEG |                       CVC-ClinicDB |
| -------------------------------- | --------------------------: | ---------------------------------: |
| Role                             |             Primary dataset |        External evaluation dataset |
| Total images                     |                       1,000 |                                612 |
| Image-mask pairs                 |                       1,000 |                                612 |
| Resolution                       |    333 distinct resolutions |                       384 × 288 px |
| Image format used                |                         JPG |                                PNG |
| Processed mask format            |                         PNG |                                PNG |
| Mask threshold                   |                         127 |                                127 |
| Minimum component area for boxes |                       0.05% |                              0.05% |
| Total mask-derived boxes         |                       1,063 |                                646 |
| Mean polyp area (% of image)     | 15.391% across full dataset |             9.1657% across dataset |
| Evaluation protocol              |  Fixed 200-image test split | All 612 images, external test only |

### 3.2 Statistical comparison

Statistics were calculated from the processed images and masks using the same analysis procedure for both datasets.

| Metric                                   | Kvasir-SEG | CVC-ClinicDB |
| ---------------------------------------- | ---------: | -----------: |
| Images analysed                          |      1,000 |          612 |
| Mean image width (px)                    |    625.292 |          384 |
| Mean image height (px)                   |    545.228 |          288 |
| Mean polyp area (%)                      |     15.391 |        9.166 |
| Mean mask-connected components per image |      1.063 |        1.056 |
| Mean bounding-box aspect ratio           |      0.982 |        1.163 |
| Mean brightness                          |     97.482 |       76.195 |
| Mean grayscale contrast                  |     59.445 |       54.166 |

### 3.3 Interpretation and domain shift

Kvasir-SEG has a higher mean polyp area (15.391%) than CVC-ClinicDB (9.166%). Its images are also brighter on average (97.482 vs. 76.195) and have higher grayscale contrast (59.445 vs. 54.166). The mean bounding-box aspect ratio is higher for ClinicDB (1.163) than for Kvasir-SEG (0.982).

These differences suggest variation in image characteristics and polyp size between the datasets, providing evidence of potential domain shift. Kvasir-SEG contains images with varied resolutions, while ClinicDB images have a fixed resolution of 384 × 288 pixels.

Both datasets were analysed using the same procedure. The connected-region count is an estimate derived from mask components and should not be interpreted as a verified count of distinct polyps. Bounding-box aspect ratios are also derived from the mask-connected components. These descriptive statistics indicate differences between the datasets but do not establish their causes or quantify their effect on model performance.

The five distribution plots and the underlying per-image statistics are saved in `notebooks/`. The summary table is saved as `notebooks/dataset_comparison.csv`, and the per-image results are saved as `notebooks/dataset_statistics.csv`. The analysis notebook is `notebooks/02_dataset_statistics.ipynb`.

CVC-ClinicDB remains reserved for external evaluation and must not be used for model training or confidence-threshold selection.

---
