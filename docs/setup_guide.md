# Setup guide: environment, baseline and sanity checks (Member 3)

Phase 2, tasks 3.1–3.7. Everything here was run on **2026-10-09** and every number below is copied from the outputs of `notebooks/03_environment_smoke_test.ipynb`.

**To reproduce:** open the notebook in Google Colab, set *Runtime → Change runtime type → T4 GPU*, then *Runtime → Run all* (about 10 minutes; the first run also downloads about 1 GB of weights to Google Drive).

---

## 1. Baseline repository (task 3.1)

| Item | Value |
|---|---|
| Repository | https://github.com/sajjad-sh33/YOLO_SAM2 (Mansoori et al., 2024, arXiv:2409.09484) |
| Commit used | `d86e375a215210209576e3d15932fe9cdb6c644d` (10 Nov 2024) |
| Where it lives | `external/YOLO_SAM2/`, git-ignored, never committed |
| License | **none**, so it is cited, not copied (DL-16) |
| Contents | `Creat_YAML.py`, `Test.py`, `README.md`, `sam2/`, `sam2_configs/`, `YAML/`, `imgs/`, `YOLO_Checkpoints/`, `checkpoints/` |
| Shipped YOLO weights | 6 files in `YOLO_Checkpoints/`, including `Kvasir_yolov8m.pt` (52,070,283 bytes). Used for the smoke test **only**, never for a reported result, because the split they were trained on is unknown (DL-17). |
| SAM 2 weights | not in the repository; `checkpoints/download_ckpts.sh` downloads them |

## 2. Environment (task 3.2)

Nobody in the group has an NVIDIA GPU, so all model work runs on **Google Colab with a T4 GPU**. Data preparation still runs locally on CPU (`requirements.txt`, DL-13).

| Package | Version | How it is installed |
|---|---|---|
| torch | 2.11.0+cu130 | preinstalled on Colab |
| torchvision | 0.26.0+cu130 | preinstalled on Colab |
| ultralytics | 8.4.174 | `pip install ultralytics==8.4.174` |
| SAM 2 | git commit `2b90b9f5ceec907a1c18123530e92e794ad901a4` | `SAM2_BUILD_CUDA=0 pip install "git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4"` |

These are pinned in `requirements-colab.txt`, which is kept separate from `requirements.txt` so that CI and the local CPU environment do not download PyTorch (DL-26).

`SAM2_BUILD_CUDA=0` skips compiling SAM 2's optional CUDA extension, which is only used by an optional mask post-processing step and is not needed for box-prompted image segmentation. It makes the install faster and avoids compiler errors.

### Steps (what the notebook does)

1. Mount Google Drive and clone our repository and the baseline:
   ```bash
   git clone https://github.com/kashishfatima999/Confidence-Aware-Self-Prompting-for-Robust-Polyp-Segmentation.git /content/proj
   cd /content/proj
   git clone https://github.com/sajjad-sh33/YOLO_SAM2.git external/YOLO_SAM2
   ```
2. Install the packages in the table above.
3. Download the weights into `MyDrive/F26-13/weights/` (section 3), so later sessions reuse them.
4. Download and prepare both datasets (section 4).
5. Run the smoke test, build the corruption set and draw the overlay check (sections 6–8).

## 3. Model weights (task 3.3)

Stored in Google Drive at `MyDrive/F26-13/weights/` (not in git, DL-12).

| File | Bytes | Source |
|---|---|---|
| `sam2_hiera_large.pt` | 897,952,466 | https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt (same file the baseline's `download_ckpts.sh` fetches) |
| `yolov8m.pt` | 52,136,884 | Ultralytics assets v8.4.0, downloaded by `YOLO('yolov8m.pt')`. COCO-pretrained starting point for our own YOLO training in Phase 3. |

The SAM 2 model is built with the config `configs/sam2/sam2_hiera_l.yaml` from the installed `sam2` package.

## 4. Datasets on Colab

| Dataset | Source used on Colab | Result |
|---|---|---|
| Kvasir-SEG | Kaggle `debeshjha1/kvasirseg` (uploaded by the dataset's author), see challenge C1 | `prepare_kvasir.py`: 1,000 / 1,000 / 1,000 verified; split reused 700 / 100 / 200 (seed 42); boxes train 749, val 107, test 207 |
| CVC-ClinicDB | Kaggle `balraj98/cvcclinicdb`, PNG version | `prepare_clinicdb.py`: 612 images, 612 masks, 646 boxes, all 384×288 |

The Kvasir-SEG numbers match Member 1's local run exactly (same split files, same box counts), which confirms the Kaggle copy gives the same processed dataset.

## 5. Problems found in the baseline code (task 3.1)

Line numbers refer to commit `d86e375` and were found with the `grep` and `py_compile` checks in the notebook. B6, B7 and B10 were not re-checked line by line and come from `project-division/Project-Division-Phase-2.md`, section 2.

| # | File, line | Problem | Consequence for us |
|---|---|---|---|
| B1 | `Test.py` 273, 329, 386, 456, 514, 573, 656 | `yolo_model.predict([image], imgsz=640, conf=0.5)`: confidence threshold hard-coded at 0.5 | Detections below 0.5 never reach SAM 2. These uncertain detections are what our method targets. |
| B2 | same lines | `imgsz=640` | The paper says image size 680. Paper and code disagree. |
| B3 | `Test.py` 271, 327, 384, 454, 512, 571, 654 | Ground truth read as `cv2.imread(...) / 255` with no threshold | JPEG artifacts (values like 0.98) are not counted as polyp. We binarise at 127 (DL-04). |
| B4 | `Test.py` imports (lines 1–11, 234–235) | `argparse` and `matplotlib` are used but never imported; `cv2` and `numpy` are imported twice | The script fails even after the indentation is fixed. |
| B5 | `Test.py` 251 | `IndentationError: unexpected indent` | The script does not run as published. A rewrite is Phase 3 work. |
| B6 | `Test.py` (evaluation loop; from the division plan, section 2) | Images with no detection add nothing to intersection or union | Missed polyps do not lower the paper's score. We count them as empty predictions (DL-18); section 6 shows a real case. |
| B7 | `Test.py` (evaluation loop; from the division plan, section 2) | Dice/IoU pooled over all pixels of the dataset | We also report per-image means (DL-19). |
| B8 | `Creat_YAML.py` 30–33 and similar | `!mkdir` (notebook syntax) inside a `.py` file | Does not run as plain Python. |
| B9 | `Creat_YAML.py` 52 and similar | Paths split on `'/'` | Breaks on Windows. |
| B10 | `Creat_YAML.py` 49 and similar | Unsorted `glob`, no seed, first 800 = train (division plan, section 2) | The paper's split cannot be reproduced. |
| B11 | `Creat_YAML.py` 61, 78 and similar | `shutil.move` | Destroys the raw download. We always copy (DL-01). |
| B12 | `README.md` 36 | `yolov8l.pt imgsz=640 epochs=50` | The paper says YOLOv8-m at 680; the shipped weights are YOLOv8-m. |

## 6. Smoke test: YOLO → SAM 2 (task 3.4)

The first 5 images of `splits/kvasir_test.txt`, run with the baseline's shipped `Kvasir_yolov8m.pt`. YOLO was run with `conf=0.05` so that every candidate box is printed; only boxes with confidence ≥ 0.5 were passed to SAM 2, as in `Test.py`. Figure: `docs/figures/smoke_test.png`.

| Image | YOLO boxes (confidence) | Dice |
|---|---|---|
| cju0roawvklrq0799vmjorwfv | 2 boxes: 0.974, 0.902 (both kept; two polyps) | 0.969 |
| cju0s690hkp960855tjuaqvv0 | 1 box: 0.920 | 0.967 |
| cju0u2g7pmnux0801vkk47ivj | 1 box: 0.963 | 0.965 |
| cju15ptjtppz40988odsm9azx | **no box, even at confidence 0.05** | **0.000** |
| cju160wshltz10993i1gmqxbe | 1 box: 0.952 | 0.958 |

- **The pipeline runs end to end** on Colab: the YOLO weights load, SAM 2 builds from the checkpoint, and box prompts give masks.
- No detection fell between 0.05 and 0.5; all detections were 0.90–0.97.
- **One polyp was missed completely.** Mean Dice over all 5 images is **0.772**; if the missed image is skipped, as the baseline's `Test.py` does, it is **0.965**. One missed polyp in five moves the mean by 0.19, which is why DL-18 counts misses.
- When YOLO does find the polyp, SAM 2's mask is very close to the ground truth (Dice 0.96–0.97). On these images the detection stage is the weak point, not the segmentation stage, which supports the project's focus on the box prompt.
- **Not a reported result:** the shipped weights may have been trained on these test images (DL-17), so these scores may be optimistic.

## 7. Image-shift test set (task 3.6)

`src/data/make_corruptions.py` (decision DL-20) applies three corruptions at three strengths to the **200 Kvasir-SEG test images only**. Training and validation images are never touched.

| Corruption | Strength 1 | Strength 2 | Strength 3 |
|---|---|---|---|
| `blur` (Gaussian, sigma in px) | 1 | 2 | 4 |
| `brightness_contrast` (contrast ×, brightness shift) | ×0.8, −20 | ×0.6, −40 | ×0.4, −60 |
| `noise` (Gaussian, std on 0–255) | 10 | 20 | 40 |

- Output: `data/processed/kvasir-corrupted/<corruption>_s<strength>/{images,masks,labels}/test/`, 9 settings × 200 images, plus `manifest.json` with every parameter. Each folder has the same layout as `data/processed/kvasir`, so YOLO can evaluate it by changing `path` in a dataset YAML.
- Masks and YOLO labels are copied unchanged, because none of these corruptions moves the polyp.
- Corrupted images are saved as PNG so no extra JPEG compression is added. Noise is seeded per image, so reruns give byte-identical files.
- `tests/test_make_corruptions.py` checks all 9 settings are written, masks and labels are unchanged, reruns are identical, the source is never modified, and a missing input stops the script before anything is written.
- Figure: `docs/figures/corruption_examples.png`. The polyp stays visible at every strength.

## 8. Overlay check of the processed data (task 3.5)

20 random test images per dataset (`random.Random(0)`), with the mask outline in green and the YOLO label box in red. Figures: `docs/figures/overlay_kvasir.png`, `docs/figures/overlay_clinicdb.png`.

- **Kvasir-SEG: 20/20 correct.** Multi-polyp images get one box per polyp, and tiny polyps get small boxes in the right place. One image (`cju2oi8sq0i2y0801mektzvw8`) looked unlabelled at thumbnail size. At full size, its polyp covers 76.3% of the frame and the label is `0 0.500742 0.500000 0.998516 1.000000`, so the outline and box run along the black border. The label is correct.
- **CVC-ClinicDB: 20/20 correct.** Two-polyp frames (547, 570) have two boxes, and very small polyps (379, 588) and polyps touching the frame edge (472, 559) are boxed correctly.
- **Conclusion:** the mask → box → YOLO label conversion in `prepare_kvasir.py` and `prepare_clinicdb.py` is correct on the sampled images.
- **Note for Phase 3:** a polyp that fills the frame gives a box covering the whole image, which tells SAM 2 almost nothing about where the polyp is. Such near-full-frame boxes may need separate handling in the prompt refinement.

## 9. Challenges faced (for the final report)

| # | Problem | What we did |
|---|---|---|
| C1 | The official Kvasir-SEG link (`datasets.simula.no/downloads/kvasir-seg.zip`) returned 0 bytes from Colab. | Used the Kaggle copy by the dataset's author (`debeshjha1/kvasirseg`). It stores boxes as one CSV per image (`bbox/*.csv`, columns `class_name,xmin,ymin,xmax,ymax`) instead of `kavsir_bboxes.json`, so the notebook rebuilds the JSON in the official format. The JSON only feeds the verification and the box-agreement report; the YOLO labels themselves are derived from the masks (DL-06), so they do not depend on it. `prepare_kvasir.py` then verified 1,000 / 1,000 / 1,000 and produced the same split and box counts as Member 1's local run (DL-26). |
| C2 | `prepare_kvasir.py`'s default `--raw` path (`data/raw/kvasir-seg`) is one level above the folder that contains `images/`. | Run with `--raw data/raw/kvasir-seg/Kvasir-SEG`. Reported to Member 1. |
| C3 | `prepare_clinicdb.py` was on `main` but not on `project-divison`, so a Colab clone of `project-divison` could not prepare CVC-ClinicDB. | The group switched to one branch per member merged into `main` (DL-21, revised). The notebook clones `main`. |
| C4 | The baseline's `Test.py` and `Creat_YAML.py` do not run as published (B4, B5, B8). | Not fixed. We write our own preprocessing (done) and evaluation (Phase 3). |
| C5 | Colab deletes everything when the session ends. | Weights and figures go to Google Drive; datasets are re-downloaded and re-prepared each session (about 2 minutes). |
| C6 | Adding the model packages to `requirements.txt` would make CI and the local CPU environment download PyTorch. | Model packages are pinned in a separate `requirements-colab.txt` (DL-26). |
