"""Verify, clean and convert CVC-ClinicDB as an external, test-only dataset (decisions DL-03, DL-24).

Input  (never modified):  data/raw/cvc-clinicdb/PNG/{Original,Ground Truth}/<id>.png
                          (the Kaggle mirror balraj98/cvcclinicdb; the PNG version is used)
Output: data/processed/cvc-clinicdb/images/test/<id>.png   copies of the originals
        data/processed/cvc-clinicdb/masks/test/<id>.png    binary masks (0 / 255)
        data/processed/cvc-clinicdb/labels/test/<id>.txt   YOLO boxes derived from the masks
        data/processed/cvc-clinicdb/reports/verification.json

All 612 images go into a single `test` split: CVC-ClinicDB is never used for training or tuning.
Masks, boxes and labels use the same shared helpers as Kvasir-SEG (src/data/common.py).

Run from the repository root:
    python src/data/prepare_clinicdb.py
"""

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from PIL import Image

import common

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_IMAGES = 612


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, default=REPO_ROOT / "data/raw/cvc-clinicdb/PNG")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data/processed/cvc-clinicdb")
    parser.add_argument("--expected-images", type=int, default=EXPECTED_IMAGES,
                        help="number of images the raw folder must contain (tests use small synthetic sets)")
    return parser.parse_args()


def verify_raw(raw, expected_images=EXPECTED_IMAGES):
    """Check the raw download is complete and consistent. Returns (ids, summary). Stops on any problem."""
    images = {p.stem: p for p in sorted((raw / "Original").glob("*.png"))}
    masks = {p.stem: p for p in sorted((raw / "Ground Truth").glob("*.png"))}

    problems = []
    if len(images) != expected_images:
        problems.append(f"expected {expected_images} images, found {len(images)}")
    missing = sorted(set(images) - set(masks))
    extra = sorted(set(masks) - set(images))
    if missing:
        problems.append(f"{len(missing)} images without a mask, e.g. {missing[:3]}")
    if extra:
        problems.append(f"{len(extra)} masks without an image, e.g. {extra[:3]}")

    sizes = Counter()
    for image_id, image_path in images.items():
        if image_id not in masks:
            continue
        try:
            with Image.open(image_path) as image:
                image.load()  # decodes the whole file, raises if it is corrupted
                size = image.size
            mask = common.load_binary_mask(masks[image_id])
        except Exception as error:
            problems.append(f"{image_id}: unreadable file ({error})")
            continue
        mask_size = (mask.shape[1], mask.shape[0])
        if mask_size != size:
            problems.append(f"{image_id}: image is {size} but mask is {mask_size}")
        if not mask.any():
            problems.append(f"{image_id}: mask is empty after thresholding")
        elif not common.mask_to_boxes(mask):
            problems.append(f"{image_id}: no polyp region above the minimum size")
        sizes[size] += 1

    if problems:
        raise SystemExit("Raw CVC-ClinicDB failed verification:\n  " + "\n  ".join(problems))

    widths = [w for (w, h) in sizes.elements()]
    heights = [h for (w, h) in sizes.elements()]
    summary = {
        "dataset": "CVC-ClinicDB",
        "source_folder": str(raw),
        "images": len(images),
        "masks": len(masks),
        # Any unreadable file or empty mask is a problem, and problems stop the run above.
        "unreadable_files": 0,
        "empty_masks": 0,
        "width_min": min(widths),
        "width_max": max(widths),
        "height_min": min(heights),
        "height_max": max(heights),
        "distinct_resolutions": len(sizes),
        "image_resolution": "x".join(map(str, sizes.most_common(1)[0][0])) if len(sizes) == 1 else "varies",
    }
    return sorted(images), summary


def main():
    args = parse_args()
    ids, summary = verify_raw(args.raw, args.expected_images)
    print(f"Raw data verified: {summary['images']} images and masks, all readable, no empty masks.")

    for folder in ("images", "masks", "labels"):
        shutil.rmtree(args.out / folder, ignore_errors=True)  # generated output only; raw data is elsewhere
    reports_dir = args.out / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    boxes_per_image = Counter()
    specks_ignored = 0
    area_fractions = []
    total_boxes = 0
    for image_id in ids:
        image_path = args.raw / "Original" / f"{image_id}.png"
        mask = common.load_binary_mask(args.raw / "Ground Truth" / f"{image_id}.png")
        height, width = mask.shape
        boxes = common.mask_to_boxes(mask)
        specks_ignored += len(common.mask_to_boxes(mask, min_area_fraction=0)) - len(boxes)

        (args.out / "images" / "test").mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, args.out / "images" / "test" / image_path.name)
        common.save_binary_mask(mask, args.out / "masks" / "test" / f"{image_id}.png")
        common.write_yolo_label(args.out / "labels" / "test" / f"{image_id}.txt", boxes, width, height)

        boxes_per_image[len(boxes)] += 1
        total_boxes += len(boxes)
        area_fractions.append(float(mask.mean()))

    summary.update({
        "labels": len(ids),
        "split": {"test": len(ids)},
        "total_boxes": total_boxes,
        "boxes_per_image": dict(sorted(boxes_per_image.items())),
        "specks_ignored_for_boxes": specks_ignored,
        "mean_polyp_area_percent": round(100 * float(np.mean(area_fractions)), 4),
        "median_polyp_area_percent": round(100 * float(np.median(area_fractions)), 4),
        "mask_threshold": common.MASK_THRESHOLD,
        "min_component_area_fraction": common.MIN_COMPONENT_AREA_FRACTION,
        "training_on_clinicdb": False,
    })
    (reports_dir / "verification.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Processed dataset written to {args.out}")


if __name__ == "__main__":
    main()
