
"""Prepare CVC-ClinicDB as an external test-only YOLO dataset."""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW = REPO_ROOT / "data/raw/cvc-clinicdb/PNG"
OUT = REPO_ROOT / "data/processed/cvc-clinicdb"
EXPECTED_IMAGES = 612


def main():
    images_dir = RAW / "Original"
    masks_dir = RAW / "Ground Truth"

    images = {p.stem: p for p in images_dir.glob("*.png")}
    masks = {p.stem: p for p in masks_dir.glob("*.png")}

    if len(images) != EXPECTED_IMAGES or len(masks) != EXPECTED_IMAGES:
        raise SystemExit(
            f"Expected {EXPECTED_IMAGES} images and masks; "
            f"found {len(images)} images and {len(masks)} masks."
        )

    if images.keys() != masks.keys():
        raise SystemExit("Image and mask filenames do not match.")

    # Verify every image and mask before creating outputs.
    for image_id in sorted(images):
        try:
            with Image.open(images[image_id]) as im:
                im.load()
                image_size = im.size

            mask = common.load_binary_mask(masks[image_id])
        except Exception as error:
            raise SystemExit(f"{image_id}: unreadable image or mask: {error}")

        mask_size = (mask.shape[1], mask.shape[0])
        if image_size != mask_size:
            raise SystemExit(
                f"{image_id}: image size {image_size} != mask size {mask_size}"
            )
        if not mask.any():
            raise SystemExit(f"{image_id}: mask is empty after thresholding")

        boxes = common.mask_to_boxes(mask)
        if not boxes:
            raise SystemExit(f"{image_id}: no polyp region found for YOLO labels")

    # Only generated ClinicDB outputs are cleared; raw files are never changed.
    for folder in ("images", "masks", "labels"):
        shutil.rmtree(OUT / folder, ignore_errors=True)

    image_count = 0
    total_boxes = 0
    empty_after_threshold = 0
    area_percentages = []

    for image_id in sorted(images):
        image_path = images[image_id]
        mask = common.load_binary_mask(masks[image_id])
        height, width = mask.shape
        boxes = common.mask_to_boxes(mask)

        image_out = OUT / "images/test" / f"{image_id}.png"
        mask_out = OUT / "masks/test" / f"{image_id}.png"
        label_out = OUT / "labels/test" / f"{image_id}.txt"

        image_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, image_out)
        common.save_binary_mask(mask, mask_out)
        common.write_yolo_label(label_out, boxes, width, height)

        image_count += 1
        total_boxes += len(boxes)
        area_percentages.append(float(mask.mean()) * 100)

    report = {
        "dataset": "CVC-ClinicDB",
        "source_folder": "data/raw/cvc-clinicdb/PNG",
        "images": image_count,
        "masks": image_count,
        "labels": image_count,
        "split": {"test": image_count},
        "image_resolution": "384x288",
        "total_boxes": total_boxes,
        "mean_polyp_area_percent": round(float(np.mean(area_percentages)), 4),
        "mask_threshold": common.MASK_THRESHOLD,
        "min_component_area_fraction": common.MIN_COMPONENT_AREA_FRACTION,
        "training_on_clinicdb": False,
    }

    reports_dir = OUT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "verification.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    print(json.dumps(report, indent=2))
    print(f"Processed ClinicDB written to: {OUT}")


if __name__ == "__main__":
    main()