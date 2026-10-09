"""Verify, clean, split and convert Kvasir-SEG for the YOLO -> SAM 2 pipeline.

Input  (never modified):  data/raw/kvasir-seg/Kvasir-SEG/{images,masks,kavsir_bboxes.json}
Output: data/processed/kvasir/images/{train,val,test}/<id>.jpg   copies of the originals
        data/processed/kvasir/masks/{train,val,test}/<id>.png    binary masks (0 / 255)
        data/processed/kvasir/labels/{train,val,test}/<id>.txt   YOLO boxes derived from the masks
        data/processed/kvasir/reports/                           verification summary + box comparison
        splits/kvasir_{train,val,test}.txt                       created once, then reused

Run from the repository root:
    python src/data/prepare_kvasir.py
"""

import argparse
import csv
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from PIL import Image

import common

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLIT_NAMES = ("train", "val", "test")
EXPECTED_IMAGES = 1000
# Official boxes whose overall extent overlaps the mask extent less than this are reported.
BOX_AGREEMENT_IOU = 0.9


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, default=REPO_ROOT / "data/raw/kvasir-seg")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data/processed/kvasir")
    parser.add_argument("--splits-dir", type=Path, default=REPO_ROOT / "splits")
    parser.add_argument("--train", type=int, default=700)
    parser.add_argument("--val", type=int, default=100)
    parser.add_argument("--test", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-images", type=int, default=EXPECTED_IMAGES,
                        help="number of images the raw folder must contain (tests use small synthetic sets)")
    return parser.parse_args()


def verify_raw(raw, expected_images=EXPECTED_IMAGES):
    """Check the raw download is complete and consistent. Returns (ids, official boxes, summary)."""
    images = {p.stem: p for p in sorted((raw / "images").glob("*.jpg"))}
    masks = {p.stem: p for p in sorted((raw / "masks").glob("*.jpg"))}
    official = json.loads((raw / "kavsir_bboxes.json").read_text())

    problems = []
    if len(images) != expected_images:
        problems.append(f"expected {expected_images} images, found {len(images)}")
    for name, other in (("mask", masks), ("box entry", official)):
        missing = sorted(set(images) - set(other))
        extra = sorted(set(other) - set(images))
        if missing:
            problems.append(f"{len(missing)} images without a {name}, e.g. {missing[:3]}")
        if extra:
            problems.append(f"{len(extra)} {name} items without an image, e.g. {extra[:3]}")

    widths, heights = [], []
    for image_id, image_path in images.items():
        if image_id not in masks or image_id not in official:
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
        entry = official[image_id]
        if (entry["width"], entry["height"]) != size:
            problems.append(f"{image_id}: image is {size} but box file says {(entry['width'], entry['height'])}")
        widths.append(size[0])
        heights.append(size[1])

    if problems:
        raise SystemExit("Raw Kvasir-SEG failed verification:\n  " + "\n  ".join(problems))

    summary = {
        "images": len(images),
        "masks": len(masks),
        "official_box_entries": len(official),
        # Any unreadable file or empty mask is a problem, and problems stop the run above.
        "unreadable_files": 0,
        "empty_masks": 0,
        "width_min": min(widths),
        "width_max": max(widths),
        "height_min": min(heights),
        "height_max": max(heights),
        "distinct_resolutions": len(set(zip(widths, heights))),
    }
    return sorted(images), official, summary


def load_or_create_splits(ids, splits_dir, sizes, seed):
    """Reuse the saved split files if they exist, otherwise create them with a fixed seed."""
    paths = {name: splits_dir / f"kvasir_{name}.txt" for name in SPLIT_NAMES}
    if all(p.exists() for p in paths.values()):
        splits = {name: paths[name].read_text().split() for name in SPLIT_NAMES}
        created = False
    else:
        if sum(sizes.values()) != len(ids):
            raise SystemExit(f"split sizes {sizes} do not add up to {len(ids)} images")
        shuffled = list(ids)  # ids arrive sorted, so the shuffle depends only on the seed
        random.Random(seed).shuffle(shuffled)
        splits, start = {}, 0
        for name in SPLIT_NAMES:
            splits[name] = sorted(shuffled[start:start + sizes[name]])
            start += sizes[name]
        splits_dir.mkdir(parents=True, exist_ok=True)
        for name in SPLIT_NAMES:
            paths[name].write_text("\n".join(splits[name]) + "\n")
        created = True

    assigned = [image_id for name in SPLIT_NAMES for image_id in splits[name]]
    if len(assigned) != len(set(assigned)):
        raise SystemExit("split files overlap: an image appears in more than one split")
    if set(assigned) != set(ids):
        raise SystemExit("split files do not cover exactly the images in the raw dataset")
    return splits, created


def main():
    args = parse_args()
    ids, official, summary = verify_raw(args.raw, args.expected_images)
    splits, created = load_or_create_splits(
        ids, args.splits_dir, {"train": args.train, "val": args.val, "test": args.test}, args.seed
    )
    print(f"Raw data verified: {summary['images']} images, masks and box entries.")
    print(f"Split files {'created' if created else 'reused'}: " + ", ".join(f"{n}={len(splits[n])}" for n in SPLIT_NAMES))

    for folder in ("images", "masks", "labels"):
        shutil.rmtree(args.out / folder, ignore_errors=True)  # generated output only; raw data is elsewhere
    reports_dir = args.out / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    comparison_rows = []
    split_stats = {}
    boxes_per_image = Counter()
    specks_ignored = 0
    for split in SPLIT_NAMES:
        area_fractions, box_count = [], 0
        for image_id in splits[split]:
            image_path = args.raw / "images" / f"{image_id}.jpg"
            mask = common.load_binary_mask(args.raw / "masks" / f"{image_id}.jpg")
            height, width = mask.shape
            if not mask.any():
                raise SystemExit(f"{image_id}: mask is empty after thresholding")

            boxes = common.mask_to_boxes(mask)
            all_regions = common.mask_to_boxes(mask, min_area_fraction=0)
            specks_ignored += len(all_regions) - len(boxes)
            if not boxes:
                raise SystemExit(f"{image_id}: no polyp region above the minimum size")

            (args.out / "images" / split).mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, args.out / "images" / split / image_path.name)
            common.save_binary_mask(mask, args.out / "masks" / split / f"{image_id}.png")
            common.write_yolo_label(args.out / "labels" / split / f"{image_id}.txt", boxes, width, height)

            official_boxes = [(b["xmin"], b["ymin"], b["xmax"], b["ymax"]) for b in official[image_id]["bbox"]]
            extent_iou = common.box_iou(common.boxes_extent(boxes), common.boxes_extent(official_boxes))
            comparison_rows.append({
                "image_id": image_id,
                "split": split,
                "official_boxes": len(official_boxes),
                "mask_boxes": len(boxes),
                "extent_iou": round(extent_iou, 4),
                "agrees": extent_iou >= BOX_AGREEMENT_IOU and len(official_boxes) == len(boxes),
            })
            boxes_per_image[len(boxes)] += 1
            box_count += len(boxes)
            area_fractions.append(float(mask.mean()))

        split_stats[split] = {
            "images": len(splits[split]),
            "boxes": box_count,
            "polyp_area_percent_mean": round(100 * float(np.mean(area_fractions)), 2),
            "polyp_area_percent_median": round(100 * float(np.median(area_fractions)), 2),
        }

    with open(reports_dir / "box_comparison.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0]))
        writer.writeheader()
        writer.writerows(comparison_rows)

    extent_ious = [row["extent_iou"] for row in comparison_rows]
    summary.update({
        "seed": args.seed,
        "splits": split_stats,
        "boxes_per_image": dict(sorted(boxes_per_image.items())),
        "specks_ignored_for_boxes": specks_ignored,
        "official_vs_mask_extent_iou_mean": round(float(np.mean(extent_ious)), 4),
        "official_vs_mask_extent_iou_below_0.9": sum(iou < BOX_AGREEMENT_IOU for iou in extent_ious),
        "official_vs_mask_box_count_differs": sum(r["official_boxes"] != r["mask_boxes"] for r in comparison_rows),
    })
    (reports_dir / "verification.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps(summary, indent=2))
    print(f"Processed dataset written to {args.out}")


if __name__ == "__main__":
    main()
