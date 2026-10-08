"""Shared helpers for dataset preparation.

Used by prepare_kvasir.py and prepare_clinicdb.py so both datasets get
identical mask cleaning, box derivation and YOLO label formatting.

Box convention everywhere in this project: (xmin, ymin, xmax, ymax) in pixels,
with xmax/ymax exclusive (a box covering the whole image is (0, 0, width, height)).
The Kvasir-SEG bounding-box file agrees with this to within 1 pixel per side.
"""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# Masks are stored as 0..255 grayscale. Anything above this value is polyp.
MASK_THRESHOLD = 127

# Connected components smaller than this fraction of the image are treated as
# compression specks, not polyps, when deriving boxes from a mask.
MIN_COMPONENT_AREA_FRACTION = 0.0005

POLYP_CLASS_ID = 0


def load_binary_mask(path):
    """Read a mask image and return a boolean array (True = polyp)."""
    gray = np.array(Image.open(path).convert("L"))
    return gray > MASK_THRESHOLD


def save_binary_mask(mask, path):
    """Save a boolean mask as a lossless PNG with values 0 and 255."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)


def mask_to_boxes(mask, min_area_fraction=MIN_COMPONENT_AREA_FRACTION):
    """Return one box per connected polyp region, sorted left to right."""
    height, width = mask.shape
    min_area = min_area_fraction * height * width
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    boxes = []
    for label in range(1, count):  # label 0 is the background
        x, y, w, h, area = stats[label]
        if area >= min_area:
            boxes.append((int(x), int(y), int(x + w), int(y + h)))
    return sorted(boxes)


def box_iou(a, b):
    """Intersection over union of two (xmin, ymin, xmax, ymax) boxes."""
    inter_w = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    inter_h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = inter_w * inter_h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def boxes_extent(boxes):
    """Smallest single box that contains every box in the list."""
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def box_to_yolo(box, image_width, image_height):
    """Convert a pixel box to YOLO format: (x_center, y_center, width, height), all 0..1."""
    xmin, ymin, xmax, ymax = box
    return (
        (xmin + xmax) / 2 / image_width,
        (ymin + ymax) / 2 / image_height,
        (xmax - xmin) / image_width,
        (ymax - ymin) / image_height,
    )


def write_yolo_label(path, boxes, image_width, image_height):
    """Write one YOLO label file: one line per box, 'class x_center y_center width height'."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for box in boxes:
        xc, yc, w, h = box_to_yolo(box, image_width, image_height)
        lines.append(f"{POLYP_CLASS_ID} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""))
