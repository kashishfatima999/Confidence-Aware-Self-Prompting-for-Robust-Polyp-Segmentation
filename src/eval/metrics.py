"""Segmentation and detection metrics for the YOLO -> SAM 2 experiments (decisions DL-18, DL-19, DL-32).

Conventions
- A prediction and a ground truth are boolean HxW arrays (True = polyp).
- An image with no prediction is scored as an empty mask: Dice = IoU = 0 (DL-18). It is NOT skipped.
- "Per-image" metrics are the mean over images; "pooled" metrics sum intersections and unions over
  all pixels of the dataset first (the paper's definition). Both are reported (DL-19).
- Boxes are (xmin, ymin, xmax, ymax) with exclusive maxima, as everywhere else in this repository.

Pure numpy, so it runs (and is tested) without a GPU.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
import common  # noqa: E402

EPS = 1e-8


def dice(pred, gt):
    """Dice coefficient of two boolean masks. Empty prediction vs non-empty truth = 0.0."""
    pred = np.asarray(pred, bool)
    gt = np.asarray(gt, bool)
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0  # both empty: nothing to miss
    return float(2.0 * np.logical_and(pred, gt).sum() / denom)


def iou(pred, gt):
    """Intersection over union of two boolean masks. Empty prediction vs non-empty truth = 0.0."""
    pred = np.asarray(pred, bool)
    gt = np.asarray(gt, bool)
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(pred, gt).sum() / union)


def per_image_scores(pred, gt):
    """Everything the per-image CSV needs for one image."""
    pred = np.asarray(pred, bool)
    gt = np.asarray(gt, bool)
    return {
        "dice": dice(pred, gt),
        "iou": iou(pred, gt),
        "gt_area_percent": 100.0 * float(gt.mean()),
        "pred_area_percent": 100.0 * float(pred.mean()),
        "pred_empty": bool(not pred.any()),
    }


class PooledScores:
    """Accumulates pixel counts over a dataset to give the paper-style pooled Dice / IoU."""

    def __init__(self):
        self.intersection = 0
        self.union = 0
        self.pred_sum = 0
        self.gt_sum = 0

    def add(self, pred, gt):
        pred = np.asarray(pred, bool)
        gt = np.asarray(gt, bool)
        self.intersection += int(np.logical_and(pred, gt).sum())
        self.union += int(np.logical_or(pred, gt).sum())
        self.pred_sum += int(pred.sum())
        self.gt_sum += int(gt.sum())

    def result(self):
        return {
            "pooled_dice": 2.0 * self.intersection / (self.pred_sum + self.gt_sum + EPS),
            "pooled_iou": self.intersection / (self.union + EPS),
        }


def match_boxes(pred_boxes, gt_boxes, iou_threshold=0.5):
    """Greedy one-to-one matching of predicted to ground-truth boxes by IoU.

    Predictions are taken in the given order (sort them by confidence first). Returns
    (tp, fp, fn): a prediction is a true positive if it matches an unmatched GT box with
    IoU >= threshold.
    """
    unmatched = list(range(len(gt_boxes)))
    tp = 0
    for pred in pred_boxes:
        best, best_iou = None, iou_threshold
        for j in unmatched:
            value = common.box_iou(pred, gt_boxes[j])
            if value >= best_iou:
                best, best_iou = j, value
        if best is not None:
            unmatched.remove(best)
            tp += 1
    fp = len(pred_boxes) - tp
    fn = len(unmatched)
    return tp, fp, fn


def precision_recall(tp, fp, fn):
    return {
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
    }


def bootstrap_ci(values, n_resamples=1000, alpha=0.05, seed=42):
    """Percentile bootstrap confidence interval of the mean of `values`."""
    values = np.asarray(values, float)
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples)
    for i in range(n_resamples):
        means[i] = values[rng.integers(0, len(values), len(values))].mean()
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def summarize(rows, pooled, tp=0, fp=0, fn=0):
    """Dataset summary from the per-image rows (dicts with 'dice', 'iou', 'pred_empty') and a PooledScores."""
    dices = [r["dice"] for r in rows]
    ious = [r["iou"] for r in rows]
    summary = {
        "images": len(rows),
        "mean_dice": float(np.mean(dices)) if rows else float("nan"),
        "mean_dice_ci95": list(bootstrap_ci(dices)),
        "mean_iou": float(np.mean(ious)) if rows else float("nan"),
        "mean_iou_ci95": list(bootstrap_ci(ious)),
        "median_dice": float(np.median(dices)) if rows else float("nan"),
        "images_with_empty_prediction": int(sum(r["pred_empty"] for r in rows)),
        "detection_tp": tp,
        "detection_fp": fp,
        "detection_fn": fn,
    }
    summary.update(pooled.result())
    summary.update(precision_recall(tp, fp, fn))
    return summary
