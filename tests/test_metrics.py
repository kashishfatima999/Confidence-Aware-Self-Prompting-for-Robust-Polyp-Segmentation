"""Unit tests for src/eval/metrics.py (synthetic masks, no data, no GPU)."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "eval"))
import metrics  # noqa: E402


def square(h=20, w=30, box=None):
    m = np.zeros((h, w), bool)
    if box:
        x0, y0, x1, y1 = box
        m[y0:y1, x0:x1] = True
    return m


class DiceIoUTests(unittest.TestCase):
    def test_identical_masks(self):
        m = square(box=(2, 2, 12, 12))
        self.assertAlmostEqual(metrics.dice(m, m), 1.0)
        self.assertAlmostEqual(metrics.iou(m, m), 1.0)

    def test_half_overlap(self):
        a = square(box=(0, 0, 10, 10))   # 100 px
        b = square(box=(5, 0, 15, 10))   # 100 px, 50 px overlap
        self.assertAlmostEqual(metrics.dice(a, b), 2 * 50 / 200)
        self.assertAlmostEqual(metrics.iou(a, b), 50 / 150)

    def test_empty_prediction_scores_zero_not_skipped(self):
        gt = square(box=(2, 2, 12, 12))
        self.assertEqual(metrics.dice(square(), gt), 0.0)
        self.assertEqual(metrics.iou(square(), gt), 0.0)
        self.assertTrue(metrics.per_image_scores(square(), gt)["pred_empty"])

    def test_both_empty_is_perfect(self):
        self.assertEqual(metrics.dice(square(), square()), 1.0)

    def test_per_image_scores_fields(self):
        s = metrics.per_image_scores(square(box=(0, 0, 10, 10)), square(box=(0, 0, 10, 10)))
        self.assertAlmostEqual(s["gt_area_percent"], 100 * 100 / 600)
        self.assertFalse(s["pred_empty"])


class PooledTests(unittest.TestCase):
    def test_pooled_differs_from_mean(self):
        # image 1: perfect on a big polyp; image 2: total miss on a tiny polyp
        p = metrics.PooledScores()
        big = square(box=(0, 0, 20, 20))
        tiny = square(box=(0, 0, 2, 2))
        p.add(big, big)
        p.add(square(), tiny)
        r = p.result()
        self.assertAlmostEqual(r["pooled_dice"], 2 * 400 / (400 + 404), places=6)
        self.assertAlmostEqual(r["pooled_iou"], 400 / 404, places=6)
        mean_dice = np.mean([metrics.dice(big, big), metrics.dice(square(), tiny)])
        self.assertAlmostEqual(mean_dice, 0.5)  # per-image mean punishes the miss much more


class BoxMatchingTests(unittest.TestCase):
    def test_counts(self):
        gt = [(0, 0, 10, 10), (50, 50, 60, 60)]
        pred = [(0, 0, 10, 10), (51, 51, 61, 61), (90, 90, 95, 95)]
        tp, fp, fn = metrics.match_boxes(pred, gt)
        self.assertEqual((tp, fp, fn), (2, 1, 0))

    def test_one_gt_matched_once(self):
        gt = [(0, 0, 10, 10)]
        pred = [(0, 0, 10, 10), (1, 1, 11, 11)]
        self.assertEqual(metrics.match_boxes(pred, gt), (1, 1, 0))

    def test_no_predictions(self):
        self.assertEqual(metrics.match_boxes([], [(0, 0, 10, 10)]), (0, 0, 1))
        self.assertEqual(metrics.precision_recall(0, 0, 1), {"precision": 0.0, "recall": 0.0})


class SummaryTests(unittest.TestCase):
    def test_bootstrap_ci_contains_mean_and_is_deterministic(self):
        values = [0.9, 0.95, 0.8, 0.0, 0.85]
        lo, hi = metrics.bootstrap_ci(values)
        self.assertLessEqual(lo, np.mean(values))
        self.assertGreaterEqual(hi, np.mean(values))
        self.assertEqual(metrics.bootstrap_ci(values), metrics.bootstrap_ci(values))

    def test_summarize_fields(self):
        gt = square(box=(0, 0, 10, 10))
        rows = [metrics.per_image_scores(gt, gt), metrics.per_image_scores(square(), gt)]
        pooled = metrics.PooledScores()
        pooled.add(gt, gt)
        pooled.add(square(), gt)
        s = metrics.summarize(rows, pooled, tp=1, fp=0, fn=1)
        self.assertEqual(s["images"], 2)
        self.assertAlmostEqual(s["mean_dice"], 0.5)
        self.assertEqual(s["images_with_empty_prediction"], 1)
        self.assertAlmostEqual(s["recall"], 0.5)
        self.assertAlmostEqual(s["pooled_dice"], 2 * 100 / 300, places=6)
        self.assertEqual(len(s["mean_dice_ci95"]), 2)


if __name__ == "__main__":
    unittest.main()
