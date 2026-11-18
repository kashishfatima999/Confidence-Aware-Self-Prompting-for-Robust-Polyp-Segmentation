import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "eval"))
import compare  # noqa: E402


def frame(rows):
    cols = ["image_id", "dice", "iou", "gt_area_percent", "pred_area_percent", "pred_empty", "box_tp", "box_fp", "box_fn"]
    return pd.DataFrame(rows, columns=cols).set_index("image_id")


class PixelMetricTests(unittest.TestCase):
    def test_perfect_prediction(self):
        df = compare.add_pixel_metrics(frame([("a", 1.0, 1.0, 10.0, 10.0, False, 1, 0, 0)]))
        r = df.iloc[0]
        self.assertAlmostEqual(r.pixel_precision, 1.0)
        self.assertAlmostEqual(r.pixel_recall, 1.0)
        self.assertAlmostEqual(r.pixel_accuracy, 1.0)
        self.assertAlmostEqual(r.area_error, 0.0)

    def test_empty_prediction(self):
        r = compare.add_pixel_metrics(frame([("a", 0.0, 0.0, 12.0, 0.0, True, 0, 0, 1)])).iloc[0]
        self.assertEqual(r.pixel_precision, 0.0)
        self.assertEqual(r.pixel_recall, 0.0)
        self.assertAlmostEqual(r.pixel_accuracy, 0.88)
        self.assertAlmostEqual(r.area_error, 12.0)

    def test_half_overlap(self):
        # GT 10% and prediction 10% of the image, overlapping on 5%: Dice 0.5
        r = compare.add_pixel_metrics(frame([("a", 0.5, 1 / 3, 10.0, 10.0, False, 0, 1, 1)])).iloc[0]
        self.assertAlmostEqual(r.pixel_precision, 0.5)
        self.assertAlmostEqual(r.pixel_recall, 0.5)
        self.assertAlmostEqual(r.pixel_accuracy, 0.90)

    def test_over_segmentation(self):
        # prediction 20% contains the whole 10% GT: Dice = 2*10/30
        r = compare.add_pixel_metrics(frame([("a", 2 / 3, 0.5, 10.0, 20.0, False, 1, 0, 0)])).iloc[0]
        self.assertAlmostEqual(r.pixel_precision, 0.5)
        self.assertAlmostEqual(r.pixel_recall, 1.0)
        self.assertAlmostEqual(r.area_error, 10.0)


class PairedTests(unittest.TestCase):
    def test_counts_and_direction(self):
        a = compare.add_pixel_metrics(frame([("x", 0.9, 0.8, 10, 10, False, 1, 0, 0), ("y", 0.5, 0.3, 10, 12, False, 1, 0, 0),
                                             ("z", 0.7, 0.5, 10, 9, False, 1, 0, 0)]))
        b = compare.add_pixel_metrics(frame([("x", 0.8, 0.7, 10, 10, False, 1, 0, 0), ("y", 0.6, 0.4, 10, 14, False, 1, 0, 0),
                                             ("z", 0.7, 0.5, 10, 9, False, 1, 0, 0)]))
        d = compare.paired(a, b, "dice")
        self.assertAlmostEqual(d["mean_diff"], 0.0)
        self.assertEqual((d["better"], d["worse"], d["tie"]), (1, 1, 1))
        e = compare.paired(a, b, "area_error")
        self.assertEqual((e["better"], e["worse"]), (1, 0))

    def test_build_on_tiny_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            for run, dice in [("baseline", 0.6), ("threshold_only", 0.8), ("casp", 0.8)]:
                path = Path(tmp) / run / "kvasir_test"
                path.mkdir(parents=True)
                frame([("a", dice, dice, 10, 10, False, 1, 0, 0), ("b", 0.0, 0.0, 5, 0, True, 0, 0, 1)]).to_csv(path / "per_image.csv")
            summary, pairs = compare.build(tmp)
            self.assertEqual(list(summary.run), ["B", "T", "C"])
            self.assertEqual(int(summary.iloc[0].missed), 1)
            c_vs_b = pairs[(pairs.run == "C") & (pairs.reference == "B") & (pairs.metric == "dice")].iloc[0]
            self.assertAlmostEqual(c_vs_b.mean_diff, 0.1)
            self.assertEqual(len(pairs[(pairs.run == "B")]), 0)


if __name__ == "__main__":
    unittest.main()
