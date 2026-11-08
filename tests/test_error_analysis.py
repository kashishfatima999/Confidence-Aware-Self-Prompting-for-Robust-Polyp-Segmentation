import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "eval"))
import error_analysis as ea  # noqa: E402


def row(dice, gt=10.0, pred=10.0, empty=False, tp=1, fp=0):
    return {"dice": dice, "gt_area_percent": gt, "pred_area_percent": pred, "pred_empty": empty, "box_tp": tp, "box_fp": fp}


class TaxonomyTests(unittest.TestCase):
    def test_missed(self):
        self.assertEqual(ea.label(row(0.0, pred=0.0, empty=True, tp=0)), "missed")

    def test_false_positive(self):
        self.assertEqual(ea.label(row(0.0, tp=0, fp=1)), "false_positive")

    def test_extra_box_wins_over_dice(self):
        self.assertEqual(ea.label(row(0.95, tp=1, fp=1)), "extra_box")
        self.assertEqual(ea.label(row(0.60, pred=15.0, tp=1, fp=1)), "extra_box")

    def test_good(self):
        self.assertEqual(ea.label(row(0.7)), "good")
        self.assertEqual(ea.label(row(0.9, tp=0, fp=1)), "good")

    def test_under_and_over(self):
        self.assertEqual(ea.label(row(0.5, pred=4.0)), "under_segmentation")
        self.assertEqual(ea.label(row(0.5, pred=25.0)), "over_segmentation")

    def test_every_label_listed(self):
        self.assertEqual(set(ea.LABELS), {"good", "extra_box", "under_segmentation", "over_segmentation", "false_positive", "missed"})


if __name__ == "__main__":
    unittest.main()
