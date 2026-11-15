import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "eval"))
import filter_analysis as fa  # noqa: E402


def frame(rows):
    df = pd.DataFrame(rows).set_index("image_id")
    return df


class FallbackTests(unittest.TestCase):
    def setUp(self):
        self.b = frame([
            {"image_id": "a", "max_confidence": 0.9, "dice": 0.95},   # confident box -> take B
            {"image_id": "b", "max_confidence": 0.4, "dice": 0.0},    # only uncertain -> take T
            {"image_id": "c", "max_confidence": 0.1, "dice": 0.0},    # nothing kept by either
        ])
        self.t = frame([
            {"image_id": "a", "max_confidence": 0.9, "dice": 0.70},
            {"image_id": "b", "max_confidence": 0.4, "dice": 0.80},
            {"image_id": "c", "max_confidence": 0.1, "dice": 0.0},
        ])

    def test_rows_come_from_the_right_run(self):
        f = fa.fallback(self.b, self.t)
        self.assertEqual(list(f["dice"]), [0.95, 0.80, 0.0])
        self.assertEqual(list(f["source"]), ["B", "other", "other"])

    def test_threshold_is_inclusive(self):
        self.b.loc["b", "max_confidence"] = self.t.loc["b", "max_confidence"] = 0.5
        self.assertEqual(fa.fallback(self.b, self.t).loc["b", "dice"], 0.0)

    def test_different_caches_are_refused(self):
        self.t.loc["a", "max_confidence"] = 0.3
        with self.assertRaises(ValueError):
            fa.fallback(self.b, self.t)


class LabelBoxTests(unittest.TestCase):
    def rec(self, box, conf, score=0.9):
        return {"box": box, "final_box": box, "confidence": conf, "score": score}

    def test_greedy_matching_in_confidence_order(self):
        gt = [(0, 0, 10, 10)]
        rows = fa.label_boxes([self.rec([0, 0, 10, 10], 0.3), self.rec([1, 1, 10, 10], 0.8)], gt)
        self.assertEqual([r["confidence"] for r in rows], [0.8, 0.3])
        self.assertEqual([r["tp"] for r in rows], [True, False])   # the second box finds the GT already taken
        self.assertTrue(rows[1]["higher_box_in_image"] and rows[1]["band"])

    def test_dropped_boxes_do_not_take_a_match(self):
        gt = [(0, 0, 10, 10)]
        rows = fa.label_boxes([self.rec([0, 0, 10, 10], 0.1), self.rec([0, 0, 10, 10], 0.3)], gt)
        self.assertEqual([(r["kept"], r["tp"]) for r in rows], [(True, True), (False, False)])

    def test_far_box_is_false(self):
        rows = fa.label_boxes([self.rec([50, 50, 60, 60], 0.9)], [(0, 0, 10, 10)])
        self.assertFalse(rows[0]["tp"])
        self.assertFalse(rows[0]["band"])


class AurocTests(unittest.TestCase):
    def test_perfect_and_reversed(self):
        self.assertEqual(fa.auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]), 1.0)
        self.assertEqual(fa.auroc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]), 0.0)

    def test_ties_count_half(self):
        self.assertEqual(fa.auroc([0.5, 0.5], [0, 1]), 0.5)

    def test_one_class_is_nan(self):
        self.assertTrue(pd.isna(fa.auroc([0.1, 0.2], [1, 1])))


class BoxTableTests(unittest.TestCase):
    def test_reads_run_and_oracle_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            for run, recs in [("threshold_only", [{"image_id": "x", "box": [0, 0, 10, 10], "final_box": [0, 0, 10, 10],
                                                   "confidence": 0.4, "score": 0.9, "branch": "confident"}]),
                              ("oracle", [{"image_id": "x", "box": [0, 0, 10, 10], "final_box": [0, 0, 10, 10],
                                           "confidence": 1.0, "score": 0.95, "branch": "confident"}]),
                              ("baseline", [])]:
                d = Path(tmp) / run / "kvasir_test"
                d.mkdir(parents=True)
                (d / "detections.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
                (d / "per_image.csv").write_text("image_id\n")
            t = fa.box_table(tmp)
            self.assertEqual(len(t), 1)
            self.assertTrue(t.iloc[0]["tp"] and t.iloc[0]["band"])


if __name__ == "__main__":
    unittest.main()
