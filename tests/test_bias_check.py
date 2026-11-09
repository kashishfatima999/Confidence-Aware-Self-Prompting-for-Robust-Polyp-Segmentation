import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "eval"))
import bias_check


class BiasCheckTests(unittest.TestCase):
    def test_image_key_removes_extension(self):
        self.assertEqual(bias_check.image_key("ABC123.JPG"), "abc123")

    def test_bootstrap_ci_single_value(self):
        self.assertEqual(bias_check.bootstrap_ci([0.25]), (0.25, 0.25))

    def test_tertile_labels(self):
        ref = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9])
        got = bias_check._tertile_labels(ref, pd.Series([1, 5, 9])).tolist()
        self.assertEqual(got, ["low", "medium", "high"])

    def test_paired_gap_uses_matching_image_ids(self):
        b = pd.DataFrame({
            "image_key": ["a", "b", "c"], "dice": [0.2, 0.4, 0.8]
        })
        c = pd.DataFrame({
            "image_key": ["a", "b", "d"], "dice": [0.3, 0.5, 0.9]
        })
        gap, lo, hi, n = bias_check._paired_gap({"B": b, "C": c})
        self.assertAlmostEqual(gap, 0.1)
        self.assertEqual(n, 2)
        self.assertAlmostEqual(lo, 0.1)
        self.assertAlmostEqual(hi, 0.1)

    def test_end_to_end_creates_factor_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for run in ["baseline", "casp", "threshold_only"]:
                folder = root / "runs" / run / "kvasir_test"
                folder.mkdir(parents=True)
                rows = []
                for i, d in enumerate([0.2, 0.4, 0.6, 0.8, 0.9, 0.7]):
                    rows.append({
                        "image_id": f"img{i}", "dice": d + (0.05 if run == "casp" else 0),
                        "gt_area_percent": i + 1, "n_gt_polyps": 1 if i < 4 else 2,
                        "pred_empty": i == 0, "box_fp": i % 2,
                    })
                pd.DataFrame(rows).to_csv(folder / "per_image.csv", index=False)
            stats = pd.DataFrame({
                "dataset": ["Kvasir-SEG"] * 6,
                "split": ["test"] * 6,
                "image": [f"img{i}.jpg" for i in range(6)],
                "width": [300, 400, 500, 600, 700, 800],
                "height": [300, 400, 500, 600, 700, 800],
                "brightness": [10, 20, 30, 40, 50, 60],
                "contrast": [5, 10, 15, 20, 25, 30],
            })
            stats_path = root / "stats.csv"
            stats.to_csv(stats_path, index=False)
            tables, _ = bias_check.build_bias_tables(root / "runs", stats_path)
            self.assertIn("polyp_size", tables)
            self.assertIn("brightness", tables)
            self.assertIn("dataset", tables)
            self.assertFalse(tables["polyp_size"].empty)
            row = tables["polyp_size"].iloc[0]
            self.assertIn("c_minus_b", row.index)
            self.assertIn("miss_rate_C", row.index)


if __name__ == "__main__":
    unittest.main()