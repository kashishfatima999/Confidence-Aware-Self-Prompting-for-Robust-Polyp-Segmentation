import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "eval"))
import seed_compare as sc  #noqa: E402


class SeedViewTests(unittest.TestCase):
    def test_oracle_taken_from_seed_42_when_missing(self):
        with tempfile.TemporaryDirectory() as s42, tempfile.TemporaryDirectory() as s43, tempfile.TemporaryDirectory() as view:
            for run in sc.RUNS:
                (Path(s42) / run).mkdir()
            for run in ["baseline", "threshold_only", "casp"]:
                (Path(s43) / run).mkdir()
            v = sc.seed_view(s43, s42, view)
            self.assertEqual((v / "baseline").resolve(), (Path(s43) / "baseline").resolve())
            self.assertEqual((v / "oracle").resolve(), (Path(s42) / "oracle").resolve())

    def test_markdown_puts_seeds_side_by_side(self):
        import pandas as pd
        df = pd.DataFrame([{"seed": 42, "dataset": "kvasir_test", "quantity": "Dice B", "value": 0.8, "ci_low": 0.7, "ci_high": 0.9},
                           {"seed": 43, "dataset": "kvasir_test", "quantity": "Dice B", "value": 0.81, "ci_low": 0.7, "ci_high": 0.9}])
        self.assertIn("| Dice B | 0.800 [0.700, 0.900] | 0.810 [0.700, 0.900] |", sc.markdown(df))


if __name__ == "__main__":
    unittest.main()
