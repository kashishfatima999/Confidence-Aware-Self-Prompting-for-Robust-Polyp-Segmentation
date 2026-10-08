"""End-to-end test of src/data/prepare_kvasir.py on a small synthetic dataset.

Builds 6 fake Kvasir-style images (JPG image, JPG mask, official box file) in a
temporary folder, runs the script with a 3/1/2 split, and checks every output.
No real data is needed, so this runs on CI.
"""

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "src" / "data" / "prepare_kvasir.py"

# id -> (width, height, polyp regions as (xmin, ymin, xmax, ymax), speck regions)
SYNTHETIC = {
    "img01": (200, 100, [(10, 20, 60, 70)], []),
    "img02": (160, 120, [(5, 5, 50, 50), (100, 60, 150, 110)], []),  # two polyps
    "img03": (200, 100, [(0, 0, 200, 100)], []),  # polyp fills the frame
    "img04": (120, 120, [(30, 30, 90, 90)], [(2, 2, 4, 4)]),  # polyp + one speck (4 px = 0.03%)
    "img05": (300, 150, [(100, 20, 220, 130)], []),
    "img06": (150, 150, [(60, 10, 140, 70)], []),
}


def build_raw_dataset(raw):
    (raw / "images").mkdir(parents=True)
    (raw / "masks").mkdir()
    official = {}
    for image_id, (w, h, polyps, specks) in SYNTHETIC.items():
        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
        Image.fromarray(image).save(raw / "images" / f"{image_id}.jpg", quality=90)
        mask = np.zeros((h, w), dtype=np.uint8)
        for xmin, ymin, xmax, ymax in polyps + specks:
            mask[ymin:ymax, xmin:xmax] = 255
        # save as RGB JPG like the real dataset; high quality keeps the regions intact
        Image.fromarray(np.stack([mask] * 3, axis=-1)).save(raw / "masks" / f"{image_id}.jpg", quality=100)
        official[image_id] = {
            "height": h,
            "width": w,
            "bbox": [{"label": "polyp", "xmin": x0, "ymin": y0, "xmax": x1, "ymax": y1} for x0, y0, x1, y1 in polyps],
        }
    (raw / "kavsir_bboxes.json").write_text(json.dumps(official))
    return official


def run_script(raw, out, splits_dir, extra=()):
    cmd = [sys.executable, "-I", str(SCRIPT), "--raw", str(raw), "--out", str(out), "--splits-dir", str(splits_dir),
           "--train", "3", "--val", "1", "--test", "2", "--expected-images", str(len(SYNTHETIC)), *extra]
    return subprocess.run(cmd, capture_output=True, text=True)


class PrepareKvasirTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.raw = base / "raw" / "Kvasir-SEG"
        self.out = base / "processed"
        self.splits_dir = base / "splits"
        self.official = build_raw_dataset(self.raw)

    def tearDown(self):
        self.tmp.cleanup()

    def test_full_run_produces_consistent_outputs(self):
        result = run_script(self.raw, self.out, self.splits_dir)
        self.assertEqual(result.returncode, 0, result.stderr)

        # split files: sizes, no overlap, all ids covered
        splits = {s: (self.splits_dir / f"kvasir_{s}.txt").read_text().split() for s in ("train", "val", "test")}
        self.assertEqual([len(splits[s]) for s in ("train", "val", "test")], [3, 1, 2])
        all_ids = sorted(i for s in splits.values() for i in s)
        self.assertEqual(all_ids, sorted(SYNTHETIC))

        # per image: copy identical, mask binary, labels match the regions
        for split, ids in splits.items():
            for image_id in ids:
                w, h, polyps, _ = SYNTHETIC[image_id]
                copied = self.out / "images" / split / f"{image_id}.jpg"
                self.assertEqual(copied.read_bytes(), (self.raw / "images" / f"{image_id}.jpg").read_bytes())
                mask = np.array(Image.open(self.out / "masks" / split / f"{image_id}.png"))
                self.assertEqual(mask.shape, (h, w))
                self.assertTrue(set(np.unique(mask).tolist()) <= {0, 255})
                lines = (self.out / "labels" / split / f"{image_id}.txt").read_text().splitlines()
                self.assertEqual(len(lines), len(polyps), image_id)  # specks produce no label line
                for line in lines:
                    parts = line.split()
                    self.assertEqual(parts[0], "0")
                    self.assertTrue(all(0 < float(v) <= 1 for v in parts[1:]))

        # reports
        summary = json.loads((self.out / "reports" / "verification.json").read_text())
        self.assertEqual(summary["images"], len(SYNTHETIC))
        self.assertEqual(summary["unreadable_files"], 0)
        self.assertEqual(summary["empty_masks"], 0)
        self.assertEqual(summary["seed"], 42)
        self.assertEqual(summary["specks_ignored_for_boxes"], 1)
        self.assertEqual(summary["boxes_per_image"], {"1": 5, "2": 1})
        with open(self.out / "reports" / "box_comparison.csv", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), len(SYNTHETIC))
        self.assertTrue(all(row["agrees"] == "True" for row in rows), rows)

        # raw data untouched
        self.assertEqual(sorted(p.name for p in (self.raw / "images").iterdir()), sorted(f"{i}.jpg" for i in SYNTHETIC))

    def test_second_run_reuses_split_files(self):
        first = run_script(self.raw, self.out, self.splits_dir)
        self.assertEqual(first.returncode, 0, first.stderr)
        before = {s: (self.splits_dir / f"kvasir_{s}.txt").read_text() for s in ("train", "val", "test")}
        second = run_script(self.raw, self.out, self.splits_dir, extra=["--seed", "7"])  # different seed must be ignored
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertIn("reused", second.stdout)
        after = {s: (self.splits_dir / f"kvasir_{s}.txt").read_text() for s in ("train", "val", "test")}
        self.assertEqual(before, after)

    def test_missing_mask_stops_before_writing_output(self):
        (self.raw / "masks" / "img03.jpg").unlink()
        result = run_script(self.raw, self.out, self.splits_dir)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("without a mask", result.stderr)
        self.assertFalse(self.out.exists())
        self.assertFalse(self.splits_dir.exists())

    def test_empty_mask_is_reported(self):
        w, h = SYNTHETIC["img05"][0], SYNTHETIC["img05"][1]
        Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8)).save(self.raw / "masks" / "img05.jpg", quality=100)
        result = run_script(self.raw, self.out, self.splits_dir)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("img05: mask is empty", result.stderr)

    def test_split_sizes_must_add_up(self):
        result = run_script(self.raw, self.out, self.splits_dir, extra=["--train", "4"])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("do not add up", result.stderr)


if __name__ == "__main__":
    unittest.main()
