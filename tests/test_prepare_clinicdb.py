"""End-to-end test of src/data/prepare_clinicdb.py on a small synthetic dataset.

Builds 4 fake CVC-ClinicDB-style frames (PNG image + PNG mask in the Kaggle
mirror's Original / Ground Truth layout) in a temporary folder, runs the script,
and checks every output. No real data is needed, so this runs on CI.
"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "src" / "data" / "prepare_clinicdb.py"

# id -> polyp regions as (xmin, ymin, xmax, ymax) on a 96 x 72 frame (same aspect as 384 x 288)
SYNTHETIC = {
    "1": [(10, 10, 40, 40)],
    "2": [(5, 5, 30, 30), (60, 40, 90, 70)],  # two polyps
    "3": [(0, 0, 96, 72)],  # polyp fills the frame
    "4": [(20, 20, 70, 60)],
}
WIDTH, HEIGHT = 96, 72


def build_raw_dataset(raw):
    (raw / "Original").mkdir(parents=True)
    (raw / "Ground Truth").mkdir()
    rng = np.random.default_rng(0)
    for image_id, polyps in SYNTHETIC.items():
        image = rng.integers(0, 256, size=(HEIGHT, WIDTH, 3), dtype=np.uint8)
        Image.fromarray(image).save(raw / "Original" / f"{image_id}.png")
        mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        for xmin, ymin, xmax, ymax in polyps:
            mask[ymin:ymax, xmin:xmax] = 255
        Image.fromarray(mask).save(raw / "Ground Truth" / f"{image_id}.png")


def run_script(raw, out, extra=()):
    cmd = [sys.executable, "-I", str(SCRIPT), "--raw", str(raw), "--out", str(out),
           "--expected-images", str(len(SYNTHETIC)), *extra]
    return subprocess.run(cmd, capture_output=True, text=True)


class PrepareClinicDBTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.raw = base / "raw" / "PNG"
        self.out = base / "processed"
        build_raw_dataset(self.raw)

    def tearDown(self):
        self.tmp.cleanup()

    def test_full_run_produces_consistent_outputs(self):
        result = run_script(self.raw, self.out)
        self.assertEqual(result.returncode, 0, result.stderr)

        for image_id, polyps in SYNTHETIC.items():
            copied = self.out / "images" / "test" / f"{image_id}.png"
            self.assertEqual(copied.read_bytes(), (self.raw / "Original" / f"{image_id}.png").read_bytes())
            mask = np.array(Image.open(self.out / "masks" / "test" / f"{image_id}.png"))
            self.assertEqual(mask.shape, (HEIGHT, WIDTH))
            self.assertTrue(set(np.unique(mask).tolist()) <= {0, 255})
            lines = (self.out / "labels" / "test" / f"{image_id}.txt").read_text().splitlines()
            self.assertEqual(len(lines), len(polyps), image_id)
            for line in lines:
                parts = line.split()
                self.assertEqual(parts[0], "0")
                self.assertTrue(all(0 < float(v) <= 1 for v in parts[1:]))

        summary = json.loads((self.out / "reports" / "verification.json").read_text())
        self.assertEqual(summary["images"], len(SYNTHETIC))
        self.assertEqual(summary["split"], {"test": len(SYNTHETIC)})
        self.assertEqual(summary["total_boxes"], 5)
        self.assertEqual(summary["boxes_per_image"], {"1": 3, "2": 1})
        self.assertEqual(summary["image_resolution"], f"{WIDTH}x{HEIGHT}")
        self.assertEqual(summary["distinct_resolutions"], 1)
        self.assertEqual(summary["unreadable_files"], 0)
        self.assertEqual(summary["empty_masks"], 0)
        self.assertFalse(summary["training_on_clinicdb"])
        self.assertFalse((self.out / "images" / "train").exists())

        # raw data untouched
        self.assertEqual(sorted(p.name for p in (self.raw / "Original").iterdir()), sorted(f"{i}.png" for i in SYNTHETIC))

    def test_missing_mask_stops_before_writing_output(self):
        (self.raw / "Ground Truth" / "2.png").unlink()
        result = run_script(self.raw, self.out)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("without a mask", result.stderr)
        self.assertFalse(self.out.exists())

    def test_empty_mask_is_reported(self):
        Image.fromarray(np.zeros((HEIGHT, WIDTH), dtype=np.uint8)).save(self.raw / "Ground Truth" / "4.png")
        result = run_script(self.raw, self.out)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("4: mask is empty", result.stderr)

    def test_wrong_image_count_is_reported(self):
        result = run_script(self.raw, self.out, extra=["--expected-images", "612"])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("expected 612 images", result.stderr)


if __name__ == "__main__":
    unittest.main()
