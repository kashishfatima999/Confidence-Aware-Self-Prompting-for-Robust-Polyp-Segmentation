"""End-to-end test of src/data/make_corruptions.py on a small synthetic test split.

Builds 3 fake processed Kvasir-style test images (JPG image, PNG mask, YOLO label)
in a temporary folder, runs the script, and checks the 9 output settings.
No real data is needed, so this runs on CI.
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
SCRIPT = REPO_ROOT / "src" / "data" / "make_corruptions.py"
IDS = ["imga", "imgb", "imgc"]
SETTINGS = [f"{name}_s{s}" for name in ("blur", "brightness_contrast", "noise") for s in (1, 2, 3)]


def build_processed(src):
    for folder in ("images", "masks", "labels"):
        (src / folder / "test").mkdir(parents=True)
    rng = np.random.default_rng(0)
    for image_id in IDS:
        image = rng.integers(0, 256, size=(80, 100, 3), dtype=np.uint8)
        Image.fromarray(image).save(src / "images/test" / f"{image_id}.jpg", quality=95)
        mask = np.zeros((80, 100), dtype=np.uint8)
        mask[20:50, 30:70] = 255
        Image.fromarray(mask).save(src / "masks/test" / f"{image_id}.png")
        (src / "labels/test" / f"{image_id}.txt").write_text("0 0.500000 0.437500 0.400000 0.375000\n")


def run_script(src, out, split_file):
    cmd = [sys.executable, "-I", str(SCRIPT), "--src", str(src), "--out", str(out), "--split-file", str(split_file)]
    return subprocess.run(cmd, capture_output=True, text=True)


class MakeCorruptionsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.src = base / "kvasir"
        self.out = base / "kvasir-corrupted"
        self.split_file = base / "kvasir_test.txt"
        self.split_file.write_text("\n".join(IDS) + "\n")
        build_processed(self.src)

    def tearDown(self):
        self.tmp.cleanup()

    def test_all_settings_written_with_unchanged_masks_and_labels(self):
        result = run_script(self.src, self.out, self.split_file)
        self.assertEqual(result.returncode, 0, result.stderr)
        manifest = json.loads((self.out / "manifest.json").read_text())
        self.assertEqual([s["name"] for s in manifest["settings"]], SETTINGS)
        for setting in SETTINGS:
            for image_id in IDS:
                image = np.array(Image.open(self.out / setting / "images/test" / f"{image_id}.png"))
                original = np.array(Image.open(self.src / "images/test" / f"{image_id}.jpg"))
                self.assertEqual(image.shape, original.shape, setting)
                self.assertFalse(np.array_equal(image, original), f"{setting} did not change {image_id}")
                self.assertEqual((self.out / setting / "masks/test" / f"{image_id}.png").read_bytes(),
                                 (self.src / "masks/test" / f"{image_id}.png").read_bytes())
                self.assertEqual((self.out / setting / "labels/test" / f"{image_id}.txt").read_text(),
                                 (self.src / "labels/test" / f"{image_id}.txt").read_text())

    def test_reruns_are_identical(self):
        run_script(self.src, self.out, self.split_file)
        first = (self.out / "noise_s3/images/test/imga.png").read_bytes()
        run_script(self.src, self.out, self.split_file)
        self.assertEqual(first, (self.out / "noise_s3/images/test/imga.png").read_bytes())

    def test_source_is_not_modified(self):
        before = {p: p.read_bytes() for p in self.src.rglob("*") if p.is_file()}
        run_script(self.src, self.out, self.split_file)
        after = {p: p.read_bytes() for p in self.src.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_missing_input_stops_before_writing(self):
        (self.src / "masks/test/imgb.png").unlink()
        result = run_script(self.src, self.out, self.split_file)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.out.exists())


if __name__ == "__main__":
    unittest.main()
