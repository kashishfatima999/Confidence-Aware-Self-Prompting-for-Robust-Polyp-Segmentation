"""End-to-end test of src/pipeline/run_experiment.evaluate() with a mocked SAM 2 and a synthetic dataset.

Builds 3 fake processed images + masks and a detection cache in a temporary folder, runs the
evaluation loop for a CASP config, the baseline config and the oracle, and checks the files it writes.
No GPU, no torch, no real data.
"""

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "pipeline"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import refine  # noqa: E402
import run_experiment  # noqa: E402
from test_refine import MockPrompter  # noqa: E402

W, H = 100, 80
POLYPS = {"a": (20, 20, 60, 60), "b": (10, 10, 30, 40), "c": (50, 30, 90, 70)}


class ImageAwarePrompter:
    """MockPrompter whose 'true polyp' changes with the image that was set (keyed by image content)."""

    def __init__(self):
        self.current = None

    def set_image(self, rgb):
        self.current = MockPrompter(POLYPS[chr(ord("a") + int(rgb[0, 0, 0]))])

    def predict(self, box, point=None):
        return self.current.predict(box, point)


def build_dataset(base):
    images, masks = base / "images", base / "masks"
    images.mkdir(parents=True)
    masks.mkdir()
    cache = {}
    for k, (image_id, (x0, y0, x1, y1)) in enumerate(POLYPS.items()):
        rgb = np.full((H, W, 3), 100, np.uint8)
        rgb[0, 0, 0] = k  # lets the mock know which image it holds
        Image.fromarray(rgb).save(images / f"{image_id}.png")
        mask = np.zeros((H, W), np.uint8)
        mask[y0:y1, x0:x1] = 255
        Image.fromarray(mask).save(masks / f"{image_id}.png")
    # cache: image a has a good box at 0.9; image b a slightly small box at 0.4 (uncertain); image c nothing kept
    cache["a"] = {"width": W, "height": H, "boxes": [[20, 20, 60, 60, 0.9]]}
    cache["b"] = {"width": W, "height": H, "boxes": [[13, 13, 27, 37, 0.4], [70, 60, 95, 78, 0.3]]}
    cache["c"] = {"width": W, "height": H, "boxes": [[50, 30, 90, 70, 0.1]]}
    return images, masks, cache


class EvaluateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.images, self.masks, self.cache = build_dataset(self.base)

    def tearDown(self):
        self.tmp.cleanup()

    def run_cfg(self, cfg, name, oracle=False):
        out = self.base / "runs" / name / "synthetic"
        summary = run_experiment.evaluate(ImageAwarePrompter(), ["a", "b", "c"], self.images, self.masks, "png",
                                          self.cache, cfg, oracle, out, save_masks=True, log=lambda *_: None)
        return summary, out

    def test_casp_run_writes_everything(self):
        summary, out = self.run_cfg(refine.CaspConfig(s_min=0.3), "casp")
        self.assertEqual(summary["images"], 3)
        with open(out / "per_image.csv", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([r["image_id"] for r in rows], ["a", "b", "c"])
        self.assertEqual(float(rows[0]["dice"]), 1.0)                 # confident, perfect box
        self.assertGreater(float(rows[1]["dice"]), 0.5)              # uncertain, refined
        self.assertEqual(rows[1]["n_uncertain"], "2")
        self.assertEqual(rows[1]["n_rejected"], "1")                  # the far-away box is rejected
        self.assertEqual(rows[2]["pred_empty"], "True")               # below t_low: missed, scored 0
        self.assertEqual(float(rows[2]["dice"]), 0.0)
        self.assertEqual(summary["images_with_empty_prediction"], 1)
        self.assertEqual(summary["detection_fn"], 1)
        self.assertEqual(summary["casp"]["s_min"], 0.3)
        lines = (out / "detections.jsonl").read_text().splitlines()
        self.assertEqual(len(lines), 4)
        self.assertEqual(json.loads(lines[0])["branch"], "confident")
        self.assertTrue((out / "masks" / "a.png").exists())
        self.assertTrue((out / "summary.json").exists())

    def test_baseline_drops_everything_below_half(self):
        summary, _ = self.run_cfg(refine.CaspConfig.baseline(), "baseline")
        self.assertEqual(summary["images_with_empty_prediction"], 2)
        self.assertAlmostEqual(summary["mean_dice"], 1 / 3, places=4)
        self.assertEqual(summary["prompts_per_image"], 1 / 3)

    def test_oracle_uses_ground_truth_boxes(self):
        summary, _ = self.run_cfg(refine.CaspConfig.baseline(conf=0.0), "oracle", oracle=True)
        self.assertEqual(summary["images_with_empty_prediction"], 0)
        self.assertAlmostEqual(summary["mean_dice"], 1.0)
        self.assertTrue(summary["oracle"])


class ConfigTests(unittest.TestCase):
    def test_overrides(self):
        cfg = run_experiment.apply_overrides({"name": "x", "casp": {"t_low": 0.25}}, ["t_low=0.35", "gate=false", "self_prompt_iters=0"])
        self.assertEqual(cfg["casp"], {"t_low": 0.35, "gate": False, "self_prompt_iters": 0})
        self.assertEqual(refine.CaspConfig.from_dict(cfg["casp"]).t_low, 0.35)

    def test_json_config_and_dataset_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "c.json"
            p.write_text(json.dumps({"name": "n", "casp": {"t_low": 0.1}}))
            self.assertEqual(run_experiment.load_config(p)["casp"]["t_low"], 0.1)
        images, masks, ext = run_experiment.resolve_dataset("kvasir_corrupted/blur_s2")
        self.assertTrue(str(images).replace("\\", "/").endswith("kvasir-corrupted/blur_s2/images/test"))
        self.assertEqual(ext, "png")
        with self.assertRaises(SystemExit):
            run_experiment.resolve_dataset("nope")

    def test_yaml_experiment_configs_parse(self):
        try:
            import yaml  # noqa: F401
        except ImportError:
            self.skipTest("PyYAML not installed locally; CI installs it from requirements.txt")
        root = Path(__file__).resolve().parents[1] / "configs" / "experiments"
        names = set()
        for path in sorted(root.glob("*.yaml")):
            config = run_experiment.load_config(path)
            refine.CaspConfig.from_dict(config.get("casp", {}))
            names.add(config["name"])
        self.assertEqual(names, {"baseline", "threshold_only", "casp", "casp_all", "casp_noiter", "oracle"})


if __name__ == "__main__":
    unittest.main()
