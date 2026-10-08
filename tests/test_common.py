"""Unit tests for the shared helpers in src/data/common.py.

They use small synthetic masks, so they need no dataset and run in a few seconds.
Run from the repository root:  python -m unittest discover -s tests -v
"""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "data"))
import common  # noqa: E402


def make_mask(height=100, width=200, regions=()):
    """Boolean mask with rectangular polyp regions given as (xmin, ymin, xmax, ymax)."""
    mask = np.zeros((height, width), dtype=bool)
    for xmin, ymin, xmax, ymax in regions:
        mask[ymin:ymax, xmin:xmax] = True
    return mask


class MaskIOTests(unittest.TestCase):
    def test_threshold_is_127(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.png"
            Image.fromarray(np.array([[0, 127, 128, 255]], dtype=np.uint8)).save(path)
            self.assertEqual(common.load_binary_mask(path).tolist(), [[False, False, True, True]])

    def test_rgb_mask_is_converted_to_grey(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.png"
            rgb = np.zeros((2, 2, 3), dtype=np.uint8)
            rgb[0, 0] = 255
            Image.fromarray(rgb).save(path)
            self.assertEqual(common.load_binary_mask(path).tolist(), [[True, False], [False, False]])

    def test_save_then_load_round_trip_is_exact(self):
        mask = make_mask(regions=[(10, 20, 60, 70)])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sub" / "m.png"  # parent folder is created on demand
            common.save_binary_mask(mask, path)
            saved = np.array(Image.open(path))
            self.assertEqual(saved.ndim, 2)
            self.assertEqual(set(np.unique(saved).tolist()), {0, 255})
            self.assertTrue(np.array_equal(common.load_binary_mask(path), mask))


class MaskToBoxesTests(unittest.TestCase):
    def test_one_region_gives_one_box_with_exclusive_max(self):
        mask = make_mask(regions=[(10, 20, 60, 70)])
        self.assertEqual(common.mask_to_boxes(mask), [(10, 20, 60, 70)])

    def test_regions_are_sorted_left_to_right(self):
        mask = make_mask(regions=[(120, 10, 150, 40), (10, 50, 40, 90)])
        self.assertEqual(common.mask_to_boxes(mask), [(10, 50, 40, 90), (120, 10, 150, 40)])

    def test_speck_below_min_area_is_ignored(self):
        # image is 100 x 200 = 20,000 px; the speck is 3 x 3 = 9 px = 0.045% < 0.05%
        mask = make_mask(regions=[(10, 20, 60, 70), (150, 80, 153, 83)])
        self.assertEqual(common.mask_to_boxes(mask), [(10, 20, 60, 70)])
        self.assertEqual(len(common.mask_to_boxes(mask, min_area_fraction=0)), 2)

    def test_empty_mask_gives_no_boxes(self):
        self.assertEqual(common.mask_to_boxes(make_mask()), [])

    def test_full_image_region(self):
        mask = make_mask(height=30, width=40, regions=[(0, 0, 40, 30)])
        self.assertEqual(common.mask_to_boxes(mask), [(0, 0, 40, 30)])


class BoxMathTests(unittest.TestCase):
    def test_iou_identical_boxes(self):
        self.assertAlmostEqual(common.box_iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)

    def test_iou_half_overlap(self):
        self.assertAlmostEqual(common.box_iou((0, 0, 10, 10), (5, 0, 15, 10)), 50 / 150)

    def test_iou_disjoint_and_degenerate(self):
        self.assertEqual(common.box_iou((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)
        self.assertEqual(common.box_iou((0, 0, 0, 0), (0, 0, 0, 0)), 0.0)

    def test_boxes_extent(self):
        self.assertEqual(common.boxes_extent([(10, 20, 30, 40), (5, 25, 20, 60)]), (5, 20, 30, 60))

    def test_box_to_yolo_normalised_centre_and_size(self):
        xc, yc, w, h = common.box_to_yolo((10, 20, 60, 70), image_width=200, image_height=100)
        self.assertAlmostEqual(xc, 35 / 200)
        self.assertAlmostEqual(yc, 45 / 100)
        self.assertAlmostEqual(w, 50 / 200)
        self.assertAlmostEqual(h, 50 / 100)


class YoloLabelTests(unittest.TestCase):
    def test_label_file_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels" / "a.txt"
            common.write_yolo_label(path, [(10, 20, 60, 70), (0, 0, 200, 100)], 200, 100)
            lines = path.read_text().splitlines()
            self.assertEqual(lines, ["0 0.175000 0.450000 0.250000 0.500000", "0 0.500000 0.500000 1.000000 1.000000"])

    def test_empty_label_file_for_no_boxes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.txt"
            common.write_yolo_label(path, [], 200, 100)
            self.assertEqual(path.read_text(), "")


if __name__ == "__main__":
    unittest.main()
