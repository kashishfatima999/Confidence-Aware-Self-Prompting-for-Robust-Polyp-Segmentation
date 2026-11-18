import unittest
from unittest.mock import Mock

import numpy as np

from app.app import build_variant_masks, select_fallback_mask


class AppLogicTests(unittest.TestCase):
    def test_fallback_uses_baseline_when_any_confidence_is_at_least_half(self):
        baseline = np.array([[True, False]])
        casp = np.array([[False, True]])
        detections = [([0, 0, 10, 10], 0.32), ([1, 1, 9, 9], 0.50)]

        actual = select_fallback_mask(baseline, casp, detections)

        np.testing.assert_array_equal(actual, baseline)

    def test_fallback_uses_casp_when_all_confidences_are_below_half(self):
        baseline = np.array([[True, False]])
        casp = np.array([[False, True]])
        detections = [([0, 0, 10, 10], 0.32), ([1, 1, 9, 9], 0.49)]

        actual = select_fallback_mask(baseline, casp, detections)

        np.testing.assert_array_equal(actual, casp)

    def test_no_detections_uses_casp(self):
        baseline = np.zeros((2, 2), dtype=bool)
        casp = np.ones((2, 2), dtype=bool)

        actual = select_fallback_mask(baseline, casp, [])

        np.testing.assert_array_equal(actual, casp)

    def test_build_variants_sets_mock_sam_image_once_and_calls_pipeline_three_times(self):
        # Fake SAM2 predictor: no real checkpoint or torch/SAM2 install needed.
        fake_prompter = Mock()
        fake_prompter.predict.return_value = (
            np.ones((8, 8), dtype=bool),
            0.9,
        )
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        detections = [([1, 1, 7, 7], 0.8)]

        # Exercise the real refine.segment_image path using the mocked prompter.
        masks = build_variant_masks(image, detections, fake_prompter)

        fake_prompter.set_image.assert_called_once()
        self.assertEqual(set(masks), {"B", "T", "C", "F_C"})
        self.assertEqual(masks["B"].shape, (8, 8))
        np.testing.assert_array_equal(masks["F_C"], masks["B"])


if __name__ == "__main__":
    unittest.main()
