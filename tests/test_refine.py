"""Unit tests for src/pipeline/refine.py with a mocked SAM 2 (no GPU, no data)."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "pipeline"))
import refine  # noqa: E402

W, H = 100, 80


class MockPrompter:
    """Pretends to be SAM 2: the mask is the 'true polyp' clipped to the prompt box; the score is
    the IoU between the prompt box and the true polyp box (so a tighter box scores higher), plus a
    bonus when a point prompt is given. Records every call."""

    def __init__(self, polyp_box, point_bonus=0.0):
        self.polyp_box = polyp_box
        self.point_bonus = point_bonus
        self.calls = []

    def predict(self, box, point=None):
        self.calls.append((tuple(box), point))
        mask = np.zeros((H, W), bool)
        x0, y0, x1, y1 = box
        px0, py0, px1, py1 = self.polyp_box
        ix0, iy0, ix1, iy1 = max(x0, px0), max(y0, py0), min(x1, px1), min(y1, py1)
        if ix1 > ix0 and iy1 > iy0:
            mask[iy0:iy1, ix0:ix1] = True
        inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
        union = (x1 - x0) * (y1 - y0) + (px1 - px0) * (py1 - py0) - inter
        score = inter / union if union else 0.0
        if point is not None:
            score = min(1.0, score + self.point_bonus)
        return mask, score


class GeometryTests(unittest.TestCase):
    def test_gate(self):
        cfg = refine.CaspConfig(t_low=0.25, t_high=0.7)
        self.assertEqual(refine.gate(0.1, cfg), refine.DROPPED)
        self.assertEqual(refine.gate(0.25, cfg), refine.UNCERTAIN)
        self.assertEqual(refine.gate(0.69, cfg), refine.UNCERTAIN)
        self.assertEqual(refine.gate(0.7, cfg), refine.CONFIDENT)
        self.assertEqual(refine.gate(0.99, refine.CaspConfig(gate=False)), refine.UNCERTAIN)

    def test_baseline_config_keeps_everything_at_or_above_half(self):
        cfg = refine.CaspConfig.baseline()
        self.assertEqual(refine.gate(0.5, cfg), refine.CONFIDENT)
        self.assertEqual(refine.gate(0.49, cfg), refine.DROPPED)
        self.assertFalse(cfg.accept)

    def test_expand_box_and_clamp(self):
        self.assertEqual(refine.expand_box((10, 10, 30, 20), 0.1, W, H), (8, 9, 32, 21))
        self.assertEqual(refine.expand_box((0, 0, 100, 80), 0.2, W, H), (0, 0, 100, 80))

    def test_tight_box_and_centroid(self):
        m = np.zeros((H, W), bool)
        m[10:20, 30:50] = True
        box, (cx, cy) = refine.tight_box_and_centroid(m)
        self.assertEqual(box, (30, 10, 50, 20))
        self.assertAlmostEqual(cx, 39.5)
        self.assertAlmostEqual(cy, 14.5)
        self.assertEqual(refine.tight_box_and_centroid(np.zeros((H, W), bool)), (None, None))

    def test_candidate_prompts(self):
        cfg = refine.CaspConfig()
        names = [p[0] for p in refine.candidate_prompts((10, 10, 30, 30), cfg, W, H)]
        self.assertEqual(names, ["box", "expand_0.10", "expand_0.20", "box+center"])

    def test_config_round_trip(self):
        cfg = refine.CaspConfig(t_low=0.3, expand_fractions=(0.05,))
        self.assertEqual(refine.CaspConfig.from_dict(cfg.to_dict()), cfg)


class RefineDetectionTests(unittest.TestCase):
    def test_confident_detection_uses_one_box_prompt(self):
        sam = MockPrompter(polyp_box=(20, 20, 60, 60))
        r = refine.refine_detection(sam, (20, 20, 60, 60), 0.95, refine.CaspConfig(), W, H)
        self.assertEqual(r.branch, refine.CONFIDENT)
        self.assertEqual(r.prompts_tried, 1)
        self.assertEqual(sam.calls, [((20, 20, 60, 60), None)])
        self.assertTrue(r.accepted)
        self.assertEqual(r.mask.sum(), 1600)

    def test_uncertain_detection_tries_candidates_and_self_prompts(self):
        # YOLO box is too small; the 20% enlargement fits the polyp better and the self-prompt tightens it
        sam = MockPrompter(polyp_box=(20, 20, 60, 60))
        r = refine.refine_detection(sam, (26, 26, 54, 54), 0.4, refine.CaspConfig(s_min=0.5), W, H)
        self.assertEqual(r.branch, refine.UNCERTAIN)
        self.assertEqual(r.prompts_tried, 5)                 # 4 candidates + 1 self-prompt
        self.assertTrue(r.prompt.startswith("expand_0.20"))
        self.assertTrue(r.accepted)
        self.assertGreater(r.mask.sum(), 28 * 28)            # bigger than the original box's mask

    def test_rejection_when_sam_is_unsure(self):
        sam = MockPrompter(polyp_box=(20, 20, 60, 60))
        # box far from the polyp: masks are empty, scores are 0 -> rejected
        r = refine.refine_detection(sam, (70, 60, 95, 78), 0.4, refine.CaspConfig(), W, H)
        self.assertFalse(r.accepted)
        self.assertIsNone(r.mask)

    def test_no_rejection_when_accept_is_off(self):
        sam = MockPrompter(polyp_box=(20, 20, 60, 60))
        r = refine.refine_detection(sam, (70, 60, 95, 78), 0.4, refine.CaspConfig(accept=False), W, H)
        self.assertTrue(r.accepted)

    def test_no_self_prompt_when_iters_zero(self):
        sam = MockPrompter(polyp_box=(20, 20, 60, 60))
        r = refine.refine_detection(sam, (26, 26, 54, 54), 0.4, refine.CaspConfig(self_prompt_iters=0, s_min=0.3), W, H)
        self.assertEqual(r.prompts_tried, 4)
        self.assertNotIn("+self", r.prompt)

    def test_dropped_detection_never_calls_sam(self):
        sam = MockPrompter(polyp_box=(20, 20, 60, 60))
        r = refine.refine_detection(sam, (20, 20, 60, 60), 0.1, refine.CaspConfig(), W, H)
        self.assertEqual(r.branch, refine.DROPPED)
        self.assertEqual(sam.calls, [])


class SegmentImageTests(unittest.TestCase):
    def test_union_of_accepted_masks_and_ordering(self):
        sam = MockPrompter(polyp_box=(20, 20, 60, 60))
        dets = [((20, 20, 40, 60), 0.6), ((40, 20, 60, 60), 0.9), ((0, 0, 5, 5), 0.1)]
        union, results = refine.segment_image(sam, dets, refine.CaspConfig(s_min=0.3), W, H)
        self.assertEqual([r.confidence for r in results], [0.9, 0.6, 0.1])  # highest confidence first
        self.assertEqual([r.branch for r in results], ["confident", "uncertain", "dropped"])
        self.assertEqual(union.sum(), 1600)

    def test_record_is_json_friendly(self):
        sam = MockPrompter(polyp_box=(20, 20, 60, 60))
        _, results = refine.segment_image(sam, [((20, 20, 60, 60), 0.95)], refine.CaspConfig(), W, H)
        rec = results[0].record()
        self.assertEqual(rec["branch"], "confident")
        self.assertEqual(rec["box"], [20, 20, 60, 60])
        self.assertIsInstance(rec["score"], float)


if __name__ == "__main__":
    unittest.main()
