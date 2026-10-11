"""Confidence-Aware Self-Prompting (CASP v1): turn YOLO detections into SAM 2 prompts (decision DL-34).

Per detection with confidence c:
    c >= t_high            confident  -> box prompt, exactly as the baseline
    t_low <= c < t_high    uncertain  -> candidate prompts -> pick the one SAM 2 scores highest
                                         -> one self-prompting iteration (tight box + centroid point)
                                         -> accept only if SAM's predicted IoU >= s_min and the mask
                                            fills at least `min_box_fill` of its box
    c < t_low              dropped

The baseline is the special case t_low = t_high = 0.5 with accept=False.
This module is pure geometry plus calls to a `prompter` object with
    prompter.predict(box, point=None) -> (mask: bool HxW, score: float)
so it is unit-tested with a mocked prompter (tests/test_refine.py) and runs on a GPU only
through src/pipeline/sam2_prompter.py.

Boxes are (xmin, ymin, xmax, ymax) in pixels, exclusive maxima. Points are (x, y).
"""

import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
import common  # noqa: E402

CONFIDENT, UNCERTAIN, DROPPED = "confident", "uncertain", "dropped"


@dataclass
class CaspConfig:
    t_low: float = 0.25           # detections below this are dropped
    t_high: float = 0.7           # detections at or above this are used as they are
    s_min: float = 0.7            # minimum SAM predicted IoU to accept a refined mask
    expand_fractions: tuple = (0.10, 0.20)  # extra candidate boxes, enlarged by this fraction per side
    use_center_point: bool = True           # extra candidate: box + positive point at the box centre
    self_prompt_iters: int = 1              # tight-box + centroid-point re-prompts after selection
    min_box_fill: float = 0.10              # refined mask must cover this fraction of its box
    gate: bool = True                       # False = refine every kept detection (ablation C-all)
    accept: bool = True                     # False = never reject (baseline / threshold-only)

    @classmethod
    def baseline(cls, conf=0.5):
        return cls(t_low=conf, t_high=conf, accept=False, self_prompt_iters=0)

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        if "expand_fractions" in d:
            d["expand_fractions"] = tuple(float(x) for x in d["expand_fractions"])
        return cls(**d)

    def to_dict(self):
        d = asdict(self)
        d["expand_fractions"] = list(self.expand_fractions)
        return d


@dataclass
class DetectionResult:
    box: tuple
    confidence: float
    branch: str                       # confident / uncertain / dropped
    mask: object = None               # bool HxW or None
    score: float = float("nan")       # SAM predicted IoU of the final mask
    prompt: str = ""                  # which candidate won, e.g. "box", "expand_0.10", "box+center", then "+self"
    prompts_tried: int = 0
    accepted: bool = False
    final_box: tuple = None           # box actually prompted last (for the per-image record)

    def record(self):
        return {
            "box": [int(v) for v in self.box],
            "confidence": round(float(self.confidence), 4),
            "branch": self.branch,
            "score": None if np.isnan(self.score) else round(float(self.score), 4),
            "prompt": self.prompt,
            "prompts_tried": self.prompts_tried,
            "accepted": self.accepted,
            "final_box": None if self.final_box is None else [int(v) for v in self.final_box],
        }


def gate(confidence, cfg):
    """Which branch handles a detection of this confidence."""
    if confidence < cfg.t_low:
        return DROPPED
    if not cfg.gate or confidence < cfg.t_high:
        return UNCERTAIN
    return CONFIDENT


def clamp_box(box, width, height):
    x0, y0, x1, y1 = box
    x0 = int(max(0, min(round(x0), width - 1)))
    y0 = int(max(0, min(round(y0), height - 1)))
    x1 = int(max(x0 + 1, min(round(x1), width)))
    y1 = int(max(y0 + 1, min(round(y1), height)))
    return (x0, y0, x1, y1)


def expand_box(box, fraction, width, height):
    """Enlarge a box by `fraction` of its width/height on every side, clamped to the image."""
    x0, y0, x1, y1 = box
    dx = (x1 - x0) * fraction
    dy = (y1 - y0) * fraction
    return clamp_box((x0 - dx, y0 - dy, x1 + dx, y1 + dy), width, height)


def box_center(box):
    x0, y0, x1, y1 = box
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


def box_area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def tight_box_and_centroid(mask):
    """Bounding box (exclusive max) and centroid of a boolean mask; None if the mask is empty."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None, None
    box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    return box, (float(xs.mean()), float(ys.mean()))


def candidate_prompts(box, cfg, width, height):
    """The prompts tried for an uncertain detection, as (name, box, point)."""
    prompts = [("box", box, None)]
    for f in cfg.expand_fractions:
        prompts.append((f"expand_{f:.2f}", expand_box(box, f, width, height), None))
    if cfg.use_center_point:
        prompts.append(("box+center", box, box_center(box)))
    return prompts


def mask_fills_box(mask, box, min_fill):
    """True if the mask covers at least `min_fill` of the box area (guards against hallucinated slivers)."""
    x0, y0, x1, y1 = box
    inside = mask[y0:y1, x0:x1].sum()
    return inside >= min_fill * box_area(box)


def refine_detection(prompter, box, confidence, cfg, width, height):
    """Run one detection through CASP and return a DetectionResult."""
    box = clamp_box(box, width, height)
    result = DetectionResult(box=box, confidence=float(confidence), branch=gate(confidence, cfg))
    if result.branch == DROPPED:
        return result

    if result.branch == CONFIDENT:
        mask, score = prompter.predict(box, None)
        result.mask, result.score, result.prompt, result.prompts_tried = mask, float(score), "box", 1
        result.final_box, result.accepted = box, True
        return result

    # uncertain: try the candidates, keep SAM's favourite
    best = None
    for name, cand_box, point in candidate_prompts(box, cfg, width, height):
        mask, score = prompter.predict(cand_box, point)
        result.prompts_tried += 1
        if best is None or score > best[0]:
            best = (float(score), mask, name, cand_box)
    score, mask, name, used_box = best

    # self-prompting: the mask re-prompts the segmenter
    for _ in range(cfg.self_prompt_iters):
        tight, centroid = tight_box_and_centroid(mask)
        if tight is None:
            break
        new_mask, new_score = prompter.predict(tight, centroid)
        result.prompts_tried += 1
        if new_score >= score:          # keep the iteration only if SAM is at least as sure
            score, mask, used_box, name = float(new_score), new_mask, tight, name + "+self"

    result.mask, result.score, result.prompt, result.final_box = mask, score, name, used_box
    if cfg.accept:
        result.accepted = bool(score >= cfg.s_min and mask_fills_box(mask, used_box, cfg.min_box_fill))
    else:
        result.accepted = True
    if not result.accepted:
        result.mask = None
    return result


def segment_image(prompter, detections, cfg, width, height):
    """All detections of one image -> (union mask, list of DetectionResult).

    `detections` are (box, confidence) pairs; the prompter must already hold the image.
    """
    union = np.zeros((height, width), bool)
    results = []
    for box, confidence in sorted(detections, key=lambda d: -d[1]):
        r = refine_detection(prompter, box, confidence, cfg, width, height)
        results.append(r)
        if r.accepted and r.mask is not None:
            union |= r.mask
    return union, results
