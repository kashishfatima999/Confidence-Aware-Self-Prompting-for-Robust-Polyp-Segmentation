
from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.refine import CaspConfig, segment_image


def make_configs() -> dict[str, CaspConfig]:
    """Configurations matching the experiment YAML files."""
    return {
        "B": CaspConfig.baseline(0.5),
        "T": CaspConfig(
            t_low=0.25,
            t_high=0.25,
            accept=False,
            self_prompt_iters=0,
        ),
        "C": CaspConfig(
            t_low=0.25,
            t_high=0.60,
            s_min=0.60,
            expand_fractions=(0.10, 0.20),
            use_center_point=True,
            self_prompt_iters=1,
            min_box_fill=0.10,
            gate=True,
            accept=True,
        ),
    }


def extract_detections(
    yolo_result: Any,
) -> list[tuple[list[float], float]]:
    """Convert one YOLO result into refine.segment_image input."""
    boxes = getattr(yolo_result, "boxes", None)

    if boxes is None or len(boxes) == 0:
        return []

    xyxy = boxes.xyxy.detach().cpu().numpy()
    confidences = boxes.conf.detach().cpu().numpy()

    return [
        ([float(v) for v in box], float(conf))
        for box, conf in zip(xyxy, confidences)
    ]


def select_fallback_mask(
    baseline_mask: np.ndarray,
    casp_mask: np.ndarray,
    detections: list[tuple[list[float], float]],
    threshold: float = 0.5,
) -> np.ndarray:
    """F_C: use B if any detection >= threshold; otherwise use C."""
    has_confident_detection = any(
        confidence >= threshold
        for _, confidence in detections
    )

    if has_confident_detection:
        return baseline_mask

    return casp_mask


def build_variant_masks(
    rgb_image: np.ndarray,
    detections: list[tuple[list[float], float]],
    prompter: Any,
) -> dict[str, np.ndarray]:
    """Run B, T and C using the same YOLO detections."""
    height, width = rgb_image.shape[:2]

    prompter.set_image(rgb_image)

    masks: dict[str, np.ndarray] = {}

    for name, config in make_configs().items():
        union_mask, _ = segment_image(
            prompter=prompter,
            detections=detections,
            cfg=config,
            width=width,
            height=height,
        )

        masks[name] = np.asarray(union_mask, dtype=bool)

    masks["F_C"] = select_fallback_mask(
        masks["B"],
        masks["C"],
        detections,
    )

    return masks


def overlay_mask(
    rgb: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Show predicted segmentation in green."""
    image = Image.fromarray(
        rgb.astype(np.uint8)
    ).convert("RGBA")

    green = Image.new(
        "RGBA",
        image.size,
        (0, 210, 90, 0),
    )

    alpha = Image.fromarray(
        mask.astype(np.uint8) * 105
    )

    green.putalpha(alpha)

    result = Image.alpha_composite(image, green)

    return np.asarray(result.convert("RGB"))


@lru_cache(maxsize=1)
def load_models():
    """Load YOLO and SAM2 once per app process."""
    from ultralytics import YOLO
    from src.pipeline.sam2_prompter import SAM2Prompter

    yolo_path = os.environ.get(
        "YOLO_WEIGHTS",
        str(ROOT / "weights" / "best.pt"),
    )

    sam2_path = os.environ.get(
        "SAM2_CHECKPOINT",
        str(ROOT / "weights" / "sam2_hiera_large.pt"),
    )

    sam2_cfg = os.environ.get(
        "SAM2_MODEL_CFG",
        str(ROOT / "configs" / "sam2" / "sam2_hiera_l.yaml"),
    )

    if not Path(yolo_path).is_file():
        raise FileNotFoundError(
            f"YOLO weights not found: {yolo_path}"
        )

    if not Path(sam2_path).is_file():
        raise FileNotFoundError(
            f"SAM2 checkpoint not found: {sam2_path}"
        )

    yolo = YOLO(yolo_path)

    sam2 = SAM2Prompter(
        checkpoint=sam2_path,
        model_cfg=sam2_cfg,
    )

    return yolo, sam2


def predict(image: Image.Image):
    """Run one YOLO pass and display B, T, C and F_C."""
    import gradio as gr

    if image is None:
        raise gr.Error("Please upload an image first.")

    try:
        yolo, prompter = load_models()

        rgb = np.asarray(
            image.convert("RGB"),
            dtype=np.uint8,
        )

        # YOLO runs exactly once.
        results = yolo.predict(
            np.ascontiguousarray(rgb[:, :, ::-1]),
            conf=0.05,
            imgsz=640,
            verbose=False,
        )

        detections = (
            extract_detections(results[0])
            if results
            else []
        )

        masks = build_variant_masks(
            rgb,
            detections,
            prompter,
        )

        previews = [
            overlay_mask(rgb, masks["B"]),
            overlay_mask(rgb, masks["T"]),
            overlay_mask(rgb, masks["C"]),
            overlay_mask(rgb, masks["F_C"]),
        ]

        selected = (
            "B"
            if any(conf >= 0.5 for _, conf in detections)
            else "C"
        )

        status_text = (
            f"YOLO detections: {len(detections)} | "
            f"F_C selected: {selected} | "
            f"Confidence threshold: 0.50"
        )

        return (*previews, status_text)

    except gr.Error:
        raise

    except Exception as exc:
        raise gr.Error(f"Demo failed: {exc}") from exc


def launch_demo():
    """Create and launch the Gradio interface."""
    import gradio as gr

    with gr.Blocks(
        title="CASP Polyp Segmentation - Phase 5"
    ) as demo:

        gr.Markdown(
            "# CASP Polyp Segmentation Demo\n"
            "Upload one endoscopy frame. YOLO runs once; "
            "B, T, C and F_C reuse the same detections. "
            "Green pixels show the predicted mask."
        )

        upload = gr.Image(
            type="pil",
            label="Upload an endoscopy image",
        )

        run_button = gr.Button(
            "Run segmentation",
            variant="primary",
        )

        status = gr.Textbox(
            label="Run information",
            interactive=False,
        )

        with gr.Row():
            out_b = gr.Image(
                label="B - Baseline",
                type="numpy",
            )

            out_t = gr.Image(
                label="T - Threshold only",
                type="numpy",
            )

        with gr.Row():
            out_c = gr.Image(
                label="C - CASP",
                type="numpy",
            )

            out_f = gr.Image(
                label="F_C - Fallback",
                type="numpy",
            )

        run_button.click(
            fn=predict,
            inputs=upload,
            outputs=[
                out_b,
                out_t,
                out_c,
                out_f,
                status,
            ],
        )

    demo.launch()


if __name__ == "__main__":
    launch_demo()
