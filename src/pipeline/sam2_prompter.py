"""Thin wrapper around SAM 2's image predictor: embed an image once, prompt it many times (DL-33).

    prompter = SAM2Prompter("weights/sam2_hiera_large.pt")
    prompter.set_image(rgb_uint8_hxwx3)
    mask, score = prompter.predict(box=(x0, y0, x1, y1), point=(x, y) or None)

`score` is SAM 2's own predicted IoU for the mask, which CASP uses to pick between candidate
prompts and to accept or reject refined masks (DL-34). multimask_output is always False, as in
the baseline. Needs torch + the `sam2` package (Colab); src/pipeline/refine.py does not.
"""

import numpy as np

DEFAULT_MODEL_CFG = "configs/sam2/sam2_hiera_l.yaml"  # config name inside the installed sam2 package


class SAM2Prompter:
    def __init__(self, checkpoint, model_cfg=DEFAULT_MODEL_CFG, device=None):
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.predictor = SAM2ImagePredictor(build_sam2(model_cfg, str(checkpoint), device=self.device))
        self.image_shape = None

    def set_image(self, rgb):
        """rgb: uint8 array HxWx3. Computes the image embedding once."""
        with self.torch.inference_mode():
            self.predictor.set_image(rgb)
        self.image_shape = rgb.shape[:2]

    def predict(self, box, point=None):
        kwargs = {"box": np.asarray(box, dtype=np.float32), "multimask_output": False}
        if point is not None:
            kwargs["point_coords"] = np.asarray([point], dtype=np.float32)
            kwargs["point_labels"] = np.asarray([1], dtype=np.int32)
        with self.torch.inference_mode():
            masks, scores, _ = self.predictor.predict(**kwargs)
        return np.asarray(masks[0]) > 0, float(scores[0])
