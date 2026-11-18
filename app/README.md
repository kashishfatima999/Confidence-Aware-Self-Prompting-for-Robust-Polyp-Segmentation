# Phase 5 Gradio demo

## What it does
- Accepts one uploaded RGB image.
- Runs YOLO once with `conf=0.05` and `imgsz=640`.
- Reuses the same detections for B, T and C.
- Computes F_C using the project fallback rule: use B if any detection has
  confidence >= 0.50; otherwise use C.
- Shows all four mask overlays.

`src/pipeline/refine.py` and `src/pipeline/sam2_prompter.py` are imported
without modification.

## Run locally
From the repository root:

```cmd
python -m pip install -r app\requirements.txt
set YOLO_WEIGHTS=C:\path\to\best.pt
set SAM2_CHECKPOINT=C:\path\to\sam2_hiera_large.pt
python app\app.py
```

Optional:
```cmd
set SAM2_MODEL_CFG=configs\sam2\sam2_hiera_l.yaml
```

## Hugging Face Spaces
Create a Gradio Space and upload the repository contents. Set these Space
variables/secrets:
- `YOLO_WEIGHTS`: path inside the Space to the seed-42 `best.pt`
- `SAM2_CHECKPOINT`: path inside the Space to the SAM2 checkpoint
- `SAM2_MODEL_CFG`: optional config path; defaults to
  `configs/sam2/sam2_hiera_l.yaml`

Do not commit model weights to GitHub. Put large weights in the Space storage
or download them at Space startup from an access-controlled source.

A free CPU Space may be slow and can run out of RAM when loading SAM2. Test
with one image first; the code does not promise real-time inference.
