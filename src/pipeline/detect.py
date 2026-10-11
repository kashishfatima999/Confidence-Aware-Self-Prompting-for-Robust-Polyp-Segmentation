"""Run the trained YOLO once per dataset and cache every candidate box (decision DL-33).

    python src/pipeline/detect.py --weights runs/yolo/yolov8m_kvasir_s42/weights/best.pt --dataset kvasir_test

Writes runs/detections/<dataset>.json:
    {"weights": ..., "conf": 0.05, "imgsz": 640, "images": {
        "<image_id>": {"width": W, "height": H, "boxes": [[x0, y0, x1, y1, confidence], ...]}}}

The confidence threshold is deliberately very low (0.05): every later experiment (baseline at 0.5,
threshold-only, CASP with its own t_low / t_high) reads this one file and applies its own
threshold, so all variants see exactly the same detector output. Needs ultralytics (Colab).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiment import REPO_ROOT, image_ids, resolve_dataset  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--dataset", required=True, help="kvasir_val | kvasir_test | clinicdb | kvasir_corrupted/<setting>")
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--out", type=Path, default=None, help="default runs/detections/<dataset>.json")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    from ultralytics import YOLO

    images_dir, _, ext = resolve_dataset(args.dataset)
    ids = image_ids(images_dir, ext, args.limit)
    paths = [str(images_dir / f"{i}.{ext}") for i in ids]
    model = YOLO(str(args.weights))

    cache = {"weights": str(args.weights), "conf": args.conf, "imgsz": args.imgsz, "dataset": args.dataset, "images": {}}
    for start in range(0, len(paths), args.batch):
        batch = paths[start:start + args.batch]
        for image_id, result in zip(ids[start:start + args.batch], model.predict(batch, conf=args.conf, imgsz=args.imgsz, verbose=False)):
            h, w = result.orig_shape
            boxes = result.boxes.xyxy.cpu().numpy().tolist()
            confs = result.boxes.conf.cpu().numpy().tolist()
            cache["images"][image_id] = {
                "width": int(w), "height": int(h),
                "boxes": [[round(v, 2) for v in b] + [round(c, 4)] for b, c in zip(boxes, confs)],
            }
        print(f"{min(start + args.batch, len(paths))}/{len(paths)} images", end="\r")

    out = args.out or (REPO_ROOT / "runs" / "detections" / (args.dataset.replace("/", "_") + ".json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cache, indent=1) + "\n")
    n_boxes = sum(len(v["boxes"]) for v in cache["images"].values())
    print(f"\n{len(cache['images'])} images, {n_boxes} candidate boxes at conf>={args.conf} -> {out}")


if __name__ == "__main__":
    main()
