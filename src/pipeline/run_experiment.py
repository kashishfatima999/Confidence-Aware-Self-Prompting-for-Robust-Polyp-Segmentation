"""Run one experiment (a config) on one dataset from the detection cache and write per-image results.

    python src/pipeline/run_experiment.py --config configs/experiments/casp.yaml --dataset kvasir_test
    python src/pipeline/run_experiment.py --config configs/experiments/casp.yaml --dataset kvasir_val --set t_low=0.35 s_min=0.8 --name grid/t035_s080

Inputs : runs/detections/<dataset>.json (from detect.py), the processed images and masks, SAM 2 weights.
Outputs: runs/<name>/<dataset>/per_image.csv       one row per image (metrics + how it was handled)
         runs/<name>/<dataset>/detections.jsonl    one line per detection (branch, prompt, score, accepted)
         runs/<name>/<dataset>/summary.json        dataset metrics (DL-32) + the exact config used
         runs/<name>/<dataset>/masks/<id>.png      predicted masks (with --save-masks)

The oracle experiment (configs/experiments/oracle.yaml) ignores the cache and prompts SAM 2 with the
boxes derived from the ground-truth masks: the upper bound a perfect detector would give.
`evaluate()` is independent of SAM 2 and ultralytics, so it is tested with a mocked prompter.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT / "src" / "data"))
sys.path.insert(0, str(REPO_ROOT / "src" / "eval"))
import common  # noqa: E402
import metrics  # noqa: E402
import refine  # noqa: E402

PROCESSED = REPO_ROOT / "data" / "processed"
DATASETS = {
    "kvasir_val": (PROCESSED / "kvasir/images/val", PROCESSED / "kvasir/masks/val", "jpg"),
    "kvasir_test": (PROCESSED / "kvasir/images/test", PROCESSED / "kvasir/masks/test", "jpg"),
    "clinicdb": (PROCESSED / "cvc-clinicdb/images/test", PROCESSED / "cvc-clinicdb/masks/test", "png"),
}


def resolve_dataset(name):
    """(images_dir, masks_dir, image extension) for a dataset name; corrupted sets are kvasir_corrupted/<setting>."""
    if name in DATASETS:
        return DATASETS[name]
    if name.startswith("kvasir_corrupted/"):
        setting = name.split("/", 1)[1]
        base = PROCESSED / "kvasir-corrupted" / setting
        return (base / "images/test", base / "masks/test", "png")
    raise SystemExit(f"unknown dataset {name!r}; use one of {sorted(DATASETS)} or kvasir_corrupted/<setting>")


def image_ids(images_dir, ext, limit=None):
    ids = sorted(p.stem for p in Path(images_dir).glob(f"*.{ext}"))
    if not ids:
        raise SystemExit(f"no *.{ext} images in {images_dir}; run the prepare script first")
    return ids[:limit] if limit else ids


def load_config(path):
    text = Path(path).read_text(encoding="utf-8")
    if str(path).endswith(".json"):
        return json.loads(text)
    import yaml
    return yaml.safe_load(text)


def apply_overrides(config, overrides):
    """--set t_low=0.3 s_min=0.8 gate=false : values go into config['casp']."""
    for item in overrides or []:
        key, _, value = item.partition("=")
        if value.lower() in ("true", "false"):
            parsed = value.lower() == "true"
        else:
            try:
                parsed = float(value) if "." in value or "e" in value.lower() else int(value)
            except ValueError:
                parsed = value
        config.setdefault("casp", {})[key] = parsed
    return config


def load_detections(path):
    return json.loads(Path(path).read_text())["images"]


def detections_for(image_id, cache, oracle, gt_mask):
    """(box, confidence) pairs for one image: from the cache, or from the ground truth for the oracle."""
    if oracle:
        return [(box, 1.0) for box in common.mask_to_boxes(gt_mask)]
    entry = cache.get(image_id)
    if entry is None:
        raise SystemExit(f"{image_id} is not in the detection cache; re-run detect.py for this dataset")
    return [(tuple(b[:4]), float(b[4])) for b in entry["boxes"]]


def evaluate(prompter, ids, images_dir, masks_dir, ext, cache, cfg, oracle, out_dir, save_masks=False, log=print):
    """The experiment loop. Returns the summary dict and writes per_image.csv / detections.jsonl / summary.json."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if save_masks:
        (out_dir / "masks").mkdir(exist_ok=True)

    rows, pooled = [], metrics.PooledScores()
    tp = fp = fn = 0
    started = time.time()
    with open(out_dir / "detections.jsonl", "w") as det_file:
        for n, image_id in enumerate(ids, 1):
            rgb = np.array(Image.open(Path(images_dir) / f"{image_id}.{ext}").convert("RGB"))
            gt = common.load_binary_mask(Path(masks_dir) / f"{image_id}.png")
            h, w = gt.shape
            dets = detections_for(image_id, cache, oracle, gt)

            t0 = time.time()
            prompter.set_image(rgb)
            pred, results = refine.segment_image(prompter, dets, cfg, w, h)
            seconds = time.time() - t0

            scores = metrics.per_image_scores(pred, gt)
            pooled.add(pred, gt)
            gt_boxes = common.mask_to_boxes(gt)
            pred_boxes = [r.final_box for r in results if r.accepted and r.mask is not None]
            i_tp, i_fp, i_fn = metrics.match_boxes(pred_boxes, gt_boxes)
            tp, fp, fn = tp + i_tp, fp + i_fp, fn + i_fn

            branches = [r.branch for r in results]
            kept = [r for r in results if r.accepted]
            rows.append({
                "image_id": image_id,
                "gt_area_percent": round(scores["gt_area_percent"], 3),
                "n_gt_polyps": len(gt_boxes),
                "n_candidates": len(dets),
                "n_confident": branches.count(refine.CONFIDENT),
                "n_uncertain": branches.count(refine.UNCERTAIN),
                "n_dropped": branches.count(refine.DROPPED),
                "n_rejected": sum(1 for r in results if r.branch == refine.UNCERTAIN and not r.accepted),
                "n_kept": len(kept),
                "max_confidence": round(max((d[1] for d in dets), default=0.0), 4),
                "min_kept_confidence": round(min((r.confidence for r in kept), default=0.0), 4),
                "prompts_tried": sum(r.prompts_tried for r in results),
                "dice": round(scores["dice"], 4),
                "iou": round(scores["iou"], 4),
                "pred_area_percent": round(scores["pred_area_percent"], 3),
                "pred_empty": scores["pred_empty"],
                "box_tp": i_tp, "box_fp": i_fp, "box_fn": i_fn,
                "seconds": round(seconds, 3),
            })
            for r in results:
                det_file.write(json.dumps({"image_id": image_id, **r.record()}) + "\n")
            if save_masks:
                common.save_binary_mask(pred, out_dir / "masks" / f"{image_id}.png")
            if n % 25 == 0 or n == len(ids):
                log(f"  {n}/{len(ids)}  running mean Dice {np.mean([r['dice'] for r in rows]):.4f}")

    with open(out_dir / "per_image.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = metrics.summarize(rows, pooled, tp, fp, fn)
    summary.update({
        "experiment": out_dir.parent.name,
        "dataset": out_dir.name,
        "oracle": oracle,
        "casp": cfg.to_dict(),
        "prompts_per_image": float(np.mean([r["prompts_tried"] for r in rows])),
        "seconds_per_image": float(np.mean([r["seconds"] for r in rows])),
        "total_seconds": round(time.time() - started, 1),
    })
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True, help="configs/experiments/<name>.yaml")
    parser.add_argument("--dataset", required=True, help="kvasir_val | kvasir_test | clinicdb | kvasir_corrupted/<setting>")
    parser.add_argument("--detections", type=Path, default=None, help="default runs/detections/<dataset>.json")
    parser.add_argument("--sam2-checkpoint", type=Path, default=REPO_ROOT / "weights" / "sam2_hiera_large.pt")
    parser.add_argument("--name", default=None, help="output folder runs/<name>/ (default: the config's name)")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override CASP parameters")
    parser.add_argument("--limit", type=int, default=None, help="only the first N images (smoke test)")
    parser.add_argument("--save-masks", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = apply_overrides(load_config(args.config), args.set)
    cfg = refine.CaspConfig.from_dict(config.get("casp", {}))
    oracle = bool(config.get("oracle", False))
    images_dir, masks_dir, ext = resolve_dataset(args.dataset)
    ids = image_ids(images_dir, ext, args.limit)
    cache = {} if oracle else load_detections(args.detections or REPO_ROOT / "runs" / "detections" / (args.dataset.replace("/", "_") + ".json"))
    out_dir = REPO_ROOT / "runs" / (args.name or config["name"]) / args.dataset.replace("/", "_")

    from sam2_prompter import SAM2Prompter
    prompter = SAM2Prompter(args.sam2_checkpoint)
    print(f"{config['name']} on {args.dataset}: {len(ids)} images, casp={cfg.to_dict()}, oracle={oracle}")
    summary = evaluate(prompter, ids, images_dir, masks_dir, ext, cache, cfg, oracle, out_dir, args.save_masks)
    print(json.dumps({k: summary[k] for k in ("images", "mean_dice", "mean_dice_ci95", "mean_iou", "pooled_dice", "pooled_iou",
                                               "images_with_empty_prediction", "precision", "recall", "prompts_per_image")}, indent=2))
    print(f"written to {out_dir}")


if __name__ == "__main__":
    main()
