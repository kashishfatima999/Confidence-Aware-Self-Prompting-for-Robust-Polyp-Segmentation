"""Build the image-shift test set: corrupted copies of the Kvasir-SEG test split (decision DL-20).

Three corruptions, each at three strengths (1 = mild, 3 = strong):
    blur                 Gaussian blur, sigma = 1, 2, 4 pixels
    brightness_contrast  darker, flatter image: contrast x0.8 / x0.6 / x0.4, brightness -20 / -40 / -60
    noise                Gaussian noise, standard deviation = 10, 20, 40 (on the 0..255 scale)

Only the TEST split is corrupted. Training and validation images are never touched.
Masks and YOLO labels are copied unchanged: none of these corruptions moves the polyp.

Input  (never modified):  data/processed/kvasir/{images,masks,labels}/test/
Output: data/processed/kvasir-corrupted/<corruption>_s<strength>/{images,masks,labels}/test/
        data/processed/kvasir-corrupted/manifest.json   every parameter used + image counts

Each output folder has the same layout as data/processed/kvasir, so YOLO can evaluate it by
pointing a dataset YAML's `path` at it.

Corrupted images are saved as PNG (lossless) so that no extra JPEG artifacts are added on top
of the corruption itself. Noise is seeded per image, so every run gives identical files.

Run from the repository root (after prepare_kvasir.py):
    python src/data/make_corruptions.py
"""

import argparse
import json
import shutil
import zlib
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

# strength -> parameters. Change these only together with a decision_log.md entry.
CORRUPTIONS = {
    "blur": {1: {"sigma": 1.0}, 2: {"sigma": 2.0}, 3: {"sigma": 4.0}},
    "brightness_contrast": {
        1: {"contrast": 0.8, "brightness": -20},
        2: {"contrast": 0.6, "brightness": -40},
        3: {"contrast": 0.4, "brightness": -60},
    },
    "noise": {1: {"std": 10.0}, 2: {"std": 20.0}, 3: {"std": 40.0}},
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, default=REPO_ROOT / "data/processed/kvasir")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data/processed/kvasir-corrupted")
    parser.add_argument("--split-file", type=Path, default=REPO_ROOT / "splits/kvasir_test.txt")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def apply_corruption(image, name, params, rng):
    """Return a corrupted copy of a uint8 BGR image. The input array is not changed."""
    if name == "blur":
        # ksize (0, 0) lets OpenCV pick the kernel size from sigma (about 6*sigma + 1).
        return cv2.GaussianBlur(image, (0, 0), sigmaX=params["sigma"], sigmaY=params["sigma"])
    if name == "brightness_contrast":
        # new = contrast * (old - mean) + mean + brightness, so contrast shrinks around the
        # image's own mean brightness and then the whole image is shifted darker.
        img = image.astype(np.float32)
        mean = img.mean()
        out = params["contrast"] * (img - mean) + mean + params["brightness"]
        return np.clip(out, 0, 255).astype(np.uint8)
    if name == "noise":
        noise = rng.normal(0.0, params["std"], size=image.shape)
        return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    raise ValueError(f"unknown corruption {name!r}")


def image_seed(base_seed, image_id, corruption, strength):
    """A fixed seed per (image, corruption, strength), independent of processing order."""
    return base_seed + zlib.crc32(f"{image_id}|{corruption}|{strength}".encode())


def main():
    args = parse_args()
    test_ids = args.split_file.read_text().split()

    src_images = args.src / "images/test"
    src_masks = args.src / "masks/test"
    src_labels = args.src / "labels/test"

    # Check every input exists before writing anything.
    missing = []
    image_paths = {}
    for image_id in test_ids:
        found = sorted(src_images.glob(f"{image_id}.*"))
        if not found:
            missing.append(f"image {image_id}")
            continue
        image_paths[image_id] = found[0]
        if not (src_masks / f"{image_id}.png").exists():
            missing.append(f"mask {image_id}")
        if not (src_labels / f"{image_id}.txt").exists():
            missing.append(f"label {image_id}")
    if missing:
        raise SystemExit(
            f"{len(missing)} test files are missing in {args.src} (run prepare_kvasir.py first), "
            f"e.g. {missing[:3]}"
        )

    manifest = {
        "source": str(args.src),
        "split_file": str(args.split_file),
        "images_per_setting": len(test_ids),
        "seed": args.seed,
        "image_format": "png",
        "corruptions": {name: {f"s{s}": p for s, p in levels.items()} for name, levels in CORRUPTIONS.items()},
        "settings": [],
    }

    for name, levels in CORRUPTIONS.items():
        for strength, params in levels.items():
            setting = f"{name}_s{strength}"
            out = args.out / setting
            # Only this script's own output folder is cleared; the source is never changed.
            shutil.rmtree(out, ignore_errors=True)
            (out / "images/test").mkdir(parents=True)
            (out / "masks/test").mkdir(parents=True)
            (out / "labels/test").mkdir(parents=True)

            for image_id in test_ids:
                image = cv2.imread(str(image_paths[image_id]), cv2.IMREAD_COLOR)
                if image is None:
                    raise SystemExit(f"could not read {image_paths[image_id]}")
                rng = np.random.default_rng(image_seed(args.seed, image_id, name, strength))
                corrupted = apply_corruption(image, name, params, rng)
                cv2.imwrite(str(out / "images/test" / f"{image_id}.png"), corrupted)
                shutil.copy2(src_masks / f"{image_id}.png", out / "masks/test" / f"{image_id}.png")
                shutil.copy2(src_labels / f"{image_id}.txt", out / "labels/test" / f"{image_id}.txt")

            manifest["settings"].append({"name": setting, "corruption": name, "strength": strength,
                                         "params": params, "images": len(test_ids)})
            print(f"{setting:<24} {len(test_ids)} images  {params}")

    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Done: {len(manifest['settings'])} settings written to {args.out}")


if __name__ == "__main__":
    main()
