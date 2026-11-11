
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
RUNS = Path(r"D:\runs_phase3")

IMAGE_DIR = ROOT / "data/processed/cvc-clinicdb/images/test"
GT_DIR = ROOT / "data/processed/cvc-clinicdb/masks/test"
BASELINE_DIR = RUNS / "baseline/clinicdb/masks"
CASP_DIR = RUNS / "casp/clinicdb/masks"

OUTPUT = "docs/figures/phase4_examples_clinicdb.png"


def load_rgb(path):
    return np.array(Image.open(path).convert("RGB"))


def load_mask(path):
    return np.array(Image.open(path).convert("L")) > 0


def main():
    baseline = pd.read_csv(
        RUNS / "baseline/clinicdb/per_image.csv"
    )[["image_id", "dice"]].rename(
        columns={"dice": "baseline_dice"}
    )

    casp = pd.read_csv(
        RUNS / "casp/clinicdb/per_image.csv"
    )[["image_id", "dice"]].rename(
        columns={"dice": "casp_dice"}
    )

    results = baseline.merge(casp, on="image_id")

    # Five worst and five best CASP examples
    worst = results.sort_values(
        ["casp_dice", "image_id"]
    ).head(5)

    best = results.sort_values(
        ["casp_dice", "image_id"],
        ascending=[False, True]
    ).head(5)

    selected = [
        ("WORST CASP EXAMPLES", worst),
        ("BEST CASP EXAMPLES", best),
    ]

    fig, axes = plt.subplots(
        10, 4,
        figsize=(16, 25)
    )

    columns = [
        "Original Image",
        "Ground Truth",
        "Baseline Prediction",
        "CASP Prediction",
    ]

    for col, title in enumerate(columns):
        axes[0, col].set_title(
            title, fontsize=13, fontweight="bold"
        )

    row = 0

    for section, frame in selected:
        for _, item in frame.iterrows():
            image_id = int(item["image_id"])
            filename = f"{image_id}.png"

            paths = [
                IMAGE_DIR / filename,
                GT_DIR / filename,
                BASELINE_DIR / filename,
                CASP_DIR / filename,
            ]

            for path in paths:
                if not path.exists():
                    raise FileNotFoundError(path)

            image = load_rgb(paths[0])
            gt = load_mask(paths[1])
            base = load_mask(paths[2])
            casp_mask = load_mask(paths[3])

            if not (
                image.shape[:2]
                == gt.shape
                == base.shape
                == casp_mask.shape
            ):
                raise ValueError(
                    f"Dimension mismatch for image {image_id}"
                )

            axes[row, 0].imshow(image)
            axes[row, 1].imshow(gt, cmap="gray", vmin=0, vmax=1)
            axes[row, 2].imshow(base, cmap="gray", vmin=0, vmax=1)
            axes[row, 3].imshow(casp_mask, cmap="gray", vmin=0, vmax=1)

            axes[row, 0].set_ylabel(
                f"{section}\nID: {image_id}\n"
                f"B Dice: {item['baseline_dice']:.4f}\n"
                f"C Dice: {item['casp_dice']:.4f}",
                fontsize=9,
                rotation=0,
                labelpad=75,
                va="center",
            )

            for ax in axes[row]:
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)

            row += 1

    fig.suptitle(
        "ClinicDB: Best and Worst CASP Segmentation Examples",
        fontsize=17,
        fontweight="bold",
        y=0.995,
    )

    fig.tight_layout(rect=[0.06, 0, 1, 0.985])

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print("Figure saved:", OUTPUT)
    print("\nWorst CASP examples:")
    print(worst[["image_id", "baseline_dice", "casp_dice"]].to_string(index=False))
    print("\nBest CASP examples:")
    print(best[["image_id", "baseline_dice", "casp_dice"]].to_string(index=False))


if __name__ == "__main__":
    main()
