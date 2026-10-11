"""Collect every runs/<experiment>/<dataset>/summary.json into one results table (DL-32, DL-36).

    python src/eval/summarize.py                       # writes docs/tables/phase3/results.csv and results.md
    python src/eval/summarize.py --runs runs --out docs/tables/phase3

No number in a report is typed by hand: the tables are rebuilt from the summary files.
"""

import argparse
import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SKIP = {"yolo", "detections"}  # runs/ folders that are not experiments
COLUMNS = ["experiment", "dataset", "images", "mean_dice", "dice_ci_low", "dice_ci_high", "mean_iou", "pooled_dice",
           "pooled_iou", "images_with_empty_prediction", "precision", "recall", "detection_fp", "prompts_per_image",
           "seconds_per_image", "t_low", "t_high", "s_min", "oracle"]


def collect(runs_dir):
    rows = []
    for path in sorted(Path(runs_dir).glob("*/*/summary.json")):
        if path.parents[1].name in SKIP:
            continue
        s = json.loads(path.read_text())
        casp = s.get("casp", {})
        rows.append({
            "experiment": s.get("experiment", path.parents[1].name),
            "dataset": s.get("dataset", path.parent.name),
            "images": s["images"],
            "mean_dice": round(s["mean_dice"], 4),
            "dice_ci_low": round(s["mean_dice_ci95"][0], 4),
            "dice_ci_high": round(s["mean_dice_ci95"][1], 4),
            "mean_iou": round(s["mean_iou"], 4),
            "pooled_dice": round(s["pooled_dice"], 4),
            "pooled_iou": round(s["pooled_iou"], 4),
            "images_with_empty_prediction": s["images_with_empty_prediction"],
            "precision": round(s["precision"], 4),
            "recall": round(s["recall"], 4),
            "detection_fp": s["detection_fp"],
            "prompts_per_image": round(s.get("prompts_per_image", float("nan")), 2),
            "seconds_per_image": round(s.get("seconds_per_image", float("nan")), 3),
            "t_low": casp.get("t_low"), "t_high": casp.get("t_high"), "s_min": casp.get("s_min"),
            "oracle": s.get("oracle", False),
        })
    return rows


def markdown(rows):
    head = ["Experiment", "Dataset", "N", "Dice (mean, 95% CI)", "IoU (mean)", "Dice (pooled)", "IoU (pooled)", "Missed", "Prec.", "Recall", "Prompts/img"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        lines.append("| {experiment} | {dataset} | {images} | {mean_dice:.3f} [{dice_ci_low:.3f}, {dice_ci_high:.3f}] | {mean_iou:.3f} | "
                     "{pooled_dice:.3f} | {pooled_iou:.3f} | {images_with_empty_prediction} | {precision:.3f} | {recall:.3f} | {prompts_per_image:.1f} |".format(**r))
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=REPO_ROOT / "runs")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "docs" / "tables" / "phase3")
    args = parser.parse_args()

    rows = collect(args.runs)
    if not rows:
        raise SystemExit(f"no summary.json found under {args.runs}")
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "results.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    (args.out / "results.md").write_text(markdown(rows), encoding="utf-8")
    print(markdown(rows))
    print(f"{len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
