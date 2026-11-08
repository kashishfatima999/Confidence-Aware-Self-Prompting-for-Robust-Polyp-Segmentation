"""Paired comparison of every run against B and T, per dataset (DL-40, DL-41).

    python src/eval/compare.py --runs runs --out docs/tables/phase4
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "eval"))
sys.path.insert(0, str(ROOT / "src" / "data"))
import metrics  # noqa: E402

RUNS = ["baseline", "threshold_only", "casp", "casp_all", "casp_noiter", "oracle"]
SHORT = {"baseline": "B", "threshold_only": "T", "casp": "C", "casp_all": "C-all", "casp_noiter": "C-noiter", "oracle": "O"}
METRICS = ["dice", "iou", "pixel_precision", "pixel_recall", "pixel_accuracy", "area_error"]
LOWER_IS_BETTER = {"area_error"}
TIE = 1e-4


def add_pixel_metrics(df):
    p = df["pred_area_percent"].astype(float)
    g = df["gt_area_percent"].astype(float)
    overlap = df["dice"] * (p + g) / 2
    df = df.copy()
    df["pixel_precision"] = np.where(p > 0, overlap / p.where(p > 0, 1.0), 0.0).clip(0, 1)
    df["pixel_recall"] = np.where(g > 0, overlap / g.where(g > 0, 1.0), 1.0).clip(0, 1)
    df["pixel_accuracy"] = (1 - (p + g - 2 * overlap) / 100).clip(0, 1)
    df["area_error"] = (p - g).abs()
    return df


def load(runs_dir, run, dataset):
    path = Path(runs_dir) / run / dataset / "per_image.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, dtype={"image_id": str}).set_index("image_id")
    return add_pixel_metrics(df)


def datasets(runs_dir):
    found = sorted(p.name for p in (Path(runs_dir) / "baseline").iterdir() if (p / "per_image.csv").exists())
    return sorted(found, key=lambda d: (d != "kvasir_test", d != "clinicdb", d))


def run_summary(df):
    row = {"images": len(df), "missed": int(df["pred_empty"].sum())}
    for m in METRICS:
        lo, hi = metrics.bootstrap_ci(df[m].values)
        row[m] = df[m].mean()
        row[m + "_ci_low"], row[m + "_ci_high"] = lo, hi
    tp, fp, fn = int(df["box_tp"].sum()), int(df["box_fp"].sum()), int(df["box_fn"].sum())
    pr = metrics.precision_recall(tp, fp, fn)
    p, r = pr["precision"], pr["recall"]
    row.update(box_tp=tp, box_fp=fp, box_fn=fn, box_precision=p, box_recall=r, box_f1=2 * p * r / (p + r) if p + r else 0.0)
    return row


def paired(a, b, metric):
    d = (a[metric] - b[metric]).reindex(a.index).dropna()
    if metric in LOWER_IS_BETTER:
        better, worse = (d < -TIE).sum(), (d > TIE).sum()
    else:
        better, worse = (d > TIE).sum(), (d < -TIE).sum()
    lo, hi = metrics.bootstrap_ci(d.values, n_resamples=2000)
    return {"mean_diff": d.mean(), "ci_low": lo, "ci_high": hi, "better": int(better), "worse": int(worse),
            "tie": int(len(d) - better - worse)}


def build(runs_dir, results_csv=None):
    pooled = pd.read_csv(results_csv) if results_csv and Path(results_csv).exists() else None
    summary, pairs = [], []
    for ds in datasets(runs_dir):
        frames = {run: load(runs_dir, run, ds) for run in RUNS}
        frames = {k: v for k, v in frames.items() if v is not None}
        for run, df in frames.items():
            row = {"dataset": ds, "run": SHORT[run], **run_summary(df)}
            if pooled is not None:
                hit = pooled[(pooled.experiment == run) & (pooled.dataset == ds)]
                if len(hit):
                    h = hit.iloc[0]
                    row.update(dice=h.mean_dice, dice_ci_low=h.dice_ci_low, dice_ci_high=h.dice_ci_high, iou=h.mean_iou,
                               pooled_dice=h.pooled_dice, pooled_iou=h.pooled_iou)
            summary.append(row)
        for ref in ["baseline", "threshold_only"]:
            if ref not in frames:
                continue
            for run, df in frames.items():
                if run == ref or run == "baseline":
                    continue
                for m in METRICS:
                    pairs.append({"dataset": ds, "run": SHORT[run], "reference": SHORT[ref], "metric": m,
                                  **paired(df, frames[ref], m)})
    return pd.DataFrame(summary), pd.DataFrame(pairs)


def markdown(summary, pairs):
    out = []
    for ds, s in summary.groupby("dataset", sort=False):
        out.append(f"### {ds} ({int(s.images.iloc[0])} images)\n")
        out.append("| Run | Dice [95% CI] | IoU | Pixel prec. | Pixel recall | Pixel acc. | Area err. (pp) | Missed | Box P / R / F1 |")
        out.append("|---|---|---|---|---|---|---|---|---|")
        for _, r in s.iterrows():
            out.append(f"| {r.run} | {r.dice:.3f} [{r.dice_ci_low:.3f}, {r.dice_ci_high:.3f}] | {r.iou:.3f} | {r.pixel_precision:.3f} | "
                       f"{r.pixel_recall:.3f} | {r.pixel_accuracy:.3f} | {r.area_error:.2f} | {r.missed} | "
                       f"{r.box_precision:.3f} / {r.box_recall:.3f} / {r.box_f1:.3f} |")
        p = pairs[(pairs.dataset == ds) & (pairs.metric == "dice")]
        if len(p):
            out.append("\nPaired Dice difference (mean [95% CI], better / worse / tie):\n")
            out.append("| Run | vs B | vs T |")
            out.append("|---|---|---|")
            for run in s.run:
                cells = []
                for ref in ["B", "T"]:
                    hit = p[(p.run == run) & (p.reference == ref)]
                    cells.append("—" if hit.empty else "{:+.3f} [{:+.3f}, {:+.3f}], {} / {} / {}".format(
                        *hit.iloc[0][["mean_diff", "ci_low", "ci_high", "better", "worse", "tie"]]))
                if cells != ["—", "—"]:
                    out.append(f"| {run} | {cells[0]} | {cells[1]} |")
        out.append("")
    return "\n".join(out) + "\n"


def shown(path):
    path = Path(path).resolve()
    return path.relative_to(ROOT) if path.is_relative_to(ROOT) else path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--out", default=str(ROOT / "docs" / "tables" / "phase4"))
    parser.add_argument("--results", default=str(ROOT / "docs" / "tables" / "phase3" / "results.csv"))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary, pairs = build(args.runs, args.results)
    summary.round(4).to_csv(out / "metrics.csv", index=False)
    pairs.round(4).to_csv(out / "comparison.csv", index=False)
    (out / "comparison.md").write_text(markdown(summary, pairs), encoding="utf-8")
    print(f"{len(summary)} run summaries, {len(pairs)} paired comparisons -> {shown(out)}")


if __name__ == "__main__":
    main()
