"""Phase 5 analysis: can the false boxes that T and C add be filtered out without a GPU? (DL-44, DL-45)

    python src/eval/filter_analysis.py --runs runs --out docs/tables/phase5

Two parts, both CPU-only and both read only the Phase 3 result files:

1. Box-level signals. Every candidate box in the T run's detections.jsonl carries its YOLO confidence
   and SAM 2's predicted IoU for the mask it produced. The ground-truth boxes are the prompts of the
   oracle run (O prompts SAM 2 with the boxes derived from the true masks). Each box that T kept is
   labelled true or false positive with the same greedy IoU >= 0.5 matching as the evaluation
   (metrics.match_boxes), and we measure how well each signal separates the two (AUROC with a
   bootstrap CI). The question is the one DL-38 left open: is SAM's score a usable filter?

2. The fallback rule F, fixed before looking at the test results (DL-44): use the uncertain boxes
   (0.25 <= confidence < 0.5) only in images where YOLO found no box at 0.5 or above. Because the
   baseline B keeps exactly the boxes >= 0.5 and SAM 2 is deterministic for a given box prompt, F's
   output on every image is identical to B's (if the image has a box >= 0.5) or to T's (if it does
   not). So F's per-image metrics are taken row by row from B and T without re-running SAM 2.
   F_C does the same with C in place of T and also covers the nine corrupted settings, where T was
   not run.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "eval"))
sys.path.insert(0, str(ROOT / "src" / "data"))
import metrics  # noqa: E402
from compare import datasets, load, paired, run_summary, shown  # noqa: E402
from error_analysis import LABELS, labelled  # noqa: E402

T_LOW = 0.25      # CASP / T threshold (DL-34)
T_BASE = 0.5      # baseline threshold (DL-31); F's switch, fixed in advance (DL-44)
COMPARE_METRICS = ["dice", "iou", "pixel_precision", "pixel_recall"]


# ---------------------------------------------------------------- part 2: the fallback rule

def fallback(base, other, threshold=T_BASE):
    """Per-image rows of the fallback rule: base's row where the image has a box >= threshold, else other's.

    `base` must be the run that keeps exactly the boxes >= threshold (B); `other` the run that also keeps
    lower boxes (T or C). Both are per_image frames indexed by image_id from compare.load.
    """
    other = other.reindex(base.index)
    if not np.allclose(base["max_confidence"], other["max_confidence"], atol=1e-4):
        raise ValueError("the two runs did not read the same detection cache (max_confidence differs)")
    confident = base["max_confidence"] >= threshold
    out = other.copy()
    out.loc[confident] = base.loc[confident]
    out["source"] = np.where(confident, "B", "other")
    return out


def fallback_tables(runs_dir):
    """Run summaries, paired differences and taxonomy counts for F_T and F_C next to B, T, C."""
    summary, pairs, taxonomy = [], [], []
    for ds in datasets(runs_dir):
        frames = {name: load(runs_dir, run, ds) for name, run in
                  [("B", "baseline"), ("T", "threshold_only"), ("C", "casp"), ("O", "oracle")]}
        frames = {k: v for k, v in frames.items() if v is not None}
        if "B" not in frames:
            continue
        if "T" in frames:
            frames["F_T"] = fallback(frames["B"], frames["T"])
        if "C" in frames:
            frames["F_C"] = fallback(frames["B"], frames["C"])
        for name, df in frames.items():
            row = {"dataset": ds, "run": name, **run_summary(df)}
            if name.startswith("F_"):
                row["images_from_B"] = int((df["source"] == "B").sum())
            summary.append(row)
            c = labelled(df)["label"].value_counts()
            taxonomy.append({"dataset": ds, "run": name, "images": len(df), **{k: int(c.get(k, 0)) for k in LABELS}})
        for f in ["F_T", "F_C"]:
            if f not in frames:
                continue
            for ref in ["B", "T", "C"]:
                if ref not in frames:
                    continue
                for m in COMPARE_METRICS:
                    pairs.append({"dataset": ds, "run": f, "reference": ref, "metric": m,
                                  **paired(frames[f], frames[ref], m)})
    return pd.DataFrame(summary), pd.DataFrame(pairs), pd.DataFrame(taxonomy)


# ---------------------------------------------------------------- part 1: box-level signals

def read_jsonl(path):
    by_image = {}
    with open(path) as handle:
        for line in handle:
            if line.strip():
                rec = json.loads(line)
                by_image.setdefault(rec["image_id"], []).append(rec)
    return by_image


def label_boxes(records, gt_boxes, t_low=T_LOW, t_base=T_BASE):
    """One row per candidate box of one image with its confidence, SAM score and true/false-positive label.

    Kept boxes (confidence >= t_low) are matched greedily in confidence order, exactly as the evaluation
    does; a dropped box gets the label it would have had if it were matched against the GT boxes the kept
    boxes left over.
    """
    recs = sorted(records, key=lambda r: -r["confidence"])
    unmatched = list(range(len(gt_boxes)))
    top = max((r["confidence"] for r in recs), default=0.0)
    rows = []
    for rank, r in enumerate(recs):
        box = tuple(r["final_box"] or r["box"])
        best, best_iou = None, 0.5
        for j in unmatched:
            v = metrics.common.box_iou(box, gt_boxes[j])
            if v >= best_iou:
                best, best_iou = j, v
        kept = r["confidence"] >= t_low
        if best is not None and kept:
            unmatched.remove(best)
        rows.append({
            "confidence": r["confidence"],
            "sam_score": r["score"],
            "kept": kept,
            "tp": best is not None,
            "rank": rank,
            "band": t_low <= r["confidence"] < t_base,
            "image_has_confident_box": top >= t_base,
            "higher_box_in_image": rank > 0 and recs[0]["confidence"] >= t_base,
            "n_gt_boxes": len(gt_boxes),
        })
    return rows


def box_table(runs_dir, run="threshold_only"):
    rows = []
    for ds in datasets(runs_dir):
        det = Path(runs_dir) / run / ds / "detections.jsonl"
        gt = Path(runs_dir) / "oracle" / ds / "detections.jsonl"
        if not (det.exists() and gt.exists()):
            continue
        dets, gts = read_jsonl(det), read_jsonl(gt)
        for image_id in sorted(set(dets) | set(gts)):
            gt_boxes = [tuple(r["box"]) for r in gts.get(image_id, [])]
            for row in label_boxes(dets.get(image_id, []), gt_boxes):
                rows.append({"dataset": ds, "image_id": image_id, **row})
    return pd.DataFrame(rows)


def auroc(scores, labels):
    """Area under the ROC curve (probability a random true box outscores a random false one; ties count half)."""
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    allv = np.concatenate([pos, neg])[order]
    ranks = np.empty(len(allv))
    i = 0
    while i < len(allv):                     # average ranks over ties
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2 + 1
        i = j + 1
    r = np.empty(len(allv))
    r[order] = ranks
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def auroc_ci(scores, labels, n_resamples=2000, seed=42):
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_resamples):
        idx = rng.integers(0, len(scores), len(scores))
        v = auroc(scores[idx], labels[idx])
        if not np.isnan(v):
            vals.append(v)
    return (float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))) if vals else (float("nan"),) * 2


SIGNALS = {
    "YOLO confidence": lambda d: d["confidence"],
    "SAM 2 score": lambda d: d["sam_score"],
    "confidence x SAM score": lambda d: d["confidence"] * d["sam_score"],
    "no box >= 0.5 in image": lambda d: (~d["higher_box_in_image"]).astype(float),
}


def signal_table(boxes):
    rows = []
    for ds, d in boxes[boxes["kept"]].groupby("dataset", sort=False):
        for subset, s in [("all kept boxes", d), ("uncertain band 0.25-0.5", d[d["band"]])]:
            for name, f in SIGNALS.items():
                v = f(s)
                lo, hi = auroc_ci(v, s["tp"])
                rows.append({"dataset": ds, "boxes": subset, "signal": name, "n": len(s), "n_tp": int(s["tp"].sum()),
                             "auroc": auroc(v, s["tp"]), "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def context_table(boxes):
    """How often an uncertain box is a polyp, split by whether the image already has a box >= 0.5."""
    band = boxes[boxes["kept"] & boxes["band"]]
    rows = []
    for (ds, ctx), s in band.groupby(["dataset", "higher_box_in_image"], sort=False):
        rows.append({"dataset": ds, "image_has_box_ge_0.5": bool(ctx), "boxes": len(s), "true_polyp": int(s["tp"].sum()),
                     "share_true": s["tp"].mean(), "mean_sam_score_true": s.loc[s["tp"], "sam_score"].mean(),
                     "mean_sam_score_false": s.loc[~s["tp"], "sam_score"].mean()})
    return pd.DataFrame(rows).sort_values(["dataset", "image_has_box_ge_0.5"])


# ---------------------------------------------------------------- output

def markdown(summary, pairs, taxonomy, signals, context):
    out = ["## Fallback rule F (uncertain boxes only where no box >= 0.5)\n"]
    for ds in ["kvasir_test", "clinicdb"]:
        s = summary[summary.dataset == ds]
        if s.empty:
            continue
        out += [f"### {ds}\n", "| Run | Dice | IoU | Pixel prec. | Pixel recall | Missed | Box P / R / F1 |", "|---|---|---|---|---|---|---|"]
        for _, r in s.iterrows():
            out.append(f"| {r.run} | {r.dice:.3f} [{r.dice_ci_low:.3f}, {r.dice_ci_high:.3f}] | {r.iou:.3f} | {r.pixel_precision:.3f} | "
                       f"{r.pixel_recall:.3f} | {r.missed} | {r.box_precision:.3f} / {r.box_recall:.3f} / {r.box_f1:.3f} |")
        p = pairs[(pairs.dataset == ds) & (pairs.metric == "dice")]
        out += ["", "| Pair | Dice difference [95% CI] | better / worse / tie |", "|---|---|---|"]
        for _, r in p.iterrows():
            out.append(f"| {r.run} − {r.reference} | {r.mean_diff:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}] | {r.better} / {r.worse} / {r.tie} |")
        t = taxonomy[taxonomy.dataset == ds]
        out += ["", "| Run | " + " | ".join(LABELS) + " |", "|---|" + "---|" * len(LABELS)]
        for _, r in t.iterrows():
            out.append(f"| {r.run} | " + " | ".join(str(r[k]) for k in LABELS) + " |")
        out.append("")
    corr = pairs[pairs.dataset.str.startswith("kvasir_corrupted") & (pairs.metric == "dice") & (pairs.reference != "T")]
    if len(corr):
        out += ["### Corrupted Kvasir-SEG test (F_C)\n", "| Setting | F_C − B | F_C − C |", "|---|---|---|"]
        for ds, g in corr.groupby("dataset"):
            cell = {r.reference: f"{r.mean_diff:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}]" for _, r in g.iterrows()}
            out.append(f"| {ds.replace('kvasir_corrupted_', '')} | {cell.get('B', '—')} | {cell.get('C', '—')} |")
        out.append("")
    out += ["## Box-level signals (T run, kept boxes; AUROC for separating true from false boxes)\n",
            "| Dataset | Boxes | Signal | n (true) | AUROC [95% CI] |", "|---|---|---|---|---|"]
    for _, r in signals.iterrows():
        out.append(f"| {r.dataset} | {r.boxes} | {r.signal} | {r.n} ({r.n_tp}) | {r.auroc:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}] |")
    out += ["", "## Uncertain boxes by context\n",
            "| Dataset | Image has a box ≥ 0.5 | Boxes | True polyp | Share | SAM score true / false |", "|---|---|---|---|---|---|"]
    for _, r in context.iterrows():
        out.append(f"| {r.dataset} | {'yes' if r['image_has_box_ge_0.5'] else 'no'} | {r.boxes} | {r.true_polyp} | {r.share_true:.2f} | "
                   f"{r.mean_sam_score_true:.3f} / {r.mean_sam_score_false:.3f} |")
    return "\n".join(out) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--out", default=str(ROOT / "docs" / "tables" / "phase5"))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    summary, pairs, taxonomy = fallback_tables(args.runs)
    boxes = box_table(args.runs)
    signals, context = signal_table(boxes), context_table(boxes)

    summary.round(4).to_csv(out / "fallback_metrics.csv", index=False)
    pairs.round(4).to_csv(out / "fallback_comparison.csv", index=False)
    taxonomy.to_csv(out / "fallback_errors.csv", index=False)
    boxes.round(4).to_csv(out / "box_signals.csv", index=False)
    signals.round(4).to_csv(out / "signal_auroc.csv", index=False)
    context.round(4).to_csv(out / "band_context.csv", index=False)
    (out / "filter_analysis.md").write_text(markdown(summary, pairs, taxonomy, signals, context), encoding="utf-8")
    print(f"{len(summary)} run summaries, {len(pairs)} paired comparisons, {len(boxes)} boxes -> {shown(out)}")


if __name__ == "__main__":
    main()
