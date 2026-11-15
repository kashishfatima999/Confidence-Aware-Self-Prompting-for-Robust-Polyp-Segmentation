"""compare the two detector seeds (42 and 43) on the main phase 5 results.

    python src/eval/seed_compare.py --s42 runs/runs_phase3 --s43 runs/runs_phase5_s43

each seed folder needs baseline, threshold_only and casp for kvasir_test and clinicdb. the oracle run
does not use the detector, so if the s43 folder has no oracle the s42 oracle is used for both.
the same analysis as filter_analysis.py is run on each seed and the key numbers go side by side.
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "eval"))
from compare import shown  #noqa: E402
import filter_analysis as fa  #noqa: E402

DATASETS = ["kvasir_test", "clinicdb"]
RUNS = ["baseline", "threshold_only", "casp", "oracle"]
PAIRS = [("C", "B"), ("T", "B"), ("F_T", "B"), ("F_T", "T")]


def seed_view(seed_dir, oracle_dir, tmp):
    """folder with the seed's runs and the oracle run, made from symlinks."""
    view = Path(tmp)
    for run in RUNS:
        src = Path(seed_dir) / run
        if not src.exists() and run == "oracle":
            src = Path(oracle_dir) / run
        if src.exists():
            (view / run).symlink_to(src.resolve(), target_is_directory=True)
    return view


def seed_rows(runs_dir, seed):
    """key numbers of one seed, one row per dataset and quantity."""
    summary, pairs, taxonomy = fa.fallback_tables(runs_dir)
    signals = fa.signal_table(fa.box_table(runs_dir))
    rows = []
    for ds in DATASETS:
        s = summary[summary.dataset == ds].set_index("run")
        t = taxonomy[taxonomy.dataset == ds].set_index("run")
        for run in ["B", "T", "C", "F_T"]:
            rows.append({"seed": seed, "dataset": ds, "quantity": f"Dice {run}", "value": s.loc[run, "dice"],
                         "ci_low": s.loc[run, "dice_ci_low"], "ci_high": s.loc[run, "dice_ci_high"]})
        for run in ["T", "F_T"]:
            rows.append({"seed": seed, "dataset": ds, "quantity": f"box F1 {run}", "value": s.loc[run, "box_f1"]})
            rows.append({"seed": seed, "dataset": ds, "quantity": f"extra-box images {run}", "value": t.loc[run, "extra_box"]})
        rows.append({"seed": seed, "dataset": ds, "quantity": "missed images B", "value": s.loc["B", "missed"]})
        rows.append({"seed": seed, "dataset": ds, "quantity": "missed images T", "value": s.loc["T", "missed"]})
        for a, b in PAIRS:
            p = pairs[(pairs.dataset == ds) & (pairs.metric == "dice") & (pairs.run == a) & (pairs.reference == b)]
            if p.empty:   #C - B and T - B are not in the fallback table, work them out here
                d = fa.paired(fa.load(runs_dir, {"C": "casp", "T": "threshold_only"}[a], ds),
                              fa.load(runs_dir, "baseline", ds), "dice")
            else:
                d = p.iloc[0]
            rows.append({"seed": seed, "dataset": ds, "quantity": f"Dice {a} - {b}", "value": d["mean_diff"],
                         "ci_low": d["ci_low"], "ci_high": d["ci_high"]})
        band = signals[(signals.dataset == ds) & signals.boxes.str.startswith("uncertain")]
        for _, r in band.iterrows():
            rows.append({"seed": seed, "dataset": ds, "quantity": f"band AUROC {r.signal}", "value": r.auroc,
                         "ci_low": r.ci_low, "ci_high": r.ci_high})
    return rows


def yolo_rows(path, seed):
    if not path or not Path(path).exists():
        return []
    s = json.loads(Path(path).read_text())
    return [{"seed": seed, "dataset": "kvasir_test", "quantity": f"YOLO test {k}", "value": s["test"][k]}
            for k in ["map50", "map50_95", "precision", "recall"]]


def build(s42, s43, yolo42=None, yolo43=None):
    rows = []
    for seed, d in [(42, s42), (43, s43)]:
        with tempfile.TemporaryDirectory() as tmp:
            rows += seed_rows(seed_view(d, s42, tmp), seed)
    rows += yolo_rows(yolo42, 42) + yolo_rows(yolo43, 43)
    return pd.DataFrame(rows)


def cell(r):
    if r.quantity.startswith(("missed", "extra")):   #counts
        return f"{r.value:.0f}"
    if pd.isna(r.get("ci_low")):
        return f"{r.value:.3f}"
    return f"{r.value:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}]" if " - " in r.quantity else f"{r.value:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}]"


def markdown(df):
    out = []
    for ds in df.dataset.unique():
        d = df[df.dataset == ds]
        out += [f"### {ds}\n", "| Quantity | Seed 42 | Seed 43 |", "|---|---|---|"]
        for q in d.quantity.unique():
            vals = {s: cell(r) for s, r in ((r.seed, r) for _, r in d[d.quantity == q].iterrows())}
            out.append(f"| {q} | {vals.get(42, '—')} | {vals.get(43, '—')} |")
        out.append("")
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--s42", required=True)
    parser.add_argument("--s43", required=True)
    parser.add_argument("--yolo42", default=None, help="summary.json of the seed-42 detector")
    parser.add_argument("--yolo43", default=None, help="summary.json of the seed-43 detector")
    parser.add_argument("--out", default=str(ROOT / "docs" / "tables" / "phase5"))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = build(args.s42, args.s43, args.yolo42, args.yolo43)
    df.round(4).to_csv(out / "seed_comparison.csv", index=False)
    (out / "seed_comparison.md").write_text(markdown(df), encoding="utf-8")
    print(f"{len(df)} rows -> {shown(out)}")


if __name__ == "__main__":
    main()
