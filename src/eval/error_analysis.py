"""Failure taxonomy for every run and dataset, plus the cases the report looks at (DL-40, DL-41).

    python src/eval/error_analysis.py --runs runs --out docs/tables/phase4
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "eval"))
from compare import RUNS, SHORT, datasets, load, shown  # noqa: E402

GOOD_DICE = 0.7
LABELS = ["good", "extra_box", "under_segmentation", "over_segmentation", "false_positive", "missed"]
CASE_COLUMNS = ["dice", "label", "gt_area_percent", "pred_area_percent", "n_gt_polyps", "n_candidates",
                "max_confidence", "box_tp", "box_fp"]


def label(row):
    if row["pred_empty"]:
        return "missed"
    if row["dice"] == 0:
        return "false_positive"
    if row["box_tp"] > 0 and row["box_fp"] > 0:
        return "extra_box"
    if row["dice"] >= GOOD_DICE:
        return "good"
    if row["pred_area_percent"] < row["gt_area_percent"]:
        return "under_segmentation"
    return "over_segmentation"


def labelled(df):
    df = df.copy()
    df["label"] = df.apply(label, axis=1)
    return df


def counts(runs_dir):
    rows = []
    for ds in datasets(runs_dir):
        for run in RUNS:
            df = load(runs_dir, run, ds)
            if df is None:
                continue
            c = labelled(df)["label"].value_counts()
            rows.append({"dataset": ds, "run": SHORT[run], "images": len(df), **{k: int(c.get(k, 0)) for k in LABELS}})
    return pd.DataFrame(rows)


def worst(runs_dir, n=10):
    rows = []
    for ds in ["kvasir_test", "clinicdb"]:
        for run in RUNS:
            df = load(runs_dir, run, ds)
            if df is None:
                continue
            w = labelled(df).sort_values("dice").head(n)[CASE_COLUMNS]
            rows.append(w.assign(dataset=ds, run=SHORT[run]).reset_index())
    return pd.concat(rows, ignore_index=True)


def never_found(runs_dir, dataset="clinicdb"):
    c = labelled(load(runs_dir, "casp", dataset))
    o = load(runs_dir, "oracle", dataset)
    miss = c[c["label"] == "missed"].copy()
    miss["oracle_dice"] = o.loc[miss.index, "dice"]
    miss["why"] = ["no candidate" if n == 0 else "all candidates below t_low" for n in miss["n_candidates"]]
    return miss[["gt_area_percent", "n_gt_polyps", "n_candidates", "max_confidence", "oracle_dice", "why"]].sort_values("max_confidence")


def c_vs_b(runs_dir, dataset="kvasir_test"):
    b = labelled(load(runs_dir, "baseline", dataset))
    c = labelled(load(runs_dir, "casp", dataset))
    d = pd.DataFrame({"dice_b": b["dice"], "dice_c": c["dice"], "label_b": b["label"], "label_c": c["label"],
                      "box_fp_b": b["box_fp"], "box_fp_c": c["box_fp"], "gt_area_percent": c["gt_area_percent"],
                      "max_confidence": c["max_confidence"]})
    d["diff"] = d["dice_c"] - d["dice_b"]
    return d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--out", default=str(ROOT / "docs" / "tables" / "phase4"))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tab = counts(args.runs)
    tab.to_csv(out / "errors_counts.csv", index=False)
    worst(args.runs).round(4).to_csv(out / "errors_worst10.csv", index=False)
    never_found(args.runs).round(4).to_csv(out / "errors_clinicdb_never_found.csv")
    d = c_vs_b(args.runs)
    d[d["diff"] < -1e-4].sort_values("diff").round(4).to_csv(out / "errors_kvasir_c_loses.csv")
    d[d["diff"] > 1e-4].sort_values("diff", ascending=False).round(4).to_csv(out / "errors_kvasir_c_wins.csv")
    lines = ["| Dataset | Run | " + " | ".join(LABELS) + " |", "|---|---|" + "---|" * len(LABELS)]
    for _, r in tab.iterrows():
        lines.append(f"| {r.dataset} | {r.run} | " + " | ".join(str(r[k]) for k in LABELS) + " |")
    (out / "errors_counts.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"taxonomy for {len(tab)} run/dataset pairs -> {shown(out)}")


if __name__ == "__main__":
    main()
