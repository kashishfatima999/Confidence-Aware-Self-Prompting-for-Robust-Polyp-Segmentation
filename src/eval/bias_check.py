
"""Phase 4 subgroup bias analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RUN_MAP = {
    "baseline": "B",
    "threshold_only": "T",
    "casp": "C",
}

CORRUPT_PREFIX = "kvasir_corrupted_"
N_BOOTSTRAP = 2000
SEED = 42


def image_key(value):
    return Path(str(value)).stem.strip().lower()


def bootstrap_ci(values, n_resamples=N_BOOTSTRAP, seed=SEED):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan, np.nan

    if len(values) == 1:
        return float(values[0]), float(values[0])

    rng = np.random.default_rng(seed)
    samples = rng.choice(
        values,
        size=(n_resamples, len(values)),
        replace=True
    )

    means = samples.mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]).astype(float))


def _read_stats(stats_path):
    stats = pd.read_csv(stats_path)

    required = {
        "dataset", "image", "width", "height",
        "brightness", "contrast"
    }

    missing = required - set(stats.columns)

    if missing:
        raise ValueError(
            f"dataset_statistics.csv missing columns: {sorted(missing)}"
        )

    stats = stats.copy()
    stats["image_key"] = stats["image"].map(image_key)

    stats["domain"] = np.where(
        stats["dataset"].str.contains(
            "ClinicDB", case=False, na=False
        ),
        "clinicdb",
        "kvasir_test"
    )

    return stats.drop_duplicates(["domain", "image_key"])


def _read_run(path, run_label, dataset):
    csv_path = path / dataset / "per_image.csv"

    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path, dtype={"image_id": str})

    required = {
        "image_id", "dice", "gt_area_percent",
        "n_gt_polyps", "pred_empty", "box_fp"
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{csv_path} missing columns: {sorted(missing)}"
        )

    df = df.copy()

    df["image_key"] = df["image_id"].map(image_key)
    df["run"] = run_label
    df["source_dataset"] = dataset

    df["domain"] = (
        "clinicdb" if dataset == "clinicdb"
        else "kvasir_test"
    )

    df["miss"] = (
        df["pred_empty"]
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    for col in [
        "box_fp", "dice", "gt_area_percent", "n_gt_polyps"
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["box_fp"] = df["box_fp"].fillna(0)

    return df


def _tertile_labels(reference, values):
    ref = pd.to_numeric(reference, errors="coerce").dropna()

    if ref.empty:
        return pd.Series("unknown", index=values.index)

    q1, q2 = ref.quantile([1 / 3, 2 / 3]).tolist()

    if np.isclose(q1, q2):
        return pd.Series("all", index=values.index)

    return pd.cut(
        pd.to_numeric(values, errors="coerce"),
        bins=[-np.inf, q1, q2, np.inf],
        labels=["low", "medium", "high"],
        include_lowest=True
    ).astype("object").fillna("unknown")


def _attach_groups(frames, stats):
    merged = []

    for df in frames:
        domain_stats = stats[
            stats["domain"] == df["domain"].iloc[0]
        ]

        out = df.merge(
            domain_stats[
                [
                    "image_key", "width", "height",
                    "brightness", "contrast"
                ]
            ],
            on="image_key",
            how="left",
            validate="many_to_one"
        )

        merged.append(out)

    # Preserve original numeric reference values.
    # These must not be overwritten by subgroup labels.
    references = {}

    reference_columns = [
        "gt_area_percent",
        "brightness",
        "contrast",
        "width",
        "height",
    ]

    for df in merged:
        if (
            df["run"].iloc[0] == "B"
            and df["source_dataset"].iloc[0]
            in {"kvasir_test", "clinicdb"}
        ):
            domain = df["domain"].iloc[0]

            references[domain] = {
                col: pd.to_numeric(
                    df[col], errors="coerce"
                ).copy()
                for col in reference_columns
            }

    for df in merged:
        domain = df["domain"].iloc[0]
        ref = references.get(domain)

        if ref is None:
            ref = {
                col: pd.to_numeric(
                    df[col], errors="coerce"
                ).copy()
                for col in reference_columns
            }

        factors = [
            ("gt_area_percent", "polyp_size"),
            ("brightness", "brightness"),
            ("contrast", "contrast"),
            ("width", "width"),
            ("height", "height"),
        ]

        for col, factor in factors:
            df[factor] = _tertile_labels(
                ref[col], df[col]
            ).values

        df["polyp_count"] = np.where(
            df["n_gt_polyps"] > 1,
            "multiple",
            "single"
        )

        dataset = df["source_dataset"].iloc[0]

        if dataset.startswith(CORRUPT_PREFIX):
            suffix = dataset[len(CORRUPT_PREFIX):]
            kind, severity = suffix.rsplit("_s", 1)

            df["corruption_type"] = kind
            df["severity"] = "s" + severity
            df["corruption"] = kind + "_s" + severity

        else:
            df["corruption_type"] = "clean"
            df["severity"] = "clean"
            df["corruption"] = "clean"

    return merged


def _summary_for_group(group, label):
    return {
        f"dice_{label}": (
            float(group["dice"].mean())
            if len(group) else np.nan
        ),
        f"miss_rate_{label}": (
            float(group["miss"].mean())
            if len(group) else np.nan
        ),
        f"fp_boxes_per_image_{label}": (
            float(group["box_fp"].mean())
            if len(group) else np.nan
        ),
    }


def _paired_gap(group_frames):
    if "B" not in group_frames or "C" not in group_frames:
        return np.nan, np.nan, np.nan, 0

    b = group_frames["B"].set_index("image_key")
    c = group_frames["C"].set_index("image_key")

    paired = b[["dice"]].join(
        c[["dice"]],
        how="inner",
        lsuffix="_b",
        rsuffix="_c"
    ).dropna()

    if paired.empty:
        return np.nan, np.nan, np.nan, 0

    differences = (
        paired["dice_c"] - paired["dice_b"]
    ).to_numpy()

    lo, hi = bootstrap_ci(differences)

    return (
        float(differences.mean()),
        lo,
        hi,
        len(differences)
    )


def build_bias_tables(runs_dir, stats_path):
    runs_dir = Path(runs_dir)
    stats_path = Path(stats_path)

    stats = _read_stats(stats_path)

    baseline_dir = runs_dir / "baseline"

    dataset_names = sorted(
        p.name
        for p in baseline_dir.iterdir()
        if p.is_dir() and (p / "per_image.csv").exists()
    )

    frames = []

    for dataset in dataset_names:
        for run, label in RUN_MAP.items():
            frame = _read_run(
                runs_dir / run,
                label,
                dataset
            )

            if frame is not None:
                frames.append(frame)

    if not frames:
        raise FileNotFoundError(
            f"No per_image.csv files found under {runs_dir}"
        )

    frames = _attach_groups(frames, stats)

    all_data = pd.concat(frames, ignore_index=True)

    factor_specs = {
        "polyp_size": "Polyp size (terciles)",
        "brightness": "Image brightness (terciles)",
        "contrast": "Image contrast (terciles)",
        "width": "Image width (terciles)",
        "height": "Image height (terciles)",
        "polyp_count": "Single vs multiple polyps",
        "dataset": "Dataset/domain",
        "corruption": "Corruption type and severity",
    }

    tables = {}

    for factor, description in factor_specs.items():
        records = []
        data = all_data.copy()

        if factor == "dataset":
            data = data[
                ~data["source_dataset"].str.startswith(
                    CORRUPT_PREFIX
                )
            ]

            data[factor] = data["domain"].map({
                "kvasir_test": "Kvasir-SEG",
                "clinicdb": "CVC-ClinicDB"
            })

        elif factor == "corruption":
            data = data[
                data["source_dataset"].str.startswith(
                    CORRUPT_PREFIX
                )
            ]

        # Keep each dataset separate so image IDs from
        # different datasets cannot be matched accidentally.
        for (dataset_name, subgroup), sub in data.groupby(
            ["source_dataset", factor],
            dropna=False,
            sort=True
        ):
            if subgroup == "unknown" or pd.isna(subgroup):
                continue

            row = {
                "dataset": dataset_name,
                "factor": factor,
                "subgroup": str(subgroup),
                "n": int(sub["image_key"].nunique()),
            }

            by_run = {
                run: sub[sub["run"] == run]
                for run in ["B", "T", "C"]
            }

            for run in ["B", "T", "C"]:
                row.update(
                    _summary_for_group(by_run[run], run)
                )

            gap, lo, hi, paired_n = _paired_gap(by_run)

            row.update({
                "c_minus_b": gap,
                "ci_low": lo,
                "ci_high": hi,
                "paired_n": paired_n
            })

            records.append(row)

        tables[factor] = pd.DataFrame(records)

    return tables, factor_specs


def write_outputs(tables, descriptions, out_dir, figure_dir):
    out_dir = Path(out_dir)
    figure_dir = Path(figure_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    for factor, df in tables.items():
        csv_path = out_dir / f"bias_{factor}.csv"
        md_path = out_dir / f"bias_{factor}.md"

        df.to_csv(csv_path, index=False)

        with md_path.open("w", encoding="utf-8") as f:
            f.write(
                f"# Bias analysis: {descriptions[factor]}\n\n"
            )

            if df.empty:
                f.write("No matching data were available.\n")
            else:
                f.write("```text\n")
                f.write(df.round(4).to_string(index=False))
                f.write("\n```\n")

        if df.empty:
            continue

        plot = df.copy()

        plot["plot_label"] = (
            plot["dataset"] + "\n" + plot["subgroup"]
        )

        x = np.arange(len(plot))
        fig_width = max(8, min(24, len(plot) * 0.55))

        fig, ax = plt.subplots(figsize=(fig_width, 5))

        width = 0.24

        for offset, run in [
            (-width, "B"),
            (0, "T"),
            (width, "C")
        ]:
            col = f"dice_{run}"

            if col in plot and plot[col].notna().any():
                ax.bar(
                    x + offset,
                    plot[col],
                    width,
                    label=run
                )

        ax.set_xticks(x)
        ax.set_xticklabels(
            plot["plot_label"],
            rotation=65,
            ha="right",
            fontsize=8
        )

        ax.set_ylabel("Mean per-image Dice")
        ax.set_title(descriptions[factor])
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.25)
        ax.legend()

        fig.tight_layout()

        fig.savefig(
            figure_dir / f"phase4_bias_{factor}.png",
            dpi=180
        )

        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--runs", default="runs")
    parser.add_argument(
        "--stats",
        default="docs/tables/dataset_statistics.csv"
    )
    parser.add_argument(
        "--out",
        default="docs/tables/phase4"
    )
    parser.add_argument(
        "--figures",
        default="docs/figures"
    )

    args = parser.parse_args()

    tables, descriptions = build_bias_tables(
        args.runs,
        args.stats
    )

    write_outputs(
        tables,
        descriptions,
        args.out,
        args.figures
    )

    for factor, df in tables.items():
        print(f"{factor}: {len(df)} subgroup rows")

    print(f"CSV/MD tables: {Path(args.out).resolve()}")
    print(f"Figures: {Path(args.figures).resolve()}")


if __name__ == "__main__":
    main()
