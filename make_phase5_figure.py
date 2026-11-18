import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
CSV = ROOT / "docs/tables/phase5/fallback_errors.csv"
OUT = ROOT / "docs/figures/phase5_fallback_taxonomy.png"

runs = ["B", "T", "C", "F_C"]
errors = [
    "extra_box",
    "under_segmentation",
    "over_segmentation",
    "false_positive",
    "missed",
]

labels = [
    "Extra box",
    "Under-segmentation",
    "Over-segmentation",
    "False positive",
    "Missed",
]

with open(CSV, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

datasets = ["kvasir_test", "clinicdb"]

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

for ax, dataset in zip(axes, datasets):
    subset = [r for r in rows if r["dataset"] == dataset]
    x = np.arange(len(runs))
    bottom = np.zeros(len(runs))

    for error, label in zip(errors, labels):
        values = []
        for run in runs:
            row = next(r for r in subset if r["run"] == run)
            values.append(
                int(row[error]) / int(row["images"]) * 100
            )

        ax.bar(x, values, bottom=bottom, label=label)
        bottom += np.array(values)

    ax.set_title(
        "Kvasir-SEG Test" if dataset == "kvasir_test"
        else "ClinicDB Test"
    )
    ax.set_xticks(x, runs)
    ax.set_ylabel("Images (%)")
    ax.set_ylim(0, max(bottom) * 1.18)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

fig.suptitle("Phase 5: Error Taxonomy Across Test Sets")
handles, legend_labels = axes[0].get_legend_handles_labels()
fig.legend(handles, legend_labels, loc="lower center", ncol=5)
fig.tight_layout(rect=[0, 0.12, 1, 0.93])

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=300, bbox_inches="tight")
print(f"Figure saved: {OUT}")