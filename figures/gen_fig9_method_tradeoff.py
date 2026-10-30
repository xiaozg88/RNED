#!/usr/bin/env python3
"""Fig. 9: detection--lead-time trade-off under identical zero-FA calibration."""
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "figures"))
from paper_plot_style import *  # noqa: E402,F403


METHODS = [
    {
        "name": "Excess rate",
        "family": "ours",
        "detection": 31,
        "dataset_median_leads": [14.5, 46.3, 36.0],
    },
    {
        "name": "Shewhart",
        "family": "classical",
        "detection": 25,
        "dataset_median_leads": [0.7, 6.8, 1.1],
    },
    {
        "name": "EWMA",
        "family": "classical",
        "detection": 25,
        "dataset_median_leads": [0.6, 6.4, 1.6],
    },
    {
        "name": "CUSUM",
        "family": "classical",
        "detection": 21,
        "dataset_median_leads": [4.4, 16.3, 27.3],
    },
    {
        "name": "MLP-AE",
        "family": "deep",
        "detection": 20,
        "dataset_median_leads": [0.2, 1.2, 25.1],
    },
    {
        "name": "LSTM-AE",
        "family": "deep",
        "detection": 20,
        "dataset_median_leads": [0.2, 1.2, 25.1],
    },
    {
        "name": "Transformer-AE",
        "family": "deep",
        "detection": 20,
        "dataset_median_leads": [0.2, 1.6, 25.7],
    },
    {
        "name": "TranAD",
        "family": "deep",
        "detection": 20,
        "dataset_median_leads": [0.2, 1.6, 25.7],
    },
]


def main() -> None:
    fig, ax = plt.subplots(figsize=(3.45, 2.65))
    styles = {
        "ours": dict(color=COLORS[0], marker="*", s=165, edgecolor="white", zorder=5),
        "classical": dict(color=COLORS[1], marker="o", s=35, edgecolor="white"),
        "deep": dict(color=COLORS[2], marker="^", s=42, edgecolor="white"),
    }
    legend_labels = {
        "ours": "Proposed",
        "classical": "Classical",
        "deep": "Deep",
    }
    plotted = {"ours", "classical", "deep"}
    rows = []
    for method in METHODS:
        macro_lead = float(np.median(method["dataset_median_leads"]))
        row = {
            "method": method["name"],
            "family": method["family"],
            "detections_of_34": method["detection"],
            "macro_median_lead_pct_life": macro_lead,
        }
        rows.append(row)
        ax.scatter(
            macro_lead,
            method["detection"],
            label=legend_labels[method["family"]]
            if method["family"] in plotted
            else None,
            **styles[method["family"]],
        )
        plotted.discard(method["family"])

    ax.annotate(
        "Excess rate\n31/34",
        xy=(36.0, 31),
        xytext=(25.5, 30.0),
        ha="right",
        va="center",
        color=COLORS[0],
        fontsize=FONT_SIZE - 1,
    )
    ax.annotate("CUSUM 21/34", xy=(16.3, 21), xytext=(13.0, 23.2),
                fontsize=FONT_SIZE - 2, color=COLORS[1])
    ax.annotate("instant/short memory", xy=(1.6, 25), xytext=(4.0, 25.8),
                fontsize=FONT_SIZE - 2, color=COLORS[1])
    ax.annotate("deep reconstruction", xy=(1.6, 20), xytext=(4.0, 17.5),
                fontsize=FONT_SIZE - 2, color=COLORS[2])

    ax.set_xlabel("macro median lead (% remaining life)")
    ax.set_ylabel("detected failed bearings (/34)")
    ax.set_xlim(0, 45)
    ax.set_ylim(14, 34)
    ax.set_yticks([15, 20, 25, 30, 34])
    ax.grid(axis="both", color="0.88", lw=0.6)
    ax.legend(
        frameon=False,
        loc="lower right",
        handletextpad=0.2,
        borderaxespad=0.1,
    )
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(ROOT / "figures" / f"fig9_method_tradeoff.{suffix}")
    print(f"saved {ROOT / 'figures' / 'fig9_method_tradeoff.{pdf,png}'}")

    output = ROOT / "results" / "fig9_method_tradeoff.csv"
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("saved", output)


if __name__ == "__main__":
    main()
