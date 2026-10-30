#!/usr/bin/env python3
"""Fig 10: LDM_UPB sensitivity to operating-condition changes.

The two representative bearings contrast an alarm immediately after a large
speed/load transition (B01) with a later alarm under nearly fixed settings
(B11). All curves are aligned to the first 60 snapshots used by the frozen
protocol, and the alarm marker is the frozen hierarchical first crossing.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "figures"))
from paper_plot_style import *  # noqa: E402,F401

FEATURE_CSV = ROOT / "data" / "features" / "ldm_upb_features.csv"
RESULT_JSON = (
    ROOT
    / "results"
    / "results_run_to_failure_public_locked_validation_ldm_upb.json"
)
RAW_DIR = ROOT / "data" / "raw_archives" / "ldm_upb_hf"
BEARINGS = ("B01", "B11")


def load_rms() -> dict[str, dict[str, np.ndarray]]:
    grouped: dict[str, dict[str, list[tuple[int, float]]]] = {}
    with FEATURE_CSV.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["bearing"] in BEARINGS:
                grouped.setdefault(row["bearing"], {}).setdefault(
                    row["axis"], []
                ).append((int(row["snapshot"]), float(row["rms"])))
    output: dict[str, dict[str, np.ndarray]] = {}
    for bearing, axes in grouped.items():
        output[bearing] = {
            axis: np.asarray(
                [value for _, value in sorted(values)], dtype=np.float64
            )
            for axis, values in axes.items()
        }
    return output


def load_conditions() -> dict[str, dict[str, np.ndarray]]:
    columns = {
        "setDynLoad / N": "dynamic load",
        "setStatLoad / N": "static load",
        "setSpeed / rpm": "speed",
    }
    output: dict[str, dict[str, np.ndarray]] = {}
    for bearing in BEARINGS:
        path = RAW_DIR / bearing / f"{bearing}_operatingConditions.csv"
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        output[bearing] = {
            label: np.asarray(
                [float(row[key]) for row in rows], dtype=np.float64
            )
            for key, label in columns.items()
        }
    return output


def main() -> None:
    rms_data = load_rms()
    condition_data = load_conditions()
    validation = json.loads(RESULT_JSON.read_text())
    per_bearing = {
        row["bearing_id"]: row for row in validation["per_bearing"]
    }

    fig, axes = plt.subplots(
        4,
        2,
        figsize=(6.8, 5.8),
        sharex="col",
        constrained_layout=True,
    )

    sidecar = []
    for column, bearing in enumerate(BEARINGS):
        result = per_bearing[bearing]
        alarm_snapshot = 60 + result["hierarchical_alarm_index"]
        total_snapshots = 60 + result["n_monitor"]
        life_fraction = (60 + np.arange(result["n_monitor"])) / total_snapshots
        alarm_fraction = alarm_snapshot / total_snapshots

        ax = axes[0, column]
        for axis, values in sorted(rms_data[bearing].items()):
            baseline = float(np.median(values[10:60]))
            ax.plot(
                life_fraction,
                values[60:] / baseline,
                lw=1.1,
                label=f"axis {axis}",
            )
        ax.axvline(
            alarm_fraction,
            color=COLORS[1],
            lw=1.2,
            ls="--",
            label="frozen alarm",
        )
        ax.axhline(1.2, color="0.35", lw=0.8, ls=":")
        ax.set_ylabel("RMS / baseline")
        ax.set_title(
            f"{bearing}: alarm at {100 * alarm_fraction:.1f}% life",
            fontsize=FONT_SIZE + 1,
        )
        ax.legend(frameon=False, loc="upper left", ncol=1)

        baseline_index = 59
        for row_index, condition in enumerate(
            ("dynamic load", "static load", "speed"), start=1
        ):
            values = condition_data[bearing][condition]
            ax = axes[row_index, column]
            ax.plot(life_fraction, values[60:], color=COLORS[2], lw=1.1)
            ax.axvline(
                alarm_fraction,
                color=COLORS[1],
                lw=1.2,
                ls="--",
            )
            baseline_value = float(values[baseline_index])
            alarm_value = float(values[alarm_snapshot])
            relative_change = 100.0 * (
                alarm_value / baseline_value - 1.0
            )
            ax.set_ylabel(condition)
            ax.text(
                0.98,
                0.92,
                f"change at alarm: {relative_change:+.1f}%",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=FONT_SIZE - 2,
            )

            sidecar.append(
                {
                    "bearing": bearing,
                    "condition": condition,
                    "baseline_snapshot": baseline_index,
                    "alarm_snapshot": alarm_snapshot,
                    "baseline_value": baseline_value,
                    "alarm_value": alarm_value,
                    "relative_change_pct": relative_change,
                }
            )

        axes[-1, column].set_xlabel("life fraction")
        sidecar.append(
            {
                "bearing": bearing,
                "hierarchical_alarm_index": result[
                    "hierarchical_alarm_index"
                ],
                "alarm_snapshot": alarm_snapshot,
                "n_snapshots": 60 + result["n_monitor"],
                "lead_pct_life": result["hierarchical_lead_pct_life"],
            }
        )

    save_fig(fig, "fig10_ldm_upb_condition_sensitivity")
    sidecar_path = (
        ROOT
        / "results"
        / "results_fig10_ldm_upb_condition_sensitivity.json"
    )
    sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n")
    print(f"saved {sidecar_path}")


if __name__ == "__main__":
    main()
