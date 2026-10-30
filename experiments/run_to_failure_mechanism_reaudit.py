#!/usr/bin/env python3
"""Audit time leakage and axis-count confounding in mechanism summaries."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "experiments"))
import run_to_failure_early_warning as probe  # noqa: E402
import run_to_failure_mechanism_value_paths as mechanism  # noqa: E402

AGE = 90
FEATURES = [
    "historical",
    "cross_axis",
    "second_persistence",
    "irreversibility",
    "coactivation_fraction",
    "first_active_fraction",
    "propagation_delay",
    "max_ratio",
]
LOWER_IS_MORE_ALARMING = {
    "first_active_fraction",
    "propagation_delay",
}


def early_snapshot(bearing: dict) -> dict:
    first_raw = np.minimum(bearing["first_active"], AGE)
    first = int(np.min(first_raw))
    onsets = np.sort(first_raw)
    propagation_delay = int(onsets[1]) if onsets.size > 1 else AGE
    index = AGE - 1
    return {
        "bearing_id": bearing["bearing_id"],
        "dataset": bearing["dataset"],
        "failed": bearing["failed"],
        "axis_count": len(bearing["axes"]),
        "historical": float(bearing["historical_rate"][index]),
        "cross_axis": float(bearing["cross_axis"][index]),
        "second_persistence": float(bearing["persistence"][index]),
        "irreversibility": float(bearing["irreversibility"][index]),
        "coactivation_fraction": float(
            np.mean(np.sum(bearing["active"][:, :AGE], axis=0) >= 2)
        ),
        "first_active_fraction": first / AGE,
        "propagation_delay": propagation_delay,
        "max_ratio": float(np.max(bearing["ratios"][:, :AGE])),
    }


def separation(frame: pd.DataFrame, feature: str) -> dict:
    failed = frame[frame["failed"]]
    healthy = frame[~frame["failed"]]
    alternative = (
        "less"
        if feature in LOWER_IS_MORE_ALARMING
        else "greater"
    )
    standard_auc = roc_auc_score(
        [1] * len(failed) + [0] * len(healthy),
        failed[feature].tolist() + healthy[feature].tolist(),
    )
    directional_auc = (
        1.0 - standard_auc
        if feature in LOWER_IS_MORE_ALARMING
        else standard_auc
    )
    test = mannwhitneyu(
        failed[feature],
        healthy[feature],
        alternative=alternative,
        method="exact",
    )
    return {
        "n_failed": int(len(failed)),
        "n_healthy": int(len(healthy)),
        "auc_directional": float(directional_auc),
        "exact_mann_whitney_p": float(test.pvalue),
    }


def leave_one_out_auc(frame: pd.DataFrame, feature: str) -> dict:
    values = []
    failed_ids = frame.loc[frame["failed"], "bearing_id"].tolist()
    for bearing_id in failed_ids:
        subset = frame[frame["bearing_id"] != bearing_id]
        values.append(
            roc_auc_score(
                subset["failed"].astype(int).to_numpy(),
                subset[feature].to_numpy(),
            )
        )
    return {
        "full_auc": float(
            roc_auc_score(
                frame["failed"].astype(int).to_numpy(),
                frame[feature].to_numpy(),
            )
        ),
        "loo_min_auc": float(np.min(values)),
        "loo_median_auc": float(np.median(values)),
        "loo_max_auc": float(np.max(values)),
    }


def hierarchy_loco(bearings: list[dict]) -> dict:
    healthy = [bearing for bearing in bearings if not bearing["failed"]]
    output = []
    for margin in (1.0, 1.01):
        rows = []
        for held_out in healthy:
            calibration = [
                bearing
                for bearing in healthy
                if bearing is not held_out
            ]
            historical_threshold = (
                max(
                    np.max(bearing["historical_rate"])
                    for bearing in calibration
                )
                * margin
            )
            multi_axis = [
                bearing
                for bearing in calibration
                if len(bearing["axes"]) >= 2
            ]
            cross_axis_threshold = (
                max(np.max(bearing["cross_axis"]) for bearing in multi_axis)
                * margin
            )
            historical_alarm = probe.alarm_index(
                held_out["historical_rate"],
                historical_threshold,
                min_window=mechanism.MIN_MONITOR,
            )
            cross_axis_alarm = probe.alarm_index(
                held_out["cross_axis"],
                cross_axis_threshold,
                min_window=mechanism.MIN_MONITOR,
            )
            false_alarm = bool(
                historical_alarm < held_out["n_monitor"]
                or cross_axis_alarm < held_out["n_monitor"]
            )
            rows.append(
                {
                    "bearing_id": held_out["bearing_id"],
                    "axis_count": len(held_out["axes"]),
                    "false_alarm": false_alarm,
                    "historical_threshold": float(historical_threshold),
                    "cross_axis_threshold": float(cross_axis_threshold),
                }
            )
        output.append(
            {
                "calibration_margin": margin,
                "false_alarms": sum(row["false_alarm"] for row in rows),
                "n_held_out": len(rows),
                "details": rows,
            }
        )
    return {"leave_one_censored_out": output}


def main() -> None:
    streams = probe.load_all_streams(ROOT / "data" / "features")
    bearings = mechanism.build_bearings(streams)
    eligible = [
        early_snapshot(bearing)
        for bearing in bearings
        if bearing["n_monitor"] >= AGE
    ]
    frame = pd.DataFrame(eligible)
    groups = {
        "all": frame,
        "ims_same_platform": frame[frame["dataset"] == "ims"],
        "femto_vs_ims": frame[
            (frame["dataset"] == "femto") | (~frame["failed"])
        ],
        "xjtu_vs_ims": frame[
            (frame["dataset"] == "xjtu") | (~frame["failed"])
        ],
        "two_axis_only": frame[frame["axis_count"] == 2],
    }

    axis_table = (
        frame.groupby(["failed", "axis_count"]).size().rename("n").reset_index()
    )
    profiles = pd.DataFrame(mechanism.mechanism_profiles(bearings))
    axis_counts = {
        bearing["bearing_id"]: len(bearing["axes"]) for bearing in bearings
    }
    profiles["axis_count"] = profiles["bearing_id"].map(axis_counts)
    two_axis_profiles = profiles[profiles["axis_count"] == 2]

    failed_ids = profiles.loc[profiles["failed"], "bearing_id"].tolist()
    maxima = mechanism.score_maxima(bearings)
    result = {
        "status": "exploratory_reaudit_only",
        "purpose": [
            "test whether full-trajectory mechanism summaries remain visible at a fixed early horizon",
            "quantify axis-count confounding",
            "verify held-out calibration behavior without reusing all healthy trajectories",
        ],
        "fixed_horizon": {
            "monitored_snapshots": AGE,
            "raw_snapshots_including_baseline": AGE + 60,
            "n_eligible_bearings": int(len(frame)),
            "n_failed": int(frame["failed"].sum()),
            "n_healthy": int((~frame["failed"]).sum()),
            "separation_by_group": {
                group: {
                    feature: separation(subset, feature)
                    for feature in FEATURES
                }
                for group, subset in groups.items()
            },
        },
        "axis_availability": {
            "fixed_horizon_table": axis_table.to_dict(orient="records"),
            "healthy_multi_axis_bearings": int(
                sum(
                    not bearing["failed"] and len(bearing["axes"]) >= 2
                    for bearing in bearings
                )
            ),
            "healthy_single_axis_bearings": int(
                sum(
                    not bearing["failed"] and len(bearing["axes"]) == 1
                    for bearing in bearings
                )
            ),
            "interpretation": (
                "Second-axis statistics are zero by construction for six of eight "
                "healthy bearings; unstratified cross-axis AUC is confounded by "
                "axis availability."
            ),
        },
        "full_trajectory_two_axis_only": {
            "n_failed": int(two_axis_profiles["failed"].sum()),
            "n_healthy": int((~two_axis_profiles["failed"]).sum()),
            "max_second_axis_recent_rate": separation(
                two_axis_profiles.rename(
                    columns={"max_second_axis_recent_rate": "cross_axis"}
                ),
                "cross_axis",
            ),
            "caution": "only two healthy multi-axis trajectories",
        },
        "full_profile_influence": {
            "max_historical_rate": leave_one_out_auc(
                profiles, "max_historical_rate"
            ),
            "max_second_axis_recent_rate": leave_one_out_auc(
                profiles, "max_second_axis_recent_rate"
            ),
        },
        "hierarchy_calibration": {
            **hierarchy_loco(bearings),
            "all_healthy_thresholds_with_1p01_margin": {
                name: float(value * 1.01)
                for name, value in maxima.items()
            },
        },
    }
    output = ROOT / "results" / "results_run_to_failure_mechanism_reaudit.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2))
    print("saved", output)


if __name__ == "__main__":
    main()
