#!/usr/bin/env python3
"""Post-hoc parameter sensitivity for the adaptive event-triggered state machine."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "experiments"))
import run_to_failure_state_machine as core  # noqa: E402


BASE_INTERVALS = (10, 20, 30, 40)
ESCALATED_INTERVALS = (2, 3, 5, 10)
CONFIRMATION_EVENTS = (2, 3, 5)


def evaluate(bearings, primary_threshold, base_interval, escalated_interval, confirmation_events):
    detected = set()
    leads = []
    per_dataset = {"ims": [0, 0], "xjtu": [0, 0], "femto": [0, 0]}
    false_bearings = 0
    pseudo_bearings = 0
    false_bearing_ids = []
    ims_false_axes = 0
    ims_axes = 0
    ims_false_bearings = 0
    ims_bearings = 0
    advanced_computations = 0
    monitored_windows = 0
    axis_duty_cycles = []

    for bearing_key, bearing in bearings.items():
        axis_results = {
            axis: core.axis_adaptive_state(
                rows,
                primary_threshold,
                base_interval,
                escalated_interval,
                confirmation_events,
            )
            for axis, rows in bearing["axes"].items()
        }
        first_trigger = min(result["selected_trigger"] for result in axis_results.values())
        length = min(result["n_monitor"] for result in axis_results.values())
        failed = bearing["dataset"] != "ims" or bearing_key[4:] in core.IMS_FAILED
        for result in axis_results.values():
            advanced_computations += result["advanced_computations"]
            monitored_windows += result["n_monitor"]
            axis_duty_cycles.append(result["advanced_computations"] / result["n_monitor"])

        if bearing["dataset"] in ("xjtu", "femto") and any(len(rows) >= 150 for rows in bearing["axes"].values()):
            pseudo_bearings += 1
            alarm = any(
                result["selected_trigger"] < 90
                for axis, result in axis_results.items()
                if len(bearing["axes"][axis]) >= 150
            )
            false_bearings += int(alarm)
            if alarm:
                false_bearing_ids.append(bearing_key)

        if bearing["dataset"] == "ims" and bearing_key[4:] not in core.IMS_FAILED:
            ims_bearings += 1
            axis_alarm = [
                result["selected_trigger"] < result["n_monitor"]
                for result in axis_results.values()
            ]
            ims_axes += len(axis_alarm)
            ims_false_axes += sum(axis_alarm)
            ims_false_bearings += int(any(axis_alarm))

        if failed:
            per_dataset[bearing["dataset"]][1] += 1
            if first_trigger < length:
                detected.add(bearing_key)
                per_dataset[bearing["dataset"]][0] += 1
                leads.append(100.0 * (length - first_trigger) / length)

    return {
        "base_interval": base_interval,
        "escalated_interval": escalated_interval,
        "confirmation_events": confirmation_events,
        "detected": f"{len(detected)}/34",
        "median_lead_pct_life": float(np.median(leads)),
        "per_dataset_detection": {name: f"{hit}/{total}" for name, (hit, total) in per_dataset.items()},
        "early_pseudo_health_bearing_false_alarms": false_bearings,
        "n_pseudo_health_bearings": pseudo_bearings,
        "false_alarm_bearing_ids": false_bearing_ids,
        "ims_censored_false_axes": ims_false_axes,
        "n_ims_censored_axes": ims_axes,
        "ims_censored_false_bearings": ims_false_bearings,
        "n_ims_censored_bearings": ims_bearings,
        "advanced_feature_computations": advanced_computations,
        "weighted_duty_cycle_pct": 100.0 * advanced_computations / monitored_windows,
        "mean_axis_duty_cycle_pct": 100.0 * float(np.mean(axis_duty_cycles)),
        "pass_detection": len(detected) == 34,
        "pass_false_alarm": false_bearings <= 6 and ims_false_axes == 0,
        "pass_duty": advanced_computations / monitored_windows < 0.10,
    }


def main():
    streams = core.load_merged_streams()
    bearings = core.build_bearings(streams)
    primary_threshold = core.calibrate_thresholds(streams)[core.PRIMARY]
    combinations = []
    for base_interval in BASE_INTERVALS:
        for escalated_interval in ESCALATED_INTERVALS:
            if escalated_interval >= base_interval:
                continue
            for confirmation_events in CONFIRMATION_EVENTS:
                combinations.append(
                    evaluate(
                        bearings,
                        primary_threshold,
                        base_interval,
                        escalated_interval,
                        confirmation_events,
                    )
                )
    passing = [row for row in combinations if row["pass_detection"] and row["pass_false_alarm"] and row["pass_duty"]]
    output = {
        "interpretation": "post-hoc sensitivity after adaptive escalation was discovered; no independent validation claim",
        "n_combinations": len(combinations),
        "n_pass_all_criteria": len(passing),
        "criteria": {
            "detection": "34/34",
            "early_pseudo_health_bearing_false_alarms": "<=6/27",
            "ims_censored_false_axes": 0,
            "weighted_duty_cycle": "<10%",
        },
        "passing_combinations": passing,
        "combinations": combinations,
    }
    destination = ROOT / "results/results_run_to_failure_adaptive_state_sensitivity.json"
    destination.write_text(json.dumps(output, indent=2))
    print(json.dumps({key: output[key] for key in ("n_combinations", "n_pass_all_criteria", "criteria")}, indent=2))
    print("best passing combinations:")
    for row in sorted(
        passing,
        key=lambda row: (
            row["early_pseudo_health_bearing_false_alarms"],
            row["weighted_duty_cycle_pct"],
            -row["median_lead_pct_life"],
        ),
    ):
        print(
            row["base_interval"],
            row["escalated_interval"],
            row["confirmation_events"],
            row["detected"],
            row["median_lead_pct_life"],
            row["early_pseudo_health_bearing_false_alarms"],
            row["weighted_duty_cycle_pct"],
        )
    print("saved", destination)


if __name__ == "__main__":
    main()
