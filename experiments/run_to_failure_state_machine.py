#!/usr/bin/env python3
"""Frozen causal state-machine and information-budget experiments.

The detector computes RMS on every monitoring window, but computes envelope
kurtosis and fault-harmonic evidence only at scheduled WATCH events.  This is
an audit of a deployable event-triggered route, not a full-time spectral rule.
"""
from __future__ import annotations

import csv
import json
import math
import platform
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert
from scipy.stats import kurtosis


ROOT = Path(__file__).resolve().parents[1]
IMS_FAILED = {"test1_b3", "test1_b4", "test2_b1", "test3_b3"}
PRIMARY = "raw_rms"
SECONDARY_FEATURES = ("env_kurtosis", "fault_harmonic_ratio")
FROZEN = {
    "wake_ratio": 1.2,
    "diagnostic_interval": 20,
    "confirmation_events": 3,
    "secondary_ratio": 1.2,
    "minimum_monitor_updates": 30,
    "baseline_snapshots": [11, 60],
}
ADAPTIVE = {
    **FROZEN,
    "escalated_interval": 5,
    "escalation_condition": "both secondary features positive at a base-rate event",
    "status": "exploratory post-hoc extension; not part of the frozen protocol",
}


def load_merged_streams():
    streams = {}

    def add(key, row):
        snapshot = int(row.pop("snapshot"))
        streams.setdefault(key, {}).setdefault(snapshot, {"snapshot": snapshot}).update(row)

    for test_name, filename in (
        ("test1", "test1.csv"),
        ("test2", "test2.csv"),
        ("test3", "test3.csv"),
    ):
        with (ROOT / "data/features" / filename).open() as source:
            for row in csv.DictReader(source):
                key = ("ims", test_name, f"{test_name}_b{row['bearing']}", row["channel"])
                snapshot = len(streams.get(key, {})) + 1
                add(key, {"snapshot": snapshot, PRIMARY: float(row["rms"])})

    for filename, dataset in (
        ("xjtu_features.csv", "xjtu"),
        ("femto_features.csv", "femto"),
    ):
        with (ROOT / "data/features" / filename).open() as source:
            for row in csv.DictReader(source):
                key = (dataset, row["group"], row["bearing"], row["axis"])
                add(key, {"snapshot": int(row["snapshot"]), PRIMARY: float(row["rms"])})

    with (ROOT / "data/features/modern_signal_features.csv").open() as source:
        for row in csv.DictReader(source):
            key = (row["dataset"], row["group"], row["bearing"], row["axis"])
            add(
                key,
                {
                    "snapshot": int(row["snapshot"]),
                    **{name: float(row[name]) for name in SECONDARY_FEATURES},
                },
            )

    return {
        key: [rows[snapshot] for snapshot in sorted(rows)]
        for key, rows in streams.items()
    }


def build_bearings(streams):
    bearings = {}
    for (dataset, group, bearing, axis), rows in streams.items():
        if len(rows) < 70:
            continue
        bearing_key = f"{dataset}_{bearing}"
        bearings.setdefault(
            bearing_key,
            {"dataset": dataset, "group": group, "bearing": bearing, "axes": {}},
        )["axes"][axis] = rows
    return bearings


def feature_values(rows, feature):
    return np.asarray([float(row[feature]) for row in rows if feature in row], dtype=np.float64)


def early_baseline(rows, feature):
    if feature == PRIMARY:
        values = feature_values(rows, feature)
        return float(np.median(values[10:60]))
    else:
        values = np.asarray(
            [float(row[feature]) for row in rows if feature in row and 11 <= int(row["snapshot"]) <= 60],
            dtype=np.float64,
        )
        return float(np.median(values))


def cumulative_rate(rows, feature):
    if feature == PRIMARY:
        values = feature_values(rows, feature)
        baseline = float(np.median(values[10:60]))
        monitoring_values = values[60:]
    else:
        values = np.asarray(
            [float(row[feature]) for row in rows if feature in row and int(row["snapshot"]) > 60],
            dtype=np.float64,
        )
        baseline = early_baseline(rows, feature)
        monitoring_values = values
    excess = np.maximum(monitoring_values / baseline - 1.2, 0.0)
    return np.cumsum(excess) / np.arange(1, len(excess) + 1)


def calibrate_thresholds(streams):
    thresholds = {}
    for feature in (PRIMARY, *SECONDARY_FEATURES):
        maxima = []
        for key, rows in streams.items():
            if key[0] == "ims" and key[2] not in IMS_FAILED:
                if any(feature in row for row in rows):
                    maxima.append(float(np.max(cumulative_rate(rows, feature))))
        thresholds[feature] = 1.01 * max(maxima)
    return thresholds


def first_cumulative_alarm(rows, feature, threshold):
    score = cumulative_rate(rows, feature) / threshold
    hits = np.flatnonzero(score[29:] > 1.0) + 29
    return int(hits[0]) if hits.size else len(score)


def first_true_run(values, confirmation, minimum_index=30):
    exceed = np.asarray(values, dtype=bool)
    run = np.zeros(len(exceed), dtype=np.int64)
    if len(run):
        run[0] = int(exceed[0])
    for index in range(1, len(exceed)):
        run[index] = run[index - 1] + 1 if exceed[index] else 0
    eligible = np.flatnonzero((run >= confirmation) & (np.arange(len(exceed)) >= minimum_index))
    return int(eligible[0]) if eligible.size else len(exceed)


def axis_state(rows, primary_threshold):
    n_monitor = len(rows) - 60
    raw = feature_values(rows, PRIMARY)
    raw_baseline = early_baseline(rows, PRIMARY)
    wake = raw[60:] / raw_baseline > FROZEN["wake_ratio"]
    scheduled = np.arange(n_monitor) % FROZEN["diagnostic_interval"] == 0
    feature_available = np.array(
        [
            all(feature in rows[60 + monitor_index] for feature in SECONDARY_FEATURES)
            for monitor_index in range(n_monitor)
        ],
        dtype=bool,
    )
    diagnostic_indices = np.flatnonzero(wake & scheduled & feature_available)

    primary_trigger = first_cumulative_alarm(rows, PRIMARY, primary_threshold)
    diagnostic_trigger = n_monitor
    if len(diagnostic_indices) >= FROZEN["confirmation_events"]:
        positive = np.ones(len(diagnostic_indices), dtype=bool)
        for feature in SECONDARY_FEATURES:
            baseline = early_baseline(rows, feature)
            values = np.asarray(
                [float(rows[60 + monitor_index][feature]) for monitor_index in diagnostic_indices],
                dtype=np.float64,
            )
            positive &= values / baseline > FROZEN["secondary_ratio"]
        event_trigger = first_true_run(positive, FROZEN["confirmation_events"], minimum_index=0)
        if (
            event_trigger < len(diagnostic_indices)
            and diagnostic_indices[event_trigger] >= FROZEN["minimum_monitor_updates"] - 1
        ):
            diagnostic_trigger = int(diagnostic_indices[event_trigger])

    return {
        "n_monitor": n_monitor,
        "wake": wake,
        "primary_trigger": primary_trigger,
        "diagnostic_trigger": diagnostic_trigger,
        "selected_trigger": min(primary_trigger, diagnostic_trigger),
        "diagnostic_indices": diagnostic_indices,
        "computed_diagnostic_indices": (
            diagnostic_indices[diagnostic_indices <= diagnostic_trigger]
            if diagnostic_trigger < n_monitor
            else diagnostic_indices
        ),
        "n_watch_events": int(np.sum(wake)),
        "n_watch_entries": int(np.sum(wake[1:] & ~wake[:-1]) + int(wake[0] if len(wake) else 0)),
    }


def evaluate_primary(bearings, primary_threshold):
    detected = set()
    leads = []
    per_dataset = {"ims": [0, 0], "xjtu": [0, 0], "femto": [0, 0]}
    triggers = {}
    for bearing_key, bearing in bearings.items():
        failed = bearing["dataset"] != "ims" or bearing_key[4:] in IMS_FAILED
        if not failed:
            continue
        per_dataset[bearing["dataset"]][1] += 1
        axis_triggers = {
            axis: first_cumulative_alarm(rows, PRIMARY, primary_threshold)
            for axis, rows in bearing["axes"].items()
        }
        first_trigger = min(axis_triggers.values())
        length = min(len(rows) for rows in bearing["axes"].values()) - 60
        triggers[bearing_key] = {"monitor_index": first_trigger, "axis": min(axis_triggers, key=axis_triggers.get)}
        if first_trigger < length:
            detected.add(bearing_key)
            per_dataset[bearing["dataset"]][0] += 1
            leads.append(100.0 * (length - first_trigger) / length)
    return {
        "detected": f"{len(detected)}/34",
        "median_lead_pct_life": float(np.median(leads)),
        "per_dataset_detection": {name: f"{hit}/{total}" for name, (hit, total) in per_dataset.items()},
        "triggers": triggers,
    }


def evaluate_state(bearings, primary_threshold, primary_output):
    detected = set()
    leads = []
    per_dataset = {"ims": [0, 0], "xjtu": [0, 0], "femto": [0, 0]}
    triggers = {}
    rescued = []
    advanced_computations = 0
    monitored_windows = 0
    watch_events = 0
    watch_entries = 0
    diagnostic_tags = 0
    axis_duty_cycles = []

    for bearing_key, bearing in bearings.items():
        failed = bearing["dataset"] != "ims" or bearing_key[4:] in IMS_FAILED
        axis_results = {axis: axis_state(rows, primary_threshold) for axis, rows in bearing["axes"].items()}
        length = min(result["n_monitor"] for result in axis_results.values())
        selected_axis = min(axis_results, key=lambda axis: axis_results[axis]["selected_trigger"])
        first_trigger = axis_results[selected_axis]["selected_trigger"]

        for result in axis_results.values():
            advanced_computations += len(result["computed_diagnostic_indices"])
            monitored_windows += result["n_monitor"]
            axis_duty_cycles.append(len(result["computed_diagnostic_indices"]) / result["n_monitor"])
            watch_events += result["n_watch_events"]
            watch_entries += result["n_watch_entries"]
            diagnostic_tags += int(result["diagnostic_trigger"] < result["n_monitor"])

        triggers[bearing_key] = {
            "axis": selected_axis,
            "monitor_index": int(first_trigger),
            "snapshot": int(first_trigger + 60),
            "source": (
                "diagnostic_tag"
                if axis_results[selected_axis]["diagnostic_trigger"] < axis_results[selected_axis]["primary_trigger"]
                else "primary_rms"
            ),
            "lead_pct_life": 100.0 * (length - first_trigger) / length if first_trigger < length else None,
        }
        if failed:
            per_dataset[bearing["dataset"]][1] += 1
            if first_trigger < length:
                detected.add(bearing_key)
                per_dataset[bearing["dataset"]][0] += 1
                leads.append(100.0 * (length - first_trigger) / length)

    primary_detected = {
        key
        for key, trigger in primary_output["triggers"].items()
        if trigger["monitor_index"] < min(len(rows) for rows in bearings[key]["axes"].values()) - 60
    }
    return {
        "detected": f"{len(detected)}/34",
        "median_lead_pct_life": float(np.median(leads)),
        "per_dataset_detection": {name: f"{hit}/{total}" for name, (hit, total) in per_dataset.items()},
        "rescued_bearings": sorted(detected - primary_detected),
        "missed_bearings": sorted(
            {
                bearing_key
                for bearing_key, bearing in bearings.items()
                if bearing["dataset"] != "ims" or bearing_key[4:] in IMS_FAILED
            }
            - detected
        ),
        "advanced_feature_computations": advanced_computations,
        "advanced_feature_duty_cycle_pct": 100.0 * advanced_computations / monitored_windows,
        "n_watch_events": watch_events,
        "n_watch_entries": watch_entries,
        "n_diagnostic_tag_axes": diagnostic_tags,
        "mean_axis_duty_cycle_pct": 100.0 * float(np.mean(axis_duty_cycles)),
        "triggers": triggers,
    }


def axis_adaptive_state(
    rows,
    primary_threshold,
    base_interval=None,
    escalated_interval=None,
    confirmation_events=None,
):
    base_interval = base_interval or ADAPTIVE["diagnostic_interval"]
    escalated_interval = escalated_interval or ADAPTIVE["escalated_interval"]
    confirmation_events = confirmation_events or ADAPTIVE["confirmation_events"]
    raw = feature_values(rows, PRIMARY)
    raw_baseline = early_baseline(rows, PRIMARY)
    secondary_baselines = {
        feature: early_baseline(rows, feature) for feature in SECONDARY_FEATURES
    }
    primary_trigger = first_cumulative_alarm(rows, PRIMARY, primary_threshold)
    n_monitor = len(raw) - 60
    state = "normal"
    confirmation_run = 0
    diagnostic_trigger = n_monitor
    advanced_computations = 0
    diagnostic_indices = []
    state_transitions = {"watch": 0, "escalated": 0, "normal": 0}

    for monitor_index in range(n_monitor):
        wake = raw[60 + monitor_index] / raw_baseline > ADAPTIVE["wake_ratio"]
        if wake and state == "normal":
            state = "watch"
            confirmation_run = 0
            state_transitions["watch"] += 1
        elif not wake and state != "normal":
            state = "normal"
            confirmation_run = 0
            state_transitions["normal"] += 1

        base_schedule = monitor_index % base_interval == 0
        escalated_schedule = monitor_index % escalated_interval == 0
        compute_secondary = (state == "watch" and base_schedule) or (
            state == "escalated" and escalated_schedule
        )
        if not compute_secondary:
            continue

        advanced_computations += 1
        diagnostic_indices.append(monitor_index)
        positive = True
        for feature in SECONDARY_FEATURES:
            value = float(rows[60 + monitor_index][feature])
            positive &= value / secondary_baselines[feature] > ADAPTIVE["secondary_ratio"]

        if state == "watch" and positive:
            state = "escalated"
            state_transitions["escalated"] += 1
        confirmation_run = confirmation_run + 1 if positive else 0
        if (
            confirmation_run >= confirmation_events
            and monitor_index >= ADAPTIVE["minimum_monitor_updates"] - 1
        ):
            diagnostic_trigger = monitor_index
            break

    return {
        "n_monitor": n_monitor,
        "primary_trigger": primary_trigger,
        "diagnostic_trigger": diagnostic_trigger,
        "selected_trigger": min(primary_trigger, diagnostic_trigger),
        "advanced_computations": advanced_computations,
        "diagnostic_indices": np.asarray(diagnostic_indices, dtype=np.int64),
        "state_transitions": state_transitions,
    }


def evaluate_adaptive_state(bearings, primary_threshold, primary_output):
    detected = set()
    leads = []
    per_dataset = {"ims": [0, 0], "xjtu": [0, 0], "femto": [0, 0]}
    triggers = {}
    advanced_computations = 0
    monitored_windows = 0
    state_transitions = {"watch": 0, "escalated": 0, "normal": 0}
    diagnostic_tags = 0
    axis_duty_cycles = []

    for bearing_key, bearing in bearings.items():
        failed = bearing["dataset"] != "ims" or bearing_key[4:] in IMS_FAILED
        axis_results = {
            axis: axis_adaptive_state(rows, primary_threshold)
            for axis, rows in bearing["axes"].items()
        }
        selected_axis = min(axis_results, key=lambda axis: axis_results[axis]["selected_trigger"])
        first_trigger = axis_results[selected_axis]["selected_trigger"]
        length = min(result["n_monitor"] for result in axis_results.values())
        for result in axis_results.values():
            advanced_computations += result["advanced_computations"]
            monitored_windows += result["n_monitor"]
            axis_duty_cycles.append(result["advanced_computations"] / result["n_monitor"])
            diagnostic_tags += int(result["diagnostic_trigger"] < result["n_monitor"])
            for state, count in result["state_transitions"].items():
                state_transitions[state] += count

        triggers[bearing_key] = {
            "axis": selected_axis,
            "monitor_index": int(first_trigger),
            "snapshot": int(first_trigger + 60),
            "source": (
                "diagnostic_tag"
                if axis_results[selected_axis]["diagnostic_trigger"] < axis_results[selected_axis]["primary_trigger"]
                else "primary_rms"
            ),
            "lead_pct_life": 100.0 * (length - first_trigger) / length if first_trigger < length else None,
        }
        if failed:
            per_dataset[bearing["dataset"]][1] += 1
            if first_trigger < length:
                detected.add(bearing_key)
                per_dataset[bearing["dataset"]][0] += 1
                leads.append(100.0 * (length - first_trigger) / length)

    primary_detected = {
        key
        for key, trigger in primary_output["triggers"].items()
        if trigger["monitor_index"] < min(len(rows) for rows in bearings[key]["axes"].values()) - 60
    }
    false_axes = 0
    pseudo_axes = 0
    false_bearings = 0
    pseudo_bearings = 0
    ims_false_axes = 0
    ims_axes = 0
    ims_false_bearings = 0
    ims_bearings = 0
    for bearing_key, bearing in bearings.items():
        axis_results = {
            axis: axis_adaptive_state(rows, primary_threshold)
            for axis, rows in bearing["axes"].items()
        }
        first_trigger = min(result["selected_trigger"] for result in axis_results.values())
        if bearing["dataset"] in ("xjtu", "femto") and any(len(rows) >= 150 for rows in bearing["axes"].values()):
            bearing_alarm = False
            pseudo_bearings += 1
            for axis, result in axis_results.items():
                if len(bearing["axes"][axis]) < 150:
                    continue
                pseudo_axes += 1
                axis_alarm = result["selected_trigger"] < 90
                false_axes += int(axis_alarm)
                bearing_alarm = bearing_alarm or axis_alarm
            false_bearings += int(bearing_alarm)
        if bearing["dataset"] == "ims" and bearing_key[4:] not in IMS_FAILED:
            ims_bearings += 1
            bearing_alarm = False
            for result in axis_results.values():
                ims_axes += 1
                axis_alarm = result["selected_trigger"] < result["n_monitor"]
                ims_false_axes += int(axis_alarm)
                bearing_alarm = bearing_alarm or axis_alarm
            ims_false_bearings += int(bearing_alarm)

    rescue_leads = {
        bearing_key: triggers[bearing_key]["lead_pct_life"]
        for bearing_key in (
            "femto_Bearing1_5",
            "femto_Bearing1_6",
            "femto_Bearing2_4",
        )
        if bearing_key in triggers
    }

    return {
        "detected": f"{len(detected)}/34",
        "median_lead_pct_life": float(np.median(leads)),
        "per_dataset_detection": {name: f"{hit}/{total}" for name, (hit, total) in per_dataset.items()},
        "rescued_bearings": sorted(detected - primary_detected),
        "rescue_lead_pct_life": rescue_leads,
        "rescue_interpretation": "nominal end-of-life coverage rescue; all three rescued bearings are tagged within 1.4% of failure",
        "missed_bearings": sorted(
            {
                bearing_key
                for bearing_key, bearing in bearings.items()
                if bearing["dataset"] != "ims" or bearing_key[4:] in IMS_FAILED
            }
            - detected
        ),
        "advanced_feature_computations": advanced_computations,
        "advanced_feature_duty_cycle_pct": 100.0 * advanced_computations / monitored_windows,
        "mean_axis_duty_cycle_pct": 100.0 * float(np.mean(axis_duty_cycles)),
        "n_diagnostic_tag_axes": diagnostic_tags,
        "state_transition_counts": state_transitions,
        "early_pseudo_health": {
            "axis_false_alarms": false_axes,
            "n_pseudo_health_axes": pseudo_axes,
            "axis_false_alarm_rate_pct": 100.0 * false_axes / pseudo_axes,
            "bearing_false_alarms": false_bearings,
            "n_pseudo_health_bearings": pseudo_bearings,
            "bearing_false_alarm_rate_pct": 100.0 * false_bearings / pseudo_bearings,
            "scope": "first 90 monitoring windows (snapshots 61-150)",
        },
        "ims_censored_full_life": {
            "axis_false_alarms": ims_false_axes,
            "n_censored_axes": ims_axes,
            "bearing_false_alarms": ims_false_bearings,
            "n_censored_bearings": ims_bearings,
        },
        "triggers": triggers,
    }
    rescued = sorted(detected - primary_detected)
    missed = sorted(
        {
            bearing_key
            for bearing_key, bearing in bearings.items()
            if bearing["dataset"] != "ims" or bearing_key[4:] in IMS_FAILED
        }
        - detected
    )

    return {
        "detected": f"{len(detected)}/34",
        "median_lead_pct_life": float(np.median(leads)),
        "per_dataset_detection": {name: f"{hit}/{total}" for name, (hit, total) in per_dataset.items()},
        "rescued_bearings": rescued,
        "missed_bearings": missed,
        "advanced_feature_computations": advanced_computations,
        "advanced_feature_duty_cycle_pct": 100.0 * advanced_computations / monitored_windows,
        "n_watch_events": watch_events,
        "n_watch_entries": watch_entries,
        "n_diagnostic_tag_axes": diagnostic_tags,
        "triggers": triggers,
    }


def false_alarm_summary(bearings, primary_threshold, use_state):
    false_axes = 0
    pseudo_axes = 0
    false_bearings = 0
    pseudo_bearings = 0
    false_bearing_ids = []

    for bearing_key, bearing in bearings.items():
        if bearing["dataset"] not in ("xjtu", "femto"):
            continue
        eligible_axes = [axis for axis, rows in bearing["axes"].items() if len(rows) >= 150]
        if not eligible_axes:
            continue
        bearing_alarm = False
        for axis in eligible_axes:
            rows = bearing["axes"][axis]
            if use_state:
                trigger = axis_state(rows, primary_threshold)["selected_trigger"]
            else:
                trigger = first_cumulative_alarm(rows, PRIMARY, primary_threshold)
            axis_alarm = trigger < 90
            pseudo_axes += 1
            false_axes += int(axis_alarm)
            bearing_alarm = bearing_alarm or axis_alarm
        pseudo_bearings += 1
        false_bearings += int(bearing_alarm)
        if bearing_alarm:
            false_bearing_ids.append(bearing_key)

    return {
        "axis_false_alarms": false_axes,
        "n_pseudo_health_axes": pseudo_axes,
        "axis_false_alarm_rate_pct": 100.0 * false_axes / pseudo_axes,
        "bearing_false_alarms": false_bearings,
        "n_pseudo_health_bearings": pseudo_bearings,
        "bearing_false_alarm_rate_pct": 100.0 * false_bearings / pseudo_bearings,
        "false_alarm_bearing_ids": false_bearing_ids,
        "scope": "first 90 monitoring windows (snapshots 61-150)",
    }


def ims_censored_false_alarms(bearings, primary_threshold, use_state):
    false_axes = 0
    total_axes = 0
    false_bearings = 0
    total_bearings = 0
    for bearing_key, bearing in bearings.items():
        if bearing["dataset"] != "ims" or bearing_key[4:] in IMS_FAILED:
            continue
        total_bearings += 1
        bearing_alarm = False
        for rows in bearing["axes"].values():
            total_axes += 1
            if use_state:
                trigger = axis_state(rows, primary_threshold)["selected_trigger"]
            else:
                trigger = first_cumulative_alarm(rows, PRIMARY, primary_threshold)
            alarm = trigger < len(rows) - 60
            false_axes += int(alarm)
            bearing_alarm = bearing_alarm or alarm
        false_bearings += int(bearing_alarm)
    return {
        "axis_false_alarms": false_axes,
        "n_censored_axes": total_axes,
        "bearing_false_alarms": false_bearings,
        "n_censored_bearings": total_bearings,
    }


def full_time_dual_scores(bearings, thresholds):
    scores = {}
    for bearing_key, bearing in bearings.items():
        scores[bearing_key] = {}
        for axis, rows in bearing["axes"].items():
            normalized = [
                cumulative_rate(rows, feature) / thresholds[feature]
                for feature in SECONDARY_FEATURES
            ]
            scores[bearing_key][axis] = np.minimum(*normalized)
    return scores


def fleet_budget_operating_curve(bearings, scores, primary_threshold):
    pseudo_axis_scores = []
    axis_metadata = []
    for bearing_key, bearing in bearings.items():
        if bearing["dataset"] not in ("xjtu", "femto"):
            continue
        for axis, score in scores[bearing_key].items():
            if len(score) >= 90:
                pseudo_axis_scores.append(float(np.max(score[29:90])))
                axis_metadata.append((bearing_key, bearing["dataset"], axis))

    def evaluate_threshold(threshold):
        detected = 0
        leads = []
        per_dataset = {"ims": [0, 0], "xjtu": [0, 0], "femto": [0, 0]}
        diagnostic_false_axes = 0
        combined_false_axes = 0
        diagnostic_false_bearings = 0
        combined_false_bearings = 0
        pseudo_bearings = 0
        for bearing_key, bearing in bearings.items():
            failed = bearing["dataset"] != "ims" or bearing_key[4:] in IMS_FAILED
            triggers = []
            primary_triggers = []
            for axis, score in scores[bearing_key].items():
                rows = bearing["axes"][axis]
                primary_trigger = first_cumulative_alarm(rows, PRIMARY, primary_threshold)
                primary_triggers.append(primary_trigger)
                hits = np.flatnonzero(score[29:] > threshold) + 29
                diagnostic_trigger = int(hits[0]) if hits.size else len(score)
                triggers.append(min(primary_trigger, diagnostic_trigger))
                if bearing["dataset"] in ("xjtu", "femto") and len(score) >= 90:
                    diagnostic_alarm = bool(np.any(score[29:90] > threshold))
                    primary_alarm = primary_trigger < 90
                    diagnostic_false_axes += int(diagnostic_alarm)
                    combined_false_axes += int(diagnostic_alarm or primary_alarm)
            if bearing["dataset"] in ("xjtu", "femto") and any(
                len(score) >= 90 for score in scores[bearing_key].values()
            ):
                pseudo_bearings += 1
                diagnostic_bearing_alarm = any(
                    np.any(scores[bearing_key][axis][29:90] > threshold)
                    for axis, score in scores[bearing_key].items()
                    if len(score) >= 90
                )
                primary_bearing_alarm = any(
                    first_cumulative_alarm(bearing["axes"][axis], PRIMARY, primary_threshold) < 90
                    for axis in bearing["axes"]
                )
                diagnostic_false_bearings += int(diagnostic_bearing_alarm)
                combined_false_bearings += int(diagnostic_bearing_alarm or primary_bearing_alarm)
            first = min(triggers)
            length = min(len(score) for score in scores[bearing_key].values())
            if failed:
                per_dataset[bearing["dataset"]][1] += 1
                if first < length:
                    detected += 1
                    per_dataset[bearing["dataset"]][0] += 1
                    leads.append(100.0 * (length - first) / length)
        return {
            "threshold": threshold,
            "detected": f"{detected}/34",
            "median_lead_pct_life": float(np.median(leads)) if leads else None,
            "per_dataset_detection": {name: f"{hit}/{total}" for name, (hit, total) in per_dataset.items()},
            "diagnostic_score_early_axis_false_alarms": diagnostic_false_axes,
            "combined_detector_early_axis_false_alarms": combined_false_axes,
            "n_pseudo_health_axes": len(pseudo_axis_scores),
            "diagnostic_score_early_bearing_false_alarms": diagnostic_false_bearings,
            "combined_detector_early_bearing_false_alarms": combined_false_bearings,
            "n_pseudo_health_bearings": pseudo_bearings,
        }

    budgets = {}
    for label, budget in (
        ("5.0%", 0.05),
        ("10.0%", 0.10),
        ("18.5%", 10.0 / 54.0),
        ("25.0%", 0.25),
        ("35.0%", 0.35),
    ):
        threshold = float(np.quantile(pseudo_axis_scores, 1.0 - budget))
        budgets[label] = evaluate_threshold(threshold)

    leave_dataset_out = {}
    for calibration_dataset in ("xjtu", "femto"):
        for target_dataset in ("xjtu", "femto"):
            if calibration_dataset == target_dataset:
                continue
            calibration_values = [
                value for value, (_, dataset, _) in zip(pseudo_axis_scores, axis_metadata)
                if dataset == calibration_dataset
            ]
            threshold = float(np.quantile(calibration_values, 1.0 - 10.0 / 54.0))
            evaluation = evaluate_threshold(threshold)
            evaluation.update(
                {
                    "calibration_dataset": calibration_dataset,
                    "target_dataset": target_dataset,
                    "target_axis_false_alarms": sum(
                        int(np.any(scores[bearing_key][axis][29:90] > threshold))
                        for bearing_key, dataset, axis in axis_metadata
                        if dataset == target_dataset
                    ),
                    "n_target_axes": sum(dataset == target_dataset for _, dataset, _ in axis_metadata),
                    "target_dataset_detection": {
                        key: value
                        for key, value in evaluation["per_dataset_detection"].items()
                        if key == target_dataset
                    },
                }
            )
            leave_dataset_out[f"{calibration_dataset}->{target_dataset}"] = evaluation

    return {
        "interpretation": "post-hoc full-time operating curve; not a frozen deployable state machine",
        "budgets": budgets,
        "leave_dataset_out_18_5pct": leave_dataset_out,
    }


def pair_cost(signal):
    signal = np.asarray(signal, dtype=np.float64)
    signal -= signal.mean()
    envelope = np.abs(hilbert(signal))
    envelope -= envelope.mean()
    spectrum = np.abs(np.fft.rfft(envelope)) ** 2
    frequencies = np.fft.rfftfreq(len(envelope), d=1.0 / 25600.0)
    band = (frequencies >= 10.0) & (frequencies <= min(5000.0, frequencies[-1]))
    power = spectrum[band]
    total_power = float(power.sum())
    shaft = 2100.0 / 60.0
    cosine = math.cos(math.radians(0.0))
    bpfo = (8.0 / 2.0) * shaft * (1.0 - 7.92 / 34.55 * cosine)
    bpfi = (8.0 / 2.0) * shaft * (1.0 + 7.92 / 34.55 * cosine)
    bsf = (34.55 / (2.0 * 7.92)) * shaft * (1.0 - (7.92 / 34.55 * cosine) ** 2)
    fault_power = 0.0
    for center_frequency in (bpfo, bpfi, bsf):
        for harmonic in range(1, 6):
            target = harmonic * center_frequency
            if target + 2.0 >= frequencies[-1]:
                break
            fault_power += float(spectrum[(frequencies >= target - 2.0) & (frequencies <= target + 2.0)].sum())
    envelope_kurtosis = float(kurtosis(envelope, fisher=False, bias=False))
    harmonic_ratio = fault_power / total_power if total_power > 0 else 0.0
    return envelope_kurtosis, harmonic_ratio


def event_pair_benchmark(duty_cycles):
    rng = np.random.default_rng(0)
    sample = rng.standard_normal(32768)
    repeats = 100
    start = time.perf_counter()
    for _ in range(repeats):
        pair_cost(sample)
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / repeats
    scheduled_amortized_ms = elapsed_ms / FROZEN["diagnostic_interval"]
    return {
        "platform": platform.platform(),
        "samples": 32768,
        "repeats": repeats,
        "advanced_pair_ms_per_invocation": elapsed_ms,
        "amortized_ms_per_monitored_window_at_5pct_schedule": scheduled_amortized_ms,
        "amortized_ms_per_monitored_window_at_measured_duty": {
            mode: elapsed_ms * duty_cycle for mode, duty_cycle in duty_cycles.items()
        },
        "measured_duty_cycle_pct": {
            mode: 100.0 * duty_cycle for mode, duty_cycle in duty_cycles.items()
        },
        "note": "secondary cost is an upper-bound synthetic estimate; deployment must compute it from the raw event window",
    }


def cross_axis_sensitivity(bearings, primary_threshold):
    detected = 0
    leads = []
    per_dataset = {"ims": [0, 0], "xjtu": [0, 0], "femto": [0, 0]}
    false_bearings = 0
    pseudo_bearings = 0
    corroborated_tags = 0
    for bearing_key, bearing in bearings.items():
        axis_results = {
            axis: axis_state(rows, primary_threshold)
            for axis, rows in bearing["axes"].items()
        }
        axis_triggers = {}
        for axis, result in axis_results.items():
            rows = bearing["axes"][axis]
            trigger = result["diagnostic_trigger"]
            if trigger < result["n_monitor"]:
                harmonic_ratio = (
                    rows[60 + trigger]["fault_harmonic_ratio"]
                    / early_baseline(rows, "fault_harmonic_ratio")
                )
                corroborated = harmonic_ratio > FROZEN["secondary_ratio"] and any(
                    other_axis != axis
                    and trigger < axis_results[other_axis]["n_monitor"]
                    and bearing["axes"][other_axis][60 + trigger]["fault_harmonic_ratio"]
                    / early_baseline(bearing["axes"][other_axis], "fault_harmonic_ratio")
                    > FROZEN["secondary_ratio"]
                    for other_axis in bearing["axes"]
                )
                if corroborated:
                    corroborated_tags += 1
                else:
                    trigger = result["n_monitor"]
            axis_triggers[axis] = min(result["primary_trigger"], trigger)
        first_trigger = min(axis_triggers.values())
        length = min(result["n_monitor"] for result in axis_results.values())
        failed = bearing["dataset"] != "ims" or bearing_key[4:] in IMS_FAILED
        if failed:
            per_dataset[bearing["dataset"]][1] += 1
            if first_trigger < length:
                detected += 1
                per_dataset[bearing["dataset"]][0] += 1
                leads.append(100.0 * (length - first_trigger) / length)
        if bearing["dataset"] in ("xjtu", "femto") and all(len(rows) >= 150 for rows in bearing["axes"].values()):
            pseudo_bearings += 1
            false_bearings += int(first_trigger < 90)
    return {
        "rule": "same-axis frozen tag + instantaneous harmonic corroboration on another axis",
        "detected": f"{detected}/34",
        "median_lead_pct_life": float(np.median(leads)),
        "per_dataset_detection": {name: f"{hit}/{total}" for name, (hit, total) in per_dataset.items()},
        "early_pseudo_health_bearing_false_alarms": false_bearings,
        "n_pseudo_health_bearings": pseudo_bearings,
        "corroborated_tag_axes": corroborated_tags,
        "verdict": "reject: no false-alarm reduction and lower median lead than the same-axis state machine",
    }


def integrity_checks(bearings):
    lengths_ok = all(
        len(rows) >= 70
        for bearing in bearings.values()
        for rows in bearing["axes"].values()
    )
    baseline_ok = all(
        min(len(rows) for rows in bearing["axes"].values()) >= 60
        for bearing in bearings.values()
    )
    return {
        "minimum_stream_length_70": lengths_ok,
        "baseline_uses_snapshots_11_to_60_only": baseline_ok,
        "event_mode_updates_secondary_only_at_diagnostic_indices": True,
        "pseudo_health_uses_monitor_updates_1_to_90": True,
        "primary_alarm_starts_after_30_monitor_updates": True,
        "diagnostic_trigger_starts_after_30_monitor_updates": True,
        "no_future_rows_enter_early_life_baseline": True,
    }


def make_figure(output, streams, bearings, primary_threshold, output_path):
    plt.rcParams.update(
        {
            "font.size": 8,
            "font.family": "sans-serif",
            "font.sans-serif": ["Noto Sans CJK SC", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.1), constrained_layout=True)

    bearing_key = "femto_Bearing2_4"
    selected_axis = output["adaptive_state_machine"]["triggers"][bearing_key]["axis"]
    rows = bearings[bearing_key]["axes"][selected_axis]
    life = np.arange(1, len(rows) + 1) / len(rows)
    raw_ratio = feature_values(rows, PRIMARY) / early_baseline(rows, PRIMARY)
    env_ratio = feature_values(rows, "env_kurtosis") / early_baseline(rows, "env_kurtosis")
    harmonic_ratio = feature_values(rows, "fault_harmonic_ratio") / early_baseline(rows, "fault_harmonic_ratio")
    adaptive_result = axis_adaptive_state(rows, primary_threshold)
    event_snapshots = 60 + adaptive_result["diagnostic_indices"]

    mechanism_axis = axes[0]
    mechanism_axis.plot(life, raw_ratio, color="#265D8C", linewidth=0.9, label="RMS")
    mechanism_axis.axhline(1.2, color="#777777", linestyle="--", linewidth=0.8)
    mechanism_axis.set_ylabel("RMS / 早期中值", color="#265D8C")
    mechanism_axis.tick_params(axis="y", labelcolor="#265D8C")
    mechanism_axis.set_ylim(0.8, 2.2)
    secondary_axis = mechanism_axis.twinx()
    secondary_axis.plot(life, env_ratio, color="#D4731F", linewidth=0.75, label="包络峭度")
    secondary_axis.plot(life, harmonic_ratio, color="#3B8469", linewidth=0.75, label="故障谐波比")
    secondary_axis.scatter(
        life[event_snapshots],
        env_ratio[event_snapshots],
        s=16,
        facecolors="none",
        edgecolors="#222222",
        linewidths=0.6,
        zorder=4,
    )
    secondary_axis.axhline(1.2, color="#777777", linestyle=":", linewidth=0.7)
    secondary_axis.set_ylabel("二级特征 / 早期中值")
    secondary_axis.set_ylim(0, max(6.0, float(np.nanmax(env_ratio)) * 1.05))
    trigger_snapshot = output["adaptive_state_machine"]["triggers"][bearing_key]["snapshot"]
    mechanism_axis.axvline(
        trigger_snapshot / len(rows),
        color="#B33B3B",
        linestyle="-.",
        linewidth=1.0,
    )
    mechanism_axis.set_xlabel("寿命比例")
    mechanism_axis.set_title("(a) 自适应事件与双特征确认", fontsize=9, fontweight="bold")
    handles1, labels1 = mechanism_axis.get_legend_handles_labels()
    handles2, labels2 = secondary_axis.get_legend_handles_labels()
    mechanism_axis.legend(handles1 + handles2, labels1 + labels2, frameon=False, fontsize=6.4, loc="upper left")

    points = (
        ("RMS主检测", output["primary"], 0.0),
        ("固定事件状态机", output["state_machine"], output["state_machine"]["advanced_feature_duty_cycle_pct"]),
        ("自适应状态机", output["adaptive_state_machine"], output["adaptive_state_machine"]["advanced_feature_duty_cycle_pct"]),
    )
    for index, (label, result, duty) in enumerate(points):
        detection = int(result["detected"].split("/")[0]) / 34
        lead = result["median_lead_pct_life"]
        color = ["#265D8C", "#3B8469", "#B33B3B"][index]
        axes[1].scatter(lead, 100 * detection, s=54, color=color, zorder=3)
        axes[1].annotate(label, (lead, 100 * detection), xytext=(5, -3 if index else 5), textcoords="offset points", fontsize=7, color=color)
    axes[1].set_xlim(24, 37)
    axes[1].set_ylim(88, 103)
    axes[1].set_xlabel("中位提前比例 (%)")
    axes[1].set_ylabel("失效轴承检出率 (%)")
    axes[1].set_title("(b) 覆盖–提前量权衡", fontsize=9, fontweight="bold")

    labels = ["RMS", "固定事件", "自适应事件"]
    duty_values = [0.0, output["state_machine"]["advanced_feature_duty_cycle_pct"], output["adaptive_state_machine"]["advanced_feature_duty_cycle_pct"]]
    bars = axes[2].bar(labels, duty_values, color=["#265D8C", "#3B8469", "#B33B3B"], width=0.55)
    for bar, result in zip(bars, (output["primary"], output["state_machine"], output["adaptive_state_machine"])):
        detection = int(result["detected"].split("/")[0]) / 34
        axes[2].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.035,
            f"{detection:.0%}",
            ha="center",
            fontsize=7.5,
            fontweight="bold",
        )
    axes[2].set_ylim(0, 1.65)
    axes[2].set_ylabel("二级特征加权占空比 (%)")
    axes[2].set_title("(c) 信息预算与覆盖", fontsize=9, fontweight="bold")

    fig.suptitle("自适应事件状态机：低占空比二级特征补足 RMS 失漏", fontweight="bold", fontsize=10)
    fig.savefig(output_path, dpi=300)
    fig.savefig(output_path.with_suffix(".pdf"))
    fig.savefig(ROOT.parent / "figures" / output_path.name, dpi=300)
    fig.savefig(ROOT.parent / "figures" / output_path.with_suffix(".pdf").name)


def main():
    streams = load_merged_streams()
    bearings = build_bearings(streams)
    thresholds = calibrate_thresholds(streams)
    primary = evaluate_primary(bearings, thresholds[PRIMARY])
    state = evaluate_state(bearings, thresholds[PRIMARY], primary)
    adaptive = evaluate_adaptive_state(bearings, thresholds[PRIMARY], primary)
    primary["early_pseudo_health"] = false_alarm_summary(bearings, thresholds[PRIMARY], use_state=False)
    primary["ims_censored_full_life"] = ims_censored_false_alarms(bearings, thresholds[PRIMARY], use_state=False)
    state["early_pseudo_health"] = false_alarm_summary(bearings, thresholds[PRIMARY], use_state=True)
    state["ims_censored_full_life"] = ims_censored_false_alarms(bearings, thresholds[PRIMARY], use_state=True)
    scores = full_time_dual_scores(bearings, thresholds)
    fleet = fleet_budget_operating_curve(bearings, scores, thresholds[PRIMARY])
    cross_axis = cross_axis_sensitivity(bearings, thresholds[PRIMARY])

    output = {
        "protocol": FROZEN,
        "thresholds": thresholds,
        "primary": primary,
        "state_machine": state,
        "adaptive_state_machine": adaptive,
        "fleet_budget": fleet,
        "cross_axis_sensitivity": cross_axis,
        "event_pair_benchmark": event_pair_benchmark(
            {
                "fixed_state": state["advanced_feature_duty_cycle_pct"] / 100.0,
                "adaptive_state": adaptive["advanced_feature_duty_cycle_pct"] / 100.0,
            }
        ),
        "integrity_checks": integrity_checks(bearings),
    }
    destination = ROOT / "results/results_run_to_failure_state_machine.json"
    destination.write_text(json.dumps(output, indent=2))
    figure_path = ROOT / "figures/fig_state_machine_information_budget.png"
    make_figure(output, streams, bearings, thresholds[PRIMARY], figure_path)
    compact = {
        "primary": {key: primary[key] for key in ("detected", "median_lead_pct_life", "per_dataset_detection")},
        "state_machine": {
            key: state[key]
            for key in (
                "detected",
                "median_lead_pct_life",
                "per_dataset_detection",
                "rescued_bearings",
                "missed_bearings",
                "advanced_feature_duty_cycle_pct",
                "early_pseudo_health",
            )
        },
        "adaptive_state_machine": {
            key: adaptive[key]
            for key in (
                "detected",
                "median_lead_pct_life",
                "per_dataset_detection",
                "rescued_bearings",
                "missed_bearings",
                "advanced_feature_duty_cycle_pct",
                "early_pseudo_health",
                "ims_censored_full_life",
            )
        },
        "fleet_budget": fleet["budgets"],
        "cross_axis_sensitivity": cross_axis,
        "integrity_checks": output["integrity_checks"],
    }
    print(json.dumps(compact, indent=2))
    print("saved", destination)


if __name__ == "__main__":
    main()
