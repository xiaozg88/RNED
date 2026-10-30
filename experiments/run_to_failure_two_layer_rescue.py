#!/usr/bin/env python3
"""Evaluate an RMS-first, same-axis dual-confirmation rescue mechanism."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
IMS_FAILED = {"test1_b3", "test1_b4", "test2_b1", "test3_b3"}
PRIMARY = "raw_rms"
SECONDARY_FEATURES = ("env_kurtosis", "fault_harmonic_ratio")
CONFIRMATION_WINDOWS = (1, 3, 5, 10)
GRACE_WINDOWS = (0, 30, 60, 90)
FIGURE_BEAMS = ("Bearing1_5", "Bearing1_6", "Bearing2_4")


def load_merged_streams():
    streams = {}

    def add(key, row):
        snapshot = int(row.get("snap", row.get("snapshot")))
        streams.setdefault(key, {}).setdefault(snapshot, {}).update(row)

    for test_name, filename in (
        ("test1", "test1.csv"),
        ("test2", "test2.csv"),
        ("test3", "test3.csv"),
    ):
        with (ROOT / "data/features" / filename).open() as source:
            for index, row in enumerate(csv.DictReader(source), 1):
                key = ("ims", test_name, f"{test_name}_b{row['bearing']}", row["channel"])
                snapshot = len(streams.get(key, {})) + 1
                add(key, {"snap": snapshot, PRIMARY: float(row["rms"])})
    for filename, dataset in (
        ("xjtu_features.csv", "xjtu"),
        ("femto_features.csv", "femto"),
    ):
        with (ROOT / "data/features" / filename).open() as source:
            for row in csv.DictReader(source):
                key = (dataset, row["group"], row["bearing"], row["axis"])
                add(key, {"snap": int(row["snapshot"]), PRIMARY: float(row["rms"])})
    with (ROOT / "data/features/modern_signal_features.csv").open() as source:
        for row in csv.DictReader(source):
            key = (row["dataset"], row["group"], row["bearing"], row["axis"])
            values = {
                "snapshot": int(row["snapshot"]),
                **{name: float(row[name]) for name in SECONDARY_FEATURES},
            }
            add(key, values)
    return {
        key: [rows[snapshot] for snapshot in sorted(rows)]
        for key, rows in streams.items()
    }


def normalized_rate(rows, feature, threshold):
    values = np.asarray([float(row[feature]) for row in rows], dtype=np.float64)
    baseline = float(np.median(values[10:60]))
    excess = np.maximum(values[60:] / baseline - 1.2, 0.0)
    rate = np.cumsum(excess) / np.arange(1, len(excess) + 1)
    return rate / threshold


def first_index(scores, threshold=1.0, min_window=30, grace=0, confirmation=1):
    if confirmation == 1:
        start = max(0, min_window - 1)
        hits = np.flatnonzero(scores[start:] > threshold) + start
        hits = hits[hits >= grace]
        return int(hits[0]) if hits.size else len(scores)
    exceed = scores > threshold
    run_length = np.zeros(len(scores), dtype=np.int64)
    run_length[0] = int(exceed[0])
    for index in range(1, len(scores)):
        run_length[index] = run_length[index - 1] + 1 if exceed[index] else 0
    eligible = np.flatnonzero((run_length >= confirmation) & (np.arange(len(scores)) >= max(min_window - 1, grace)))
    return int(eligible[0]) if eligible.size else len(scores)


def evaluate(streams, thresholds):
    bearings = {}
    for key, rows in streams.items():
        dataset, group, bearing, axis = key
        if len(rows) < 70:
            continue
        bearing_key = f"{dataset}_{bearing}"
        bearings.setdefault(
            bearing_key,
            {"dataset": dataset, "group": group, "axes": {}},
        )["axes"][axis] = rows

    primary_thresholds = {
        feature: 1.01
        * max(
            float(np.max(normalized_rate(rows, feature, 1.0)))
            for key, rows in streams.items()
            if key[0] == "ims" and key[2] not in IMS_FAILED
        )
        for feature in (PRIMARY, *SECONDARY_FEATURES)
    }
    # normalized_rate divides by the supplied threshold; use calibrated raw-rate maxima.
    for feature, threshold in primary_thresholds.items():
        primary_thresholds[feature] = threshold

    rates = {}
    for bearing_key, bearing in bearings.items():
        rates[bearing_key] = {}
        for axis, rows in bearing["axes"].items():
            rates[bearing_key][axis] = {
                PRIMARY: normalized_rate(rows, PRIMARY, thresholds[PRIMARY]),
                **{
                    feature: normalized_rate(rows, feature, thresholds[feature])
                    for feature in SECONDARY_FEATURES
                },
            }

    output = {"primary": {}, "rules": {}}
    primary_detected = set()
    primary_leads = []
    per_dataset_primary = {"ims": [0, 0], "xjtu": [0, 0], "femto": [0, 0]}
    for bearing_key, bearing in bearings.items():
        dataset = bearing["dataset"]
        failed = bearing_key[4:] in IMS_FAILED if dataset == "ims" else True
        if not failed:
            continue
        per_dataset_primary[dataset][1] += 1
        first = min(first_index(rates[bearing_key][axis][PRIMARY]) for axis in bearing["axes"])
        length = len(next(iter(bearing["axes"].values()))) - 60
        if first < length:
            primary_detected.add(bearing_key)
            primary_leads.append(100.0 * (length - first) / length)
            per_dataset_primary[dataset][0] += 1
    output["primary"] = {
        "detected": f"{len(primary_detected)}/34",
        "median_lead_pct_life": float(np.median(primary_leads)),
        "per_dataset_detection": {name: f"{hit}/{total}" for name, (hit, total) in per_dataset_primary.items()},
    }
    baseline_proxy = None

    for grace in GRACE_WINDOWS:
        for confirmation in CONFIRMATION_WINDOWS:
            detected = set(primary_detected)
            leads = list(primary_leads)
            per_dataset = {name: list(value) for name, value in per_dataset_primary.items()}
            rescued = []
            for bearing_key, bearing in bearings.items():
                dataset = bearing["dataset"]
                failed = bearing_key[4:] in IMS_FAILED if dataset == "ims" else True
                if not failed:
                    continue
                primary_first = min(first_index(rates[bearing_key][axis][PRIMARY]) for axis in bearing["axes"])
                length = len(next(iter(bearing["axes"].values()))) - 60
                axis_alarms = []
                for axis in bearing["axes"]:
                    primary_axis = first_index(rates[bearing_key][axis][PRIMARY])
                    secondary_axis = min(
                        first_index(
                            np.minimum(
                                rates[bearing_key][axis][SECONDARY_FEATURES[0]],
                                rates[bearing_key][axis][SECONDARY_FEATURES[1]],
                            ),
                            min_window=30,
                            grace=grace,
                            confirmation=confirmation,
                        ),
                        len(rates[bearing_key][axis][PRIMARY]),
                    )
                    selected_alarm = primary_axis if primary_axis <= secondary_axis else secondary_axis
                    axis_alarms.append((selected_alarm, axis, primary_axis, secondary_axis))
                first_alarm = min(alarm[0] for alarm in axis_alarms)
                if first_alarm < length:
                    if bearing_key not in detected:
                        rescued.append(bearing_key)
                        detected.add(bearing_key)
                        leads.append(100.0 * (length - first_alarm) / length)
                        per_dataset[dataset][0] += 1
                        rules = output.setdefault("rescue_details", {}).setdefault(
                            f"grace={grace},confirm={confirmation}",
                            [],
                        )
                        selected_alarm = min(axis_alarms, key=lambda alarm: alarm[0])
                        rules.append(
                            {
                                "bearing": bearing_key,
                                "axis": selected_alarm[1],
                                "monitor_index": int(first_alarm),
                                "snapshot": int(first_alarm + 60),
                                "lead_pct_life": 100.0 * (length - first_alarm) / length,
                            }
                        )

            false_axes = 0
            pseudo_axes = 0
            false_bearings = 0
            pseudo_bearings = 0
            for bearing_key, bearing in bearings.items():
                if bearing["dataset"] not in ("xjtu", "femto"):
                    continue
                bearing_alarm = False
                for axis, rows in bearing["axes"].items():
                    if len(rows) < 150:
                        continue
                    pseudo_axes += 1
                    primary_axis = first_index(rates[bearing_key][axis][PRIMARY])
                    secondary_axis = first_index(
                        np.minimum(
                            rates[bearing_key][axis][SECONDARY_FEATURES[0]],
                            rates[bearing_key][axis][SECONDARY_FEATURES[1]],
                        ),
                        min_window=30,
                        grace=grace,
                        confirmation=confirmation,
                    )
                    axis_alarm = min(primary_axis, secondary_axis) < 90
                    false_axes += int(axis_alarm)
                    bearing_alarm = bearing_alarm or axis_alarm
                if any(len(rows) >= 150 for rows in bearing["axes"].values()):
                    pseudo_bearings += 1
                    false_bearings += int(bearing_alarm)

            rule_name = f"grace={grace},confirm={confirmation}"
            output["rules"][rule_name] = {
                "detected": f"{len(detected)}/34",
                "rescued": rescued,
                "median_lead_pct_life": float(np.median(leads)),
                "per_dataset_detection": {name: f"{hit}/{total}" for name, (hit, total) in per_dataset.items()},
                "axis_false_alarm_rate_pct": 100.0 * false_axes / pseudo_axes,
                "n_false_alarm_axes": false_axes,
                "n_pseudo_health_axes": pseudo_axes,
                "bearing_false_alarm_rate_pct": 100.0 * false_bearings / pseudo_bearings,
                "n_false_alarm_bearings": false_bearings,
                "n_pseudo_health_bearings": pseudo_bearings,
            }
            if grace == 90 and confirmation == 1:
                baseline_proxy = output["rules"][rule_name]
    output["primary"].update(
        {
            "axis_false_alarm_rate_pct": baseline_proxy["axis_false_alarm_rate_pct"],
            "n_false_alarm_axes": baseline_proxy["n_false_alarm_axes"],
            "n_pseudo_health_axes": baseline_proxy["n_pseudo_health_axes"],
            "bearing_false_alarm_rate_pct": baseline_proxy["bearing_false_alarm_rate_pct"],
            "n_false_alarm_bearings": baseline_proxy["n_false_alarm_bearings"],
            "n_pseudo_health_bearings": baseline_proxy["n_pseudo_health_bearings"],
        }
    )
    return output, rates, bearings


def make_figure(rates, bearings, output_path):
    plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.alpha": 0.25})
    fig, axes = plt.subplots(
        len(FIGURE_BEAMS),
        3,
        figsize=(7.1, 5.6),
        sharex=True,
        constrained_layout=True,
    )
    panels = [
        (PRIMARY, "(a) RMS rate"),
        (SECONDARY_FEATURES[0], "(b) Envelope-kurtosis rate"),
        (SECONDARY_FEATURES[1], "(c) Fault-harmonic rate"),
    ]
    selected_axes = {"Bearing1_5": "V", "Bearing1_6": "H", "Bearing2_4": "H"}
    colors = ["#265D8C", "#D4731F", "#3B8469"]
    for row, beam in enumerate(FIGURE_BEAMS):
        bearing_key = f"femto_{beam}"
        axis = selected_axes[beam]
        rows = bearings[bearing_key]["axes"][axis]
        life_fraction = np.arange(1, len(rows) + 1) / len(rows)
        for column, (feature, label) in enumerate(panels):
            axis_handle = axes[row, column]
            score = np.r_[np.full(60, np.nan), rates[bearing_key][axis][feature]]
            axis_handle.plot(life_fraction, score, color=colors[column], linewidth=1.2)
            axis_handle.axhline(1.0, color="#B33B3B", linestyle="--", linewidth=0.9)
            axis_handle.axvline(60 / len(rows), color="#777777", linestyle=":", linewidth=0.7)
            if column > 0:
                confirmation = np.minimum(
                    rates[bearing_key][axis][SECONDARY_FEATURES[0]],
                    rates[bearing_key][axis][SECONDARY_FEATURES[1]],
                )
                trigger = first_index(confirmation, confirmation=10)
                if trigger < len(confirmation):
                    x_value = (trigger + 60) / len(rows)
                    y_value = confirmation[trigger]
                    axis_handle.scatter(
                        [x_value],
                        [y_value],
                        s=16,
                        color="#B33B3B",
                        zorder=4,
                    )
            axis_handle.set_xlim(0, 1)
            axis_handle.set_ylim(-0.05, max(2.0, float(np.nanmax(score)) * 1.05))
            if row == 0:
                axis_handle.set_title(label, fontsize=8.5)
                legend_handles = [
                    Line2D([0], [0], color="#777777", linestyle=":", linewidth=0.8, label="online start"),
                    Line2D([0], [0], color="#B33B3B", linestyle="--", linewidth=0.8, label="threshold"),
                    Line2D([0], [0], marker="o", color="none", markerfacecolor="#B33B3B", markersize=3.5, label="dual confirmation"),
                ]
                axis_handle.legend(handles=legend_handles, loc="upper left", fontsize=5.8, frameon=False)
            if column == 0:
                axis_handle.set_ylabel(f"{beam}\nscore / threshold")
            if row == len(FIGURE_BEAMS) - 1:
                axis_handle.set_xlabel("Life fraction")
    fig.suptitle("RMS-missed FEMTO bearings show complementary impulsivity and harmonic evidence", fontsize=10, fontweight="bold")
    fig.savefig(output_path, dpi=300)
    fig.savefig(output_path.with_suffix(".pdf"))
    fig.savefig(ROOT.parent / "figures" / output_path.name, dpi=300)
    fig.savefig(ROOT.parent / "figures" / output_path.with_suffix(".pdf").name)


def make_tradeoff_figure(output, output_path):
    plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.alpha": 0.25})
    fig, axis = plt.subplots(figsize=(4.6, 3.3), constrained_layout=True)
    points = [
        ("RMS primary", 18.5185, 31 / 34, 28.82, "#265D8C", "o"),
        ("Envelope kurtosis", 25.9259, 19 / 34, 66.47, "#D4731F", "s"),
        ("Fault harmonic", 90.7407, 30 / 34, 95.64, "#3B8469", "^"),
        ("Immediate rescue", 29.6296, 1.0, 32.65, "#B33B3B", "D"),
    ]
    for label, false_alarm, detection, lead, color, marker in points:
        axis.scatter(false_alarm, 100 * detection, s=28 + 2.2 * lead, color=color, marker=marker, zorder=3)
        axis.annotate(label, (false_alarm, 100 * detection), xytext=(4, 4), textcoords="offset points", fontsize=7)
    delayed = output["rules"]["grace=90,confirm=10"]
    axis.scatter(
        delayed["axis_false_alarm_rate_pct"],
        100 * int(delayed["detected"].split("/")[0]) / 34,
        s=100,
        facecolors="none",
        edgecolors="#B33B3B",
        linewidths=1.2,
        zorder=4,
    )
    axis.annotate(
        "Delayed rescue tag\n(early-window FA only)",
        (
            delayed["axis_false_alarm_rate_pct"],
            100 * int(delayed["detected"].split("/")[0]) / 34,
        ),
        xytext=(-96, -6),
        textcoords="offset points",
        fontsize=7,
    )
    axis.set_xlim(0, 100)
    axis.set_ylim(45, 106)
    axis.set_xlabel("Early pseudo-health axis false alarms (%)")
    axis.set_ylabel("Failed-bearing detection (%)")
    axis.set_title("Two-layer rescue trade-off", fontsize=9, fontweight="bold")
    fig.savefig(output_path, dpi=300)
    fig.savefig(output_path.with_suffix(".pdf"))
    fig.savefig(ROOT.parent / "figures" / output_path.name, dpi=300)
    fig.savefig(ROOT.parent / "figures" / output_path.with_suffix(".pdf").name)


def main():
    streams = load_merged_streams()
    thresholds = {}
    for feature in (PRIMARY, *SECONDARY_FEATURES):
        maxima = []
        for key, rows in streams.items():
            if key[0] == "ims" and key[2] not in IMS_FAILED:
                values = np.asarray([float(row[feature]) for row in rows])
                baseline = float(np.median(values[10:60]))
                excess = np.maximum(values[60:] / baseline - 1.2, 0.0)
                maxima.append(float(np.max(np.cumsum(excess) / np.arange(1, len(excess) + 1))))
        thresholds[feature] = 1.01 * max(maxima)
    output, rates, bearings = evaluate(streams, thresholds)
    result_path = ROOT / "results/results_run_to_failure_two_layer_rescue.json"
    result_path.write_text(json.dumps({"thresholds": thresholds, **output}, indent=2))
    make_figure(rates, bearings, ROOT / "figures/fig_rms_missed_secondary_mechanism.png")
    make_tradeoff_figure(output, ROOT / "figures/fig_two_layer_rescue_tradeoff.png")
    print(json.dumps({"thresholds": thresholds, **output}, indent=2))


if __name__ == "__main__":
    main()
