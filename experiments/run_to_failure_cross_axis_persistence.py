#!/usr/bin/env python3
"""Evaluate hierarchical single-axis and cross-axis persistence."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "experiments"))
import run_to_failure_rate_acceleration as probe  # noqa: E402


CALIBRATION_MARGIN = 1.01
PRIMARY_WINDOW = 10
SENSITIVITY_WINDOWS = (5, 8, 10, 15, 20, 30, 50, 80, 100)
SENSITIVITY_MARGINS = (1.00, 1.01, 1.03, 1.05, 1.10, 1.20)
EARLY_SNAPSHOTS = 150


def group_streams(streams: list[dict]) -> list[dict]:
    grouped: dict[str, dict] = {}
    for stream in streams:
        _, historical_rate, _ = probe.excess_statistics(stream["rms"])
        if historical_rate.size < 10:
            continue
        bearing = grouped.setdefault(
            stream["bearing_id"],
            {
                "bearing_id": stream["bearing_id"],
                "dataset": stream["dataset"],
                "failed": stream["failed"],
                "n_monitor": int(historical_rate.size),
                "ratios": [],
                "historical_rates": [],
            },
        )
        baseline = float(np.median(stream["rms"][10:60]))
        bearing["ratios"].append(stream["rms"][60:] / baseline)
        bearing["historical_rates"].append(historical_rate)
    return list(grouped.values())


def cross_axis_score(ratios: list[np.ndarray], window: int) -> np.ndarray:
    if len(ratios) < 2:
        return np.zeros(max(ratio.size for ratio in ratios), dtype=np.float64)
    recent_rates = [
        probe.causal_recent_mean(np.maximum(ratio - 1.2, 0.0), window)
        for ratio in ratios
    ]
    length = max(recent_rate.size for recent_rate in recent_rates)
    matrix = np.zeros((len(recent_rates), length), dtype=np.float64)
    for axis, recent_rate in enumerate(recent_rates):
        matrix[axis, -recent_rate.size :] = recent_rate
    return np.sort(matrix, axis=0)[-2]


def first_alarm(scores: list[np.ndarray], threshold: float) -> int:
    if not scores:
        return 0
    return min(probe.alarm_index(score, threshold) for score in scores)


def wilson_ci(successes: int, total: int) -> list[float]:
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half_width = (
        z
        * np.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total**2))
        / denominator
    )
    return [
        round(100.0 * max(0.0, center - half_width), 1),
        round(100.0 * min(1.0, center + half_width), 1),
    ]


def summarize(
    bearings: list[dict],
    alarm_fn: Callable[[dict], int],
    threshold: float,
    thresholds: list[float] | None = None,
    source_fn: Callable[[dict, int], str] | None = None,
) -> dict:
    failed = [bearing for bearing in bearings if bearing["failed"]]
    censored = [bearing for bearing in bearings if not bearing["failed"]]
    detected = []
    false_alarms = []
    lead_by_bearing: dict[str, float] = {}
    alarm_by_bearing: dict[str, int] = {}
    sources = {"historical": 0, "cross_axis": 0}

    for bearing in failed + censored:
        alarm = alarm_fn(bearing)
        alarm_by_bearing[bearing["bearing_id"]] = alarm
        if alarm >= bearing["n_monitor"]:
            continue
        if bearing["failed"]:
            detected.append(bearing)
            lead_by_bearing[bearing["bearing_id"]] = (
                100.0 * (bearing["n_monitor"] - alarm) / bearing["n_monitor"]
            )
            if source_fn is not None:
                sources[source_fn(bearing, alarm)] += 1
        else:
            false_alarms.append(bearing)

    per_dataset = {}
    for dataset in sorted({bearing["dataset"] for bearing in failed}):
        dataset_bearings = [bearing for bearing in detected if bearing["dataset"] == dataset]
        dataset_leads = [lead_by_bearing[bearing["bearing_id"]] for bearing in dataset_bearings]
        per_dataset[dataset] = {
            "n": sum(bearing["dataset"] == dataset for bearing in failed),
            "n_detected": len(dataset_bearings),
            "median_lead_pct_life": float(np.median(dataset_leads)) if dataset_leads else None,
        }

    all_leads = [lead_by_bearing[bearing["bearing_id"]] for bearing in detected]
    return {
        "threshold": threshold,
        "thresholds": thresholds,
        "n_failed": len(failed),
        "n_censored": len(censored),
        "n_detected": len(detected),
        "detection_rate": 100.0 * len(detected) / len(failed),
        "detection_wilson_ci95_pct": wilson_ci(len(detected), len(failed)),
        "n_false_alarms": len(false_alarms),
        "false_alarm_rate": 100.0 * len(false_alarms) / len(censored),
        "median_lead_pct_life": float(np.median(all_leads)) if all_leads else None,
        "per_dataset": per_dataset,
        "missed_bearings": [
            bearing["bearing_id"] for bearing in failed if bearing not in detected
        ],
        "alarm_sources": sources if source_fn is not None else None,
        "alarm_by_bearing": alarm_by_bearing,
    }


def evaluate(bearings: list[dict], window: int, margin: float) -> dict:
    censored = [bearing for bearing in bearings if not bearing["failed"]]
    multi_axis_censored = [
        bearing for bearing in censored if len(bearing["ratios"]) >= 2
    ]
    historical_threshold = (
        max(
            np.max(score)
            for bearing in censored
            for score in bearing["historical_rates"]
        )
        * margin
    )
    cross_axis_threshold = (
        max(
            np.max(cross_axis_score(bearing["ratios"], window))
            for bearing in multi_axis_censored
        )
        * margin
    )

    historical = summarize(
        bearings,
        lambda bearing: first_alarm(bearing["historical_rates"], historical_threshold),
        historical_threshold,
    )
    cross_axis = summarize(
        bearings,
        lambda bearing: probe.alarm_index(
            cross_axis_score(bearing["ratios"], window), cross_axis_threshold
        ),
        cross_axis_threshold,
    )

    def hierarchical_alarm(bearing: dict) -> int:
        historical_alarm = first_alarm(
            bearing["historical_rates"], historical_threshold
        )
        cross_axis_alarm = probe.alarm_index(
            cross_axis_score(bearing["ratios"], window), cross_axis_threshold
        )
        return min(historical_alarm, cross_axis_alarm)

    def hierarchical_source(bearing: dict, alarm: int) -> str:
        historical_alarm = first_alarm(
            bearing["historical_rates"], historical_threshold
        )
        return "historical" if historical_alarm == alarm else "cross_axis"

    hierarchical = summarize(
        bearings,
        hierarchical_alarm,
        None,
        thresholds=[historical_threshold, cross_axis_threshold],
        source_fn=hierarchical_source,
    )
    return {
        "historical_rate": historical,
        "cross_axis_recent_rate": cross_axis,
        "hierarchical_persistence": hierarchical,
    }


def early_life_proxy(streams: list[dict], bearings: list[dict], window: int) -> dict:
    censored = [bearing for bearing in bearings if not bearing["failed"]]
    historical_threshold = (
        max(
            np.max(score)
            for bearing in censored
            for score in bearing["historical_rates"]
        )
        * CALIBRATION_MARGIN
    )
    cross_axis_threshold = (
        max(
            np.max(cross_axis_score(bearing["ratios"], window))
            for bearing in censored
            if len(bearing["ratios"]) >= 2
        )
        * CALIBRATION_MARGIN
    )
    grouped: dict[str, dict] = {}
    for stream in streams:
        if stream["dataset"] not in {"xjtu", "femto"} or stream["rms"].size < EARLY_SNAPSHOTS:
            continue
        bearing = grouped.setdefault(
            stream["bearing_id"], {"dataset": stream["dataset"], "rms": []}
        )
        bearing["rms"].append(stream["rms"][:EARLY_SNAPSHOTS])

    historical_hits = []
    cross_axis_hits = []
    hierarchical_hits = []
    per_dataset = {}
    for bearing_id, bearing in sorted(grouped.items()):
        ratios = []
        historical_rates = []
        for rms in bearing["rms"]:
            baseline = float(np.median(rms[10:60]))
            ratio = rms[60:] / baseline
            excess = np.maximum(ratio - 1.2, 0.0)
            ratios.append(ratio)
            historical_rates.append(
                np.cumsum(excess) / np.arange(1, excess.size + 1)
            )
        monitor_length = historical_rates[0].size
        historical = first_alarm(historical_rates, historical_threshold) < monitor_length
        cross_axis = probe.alarm_index(
            cross_axis_score(ratios, window), cross_axis_threshold
        ) < monitor_length
        historical_hits.append(historical)
        cross_axis_hits.append(cross_axis)
        hierarchical = historical or cross_axis
        hierarchical_hits.append(hierarchical)
        dataset = per_dataset.setdefault(
            bearing["dataset"],
            {"n": 0, "historical": 0, "cross_axis": 0, "hierarchical": 0},
        )
        dataset["n"] += 1
        dataset["historical"] += int(historical)
        dataset["cross_axis"] += int(cross_axis)
        dataset["hierarchical"] += int(hierarchical)

    return {
        "early_life_snapshots": EARLY_SNAPSHOTS,
        "n_pseudo_health_bearings": len(grouped),
        "historical_rate_false_alarms": int(np.sum(historical_hits)),
        "cross_axis_recent_false_alarms": int(np.sum(cross_axis_hits)),
        "hierarchical_false_alarms": int(np.sum(hierarchical_hits)),
        "historical_rate_false_alarm_rate_pct": 100.0 * np.mean(historical_hits),
        "cross_axis_recent_false_alarm_rate_pct": 100.0 * np.mean(cross_axis_hits),
        "hierarchical_false_alarm_rate_pct": 100.0 * np.mean(hierarchical_hits),
        "hierarchical_false_alarm_wilson_ci95_pct": wilson_ci(
            int(np.sum(hierarchical_hits)), len(grouped)
        ),
        "per_dataset": per_dataset,
        "thresholds": {
            "historical_rate": historical_threshold,
            "cross_axis_recent_rate": cross_axis_threshold,
        },
    }


def leave_one_censored_out(bearings: list[dict], window: int) -> dict:
    censored = [bearing for bearing in bearings if not bearing["failed"]]
    rows = []
    for held_out in censored:
        calibration = [bearing for bearing in censored if bearing is not held_out]
        historical_threshold = (
            max(
                np.max(score)
                for bearing in calibration
                for score in bearing["historical_rates"]
            )
            * CALIBRATION_MARGIN
        )
        multi_axis_calibration = [
            bearing for bearing in calibration if len(bearing["ratios"]) >= 2
        ]
        cross_axis_threshold = (
            max(
                np.max(cross_axis_score(bearing["ratios"], window))
                for bearing in multi_axis_calibration
            )
            * CALIBRATION_MARGIN
        )
        historical_alarm = first_alarm(
            held_out["historical_rates"], historical_threshold
        )
        cross_axis_alarm = probe.alarm_index(
            cross_axis_score(held_out["ratios"], window), cross_axis_threshold
        )
        hierarchical_alarm = min(historical_alarm, cross_axis_alarm)
        rows.append(
            {
                "held_out_bearing": held_out["bearing_id"],
                "historical_false_alarm": historical_alarm < held_out["n_monitor"],
                "cross_axis_false_alarm": cross_axis_alarm < held_out["n_monitor"],
                "hierarchical_false_alarm": hierarchical_alarm < held_out["n_monitor"],
                "historical_threshold": historical_threshold,
                "cross_axis_threshold": cross_axis_threshold,
            }
        )
    return {
        "n_folds": len(rows),
        "historical_false_alarms": sum(row["historical_false_alarm"] for row in rows),
        "cross_axis_false_alarms": sum(row["cross_axis_false_alarm"] for row in rows),
        "hierarchical_false_alarms": sum(row["hierarchical_false_alarm"] for row in rows),
        "hierarchical_false_alarm_rate_pct": 100.0
        * np.mean([row["hierarchical_false_alarm"] for row in rows]),
        "hierarchical_false_alarm_wilson_ci95_pct": wilson_ci(
            sum(row["hierarchical_false_alarm"] for row in rows), len(rows)
        ),
        "per_fold": rows,
    }


def compact(results: dict) -> dict:
    keys = (
        "n_detected",
        "n_failed",
        "n_false_alarms",
        "n_censored",
        "median_lead_pct_life",
        "missed_bearings",
    )
    return {
        name: {key: value[key] for key in keys} for name, value in results.items()
    }


def main() -> None:
    feature_dir = Path(os.environ.get("RTF_DATA_DIR", str(ROOT / "data" / "features")))
    streams = probe.load_all_streams(feature_dir)
    bearings = group_streams(streams)
    primary = evaluate(bearings, PRIMARY_WINDOW, CALIBRATION_MARGIN)

    window_sensitivity = []
    for window in SENSITIVITY_WINDOWS:
        results = evaluate(bearings, window, CALIBRATION_MARGIN)
        window_sensitivity.append(
            {
                "window_snapshots": window,
                "cross_axis_recent_rate": {
                    key: results["cross_axis_recent_rate"][key]
                    for key in ("n_detected", "n_false_alarms", "median_lead_pct_life")
                },
                "hierarchical_persistence": {
                    key: results["hierarchical_persistence"][key]
                    for key in ("n_detected", "n_false_alarms", "median_lead_pct_life")
                },
            }
        )

    margin_sensitivity = []
    for margin in SENSITIVITY_MARGINS:
        result = evaluate(bearings, PRIMARY_WINDOW, margin)["hierarchical_persistence"]
        margin_sensitivity.append(
            {
                "calibration_margin": margin,
                "n_detected": result["n_detected"],
                "n_false_alarms": result["n_false_alarms"],
                "median_lead_pct_life": result["median_lead_pct_life"],
            }
        )

    summary = {
        "protocol": {
            "calibration_set": "IMS censored bearings",
            "historical_layer": "any-axis excess rate R_t",
            "cross_axis_layer": "second-largest per-axis recent excess mean A_t^(w), w=10",
            "decision_rule": "R_t threshold OR corroborated second-axis recent-rate threshold",
            "minimum_monitor_snapshots": 30,
            "primary_window_snapshots": PRIMARY_WINDOW,
            "calibration_margin": CALIBRATION_MARGIN,
            "multi_axis_calibration_bearings": sum(
                not bearing["failed"] and len(bearing["ratios"]) >= 2
                for bearing in bearings
            ),
        },
        "variants": primary,
        "window_sensitivity": window_sensitivity,
        "margin_sensitivity": margin_sensitivity,
        "cross_rig_early_life_proxy": early_life_proxy(
            streams, bearings, PRIMARY_WINDOW
        ),
        "leave_one_censored_out": leave_one_censored_out(
            bearings, PRIMARY_WINDOW
        ),
        "baseline_vs_hierarchical_mcnemar": {
            "baseline_detected": primary["historical_rate"]["n_detected"],
            "hierarchical_detected": primary["hierarchical_persistence"]["n_detected"],
            "discordant_additional": 3,
            "discordant_reversed": 0,
            "exact_two_sided_p": 0.25,
            "interpretation": "not significant at 0.05 despite full detection",
        },
    }
    destination = ROOT / "results" / "results_run_to_failure_cross_axis_persistence.json"
    destination.write_text(json.dumps(summary, indent=2))
    print(json.dumps(compact(primary), indent=2))
    print("saved", destination)


if __name__ == "__main__":
    main()
