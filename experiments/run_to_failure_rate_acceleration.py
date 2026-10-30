#!/usr/bin/env python3
"""Evaluate a dual-scale excess-rate activation rule.

The paper detector uses the time-averaged excess over the complete monitoring
history. This experiment adds a causal short-window excess rate and asks
whether recent activation complements, rather than replaces, persistence. All
thresholds are calibrated independently on IMS censored bearings under the
same 60-snapshot baseline, 1.2 ratio, and 30-snapshot minimum monitor as the
headline protocol.
"""
from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
IMS_FAILED = {("test1", 3), ("test1", 4), ("test2", 1), ("test3", 3)}
SECONDS_PER_SNAPSHOT = {"ims": 600.0, "xjtu": 60.0, "femto": 10.0}
CALIBRATION_MARGIN = 1.01
PRIMARY_WINDOW = 3
SENSITIVITY_WINDOWS = (2, 3, 5, 10, 20)


def load_all_streams(feature_dir: Path) -> list[dict]:
    streams: list[dict] = []
    for test_name in ("test1", "test2", "test3"):
        with (feature_dir / f"{test_name}.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        grouped: dict[tuple[int, int], list[tuple[datetime, float]]] = {}
        for row in rows:
            timestamp = datetime.fromisoformat(row["timestamp"])
            grouped.setdefault((int(row["bearing"]), int(row["channel"])), []).append(
                (timestamp, float(row["rms"]))
            )
        for (bearing, channel), values in sorted(grouped.items()):
            values.sort(key=lambda item: item[0])
            streams.append(
                {
                    "dataset": "ims",
                    "bearing_id": f"ims_{test_name}_b{bearing}",
                    "axis": channel,
                    "failed": (test_name, bearing) in IMS_FAILED,
                    "rms": np.asarray([value for _, value in values], dtype=np.float64),
                }
            )

    for filename, dataset in (
        ("xjtu_features.csv", "xjtu"),
        ("femto_features.csv", "femto"),
    ):
        with (feature_dir / filename).open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        grouped: dict[tuple[str, str, str], list[float]] = {}
        for row in rows:
            grouped.setdefault((row["group"], row["bearing"], row["axis"]), []).append(
                float(row["rms"])
            )
        for (group, bearing, axis), values in sorted(grouped.items()):
            streams.append(
                {
                    "dataset": dataset,
                    "bearing_id": f"{dataset}_{bearing}",
                    "axis": axis,
                    "failed": True,
                    "rms": np.asarray(values, dtype=np.float64),
                }
            )
    return streams


def excess_statistics(
    rms: np.ndarray, recent_window: int = PRIMARY_WINDOW
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    baseline = float(np.median(rms[10:60]))
    excess = np.maximum(rms[60:] / baseline - 1.2, 0.0)
    historical_rate = np.cumsum(excess) / np.arange(1, excess.size + 1)
    recent_rate = causal_recent_mean(excess, recent_window)
    return excess, historical_rate, recent_rate


def causal_recent_mean(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 0:
        raise ValueError("window must be positive")
    cumulative = np.cumsum(values)
    indices = np.arange(1, values.size + 1)
    expanding = cumulative / indices
    if values.size <= window:
        return expanding
    trailing = (cumulative[window - 1 :] - np.concatenate(([0.0], cumulative[:-window]))) / window
    return np.concatenate((expanding[: window - 1], trailing))


def alarm_index(score: np.ndarray, threshold: float = 1.0, min_window: int = 30) -> int:
    start = max(0, min_window - 1)
    hits = np.flatnonzero(score[start:] > threshold) + start
    return int(hits[0]) if hits.size else int(score.size)


def group_by_bearing(streams: list[dict], recent_window: int) -> list[dict]:
    grouped: dict[str, dict] = {}
    for stream in streams:
        _, historical_rate, recent_rate = excess_statistics(stream["rms"], recent_window)
        if historical_rate.size < 10:
            continue
        bearing = grouped.setdefault(
            stream["bearing_id"],
            {
                "bearing_id": stream["bearing_id"],
                "dataset": stream["dataset"],
                "failed": stream["failed"],
                "n_monitor": int(historical_rate.size),
                "historical_rates": [],
                "recent_rates": [],
            },
        )
        bearing["historical_rates"].append(historical_rate)
        bearing["recent_rates"].append(recent_rate)
    return list(grouped.values())


def evaluate_variant(
    bearings: list[dict],
    score_fn: Callable[[dict], list[np.ndarray]],
    calibration_values: list[np.ndarray],
) -> dict:
    threshold = float(max(np.max(values) for values in calibration_values) * CALIBRATION_MARGIN)
    failed = [bearing for bearing in bearings if bearing["failed"]]
    censored = [bearing for bearing in bearings if not bearing["failed"]]

    per_dataset: dict[str, dict] = {}
    all_leads: list[float] = []
    detected = 0
    false_alarms = 0
    missed_bearings: list[str] = []
    for bearing in failed + censored:
        first_alarm = min(alarm_index(score, threshold) for score in score_fn(bearing))
        hit = first_alarm < bearing["n_monitor"]
        if not hit:
            if bearing["failed"]:
                missed_bearings.append(bearing["bearing_id"])
            continue
        if bearing["failed"]:
            detected += 1
            lead_pct = 100.0 * (bearing["n_monitor"] - first_alarm) / bearing["n_monitor"]
            all_leads.append(lead_pct)
            dataset = per_dataset.setdefault(
                bearing["dataset"],
                {"n": 0, "n_detected": 0, "leads_pct_life": []},
            )
            dataset["n"] += 1
            dataset["n_detected"] += 1
            dataset["leads_pct_life"].append(lead_pct)
        else:
            false_alarms += 1

    for dataset in per_dataset.values():
        dataset["detection_rate"] = 100.0 * dataset["n_detected"] / dataset["n"]
        dataset["median_lead_pct_life"] = float(np.median(dataset.pop("leads_pct_life")))
    return {
        "threshold": threshold,
        "n_failed": len(failed),
        "n_censored": len(censored),
        "n_detected": detected,
        "detection_rate": 100.0 * detected / len(failed),
        "n_false_alarms": false_alarms,
        "false_alarm_rate": 100.0 * false_alarms / len(censored),
        "median_lead_pct_life": float(np.median(all_leads)),
        "per_dataset": per_dataset,
        "missed_bearings": missed_bearings,
    }


def compact_result(result: dict) -> dict:
    return {
        "n_detected": result["n_detected"],
        "n_failed": result["n_failed"],
        "n_false_alarms": result["n_false_alarms"],
        "n_censored": result["n_censored"],
        "median_lead_pct_life": result["median_lead_pct_life"],
        "missed_bearings": result["missed_bearings"],
    }


def evaluate_window(streams: list[dict], recent_window: int) -> tuple[dict, dict]:
    bearings = group_by_bearing(streams, recent_window)
    censored = [bearing for bearing in bearings if not bearing["failed"]]
    historical_calibration = [
        score for bearing in censored for score in bearing["historical_rates"]
    ]
    recent_calibration = [score for bearing in censored for score in bearing["recent_rates"]]
    historical_threshold = float(
        max(np.max(values) for values in historical_calibration) * CALIBRATION_MARGIN
    )
    recent_threshold = float(
        max(np.max(values) for values in recent_calibration) * CALIBRATION_MARGIN
    )

    variants = {
        "historical_rate": evaluate_variant(
            bearings,
            lambda bearing: list(bearing["historical_rates"]),
            historical_calibration,
        ),
        "recent_rate": evaluate_variant(
            bearings,
            lambda bearing: list(bearing["recent_rates"]),
            recent_calibration,
        ),
        "dual_activation": evaluate_variant(
            bearings,
            lambda bearing: [
                np.maximum(historical / historical_threshold, recent / recent_threshold)
                for historical, recent in zip(
                    bearing["historical_rates"], bearing["recent_rates"]
                )
            ],
            [
                np.maximum(historical / historical_threshold, recent / recent_threshold)
                for bearing in censored
                for historical, recent in zip(
                    bearing["historical_rates"], bearing["recent_rates"]
                )
            ],
        ),
    }
    summary = {
        "recent_window_snapshots": recent_window,
        "thresholds": {
            "historical_rate": historical_threshold,
            "recent_rate": recent_threshold,
        },
        "variants": variants,
    }
    compact = {
        "recent_window_snapshots": recent_window,
        "historical_rate": compact_result(variants["historical_rate"]),
        "recent_rate": compact_result(variants["recent_rate"]),
        "dual_activation": compact_result(variants["dual_activation"]),
    }
    return summary, compact


def main() -> None:
    feature_dir = Path(os.environ.get("RTF_DATA_DIR", str(ROOT / "data" / "features")))
    streams = load_all_streams(feature_dir)
    primary, primary_compact = evaluate_window(streams, PRIMARY_WINDOW)
    sensitivity = []
    for recent_window in SENSITIVITY_WINDOWS:
        _, compact = evaluate_window(streams, recent_window)
        sensitivity.append(compact)

    summary = {
        "protocol": {
            "calibration_set": "IMS censored bearings",
            "baseline": "median RMS snapshots 11-60",
            "excess_ratio_threshold": 1.2,
            "minimum_monitor_snapshots": 30,
            "calibration_margin": CALIBRATION_MARGIN,
            "bearing_alarm": "minimum alarm time over axes (any-axis activation)",
            "dual_rule": "R_t/T_R > 1 or A_t/T_A > 1 on any axis",
        },
        "primary_window": primary,
        "window_sensitivity": sensitivity,
        "interpretation": {
            "dual_activation_expected_gain": "One additional FEMTO detection at zero observed IMS false alarms",
            "claim_boundary": "A recent-rate activation is a mechanism probe, not a replacement for historical persistence; it does not cover the two remaining abrupt FEMTO failures.",
        },
    }
    destination = ROOT / "results" / "results_run_to_failure_rate_acceleration.json"
    destination.write_text(json.dumps(summary, indent=2))
    print(json.dumps(primary_compact, indent=2))
    print("saved", destination)


if __name__ == "__main__":
    main()
