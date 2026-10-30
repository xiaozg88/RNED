#!/usr/bin/env python3
"""Audit whether modern signal features complement the frozen RMS detector."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
IMS_FAILED = {"test1_b3", "test1_b4", "test2_b1", "test3_b3"}
FEATURES = (
    "env_rms",
    "env_kurtosis",
    "fault_harmonic_ratio",
    "env_spectral_concentration",
    "env_spectral_gini",
    "env_spectral_crest",
    "env_spectral_kurtosis",
    "stft_spectral_kurtosis_max",
    "hf_env_ratio",
)
CALIBRATION_MARGIN = 1.01


def load_basic_streams() -> dict[tuple[str, str, str, str], list[float]]:
    streams: dict[tuple[str, str, str, str], list[float]] = {}
    for filename, dataset in (
        ("test1.csv", "ims"),
        ("test2.csv", "ims"),
        ("test3.csv", "ims"),
    ):
        test_name = filename[:-4]
        with (ROOT / "data/features" / filename).open(newline="") as handle:
            for row in csv.DictReader(handle):
                key = (dataset, test_name, f"{test_name}_b{row['bearing']}", row["channel"])
                streams.setdefault(key, []).append(float(row["rms"]))
    for filename, dataset in (
        ("xjtu_features.csv", "xjtu"),
        ("femto_features.csv", "femto"),
    ):
        with (ROOT / "data/features" / filename).open(newline="") as handle:
            for row in csv.DictReader(handle):
                key = (dataset, row["group"], row["bearing"], row["axis"])
                streams.setdefault(key, []).append(float(row["rms"]))
    return streams


def load_modern_streams() -> dict[tuple[str, str, str, str], list[dict[str, str]]]:
    streams: dict[tuple[str, str, str, str], list[dict[str, str]]] = {}
    with (ROOT / "data/features/modern_signal_features.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["dataset"], row["group"], row["bearing"], row["axis"])
            streams.setdefault(key, []).append(row)
    return streams


def build_bearings() -> dict[str, dict]:
    basic = load_basic_streams()
    modern = load_modern_streams()
    merged: dict[tuple[str, str, str, str], dict[str, np.ndarray]] = {}
    for key in sorted(set(basic) & set(modern)):
        modern_rows = sorted(modern[key], key=lambda row: int(row["snapshot"]))
        basic_values = basic[key]
        if len(modern_rows) != len(basic_values) or len(modern_rows) < 70:
            continue
        values = {feature: np.array([float(row[feature]) for row in modern_rows]) for feature in FEATURES}
        values["raw_rms"] = np.asarray(basic_values, dtype=np.float64)
        merged[key] = values

    bearings: dict[str, dict] = {}
    for (dataset, _group, bearing, axis), values in merged.items():
        bearing_key = f"{dataset}_{bearing}"
        bearings.setdefault(
            bearing_key,
            {"dataset": dataset, "axes": {}},
        )["axes"][axis] = values
    return bearings


def online_rate(values: np.ndarray) -> np.ndarray:
    baseline = float(np.median(values[10:60]))
    excess = np.maximum(values[60:] / baseline - 1.2, 0.0)
    return np.cumsum(excess) / np.arange(1, excess.size + 1)


def first_alarm(rate: np.ndarray, threshold: float) -> int:
    hits = np.flatnonzero(rate[29:] > threshold)
    return int(hits[0] + 29) if hits.size else int(rate.size)


def evaluate_feature(bearings: dict[str, dict], feature: str) -> dict:
    rates = {
        bearing_key: {
            axis: online_rate(values[feature]) for axis, values in bearing["axes"].items()
        }
        for bearing_key, bearing in bearings.items()
    }
    censored = [
        bearing_key
        for bearing_key in bearings
        if bearings[bearing_key]["dataset"] == "ims"
        and bearing_key[4:] not in IMS_FAILED
    ]
    threshold = CALIBRATION_MARGIN * max(
        float(np.max(rate)) for bearing_key in censored for rate in rates[bearing_key].values()
    )

    detected: list[str] = []
    leads: dict[str, float] = {}
    for bearing_key, bearing in bearings.items():
        failed = (
            bearings[bearing_key]["dataset"] != "ims"
            or bearing_key[4:] in IMS_FAILED
        )
        if not failed:
            continue
        alarms = {
            axis: first_alarm(rate, threshold)
            for axis, rate in rates[bearing_key].items()
        }
        alarm = min(alarms.values())
        length = min(len(rate) for rate in rates[bearing_key].values())
        if alarm < length:
            detected.append(bearing_key)
            leads[bearing_key] = 100.0 * (length - alarm) / length

    false_axes: set[tuple[str, str]] = set()
    false_bearings: set[str] = set()
    n_pseudo_axes = 0
    n_pseudo_bearings = 0
    for bearing_key, bearing in bearings.items():
        if bearings[bearing_key]["dataset"] not in {"xjtu", "femto"}:
            continue
        eligible_axes = [
            axis
            for axis, values in bearing["axes"].items()
            if len(values[feature]) >= 150
        ]
        if not eligible_axes:
            continue
        n_pseudo_bearings += 1
        for axis in eligible_axes:
            n_pseudo_axes += 1
            if np.any(rates[bearing_key][axis][29:90] > threshold):
                false_axes.add((bearing_key, axis))
                false_bearings.add(bearing_key)

    return {
        "threshold": threshold,
        "detected": detected,
        "leads": leads,
        "false_axes": false_axes,
        "false_bearings": false_bearings,
        "n_pseudo_axes": n_pseudo_axes,
        "n_pseudo_bearings": n_pseudo_bearings,
    }


def compare_to_rms(feature_result: dict, rms_result: dict) -> dict:
    feature_detected = set(feature_result["detected"])
    rms_detected = set(rms_result["detected"])
    both = feature_detected & rms_detected
    lead_deltas = [
        feature_result["leads"][bearing] - rms_result["leads"][bearing]
        for bearing in sorted(both)
    ]
    timing = {"earlier_than_rms": 0, "later_than_rms": 0, "same_as_rms": 0}
    for bearing in sorted(both):
        delta = feature_result["leads"][bearing] - rms_result["leads"][bearing]
        if delta > 0:
            timing["earlier_than_rms"] += 1
        elif delta < 0:
            timing["later_than_rms"] += 1
        else:
            timing["same_as_rms"] += 1
    feature_false_axes = feature_result["false_axes"]
    rms_false_axes = rms_result["false_axes"]
    union_false_axes = feature_false_axes | rms_false_axes
    return {
        "det": f"{len(feature_detected)}/34",
        "rms_union": len(feature_detected | rms_detected),
        "intersection": len(both),
        "catches_rms_misses": sorted(feature_detected - rms_detected),
        "loses_from_rms_count": len(rms_detected - feature_detected),
        "loses_from_rms": sorted(rms_detected - feature_detected),
        **timing,
        "median_lead_delta_both_detected": float(np.median(lead_deltas)) if lead_deltas else None,
        "false_axes": len(feature_false_axes),
        "n_pseudo_health_axes": feature_result["n_pseudo_axes"],
        "false_overlap_with_rms": len(feature_false_axes & rms_false_axes),
        "false_disjoint": len(feature_false_axes - rms_false_axes),
        "false_jaccard": (
            len(feature_false_axes & rms_false_axes) / len(union_false_axes)
            if union_false_axes
            else 0.0
        ),
        "false_bearings": len(feature_result["false_bearings"]),
        "n_pseudo_health_bearings": feature_result["n_pseudo_bearings"],
        "threshold": feature_result["threshold"],
    }


def main() -> None:
    bearings = build_bearings()
    rms_result = evaluate_feature(bearings, "raw_rms")
    features = {
        feature: compare_to_rms(evaluate_feature(bearings, feature), rms_result)
        for feature in FEATURES
    }
    summary = {
        "protocol": {
            "baseline": "feature median snapshots 11-60",
            "excess_ratio_threshold": 1.2,
            "calibration_set": "IMS censored bearings",
            "calibration_margin": CALIBRATION_MARGIN,
            "minimum_monitor_updates": 30,
            "bearing_alarm": "any-axis activation",
            "pseudo_health_window": "monitor updates 30-89 (snapshots 90-149)",
        },
        "raw_rms_detected": rms_result["detected"],
        "raw_rms_misses": [
            bearing
            for bearing in bearings
            if (bearings[bearing]["dataset"] != "ims" or bearing[4:] in IMS_FAILED)
            and bearing not in rms_result["detected"]
        ],
        "raw_rms_false_axes": len(rms_result["false_axes"]),
        "raw_rms_n_pseudo_health_axes": rms_result["n_pseudo_axes"],
        "raw_rms_false_bearings": len(rms_result["false_bearings"]),
        "raw_rms_n_pseudo_health_bearings": rms_result["n_pseudo_bearings"],
        "features": features,
    }
    destination = ROOT / "results/results_run_to_failure_feature_complementarity.json"
    destination.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
