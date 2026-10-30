#!/usr/bin/env python3
"""Evaluate pre-specified normalized-rate fusion rules for signal features."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
IMS_FAILED = {"test1_b3", "test1_b4", "test2_b1", "test3_b3"}
COMBINATIONS = {
    "raw_rms+env_kurtosis": ("raw_rms", "env_kurtosis"),
    "raw_rms+fault_harmonic": ("raw_rms", "fault_harmonic_ratio"),
    "raw_rms+spectral_gini": ("raw_rms", "env_spectral_gini"),
    "env_rms+env_kurtosis": ("env_rms", "env_kurtosis"),
    "env_rms+fault_harmonic": ("env_rms", "fault_harmonic_ratio"),
    "env_rms+spectral_gini": ("env_rms", "env_spectral_gini"),
    "energy+impulsivity+harmonic": ("raw_rms", "env_rms", "env_kurtosis", "fault_harmonic_ratio"),
    "energy+harmonic+gini": ("raw_rms", "env_rms", "fault_harmonic_ratio", "env_spectral_gini"),
    "raw_rms+spectral_kurtosis": ("raw_rms", "env_spectral_kurtosis"),
    "raw_rms+stft_kurtosis": ("raw_rms", "stft_spectral_kurtosis_max"),
    "raw_rms+spectral_crest": ("raw_rms", "env_spectral_crest"),
    "raw_rms+hf_ratio": ("raw_rms", "hf_env_ratio"),
    "raw_rms+three_spectral_confirmations": (
        "raw_rms",
        "env_spectral_kurtosis",
        "stft_spectral_kurtosis_max",
        "env_spectral_crest",
    ),
}
RULES = ("mean", "geometric_mean", "maximum", "minimum")


def load_basic_streams():
    streams = {}
    for test_name, filename in (("test1", "test1.csv"), ("test2", "test2.csv"), ("test3", "test3.csv")):
        with (ROOT / "data/features" / filename).open() as stream:
            for row in csv.DictReader(stream):
                bearing = f"{test_name}_b{row['bearing']}"
                key = ("ims", test_name, bearing, row["channel"])
                streams.setdefault(key, []).append(
                    {"snapshot": len(streams.get(key, [])) + 1, "raw_rms": float(row["rms"])}
                )
    for filename, dataset in (("xjtu_features.csv", "xjtu"), ("femto_features.csv", "femto")):
        with (ROOT / "data/features" / filename).open() as stream:
            for row in csv.DictReader(stream):
                key = (dataset, row["group"], row["bearing"], row["axis"])
                streams.setdefault(key, []).append(
                    {"snapshot": int(row["snapshot"]), "raw_rms": float(row["rms"])}
                )
    return streams


def load_modern_streams():
    streams = {}
    with (ROOT / "data/features/modern_signal_features.csv").open() as stream:
        for row in csv.DictReader(stream):
            key = (row["dataset"], row["group"], row["bearing"], row["axis"])
            streams.setdefault(key, []).append(row)
    return streams


def normalized_rate(rows, feature, feature_threshold):
    values = np.array([float(row[feature]) for row in rows], dtype=np.float64)
    baseline = float(np.median(values[10:60]))
    excess = np.maximum(values[60:] / baseline - 1.2, 0.0)
    rate = np.cumsum(excess) / np.arange(1, len(excess) + 1)
    return rate / feature_threshold


def first_alarm(score, threshold):
    hits = np.flatnonzero(score[29:] > threshold)
    return int(hits[0] + 29) if hits.size else len(score)


def main():
    basic = load_basic_streams()
    modern = load_modern_streams()
    keys = sorted(set(basic) & set(modern))
    merged = {}
    for key in keys:
        modern_rows = sorted(modern[key], key=lambda row: int(row["snapshot"]))
        basic_rows = sorted(basic[key], key=lambda row: int(row["snapshot"]))
        if len(modern_rows) != len(basic_rows):
            raise ValueError(f"stream length mismatch: {key}")
        rows = [{**modern_row, "raw_rms": basic_row["raw_rms"]} for modern_row, basic_row in zip(modern_rows, basic_rows)]
        if len(rows) >= 70:
            merged[key] = rows

    bearings = {}
    for key, rows in merged.items():
        dataset, group, bearing, axis = key
        bearing_key = f"{dataset}_{bearing}"
        bearings.setdefault(bearing_key, {"dataset": dataset, "axes": {}})["axes"][axis] = rows

    feature_thresholds = {}
    for feature in {
        "raw_rms",
        "env_rms",
        "env_kurtosis",
        "fault_harmonic_ratio",
        "env_spectral_gini",
        "env_spectral_kurtosis",
        "stft_spectral_kurtosis_max",
        "env_spectral_crest",
        "hf_env_ratio",
    }:
        maximum = 0.0
        for bearing_key, bearing in bearings.items():
            if not bearing_key.startswith("ims_") or bearing_key[4:] in IMS_FAILED:
                continue
            for rows in bearing["axes"].values():
                values = np.array([float(row[feature]) for row in rows])
                baseline = float(np.median(values[10:60]))
                excess = np.maximum(values[60:] / baseline - 1.2, 0.0)
                rate = np.cumsum(excess) / np.arange(1, len(excess) + 1)
                maximum = max(maximum, float(rate.max()))
        feature_thresholds[feature] = 1.01 * maximum

    output = {}
    for combination_name, features in COMBINATIONS.items():
        scores = {}
        for bearing_key, bearing in bearings.items():
            scores[bearing_key] = {}
            for axis, rows in bearing["axes"].items():
                component_rates = [normalized_rate(rows, feature, feature_thresholds[feature]) for feature in features]
                component_matrix = np.vstack(component_rates)
                scores[bearing_key][axis] = {
                    "mean": component_matrix.mean(axis=0),
                    "geometric_mean": np.exp(np.log(np.maximum(component_matrix, 1e-15)).mean(axis=0)),
                    "maximum": component_matrix.max(axis=0),
                    "minimum": component_matrix.min(axis=0),
                }
        for rule in RULES:
            censored = [key for key in bearings if key.startswith("ims_") and key[4:] not in IMS_FAILED]
            threshold = 1.01 * max(float(scores[key][axis][rule].max()) for key in censored for axis in scores[key])
            detected = []
            leads = []
            per_dataset = {"ims": [0, 0], "xjtu": [0, 0], "femto": [0, 0]}
            for bearing_key, bearing in bearings.items():
                dataset = bearing["dataset"]
                failed = bearing_key[4:] in IMS_FAILED if dataset == "ims" else True
                if not failed:
                    continue
                per_dataset[dataset][1] += 1
                axis_results = []
                for axis in bearing["axes"]:
                    score = scores[bearing_key][axis][rule]
                    alarm = first_alarm(score, threshold)
                    axis_results.append((alarm, len(score)))
                first, length = min(axis_results)
                if first < length:
                    detected.append(bearing_key)
                    leads.append(100.0 * (length - first) / length)
                    per_dataset[dataset][0] += 1
            false_alarm_axes = 0
            false_alarm_bearings = 0
            pseudo_axes = 0
            pseudo_bearings = 0
            for bearing_key, bearing in bearings.items():
                if not bearing_key.startswith(("xjtu_", "femto_")):
                    continue
                eligible_axes = [
                    axis for axis, rows in bearing["axes"].items() if len(rows) >= 150
                ]
                if not eligible_axes:
                    continue
                pseudo_bearings += 1
                bearing_alarm = False
                for axis, rows in bearing["axes"].items():
                    if len(rows) < 150:
                        continue
                    pseudo_axes += 1
                    if np.any(scores[bearing_key][axis][rule][29:90] > threshold):
                        false_alarm_axes += 1
                        bearing_alarm = True
                false_alarm_bearings += int(bearing_alarm)
            output[f"{combination_name}|{rule}"] = {
                "detected": f"{len(detected)}/34",
                "median_lead_pct_life": float(np.median(leads)) if leads else None,
                "threshold": threshold,
                "per_dataset_detection": {name: f"{hit}/{total}" for name, (hit, total) in per_dataset.items()},
                "cross_rig_pseudo_health_axis_fa_rate": 100.0 * false_alarm_axes / pseudo_axes,
                "n_false_alarm_axes": false_alarm_axes,
                "n_pseudo_health_axes": pseudo_axes,
                "cross_rig_pseudo_health_bearing_fa_rate": 100.0 * false_alarm_bearings / pseudo_bearings,
                "n_false_alarm_bearings": false_alarm_bearings,
                "n_pseudo_health_bearings": pseudo_bearings,
            }

    result_path = ROOT / "results/results_run_to_failure_signal_fusion_probe.json"
    result_path.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
