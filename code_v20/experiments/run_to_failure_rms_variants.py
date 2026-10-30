#!/usr/bin/env python3
"""Evaluate lightweight, causal variants of the frozen RMS detector."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
IMS_FAILED = {"test1_b3", "test1_b4", "test2_b1", "test3_b3"}
CALIBRATION_MARGIN = 1.01
MIN_MONITOR_UPDATES = 30
IMPACT_WEIGHTS = (0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50)


def load_axis_streams() -> dict[tuple[str, str, str, str], dict]:
    streams: dict[tuple[str, str, str, str], dict] = {}

    basic: dict[tuple[str, str, str, str], list[float]] = {}
    for filename, dataset in (
        ("test1.csv", "ims"),
        ("test2.csv", "ims"),
        ("test3.csv", "ims"),
    ):
        test_name = filename[:-4]
        with (ROOT / "data/features" / filename).open(newline="") as handle:
            for row in csv.DictReader(handle):
                key = (dataset, test_name, f"{test_name}_b{row['bearing']}", row["channel"])
                basic.setdefault(key, []).append(float(row["rms"]))

    for filename, dataset in (
        ("xjtu_features.csv", "xjtu"),
        ("femto_features.csv", "femto"),
    ):
        with (ROOT / "data/features" / filename).open(newline="") as handle:
            for row in csv.DictReader(handle):
                key = (dataset, row["group"], row["bearing"], row["axis"])
                basic.setdefault(key, []).append(float(row["rms"]))

    modern: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    with (ROOT / "data/features/modern_signal_features.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["dataset"], row["group"], row["bearing"], row["axis"])
            modern[key].append(row)

    for key, raw_values in basic.items():
        if key not in modern:
            continue
        modern_rows = sorted(modern[key], key=lambda row: int(row["snapshot"]))
        if len(modern_rows) != len(raw_values) or len(raw_values) < 70:
            continue
        streams[key] = {
            "raw": np.asarray(raw_values, dtype=np.float64),
            "env": np.asarray(
                [float(row["env_rms"]) for row in modern_rows], dtype=np.float64
            ),
        }
    return streams


def causal_median(values: np.ndarray, window: int) -> np.ndarray:
    return np.asarray(
        [float(np.median(values[max(0, index - window + 1) : index + 1])) for index in range(len(values))]
    )


def block_rms(values: np.ndarray, window: int) -> np.ndarray:
    return np.asarray(
        [
            float(np.sqrt(np.mean(np.square(values[max(0, index - window + 1) : index + 1]))))
            for index in range(len(values))
        ]
    )


def impact_aware_rms(values: dict, weight: float) -> np.ndarray:
    raw = values["raw"]
    envelope = values["env"]
    raw_ratio = raw / float(np.median(raw[10:60]))
    envelope_ratio = envelope / float(np.median(envelope[10:60]))
    return np.sqrt((1.0 - weight) * raw_ratio**2 + weight * envelope_ratio**2)


def build_features() -> dict[tuple[str, str, str, str], dict[str, np.ndarray]]:
    streams = load_axis_streams()
    features: dict[tuple[str, str, str, str], dict[str, np.ndarray]] = {}
    for key, values in streams.items():
        variants = {
            "raw_rms": values["raw"],
            "causal_median3_rms": causal_median(values["raw"], 3),
            "causal_median5_rms": causal_median(values["raw"], 5),
            "block3_rms": block_rms(values["raw"], 3),
            "block5_rms": block_rms(values["raw"], 5),
            "env_rms": values["env"],
        }
        for weight in IMPACT_WEIGHTS:
            name = f"impact_aware_rms_w{int(round(weight * 100)):02d}"
            variants[name] = impact_aware_rms(values, weight)
        features[key] = variants
    return features


def build_bearings(
    features: dict[tuple[str, str, str, str], dict[str, np.ndarray]]
) -> dict[str, dict]:
    bearings: dict[str, dict] = {}
    for (dataset, _group, bearing, axis), variants in features.items():
        bearing_key = f"{dataset}_{bearing}"
        bearings.setdefault(bearing_key, {"dataset": dataset, "axes": {}})["axes"][axis] = variants
    return bearings


def online_rate(values: np.ndarray) -> np.ndarray:
    baseline = float(np.median(values[10:60]))
    excess = np.maximum(values[60:] / baseline - 1.2, 0.0)
    return np.cumsum(excess) / np.arange(1, excess.size + 1)


def first_alarm(rate: np.ndarray, threshold: float) -> int:
    hits = np.flatnonzero(rate[MIN_MONITOR_UPDATES - 1 :] > threshold)
    return int(hits[0] + MIN_MONITOR_UPDATES - 1) if hits.size else int(rate.size)


def evaluate(bearings: dict[str, dict], variant: str) -> dict:
    rates = {
        bearing_key: {
            axis: online_rate(axis_variants[variant])
            for axis, axis_variants in bearing["axes"].items()
        }
        for bearing_key, bearing in bearings.items()
    }
    censored = [
        bearing_key
        for bearing_key, bearing in bearings.items()
        if bearing["dataset"] == "ims" and bearing_key[4:] not in IMS_FAILED
    ]
    threshold = CALIBRATION_MARGIN * max(
        float(rate.max())
        for bearing_key in censored
        for rate in rates[bearing_key].values()
    )

    detected: list[str] = []
    leads: dict[str, float] = {}
    per_dataset = defaultdict(lambda: [0, 0])
    for bearing_key, bearing in bearings.items():
        dataset = bearing["dataset"]
        failed = dataset != "ims" or bearing_key[4:] in IMS_FAILED
        if not failed:
            continue
        alarms = {
            axis: first_alarm(rate, threshold)
            for axis, rate in rates[bearing_key].items()
        }
        alarm = min(alarms.values())
        length = min(len(rate) for rate in rates[bearing_key].values())
        per_dataset[dataset][1] += 1
        if alarm < length:
            detected.append(bearing_key)
            leads[bearing_key] = 100.0 * (length - alarm) / length
            per_dataset[dataset][0] += 1

    false_axes: list[str] = []
    false_bearings: list[str] = []
    n_pseudo_axes = 0
    n_pseudo_bearings = 0
    for bearing_key, bearing in bearings.items():
        if bearing["dataset"] not in {"xjtu", "femto"}:
            continue
        eligible_axes = [
            axis for axis, rate in rates[bearing_key].items() if len(rate) >= 90
        ]
        if not eligible_axes:
            continue
        n_pseudo_bearings += 1
        bearing_flagged = False
        for axis in eligible_axes:
            n_pseudo_axes += 1
            if np.any(rates[bearing_key][axis][29:90] > threshold):
                false_axes.append(f"{bearing_key}|{axis}")
                bearing_flagged = True
        if bearing_flagged:
            false_bearings.append(bearing_key)

    return {
        "detected": f"{len(detected)}/34",
        "n_detected": len(detected),
        "detected_bearings": detected,
        "median_lead_pct_life": float(np.median(list(leads.values()))) if leads else None,
        "per_dataset_detection": {
            name: f"{hit}/{total}" for name, (hit, total) in sorted(per_dataset.items())
        },
        "threshold": threshold,
        "n_false_alarm_axes": len(false_axes),
        "n_pseudo_health_axes": n_pseudo_axes,
        "cross_rig_pseudo_health_axis_fa_rate": 100.0 * len(false_axes) / n_pseudo_axes,
        "false_alarm_axes": false_axes,
        "n_false_alarm_bearings": len(false_bearings),
        "n_pseudo_health_bearings": n_pseudo_bearings,
        "cross_rig_pseudo_health_bearing_fa_rate": 100.0 * len(false_bearings) / n_pseudo_bearings,
        "leads_pct_life": leads,
    }


def compare_to_raw(result: dict, raw_result: dict) -> dict:
    detected = set(result["detected_bearings"])
    raw_detected = set(raw_result["detected_bearings"])
    shared = sorted(detected & raw_detected)
    lead_deltas = [
        result["leads_pct_life"][bearing] - raw_result["leads_pct_life"][bearing]
        for bearing in shared
    ]
    timing = {"earlier_than_raw": 0, "same_as_raw": 0, "later_than_raw": 0}
    for bearing in shared:
        delta = result["leads_pct_life"][bearing] - raw_result["leads_pct_life"][bearing]
        if delta > 0:
            timing["earlier_than_raw"] += 1
        elif delta < 0:
            timing["later_than_raw"] += 1
        else:
            timing["same_as_raw"] += 1
    return {
        "catches_raw_misses": sorted(detected - raw_detected),
        "loses_from_raw": sorted(raw_detected - detected),
        "median_lead_delta_both_detected": float(np.median(lead_deltas)) if lead_deltas else None,
        **timing,
    }


def threshold_sweep(bearings: dict[str, dict], variant: str, grid_size: int = 800) -> list[dict]:
    rates = {
        bearing_key: {
            axis: online_rate(axis_variants[variant])
            for axis, axis_variants in bearing["axes"].items()
        }
        for bearing_key, bearing in bearings.items()
    }
    maximum = max(float(rate.max()) for axis_rates in rates.values() for rate in axis_rates.values())
    sweep = []
    for index in range(grid_size + 1):
        threshold = maximum * index / grid_size
        detected = 0
        leads = []
        false_axes = 0
        pseudo_axes = 0
        for bearing_key, bearing in bearings.items():
            failed = bearing["dataset"] != "ims" or bearing_key[4:] in IMS_FAILED
            if failed:
                alarms = [first_alarm(rate, threshold) for rate in rates[bearing_key].values()]
                alarm = min(alarms)
                length = min(len(rate) for rate in rates[bearing_key].values())
                if alarm < length:
                    detected += 1
                    leads.append(100.0 * (length - alarm) / length)
            if bearing["dataset"] in {"xjtu", "femto"}:
                for rate in rates[bearing_key].values():
                    if len(rate) < 90:
                        continue
                    pseudo_axes += 1
                    false_axes += int(np.any(rate[29:90] > threshold))
        sweep.append(
            {
                "threshold": threshold,
                "n_detected": detected,
                "median_lead_pct_life": float(np.median(leads)) if leads else None,
                "n_false_alarm_axes": false_axes,
                "n_pseudo_health_axes": pseudo_axes,
            }
        )
    return sweep


def main() -> None:
    bearings = build_bearings(build_features())
    variant_names = list(next(iter(bearings.values()))["axes"].values())[0].keys()
    raw_result = {
        **evaluate(bearings, "raw_rms"),
        "catches_raw_misses": [],
        "loses_from_raw": [],
        "median_lead_delta_both_detected": 0.0,
        "earlier_than_raw": 0,
        "same_as_raw": 31,
        "later_than_raw": 0,
    }
    results = {
        variant: {**evaluate(bearings, variant), **compare_to_raw(evaluate(bearings, variant), raw_result)}
        for variant in variant_names
    }
    output = {
        "protocol": {
            "status": "exploratory post-hoc screen; not a locked validation",
            "baseline": "per-axis variant median over snapshots 11-60",
            "excess_ratio_threshold": 1.2,
            "calibration_set": "eight IMS censored bearings",
            "calibration_margin": CALIBRATION_MARGIN,
            "minimum_monitor_updates": MIN_MONITOR_UPDATES,
            "bearing_alarm": "any-axis activation",
            "pseudo_health_window": "monitor updates 30-89",
            "variants": {
                "impact_aware_rms": "sqrt((1-w)(RMS/b_RMS)^2+w(envRMS/b_envRMS)^2); w in {0.02,0.05,0.10,0.20,0.30,0.50}",
                "causal_median_rms": "median of the current and preceding RMS snapshots",
                "block_rms": "RMS of the current and preceding RMS snapshots",
            },
        },
        "results": results,
    }
    result_path = ROOT / "results/results_run_to_failure_rms_variants.json"
    result_path.write_text(json.dumps(output, indent=2))

    csv_path = ROOT / "results/results_run_to_failure_rms_variants.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "variant",
                "detected",
                "median_lead_pct_life",
                "cross_rig_pseudo_health_axis_fa_rate",
                "n_false_alarm_axes",
                "n_false_alarm_bearings",
                "catches_raw_misses",
                "loses_from_raw",
                "median_lead_delta_both_detected",
            ],
        )
        writer.writeheader()
        for variant, result in results.items():
            writer.writerow(
                {
                    "variant": variant,
                    "detected": result["detected"],
                    "median_lead_pct_life": result["median_lead_pct_life"],
                    "cross_rig_pseudo_health_axis_fa_rate": result["cross_rig_pseudo_health_axis_fa_rate"],
                    "n_false_alarm_axes": result["n_false_alarm_axes"],
                    "n_false_alarm_bearings": result["n_false_alarm_bearings"],
                    "catches_raw_misses": ";".join(result.get("catches_raw_misses", [])),
                    "loses_from_raw": ";".join(result.get("loses_from_raw", [])),
                    "median_lead_delta_both_detected": result.get("median_lead_delta_both_detected"),
                }
            )
    sweep_path = ROOT / "results/results_run_to_failure_rms_variants_threshold_sweep.csv"
    sweep_rows = []
    with sweep_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "variant",
                "threshold",
                "n_detected",
                "median_lead_pct_life",
                "n_false_alarm_axes",
                "n_pseudo_health_axes",
            ],
        )
        writer.writeheader()
        impact_names = [f"impact_aware_rms_w{int(round(weight * 100)):02d}" for weight in IMPACT_WEIGHTS]
        for variant in ("raw_rms", "env_rms", *impact_names):
            for point in threshold_sweep(bearings, variant):
                row = {"variant": variant, **point}
                sweep_rows.append(row)
                writer.writerow(row)

    fixed_budget_path = ROOT / "results/results_run_to_failure_rms_variants_fixed_budget.csv"
    with fixed_budget_path.open("w", newline="") as handle:
        fieldnames = [
            "variant",
            "default_detection",
            "default_median_lead_pct_life",
            "default_n_false_alarm_axes",
            "budget_threshold",
            "budget_detection",
            "budget_median_lead_pct_life",
            "budget_n_false_alarm_axes",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for variant in sorted({row["variant"] for row in sweep_rows}):
            eligible = [row for row in sweep_rows if row["variant"] == variant and int(row["n_false_alarm_axes"]) <= 10]
            best = max(
                eligible,
                key=lambda row: (
                    int(row["n_detected"]),
                    float(row["median_lead_pct_life"]) if row["median_lead_pct_life"] is not None else -1.0,
                ),
            )
            default = results[variant]
            writer.writerow(
                {
                    "variant": variant,
                    "default_detection": default["detected"],
                    "default_median_lead_pct_life": default["median_lead_pct_life"],
                    "default_n_false_alarm_axes": default["n_false_alarm_axes"],
                    "budget_threshold": best["threshold"],
                    "budget_detection": f"{best['n_detected']}/34",
                    "budget_median_lead_pct_life": best["median_lead_pct_life"],
                    "budget_n_false_alarm_axes": best["n_false_alarm_axes"],
                }
            )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
