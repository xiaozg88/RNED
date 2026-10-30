#!/usr/bin/env python3
"""Deployment-oriented experiments for the frozen RMS-rate detector.

The modules are descriptive deployment analyses, not new detector tuning:
1) confirmation delay and fleet false-alarm proxy;
2) normalized maintenance-cost scenarios;
3) sparse updates, missing bursts, and RMS-domain quantization;
4) finite conformal calibration-capacity planning;
5) time-resolved mechanism separation.

The clean-data threshold is frozen at the existing IMS calibration. Perturbed
data are never used to re-tune the threshold.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
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

THRESHOLD_MARGIN = 1.01
MIN_OBSERVED_UPDATES = 30
CONFIRMATION_LEVELS = (1, 2, 3, 5, 10, 20, 30, 60)
FLEET_SIZES = (10, 50, 100, 500, 1000)
EARLY_PROXY_MONITOR_SNAPSHOTS = 90
QUANTIZATION_HEADROOM_RATIO = 8.0
BOOTSTRAP_RUNS = 10000
RNG = np.random.default_rng(20261023)


def confirmed_alarm_index(
    scores: np.ndarray, threshold: float, confirmation: int
) -> int:
    """Return the first snapshot ending `confirmation` consecutive crossings."""
    if scores.size < max(MIN_OBSERVED_UPDATES, confirmation):
        return int(scores.size)
    crossing = scores > threshold
    run = 0
    for index, is_crossing in enumerate(crossing):
        run = run + 1 if is_crossing else 0
        if index >= MIN_OBSERVED_UPDATES - 1 and run >= confirmation:
            return int(index)
    return int(scores.size)


def bearing_confirmed_alarm(
    historical_matrix: np.ndarray, threshold: float, confirmation: int
) -> int:
    return int(
        min(
            confirmed_alarm_index(axis, threshold, confirmation)
            for axis in historical_matrix
        )
    )


def summarize_confirmed(
    bearings: list[dict], threshold: float, confirmation: int
) -> dict:
    failed = [bearing for bearing in bearings if bearing["failed"]]
    censored = [bearing for bearing in bearings if not bearing["failed"]]
    alarms = {
        bearing["bearing_id"]: bearing_confirmed_alarm(
            bearing["historical_matrix"], threshold, confirmation
        )
        for bearing in bearings
    }
    detected = [
        bearing
        for bearing in failed
        if alarms[bearing["bearing_id"]] < bearing["n_monitor"]
    ]
    false_alarms = [
        bearing
        for bearing in censored
        if alarms[bearing["bearing_id"]] < bearing["n_monitor"]
    ]
    leads = [
        100.0
        * (bearing["n_monitor"] - alarms[bearing["bearing_id"]])
        / bearing["n_monitor"]
        for bearing in detected
    ]
    return {
        "confirmation_snapshots": confirmation,
        "n_detected": len(detected),
        "n_failed": len(failed),
        "n_false_alarms_ims_censored": len(false_alarms),
        "n_censored": len(censored),
        "median_lead_pct_life": float(np.median(leads)) if leads else None,
        "mean_lead_all_failed_pct_life": float(
            np.mean(
                [
                    100.0
                    * (
                        bearing["n_monitor"]
                        - min(
                            alarms[bearing["bearing_id"]],
                            bearing["n_monitor"],
                        )
                    )
                    / bearing["n_monitor"]
                    if bearing in detected
                    else 0.0
                    for bearing in failed
                ]
            )
        ),
        "missed_bearings": [
            bearing["bearing_id"]
            for bearing in failed
            if bearing not in detected
        ],
        "alarm_by_bearing": alarms,
    }


def pseudo_health_confirmation(
    bearings: list[dict], threshold: float, confirmation: int
) -> dict:
    pseudo = [
        bearing
        for bearing in bearings
        if bearing["dataset"] in {"xjtu", "femto"}
        and bearing["n_monitor"] >= EARLY_PROXY_MONITOR_SNAPSHOTS
    ]
    hits = []
    per_dataset = defaultdict(lambda: {"n": 0, "alarms": 0})
    for bearing in pseudo:
        alarm = bearing_confirmed_alarm(
            bearing["historical_matrix"], threshold, confirmation
        )
        hit = bool(alarm < EARLY_PROXY_MONITOR_SNAPSHOTS)
        hits.append(hit)
        per_dataset[bearing["dataset"]]["n"] += 1
        per_dataset[bearing["dataset"]]["alarms"] += int(hit)
    return {
        "n_bearings": len(pseudo),
        "n_alarms": int(np.sum(hits)),
        "alarm_rate": float(np.mean(hits)),
        "indicators": hits,
        "per_dataset": dict(per_dataset),
    }


def fleet_bootstrap(indicators: list[bool]) -> dict:
    values = np.asarray(indicators, dtype=np.float64)
    rates = np.empty(BOOTSTRAP_RUNS, dtype=np.float64)
    for run in range(BOOTSTRAP_RUNS):
        sample = RNG.choice(values, size=values.size, replace=True)
        rates[run] = float(np.mean(sample))
    rows = []
    for fleet_size in FLEET_SIZES:
        risks = 1.0 - (1.0 - rates) ** fleet_size
        rows.append(
            {
                "fleet_size": fleet_size,
                "median_any_false_alarm_probability": float(np.median(risks)),
                "risk_ci95": [
                    float(np.quantile(risks, 0.025)),
                    float(np.quantile(risks, 0.975)),
                ],
                "median_expected_false_alarms": float(
                    np.median(fleet_size * rates)
                ),
                "expected_ci95": [
                    float(np.quantile(fleet_size * rates, 0.025)),
                    float(np.quantile(fleet_size * rates, 0.975)),
                ],
            }
        )
    return rows


def confirmation_and_fleet_module(bearings: list[dict], threshold: float) -> tuple[dict, pd.DataFrame]:
    rows = []
    fleet_rows = []
    for confirmation in CONFIRMATION_LEVELS:
        clean = summarize_confirmed(bearings, threshold, confirmation)
        proxy = pseudo_health_confirmation(
            bearings, threshold, confirmation
        )
        fleet = fleet_bootstrap(proxy["indicators"])
        for item in fleet:
            fleet_rows.append(
                {
                    "confirmation_snapshots": confirmation,
                    **item,
                }
            )
        rows.append(
            {
                "confirmation_snapshots": confirmation,
                "n_detected": clean["n_detected"],
                "n_failed": clean["n_failed"],
                "n_ims_false_alarms": clean["n_false_alarms_ims_censored"],
                "median_lead_pct_life": clean["median_lead_pct_life"],
                "mean_lead_all_failed_pct_life": clean[
                    "mean_lead_all_failed_pct_life"
                ],
                "pseudo_health_n_alarms": proxy["n_alarms"],
                "pseudo_health_n": proxy["n_bearings"],
                "pseudo_health_alarm_rate": proxy["alarm_rate"],
            }
        )
    output = {
        "protocol": {
            "threshold": threshold,
            "rule": "k consecutive above-threshold historical-rate snapshots on any axis",
            "fleet_proxy": "bootstrap over 27 XJTU/FEMTO first-150-raw-snapshot pseudo-health bearings",
            "bootstrap_runs": BOOTSTRAP_RUNS,
            "status": "deployment_scenario_not_validation",
        },
        "confirmation_rows": rows,
        "fleet_rows": fleet_rows,
    }
    return output, pd.DataFrame(rows)


def maintenance_cost_module(confirmation_frame: pd.DataFrame) -> dict:
    lambda_values = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
    scenarios = []
    for relative_false_alarm_cost in lambda_values:
        costs = []
        for row in confirmation_frame.to_dict(orient="records"):
            miss_rate = 1.0 - row["n_detected"] / row["n_failed"]
            late_penalty = 1.0 - row["mean_lead_all_failed_pct_life"] / 100.0
            total = (
                relative_false_alarm_cost * row["pseudo_health_alarm_rate"]
                + miss_rate
                + late_penalty
            )
            costs.append(total)
        best_index = int(np.argmin(costs))
        best = confirmation_frame.iloc[best_index].to_dict()
        scenarios.append(
            {
                "relative_false_alarm_cost_lambda": relative_false_alarm_cost,
                "best_confirmation_snapshots": int(
                    best["confirmation_snapshots"]
                ),
                "best_normalized_cost": float(costs[best_index]),
                "cost_by_confirmation": {
                    int(row["confirmation_snapshots"]): float(cost)
                    for row, cost in zip(
                        confirmation_frame.to_dict(orient="records"), costs
                    )
                },
            }
        )
    return {
        "protocol": {
            "cost": "lambda * pseudo-health alarm rate + miss rate + (1 - mean lead fraction over all failed bearings)",
            "interpretation": "lambda is the relative cost of one proxy false alarm compared with the unit miss/late penalty",
            "status": "scenario_analysis_not_field_cost_study",
        },
        "scenarios": scenarios,
    }


def rates_from_selected_monitor_indices(
    rms: np.ndarray, monitor_indices: np.ndarray
) -> np.ndarray:
    baseline = float(np.median(rms[10:60]))
    ratio = rms[60:][monitor_indices] / baseline
    excess = np.maximum(ratio - 1.2, 0.0)
    return np.cumsum(excess) / np.arange(1, excess.size + 1)


def alarm_from_observed_indices(
    scores: np.ndarray, monitor_indices: np.ndarray, threshold: float
) -> int:
    if scores.size < MIN_OBSERVED_UPDATES:
        return int(monitor_indices[-1] + 1)
    hits = np.flatnonzero(scores[MIN_OBSERVED_UPDATES - 1 :] > threshold)
    if not hits.size:
        return int(monitor_indices[-1] + 1)
    update_index = int(hits[0] + MIN_OBSERVED_UPDATES - 1)
    return int(monitor_indices[update_index])


def evaluate_observation_pattern(
    bearings: list[dict], pattern_fn, threshold: float, seed: int | None = None
) -> dict:
    failed = [bearing for bearing in bearings if bearing["failed"]]
    censored = [bearing for bearing in bearings if not bearing["failed"]]
    detections = []
    median_leads = []
    false_alarms = []
    for bearing in bearings:
        pattern = pattern_fn(bearing, seed)
        scores = []
        actual_indices = []
        for stream in bearing["axes"]:
            monitor_indices = np.asarray(pattern, dtype=int)
            if monitor_indices.size < MIN_OBSERVED_UPDATES:
                continue
            scores.append(
                rates_from_selected_monitor_indices(
                    stream["rms"], monitor_indices
                )
            )
            actual_indices.append(monitor_indices)
        if not scores:
            alarm = bearing["n_monitor"]
        else:
            alarm = min(
                alarm_from_observed_indices(
                    score, indices, threshold
                )
                for score, indices in zip(scores, actual_indices)
            )
        if bearing["failed"]:
            detections.append(alarm < bearing["n_monitor"])
            if alarm < bearing["n_monitor"]:
                median_leads.append(
                    100.0
                    * (bearing["n_monitor"] - alarm)
                    / bearing["n_monitor"]
                )
        else:
            false_alarms.append(alarm < bearing["n_monitor"])
    return {
        "n_detected": int(np.sum(detections)),
        "n_failed": len(failed),
        "n_false_alarms": int(np.sum(false_alarms)),
        "n_censored": len(censored),
        "median_lead_pct_life": float(np.median(median_leads)),
    }


def causally_quantize_monitor_rms(
    rms: np.ndarray, bits: int, headroom_ratio: float
) -> np.ndarray:
    baseline = float(np.median(rms[10:60]))
    monitor = rms[60:]
    upper = headroom_ratio * baseline
    step = upper / (2**bits - 1)
    return np.clip(np.round(monitor / step) * step, 0.0, upper)


def low_power_module(bearings: list[dict], threshold: float) -> dict:
    sparse_rows = []
    for stride in (1, 2, 3, 5, 10):
        result = evaluate_observation_pattern(
            bearings,
            lambda bearing, seed, stride=stride: np.arange(
                0, bearing["n_monitor"], stride
            ),
            threshold,
        )
        sparse_rows.append({"sampling_stride": stride, **result})

    quantization_rows = []
    for bits in (8, 12, 16, 24):
        detections = []
        false_alarms = []
        leads = []
        for bearing in bearings:
            scores = []
            for stream in bearing["axes"]:
                quantized = causally_quantize_monitor_rms(
                    stream["rms"], bits, QUANTIZATION_HEADROOM_RATIO
                )
                scores.append(
                    rates_from_selected_monitor_indices(
                        np.concatenate((stream["rms"][:60], quantized)),
                        np.arange(bearing["n_monitor"]),
                    )
                )
            alarm = min(
                alarm_from_observed_indices(
                    score, np.arange(score.size), threshold
                )
                for score in scores
            )
            if bearing["failed"]:
                detections.append(alarm < bearing["n_monitor"])
                if alarm < bearing["n_monitor"]:
                    leads.append(
                        100.0
                        * (bearing["n_monitor"] - alarm)
                        / bearing["n_monitor"]
                    )
            else:
                false_alarms.append(alarm < bearing["n_monitor"])
        quantization_rows.append(
            {
                "rms_quantization_bits": bits,
                "n_detected": int(np.sum(detections)),
                "n_failed": sum(bearing["failed"] for bearing in bearings),
                "n_false_alarms": int(np.sum(false_alarms)),
                "n_censored": sum(
                    not bearing["failed"] for bearing in bearings
                ),
                "median_lead_pct_life": float(np.median(leads)),
            }
        )

    random_rows = []
    for keep_probability in (0.9, 0.8, 0.5):
        results = []
        for seed in range(100):
            def pattern(bearing, seed_value, probability=keep_probability):
                local_rng = np.random.default_rng(100000 + seed_value * 1000 + bearing["n_monitor"])
                return np.flatnonzero(
                    local_rng.random(bearing["n_monitor"]) < probability
                )

            results.append(
                evaluate_observation_pattern(
                    bearings, pattern, threshold, seed=seed
                )
            )
        random_rows.append(
            {
                "keep_probability": keep_probability,
                "mean_detection": float(
                    np.mean([row["n_detected"] for row in results])
                ),
                "detection_ci95": [
                    float(np.quantile([row["n_detected"] for row in results], 0.025)),
                    float(np.quantile([row["n_detected"] for row in results], 0.975)),
                ],
                "mean_false_alarms": float(
                    np.mean([row["n_false_alarms"] for row in results])
                ),
                "median_lead_median_across_trials": float(
                    np.median([row["median_lead_pct_life"] for row in results])
                ),
            }
        )

    burst_rows = []
    for burst_length in (10, 30, 60):
        results = []
        for seed in range(100):
            def pattern(bearing, seed_value, length=burst_length):
                local_rng = np.random.default_rng(200000 + seed_value * 1000 + bearing["n_monitor"])
                start = int(
                    local_rng.integers(
                        0, max(1, bearing["n_monitor"] - length + 1)
                    )
                )
                removed = np.arange(start, min(start + length, bearing["n_monitor"]))
                return np.setdiff1d(
                    np.arange(bearing["n_monitor"]), removed
                )

            results.append(
                evaluate_observation_pattern(
                    bearings, pattern, threshold, seed=seed
                )
            )
        burst_rows.append(
            {
                "missing_burst_snapshots": burst_length,
                "mean_detection": float(
                    np.mean([row["n_detected"] for row in results])
                ),
                "detection_ci95": [
                    float(np.quantile([row["n_detected"] for row in results], 0.025)),
                    float(np.quantile([row["n_detected"] for row in results], 0.975)),
                ],
                "mean_false_alarms": float(
                    np.mean([row["n_false_alarms"] for row in results])
                ),
                "median_lead_median_across_trials": float(
                    np.median([row["median_lead_pct_life"] for row in results])
                ),
            }
        )

    return {
        "protocol": {
            "threshold": threshold,
            "threshold_recalibration": False,
            "quantization_scope": (
                "causal RMS feature-domain proxy, not a raw-ADC claim"
            ),
            "quantization_headroom_ratio": QUANTIZATION_HEADROOM_RATIO,
            "sparse_updates": "selected monitor snapshots enter the same cumulative-mean update; actual life clock is retained",
            "random_dropout_unit": "shared missingness mask across axes within a bearing",
            "trials_per_random_condition": 100,
            "status": "stress_test",
        },
        "sparse_updates": sparse_rows,
        "rms_quantization": quantization_rows,
        "random_dropout": random_rows,
        "missing_bursts": burst_rows,
    }


def conformal_capacity_module(bearings: list[dict]) -> dict:
    healthy = [bearing for bearing in bearings if not bearing["failed"]]
    maxima = np.asarray(
        [
            float(np.max(bearing["historical_rate"]))
            for bearing in healthy
        ]
    )
    levels = [
        {"total_alpha": 0.20, "layers": 2, "per_layer_alpha": 0.10},
        {"total_alpha": 0.10, "layers": 2, "per_layer_alpha": 0.05},
        {"total_alpha": 0.05, "layers": 2, "per_layer_alpha": 0.025},
    ]
    output = []
    for level in levels:
        alpha = level["per_layer_alpha"]
        required = int(np.ceil(1.0 / alpha - 1.0))
        sample_sizes = sorted(
            set(
                [
                    required,
                    required + 10,
                    required * 2,
                    min(100, required * 4),
                ]
            )
        )
        rows = []
        for sample_size in sample_sizes:
            order_index = int(np.ceil((sample_size + 1) * (1.0 - alpha)))
            if order_index > sample_size:
                continue
            thresholds = np.empty(5000, dtype=np.float64)
            for run in range(5000):
                sample = RNG.choice(maxima, size=sample_size, replace=True)
                thresholds[run] = float(np.sort(sample)[order_index - 1])
            rows.append(
                {
                    "calibration_bearings": sample_size,
                    "order_statistic_index": order_index,
                    "threshold_median": float(np.median(thresholds)),
                    "threshold_q05": float(np.quantile(thresholds, 0.05)),
                    "threshold_q95": float(np.quantile(thresholds, 0.95)),
                    "q95_q50_ratio": float(
                        np.quantile(thresholds, 0.95)
                        / np.median(thresholds)
                    ),
                }
            )
        output.append({**level, "minimum_required": required, "bootstrap": rows})
    return {
        "protocol": {
            "calibration_unit": "one healthy bearing trajectory",
            "statistic": "maximum any-axis historical excess rate over the full trajectory",
            "bootstrap_source": "the eight IMS censored bearing maxima",
            "runs": 5000,
            "status": "capacity_planning_not_independent_validation",
        },
        "levels": output,
    }


def separation_at_snapshot(failed_values: list[float], healthy_values: list[float]) -> dict:
    auc = roc_auc_score(
        [1] * len(failed_values) + [0] * len(healthy_values),
        failed_values + healthy_values,
    )
    test = mannwhitneyu(
        failed_values, healthy_values, alternative="greater", method="exact"
    )
    return {
        "n_failed": len(failed_values),
        "n_healthy": len(healthy_values),
        "auc": float(auc),
        "exact_mann_whitney_p": float(test.pvalue),
    }


def time_resolved_module(bearings: list[dict]) -> dict:
    horizons = (30, 60, 90, 150, 300, 600, 1200)
    features = (
        "historical_rate",
        "cross_axis",
        "coactivation_fraction",
        "max_ratio",
        "cumulative_excess",
    )
    rows = []
    for horizon in horizons:
        eligible = [
            bearing
            for bearing in bearings
            if bearing["n_monitor"] >= horizon
        ]
        failed = [bearing for bearing in eligible if bearing["failed"]]
        healthy = [bearing for bearing in eligible if not bearing["failed"]]
        feature_values = {name: ([], []) for name in features}
        for bearing in failed:
            values = mechanism_features_at_snapshot(bearing, horizon)
            for name in features:
                feature_values[name][0].append(values[name])
        for bearing in healthy:
            values = mechanism_features_at_snapshot(bearing, horizon)
            for name in features:
                feature_values[name][1].append(values[name])
        row = {
            "monitored_snapshots": horizon,
            "raw_snapshots_including_baseline": horizon + 60,
            "n_failed": len(failed),
            "n_healthy": len(healthy),
        }
        for name in features:
            row[name] = separation_at_snapshot(
                feature_values[name][0], feature_values[name][1]
            )
        rows.append(row)
    return {
        "protocol": {
            "comparison": "eligible failed trajectories versus the eight IMS censored trajectories at the same observation horizon",
            "status": "post_hoc_mechanism_audit",
            "confound_note": "cross-axis statistics are unavailable for six of eight healthy bearings and are therefore descriptive",
        },
        "rows": rows,
    }


def mechanism_features_at_snapshot(bearing: dict, horizon: int) -> dict:
    index = horizon - 1
    cumulative = np.cumsum(bearing["excess"], axis=1)
    return {
        "historical_rate": float(
            np.max(bearing["historical_matrix"][:, index])
        ),
        "cross_axis": float(bearing["cross_axis"][index]),
        "coactivation_fraction": float(
            np.mean(np.sum(bearing["active"][:, :horizon], axis=0) >= 2)
        ),
        "max_ratio": float(np.max(bearing["ratios"][:, :horizon])),
        "cumulative_excess": float(np.max(cumulative[:, index])),
    }


def main() -> None:
    streams = probe.load_all_streams(ROOT / "data" / "features")
    bearings = mechanism.build_bearings(streams)
    censored = [bearing for bearing in bearings if not bearing["failed"]]
    threshold = (
        max(np.max(bearing["historical_rate"]) for bearing in censored)
        * THRESHOLD_MARGIN
    )
    confirmation, confirmation_frame = confirmation_and_fleet_module(
        bearings, threshold
    )
    result = {
        "experiment_status": "deployment_scenario_and_stress_analysis_only",
        "threshold": threshold,
        "confirmation_and_fleet": confirmation,
        "maintenance_cost": maintenance_cost_module(confirmation_frame),
        "low_power_robustness": low_power_module(bearings, threshold),
        "conformal_capacity": conformal_capacity_module(bearings),
        "time_resolved_mechanism": time_resolved_module(bearings),
    }
    output = ROOT / "results" / "results_run_to_failure_deployment_value_experiments.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    confirmation_frame.to_csv(
        ROOT / "results" / "results_deployment_confirmation.csv", index=False
    )
    print(json.dumps(result, indent=2))
    print("saved", output)


if __name__ == "__main__":
    main()
