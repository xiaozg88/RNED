#!/usr/bin/env python3
"""Explore mechanism- and value-oriented extensions of the RMS-rate detector.

All results produced by this script are exploratory. They must not replace the
paper's frozen historical-rate result without independent, pre-registered
validation. The experiment deliberately fixes the recent window at 10 snapshots
and the composite weights before evaluating failure trajectories.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "experiments"))
import run_to_failure_early_warning as probe  # noqa: E402

WINDOW = 10
MIN_MONITOR = 30
CALIBRATION_MARGIN = 1.01
COMPOSITE_WEIGHTS = {
    "historical_rate": 0.30,
    "persistence": 0.20,
    "irreversibility": 0.20,
    "cross_axis": 0.20,
    "acceleration": 0.10,
}


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if values.size == 0:
        return np.asarray([])
    out = np.empty_like(values, dtype=np.float64)
    cumulative = np.cumsum(values)
    for index in range(values.size):
        start = max(0, index - window + 1)
        denominator = index - start + 1
        total = cumulative[index] - (cumulative[start - 1] if start else 0.0)
        out[index] = total / denominator
    return out


def largest_or_secondmax(matrix: np.ndarray, rank: int) -> np.ndarray:
    if matrix.shape[0] == 0:
        return np.zeros(matrix.shape[1], dtype=np.float64)
    if rank == 1:
        return np.max(matrix, axis=0)
    if matrix.shape[0] < rank:
        return np.zeros(matrix.shape[1], dtype=np.float64)
    return np.sort(matrix, axis=0)[-rank, :]


def safe_mad(values: np.ndarray) -> float:
    mad = float(np.median(np.abs(values - np.median(values))))
    return max(1e-12, 1.4826 * mad)


def build_bearings(streams: list[dict]) -> list[dict]:
    grouped: dict[str, dict] = {}
    for stream in streams:
        entry = grouped.setdefault(
            stream["bearing_id"],
            {
                "bearing_id": stream["bearing_id"],
                "dataset": stream["dataset"],
                "failed": stream["failed"],
                "axes": [],
            },
        )
        entry["axes"].append(stream)

    bearings = []
    for bearing_id in sorted(grouped):
        source = grouped[bearing_id]
        lengths = {stream["rms"].size for stream in source["axes"]}
        if len(lengths) != 1:
            raise ValueError(f"Unequal axis lengths for {bearing_id}: {lengths}")
        monitor_length = next(iter(lengths)) - 60
        if monitor_length < 10:
            continue

        rms = np.vstack([stream["rms"] for stream in source["axes"]])
        baselines = np.asarray(
            [float(np.median(axis[10:60])) for axis in rms], dtype=np.float64
        )
        ratios = rms[:, 60:] / baselines[:, None]
        excess = np.maximum(ratios - 1.2, 0.0)
        active = (ratios > 1.2).astype(np.float64)

        historical = np.cumsum(excess, axis=1) / np.arange(
            1, excess.shape[1] + 1
        )
        historical_rate = np.max(historical, axis=0)
        persistence_axis = np.vstack(
            [rolling_mean(row, WINDOW) for row in active]
        )
        persistence = largest_or_secondmax(persistence_axis, 2)
        max_persistence = np.max(persistence_axis, axis=0)

        running_max = np.maximum.accumulate(ratios, axis=1)
        drop = np.maximum(running_max - ratios, 0.0) / np.maximum(
            running_max, 1e-12
        )
        irreversibility = np.max(1.0 - drop, axis=0)

        recent_axis = np.vstack([rolling_mean(row, WINDOW) for row in excess])
        cross_axis = largest_or_secondmax(recent_axis, 2)
        max_recent = np.max(recent_axis, axis=0)
        acceleration = max_recent - historical_rate
        coactivation_count = np.sum(active, axis=0)
        coactivation_fraction = np.mean(coactivation_count >= 2)

        first_active = np.full(ratios.shape[0], ratios.shape[1], dtype=int)
        for axis_index in range(ratios.shape[0]):
            hits = np.flatnonzero(active[axis_index])
            if hits.size:
                first_active[axis_index] = int(hits[0])
        ordered_onsets = np.sort(first_active)
        propagation_delay = (
            int(ordered_onsets[1] - ordered_onsets[0])
            if ratios.shape[0] >= 2 and ordered_onsets[1] < ratios.shape[1]
            else int(ratios.shape[1])
        )

        robust_baselines = np.asarray(
            [float(np.median(axis[10:60])) for axis in rms]
        )
        robust_scales = np.asarray(
            [safe_mad(axis[10:60]) for axis in rms], dtype=np.float64
        )
        robust_z = (rms[:, 60:] - robust_baselines[:, None]) / robust_scales[:, None]
        robust_excess = np.maximum(robust_z, 0.0)
        robust_historical = np.cumsum(robust_excess, axis=1) / np.arange(
            1, robust_excess.shape[1] + 1
        )
        robust_rate = np.max(robust_historical, axis=0)
        normalized_threshold_z = 0.2 * baselines / robust_scales
        normalized_threshold_excess = np.maximum(
            robust_z - normalized_threshold_z[:, None], 0.0
        )
        normalized_threshold_historical = np.cumsum(
            normalized_threshold_excess, axis=1
        ) / np.arange(1, normalized_threshold_excess.shape[1] + 1)
        normalized_threshold_z_rate = np.max(
            normalized_threshold_historical, axis=0
        )

        bearings.append(
            {
                **source,
                "n_monitor": int(monitor_length),
                "baselines": baselines,
                "ratios": ratios,
                "excess": excess,
                "active": active,
                "historical_matrix": historical,
                "historical_rate": historical_rate,
                "persistence": persistence,
                "max_persistence": max_persistence,
                "irreversibility": irreversibility,
                "cross_axis": cross_axis,
                "acceleration": acceleration,
                "coactivation_count": coactivation_count,
                "coactivation_fraction": float(coactivation_fraction),
                "first_active": first_active,
                "propagation_delay": propagation_delay,
                "robust_rate": robust_rate,
                "normalized_threshold_z_rate": normalized_threshold_z_rate,
            }
        )
    return bearings


def alarm_index(score: np.ndarray, threshold: float) -> int:
    return probe.alarm_index(score, threshold, min_window=MIN_MONITOR)


def score_maxima(bearings: list[dict]) -> dict[str, float]:
    censored = [bearing for bearing in bearings if not bearing["failed"]]
    return {
        name: float(
            max(np.max(bearing[name]) for bearing in censored)
        )
        for name in COMPOSITE_WEIGHTS
    }


def composite_score(bearing: dict, maxima: dict[str, float]) -> np.ndarray:
    score = np.zeros(bearing["n_monitor"], dtype=np.float64)
    for name, weight in COMPOSITE_WEIGHTS.items():
        score += weight * bearing[name] / max(maxima[name], 1e-12)
    return score


def summarize_alarm(
    bearings: list[dict],
    alarm_fn: Callable[[dict], int],
    threshold: float | None = None,
) -> dict:
    failed = [bearing for bearing in bearings if bearing["failed"]]
    censored = [bearing for bearing in bearings if not bearing["failed"]]
    alarms: dict[str, int] = {}
    detected = []
    false_alarms = []
    leads: dict[str, float] = {}
    for bearing in failed + censored:
        alarm = int(alarm_fn(bearing))
        alarms[bearing["bearing_id"]] = alarm
        if alarm < bearing["n_monitor"]:
            if bearing["failed"]:
                detected.append(bearing)
                leads[bearing["bearing_id"]] = 100.0 * (
                    bearing["n_monitor"] - alarm
                ) / bearing["n_monitor"]
            else:
                false_alarms.append(bearing)
    per_dataset = {}
    for dataset in sorted({bearing["dataset"] for bearing in failed}):
        subset = [bearing for bearing in failed if bearing["dataset"] == dataset]
        detected_subset = [bearing for bearing in subset if bearing["bearing_id"] in leads]
        dataset_leads = [leads[bearing["bearing_id"]] for bearing in detected_subset]
        per_dataset[dataset] = {
            "n": len(subset),
            "n_detected": len(detected_subset),
            "median_lead_pct_life": float(np.median(dataset_leads))
            if dataset_leads
            else None,
        }
    all_leads = list(leads.values())
    return {
        **({"threshold": float(threshold)} if threshold is not None else {}),
        "n_failed": len(failed),
        "n_detected": len(detected),
        "n_censored": len(censored),
        "n_false_alarms": len(false_alarms),
        "median_lead_pct_life": float(np.median(all_leads))
        if all_leads
        else None,
        "per_dataset": per_dataset,
        "missed_bearings": [
            bearing["bearing_id"] for bearing in failed if bearing not in detected
        ],
        "alarm_by_bearing": alarms,
    }


def mechanism_path(bearings: list[dict]) -> tuple[dict, list[dict]]:
    maxima = score_maxima(bearings)
    thresholds = {
        name: value * CALIBRATION_MARGIN for name, value in maxima.items()
    }

    methods = {
        "historical_rate": lambda bearing: alarm_index(
            bearing["historical_rate"], thresholds["historical_rate"]
        ),
        "cross_axis_recent": lambda bearing: alarm_index(
            bearing["cross_axis"], thresholds["cross_axis"]
        ),
        "persistence_only": lambda bearing: alarm_index(
            bearing["persistence"], thresholds["persistence"]
        ),
        "irreversibility_only": lambda bearing: alarm_index(
            bearing["irreversibility"], thresholds["irreversibility"]
        ),
        "acceleration_only": lambda bearing: alarm_index(
            bearing["acceleration"], thresholds["acceleration"]
        ),
    }

    def hierarchical_alarm(bearing: dict) -> int:
        return min(
            methods["historical_rate"](bearing),
            methods["cross_axis_recent"](bearing),
        )

    def event_or_alarm(bearing: dict) -> int:
        return min(
            methods["historical_rate"](bearing),
            methods["cross_axis_recent"](bearing),
            methods["persistence_only"](bearing),
            methods["irreversibility_only"](bearing),
        )

    composite_threshold = (
        max(
            np.max(composite_score(bearing, maxima))
            for bearing in bearings
            if not bearing["failed"]
        )
        * CALIBRATION_MARGIN
    )
    methods["hierarchical_h_or_cross"] = hierarchical_alarm
    methods["event_or"] = event_or_alarm
    methods["fixed_weight_composite"] = (
        lambda bearing: alarm_index(
            composite_score(bearing, maxima), composite_threshold
        )
    )

    summary = {
        "protocol": {
            "window": WINDOW,
            "minimum_monitor_snapshots": MIN_MONITOR,
            "calibration_margin": CALIBRATION_MARGIN,
            "weights": COMPOSITE_WEIGHTS,
            "status": "exploratory_post_hoc",
        },
        "thresholds": thresholds,
        "composite_threshold": float(composite_threshold),
        "results": {
            name: summarize_alarm(
                bearings,
                alarm_fn,
                thresholds.get(name, composite_threshold),
            )
            for name, alarm_fn in methods.items()
        },
    }

    early_life_proxy = {}
    for name, alarm_fn in methods.items():
        pseudo_health = [
            bearing
            for bearing in bearings
            if bearing["dataset"] in {"xjtu", "femto"}
            and bearing["n_monitor"] >= 90
        ]
        hits = []
        per_dataset = defaultdict(lambda: {"n": 0, "hits": 0})
        for bearing in pseudo_health:
            hit = bool(alarm_fn(bearing) < 90)
            hits.append(hit)
            per_dataset[bearing["dataset"]]["n"] += 1
            per_dataset[bearing["dataset"]]["hits"] += int(hit)
        early_life_proxy[name] = {
            "n_bearings": len(hits),
            "n_alarms": int(np.sum(hits)),
            "alarm_rate_pct": 100.0 * float(np.mean(hits)),
            "per_dataset": dict(per_dataset),
        }
    summary["early_life_pseudo_health"] = early_life_proxy

    profiles = mechanism_profiles(bearings)
    return summary, profiles


def statistic_with_auc(
    failed: list[float], censored: list[float], alternative: str
) -> dict:
    auc = roc_auc_score(
        [1] * len(failed) + [0] * len(censored), failed + censored
    )
    if alternative == "less":
        auc = 1.0 - auc
    test = mannwhitneyu(
        failed, censored, alternative=alternative, method="exact"
    )
    return {
        "failed_median": float(np.median(failed)),
        "censored_median": float(np.median(censored)),
        "auc_failed_higher": float(auc),
        "mann_whitney_p": float(test.pvalue),
    }


def mechanism_profiles(bearings: list[dict]) -> list[dict]:
    profiles = []
    for bearing in bearings:
        first = np.min(bearing["first_active"])
        profiles.append(
            {
                "bearing_id": bearing["bearing_id"],
                "dataset": bearing["dataset"],
                "failed": bearing["failed"],
                "n_monitor": bearing["n_monitor"],
                "max_historical_rate": float(np.max(bearing["historical_rate"])),
                "max_recent_rate": float(
                    np.max(largest_or_secondmax(bearing["excess"], 1))
                ),
                "max_second_axis_recent_rate": float(
                    np.max(bearing["cross_axis"])
                ),
                "max_persistence": float(np.max(bearing["max_persistence"])),
                "max_second_axis_persistence": float(
                    np.max(bearing["persistence"])
                ),
                "final_irreversibility": float(bearing["irreversibility"][-1]),
                "coactivation_fraction": bearing["coactivation_fraction"],
                "first_active_fraction": float(first / bearing["n_monitor"]),
                "propagation_delay_snapshots": bearing["propagation_delay"],
                "max_ratio": float(np.max(bearing["ratios"])),
            }
        )
    return profiles


def mechanism_separation(
    profiles: list[dict], dataset: str | None = None
) -> dict:
    frame = pd.DataFrame(profiles)
    if dataset is not None:
        failed = frame[(frame["failed"]) & (frame["dataset"] == dataset)]
        censored = frame[~frame["failed"]]
    else:
        failed = frame[frame["failed"]]
        censored = frame[~frame["failed"]]
    higher_features = [
        "max_historical_rate",
        "max_second_axis_recent_rate",
        "max_persistence",
        "max_second_axis_persistence",
        "final_irreversibility",
        "coactivation_fraction",
        "max_ratio",
    ]
    lower_features = [
        "first_active_fraction",
        "propagation_delay_snapshots",
    ]
    output = {}
    for feature in higher_features:
        output[feature] = statistic_with_auc(
            failed[feature].tolist(), censored[feature].tolist(), "greater"
        )
    for feature in lower_features:
        output[feature] = statistic_with_auc(
            failed[feature].tolist(), censored[feature].tolist(), "less"
        )
    return output


def conformal_path(bearings: list[dict]) -> dict:
    maxima = score_maxima(bearings)
    censored = [bearing for bearing in bearings if not bearing["failed"]]
    output = {
        "protocol": {
            "unit": "full independent healthy bearing trajectory",
            "statistic": "full-trajectory maximum",
            "status": "capacity_audit_and_loco_exploratory",
        },
        "finite_sample_capacity": {
            "available_healthy_bearings": len(censored),
            "single_layer_minimum_alpha_0p125": 7,
            "bonferroni_two_layers_alpha_0p20": {
                "per_layer_alpha": 0.10,
                "minimum_per_layer": 9,
                "defined": len(censored) >= 9,
            },
            "bonferroni_two_layers_alpha_0p10": {
                "per_layer_alpha": 0.05,
                "minimum_per_layer": 19,
                "defined": len(censored) >= 19,
            },
        },
        "leave_one_censored_out": {},
    }

    def score_for(bearing: dict, name: str) -> np.ndarray:
        if name == "fixed_weight_composite":
            return composite_score(bearing, maxima)
        return bearing[name]

    for name in [
        "historical_rate",
        "cross_axis",
        "persistence",
        "irreversibility",
        "fixed_weight_composite",
    ]:
        false_alarms = 0
        rows = []
        for held_out in censored:
            calibration = [
                bearing for bearing in censored if bearing is not held_out
            ]
            calibration_maxima = sorted(
                float(np.max(score_for(bearing, name)))
                for bearing in calibration
            )
            alpha = 0.125
            order_index = int(np.ceil((len(calibration) + 1) * (1.0 - alpha)))
            threshold = calibration_maxima[
                min(order_index, len(calibration_maxima)) - 1
            ]
            held_out_max = float(np.max(score_for(held_out, name)))
            alarm = held_out_max > threshold
            false_alarms += int(alarm)
            rows.append(
                {
                    "bearing_id": held_out["bearing_id"],
                    "threshold": threshold,
                    "held_out_maximum": held_out_max,
                    "false_alarm": bool(alarm),
                }
            )
        output["leave_one_censored_out"][name] = {
            "n": len(censored),
            "false_alarms": false_alarms,
            "false_alarm_rate_pct": 100.0 * false_alarms / len(censored),
            "details": rows,
        }
    return output


def early_life_condition_path(bearings: list[dict]) -> dict:
    censored = [bearing for bearing in bearings if not bearing["failed"]]
    original_threshold = (
        max(np.max(bearing["historical_rate"]) for bearing in censored)
        * CALIBRATION_MARGIN
    )
    robust_threshold = (
        max(np.max(bearing["robust_rate"]) for bearing in censored)
        * CALIBRATION_MARGIN
    )
    normalized_threshold_z_threshold = (
        max(
            np.max(bearing["normalized_threshold_z_rate"])
            for bearing in censored
        )
        * CALIBRATION_MARGIN
    )
    pseudo_health = [
        bearing
        for bearing in bearings
        if bearing["dataset"] in {"xjtu", "femto"} and bearing["n_monitor"] >= 90
    ]
    result = {
        "protocol": {
            "idea": "normalize per-axis RMS deviations by early-life robust noise scale",
            "baseline": "median RMS over snapshots 10:60",
            "scale": "1.4826 * MAD over snapshots 10:60",
            "segment": "first 150 raw snapshots (= first 90 monitored snapshots)",
            "status": "exploratory",
        },
        "thresholds": {
            "original_ratio_excess": original_threshold,
            "robust_z_excess": robust_threshold,
            "normalized_threshold_z_excess": normalized_threshold_z_threshold,
        },
        "original": summarize_alarm(
            bearings,
            lambda bearing: alarm_index(
                bearing["historical_rate"], original_threshold
            ),
            original_threshold,
        ),
        "robust_z": summarize_alarm(
            bearings,
            lambda bearing: alarm_index(
                bearing["robust_rate"], robust_threshold
            ),
            robust_threshold,
        ),
        "normalized_threshold_z": summarize_alarm(
            bearings,
            lambda bearing: alarm_index(
                bearing["normalized_threshold_z_rate"],
                normalized_threshold_z_threshold,
            ),
            normalized_threshold_z_threshold,
        ),
        "early_life_pseudo_health": {},
    }
    for name, score_name, threshold in [
        ("original", "historical_rate", original_threshold),
        ("robust_z", "robust_rate", robust_threshold),
        (
            "normalized_threshold_z",
            "normalized_threshold_z_rate",
            normalized_threshold_z_threshold,
        ),
    ]:
        hits = []
        per_dataset = defaultdict(lambda: {"n": 0, "hits": 0})
        for bearing in pseudo_health:
            length = 90
            hit = bool(np.any(bearing[score_name][MIN_MONITOR - 1 : length] > threshold))
            hits.append(hit)
            per_dataset[bearing["dataset"]]["n"] += 1
            per_dataset[bearing["dataset"]]["hits"] += int(hit)
        result["early_life_pseudo_health"][name] = {
            "n_bearings": len(hits),
            "n_false_alarms": int(np.sum(hits)),
            "false_alarm_rate_pct": 100.0 * float(np.mean(hits)),
            "per_dataset": dict(per_dataset),
        }
    return result


def features_at_age(bearing: dict, age: int) -> np.ndarray:
    index = age - 1
    historical = bearing["historical_rate"][index]
    persistence = bearing["max_persistence"][index]
    second_persistence = bearing["persistence"][index]
    irreversibility = bearing["irreversibility"][index]
    cross_axis = bearing["cross_axis"][index]
    acceleration = bearing["acceleration"][index]
    active_fraction = float(np.mean(bearing["active"][:, :age]))
    max_ratio = float(np.max(bearing["ratios"][:, :age]))
    robust_rate = bearing["robust_rate"][index]
    return np.asarray(
        [
            historical,
            persistence,
            second_persistence,
            irreversibility,
            cross_axis,
            acceleration,
            active_fraction,
            max_ratio,
            robust_rate,
        ],
        dtype=np.float64,
    )


def survival_value_path(bearings: list[dict]) -> dict:
    failed = [bearing for bearing in bearings if bearing["failed"]]
    feature_names = [
        "historical_rate",
        "persistence",
        "second_axis_persistence",
        "irreversibility",
        "cross_axis",
        "acceleration",
        "active_fraction",
        "max_ratio",
        "robust_rate",
    ]
    output = {
        "protocol": {
            "task": "leave-one-bearing-out prediction of total life at fixed observed age",
            "model": "median baseline versus standardized ridge regression",
            "unit": "snapshot count within dataset",
            "status": "exploratory_mechanism_value",
            "features": feature_names,
        },
        "observations": [],
    }
    for dataset in sorted({bearing["dataset"] for bearing in failed}):
        subset = [bearing for bearing in failed if bearing["dataset"] == dataset]
        for age in (90, 150, 300, 600):
            eligible = [bearing for bearing in subset if bearing["n_monitor"] > age + 30]
            if len(eligible) < 10:
                continue
            baseline_errors = []
            model_errors = []
            actual = []
            predicted = []
            mechanism_features = []
            for held_out in eligible:
                training = [bearing for bearing in eligible if bearing is not held_out]
                x_train = np.vstack([features_at_age(bearing, age) for bearing in training])
                y_train = np.asarray([bearing["n_monitor"] for bearing in training])
                scaler = StandardScaler().fit(x_train)
                model = Ridge(alpha=10.0).fit(scaler.transform(x_train), y_train)
                baseline = float(np.median(y_train))
                prediction = float(
                    model.predict(
                        scaler.transform(features_at_age(held_out, age)[None, :])
                    )[0]
                )
                baseline_errors.append(abs(baseline - held_out["n_monitor"]))
                model_errors.append(abs(prediction - held_out["n_monitor"]))
                actual.append(held_out["n_monitor"])
                predicted.append(prediction)
                mechanism_features.append(features_at_age(held_out, age))
            feature_correlations = {
                name: float(
                    pd.Series(
                        [values[index] for values in mechanism_features]
                    ).corr(pd.Series(actual), method="spearman")
                )
                for index, name in enumerate(feature_names)
            }
            output["observations"].append(
                {
                    "dataset": dataset,
                    "observed_age_snapshots": age,
                    "n": len(eligible),
                    "baseline_mae": float(np.mean(baseline_errors)),
                    "mechanism_ridge_mae": float(np.mean(model_errors)),
                    "mae_reduction_pct": 100.0
                    * (np.mean(baseline_errors) - np.mean(model_errors))
                    / np.mean(baseline_errors),
                    "spearman_actual_predicted": float(
                        pd.Series(actual).corr(pd.Series(predicted), method="spearman")
                    ),
                    "feature_spearman_with_life": feature_correlations,
                }
            )
    return output


def phenotype_path(
    bearings: list[dict], profiles: list[dict], baseline_misses: set[str]
) -> dict:
    frame = pd.DataFrame(profiles)
    failed = frame[frame["failed"]].copy()
    features = [
        "max_historical_rate",
        "max_recent_rate",
        "max_second_axis_recent_rate",
        "max_persistence",
        "max_second_axis_persistence",
        "final_irreversibility",
        "coactivation_fraction",
        "first_active_fraction",
        "propagation_delay_snapshots",
        "max_ratio",
    ]
    scaler = StandardScaler().fit(failed[features])
    clustered = KMeans(n_clusters=3, n_init=50, random_state=0).fit(
        scaler.transform(failed[features])
    )
    failed["cluster"] = clustered.labels_
    failed["historical_missed"] = failed["bearing_id"].isin(baseline_misses)
    centers = {
        str(index): {
            feature: float(value)
            for feature, value in zip(features, center)
        }
        for index, center in enumerate(
            clustered.cluster_centers_
        )
    }
    cluster_rows = []
    for cluster in sorted(failed["cluster"].unique()):
        subset = failed[failed["cluster"] == cluster]
        row = {
            "cluster": int(cluster),
            "n": int(len(subset)),
            "datasets": subset["dataset"].value_counts().to_dict(),
            "historical_missed": int(subset["historical_missed"].sum()),
            "median_first_active_fraction": float(
                subset["first_active_fraction"].median()
            ),
            "median_coactivation_fraction": float(
                subset["coactivation_fraction"].median()
            ),
            "median_max_ratio": float(subset["max_ratio"].median()),
            "bearing_ids": subset["bearing_id"].tolist(),
        }
        cluster_rows.append(row)
    return {
        "protocol": {
            "unit": "failed bearing trajectory",
            "features": features,
            "model": "k-means, k=3, standardized full-trajectory summaries",
            "status": "descriptive_exploratory",
        },
        "centers_in_standardized_space": centers,
        "clusters": cluster_rows,
        "profiles": failed[
            [
                "bearing_id",
                "dataset",
                "cluster",
                "historical_missed",
                *features,
            ]
        ].to_dict(orient="records"),
    }


def main() -> None:
    features_dir = ROOT / "data" / "features"
    streams = probe.load_all_streams(features_dir)
    bearings = build_bearings(streams)
    mechanism, profiles = mechanism_path(bearings)
    baseline_misses = set(
        mechanism["results"]["historical_rate"]["missed_bearings"]
    )
    result = {
        "experiment_status": "exploratory; no paper-main upgrade without independent validation",
        "n_bearings_evaluated": len(bearings),
        "n_failed": sum(bearing["failed"] for bearing in bearings),
        "n_censored": sum(not bearing["failed"] for bearing in bearings),
        "priority_1_cross_axis_event_mechanism": mechanism,
        "priority_1_mechanism_separation": mechanism_separation(profiles),
        "priority_1_mechanism_separation_by_dataset": {
            dataset: mechanism_separation(profiles, dataset)
            for dataset in ("femto", "xjtu")
        },
        "priority_2_trajectory_conformal_capacity": conformal_path(bearings),
        "priority_3_robust_condition_normalization": early_life_condition_path(bearings),
        "priority_4_early_life_value_prediction": survival_value_path(bearings),
        "priority_5_failure_phenotypes": phenotype_path(
            bearings, profiles, baseline_misses
        ),
    }
    output = ROOT / "results" / "results_run_to_failure_mechanism_value_paths.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False))

    profile_output = ROOT / "results" / "results_run_to_failure_mechanism_profiles.csv"
    pd.DataFrame(profiles).to_csv(profile_output, index=False)
    print(json.dumps(result, indent=2))
    print("saved", output)
    print("saved", profile_output)


if __name__ == "__main__":
    main()
