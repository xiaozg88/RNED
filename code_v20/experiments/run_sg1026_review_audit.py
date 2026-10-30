#!/usr/bin/env python3
"""Deterministic audits requested by the sg1026 review.

The script uses only the frozen feature tables. It does not retune the primary
detector and writes one JSON audit used by the v17 manuscript.
"""
from __future__ import annotations

import itertools
import json
import math
import os
import sys
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "experiments"))
import run_to_failure_classical_baselines as classical  # noqa: E402


def _rankdata(values: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    i = 0
    while i < order.size:
        j = i
        while j + 1 < order.size and values[order[j + 1]] == values[order[i]]:
            j += 1
        average_rank = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = average_rank
        i = j + 1
    return ranks


def directional_auc(labels: Sequence[bool], scores: Sequence[float]) -> float:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    ranks = _rankdata(scores)
    positive = int(labels.sum())
    negative = int(labels.size - positive)
    if positive == 0 or negative == 0:
        return float("nan")
    return float((ranks[labels].sum() - positive * (positive + 1) / 2.0) / (positive * negative))


def exact_two_sided_auc_p(labels: Sequence[bool], scores: Sequence[float]) -> float:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    observed = directional_auc(labels, scores)
    n_positive = int(labels.sum())
    indices = range(labels.size)
    null = [directional_auc(np.array([i in subset for i in indices]), scores) for subset in itertools.combinations(indices, n_positive)]
    return float(np.mean(np.abs(np.asarray(null) - 0.5) >= abs(observed - 0.5) - 1e-12))


def score_arrays(rms: np.ndarray) -> Dict[str, np.ndarray]:
    baseline = float(np.median(rms[10:60]))
    ratio = rms[60:] / baseline
    excess = np.maximum(ratio - 1.2, 0.0)
    prefix = np.concatenate(([0.0], np.cumsum(excess)))
    return {"baseline": baseline, "ratio": ratio, "excess": excess, "prefix": prefix}


def rate_score(rms: np.ndarray) -> np.ndarray:
    sequence = score_arrays(rms)
    return sequence["prefix"][1:] / np.arange(1, sequence["excess"].size + 1)


def binary_occupancy_score(rms: np.ndarray) -> np.ndarray:
    sequence = score_arrays(rms)
    indicator = (sequence["excess"] > 0).astype(float)
    return np.cumsum(indicator) / np.arange(1, indicator.size + 1)


def rolling_mean_score(values: np.ndarray, window: int) -> np.ndarray:
    prefix = np.concatenate(([0.0], np.cumsum(values)))
    index = np.arange(1, values.size + 1)
    start = np.maximum(0, index - window)
    denominator = np.minimum(index, window).astype(float)
    return (prefix[index] - prefix[start]) / denominator


def run_rule_score(rms: np.ndarray, window: int) -> np.ndarray:
    sequence = score_arrays(rms)
    indicator = sequence["excess"] > 0
    run_length = np.zeros(indicator.size, dtype=float)
    current = 0
    for index, active in enumerate(indicator):
        current = current + 1 if active else 0
        run_length[index] = current
    return run_length / float(window)


def normalized_cusum_score(rms: np.ndarray, delta: float) -> np.ndarray:
    sequence = score_arrays(rms)
    increments = sequence["excess"] - delta
    accumulator = 0.0
    score = np.empty(increments.size, dtype=float)
    for index, increment in enumerate(increments):
        accumulator = max(0.0, accumulator + increment)
        score[index] = accumulator
    return score / np.arange(1, score.size + 1)


def ewma_excess_score(rms: np.ndarray, decay: float) -> np.ndarray:
    sequence = score_arrays(rms)
    weight = 1.0 - decay
    score = np.empty(sequence["excess"].size, dtype=float)
    state = 0.0
    for index, value in enumerate(sequence["excess"]):
        state = weight * value + decay * state
        score[index] = state
    return score


def build_bearings(streams: List[dict], score_fn: Callable[[np.ndarray], np.ndarray]) -> List[dict]:
    bearings: Dict[str, dict] = {}
    for stream in streams:
        score = score_fn(stream["rms"])
        if score.size < 10:
            continue
        bearing = bearings.setdefault(
            stream["bearing_id"],
            {
                "bearing_id": stream["bearing_id"],
                "dataset": stream["dataset"],
                "failed": bool(stream["failed"]),
                "axes": {},
            },
        )
        bearing["axes"][str(stream["axis"])] = score
    return list(bearings.values())


def alarm_index(score: np.ndarray, threshold: float, minimum_updates: int = 30) -> int:
    start = max(0, minimum_updates - 1)
    hits = np.flatnonzero(score[start:] > threshold) + start
    return int(hits[0]) if hits.size else int(score.size)


def evaluate_bearings(bearings: List[dict]) -> dict:
    censored = [bearing for bearing in bearings if not bearing["failed"]]
    threshold = 1.01 * max(float(np.max(score)) for bearing in censored for score in bearing["axes"].values())
    result = {
        "zero_fa_threshold": threshold,
        "n_detected": 0,
        "total_n": 0,
        "per_dataset": {},
        "leads_pct": [],
    }
    for dataset in sorted({bearing["dataset"] for bearing in bearings}):
        failed = [bearing for bearing in bearings if bearing["dataset"] == dataset and bearing["failed"]]
        detected = 0
        leads: List[float] = []
        for bearing in failed:
            axis_hits = {
                axis: alarm_index(score, threshold)
                for axis, score in bearing["axes"].items()
            }
            alarmed_axes = [axis for axis, hit in axis_hits.items() if hit < bearing["axes"][axis].size]
            if alarmed_axes:
                detected += 1
                first = min(axis_hits[axis] for axis in alarmed_axes)
                leads.append(100.0 * (max(score.size for score in bearing["axes"].values()) - first) / max(score.size for score in bearing["axes"].values()))
        result["n_detected"] += detected
        result["total_n"] += len(failed)
        result["leads_pct"].extend(leads)
        result["per_dataset"][dataset] = {
            "n": len(failed),
            "n_detected": detected,
            "median_lead_pct_life": float(np.median(leads)) if leads else None,
        }
    result["median_lead_pct_life"] = float(np.median(result.pop("leads_pct"))) if result["leads_pct"] else None
    return result


def occupancy_and_horizon_audit(streams: List[dict]) -> dict:
    ims_streams = [stream for stream in streams if stream["dataset"] == "ims"]
    minimum_monitor = min(stream["rms"].size - 60 for stream in ims_streams)
    bearings: Dict[str, dict] = {}
    for stream in ims_streams:
        sequence = score_arrays(stream["rms"])
        full_length = sequence["excess"].size
        fixed = sequence["excess"][:minimum_monitor]
        fixed_prefix = np.concatenate(([0.0], np.cumsum(fixed)))
        bearing = bearings.setdefault(stream["bearing_id"], {"failed": bool(stream["failed"]), "axes": []})
        bearing["axes"].append(
            {
                "full_length": full_length,
                "full_rate": float(sequence["excess"].mean()),
                "fixed_rate": float(fixed.mean()),
                "full_occupancy": float(np.mean(sequence["excess"] > 0)),
                "fixed_occupancy": float(np.mean(fixed > 0)),
                "full_cumulative": float(sequence["excess"].sum()),
                "fixed_cumulative": float(fixed.sum()),
                "full_positive_mean": float(sequence["excess"][sequence["excess"] > 0].mean()) if np.any(sequence["excess"] > 0) else 0.0,
                "fixed_positive_mean": float(fixed[fixed > 0].mean()) if np.any(fixed > 0) else 0.0,
            }
        )
    rows = []
    for bearing_id, bearing in sorted(bearings.items()):
        row = {"bearing_id": bearing_id, "failed": bearing["failed"]}
        for key in next(iter(bearing["axes"])):
            if key != "full_length":
                row[key] = max(axis[key] for axis in bearing["axes"])
        row["full_length"] = int(min(axis["full_length"] for axis in bearing["axes"]))
        rows.append(row)
    labels = [row["failed"] for row in rows]
    metrics = {}
    for key in ("full_cumulative", "fixed_cumulative", "full_rate", "fixed_rate", "full_occupancy", "fixed_occupancy", "full_positive_mean", "fixed_positive_mean"):
        scores = [row[key] for row in rows]
        metrics[key] = {
            "auc_failed_higher": directional_auc(labels, scores),
            "exact_two_sided_permutation_p": exact_two_sided_auc_p(labels, scores),
            "failed_median": float(np.median([score for score, label in zip(scores, labels) if label])),
            "censored_median": float(np.median([score for score, label in zip(scores, labels) if not label])),
        }
    return {
        "status": "exploratory_mechanism_audit",
        "common_fixed_monitor_snapshots": minimum_monitor,
        "failed_median_full_monitor_length": float(np.median([row["full_length"] for row in rows if row["failed"]])),
        "censored_median_full_monitor_length": float(np.median([row["full_length"] for row in rows if not row["failed"]])),
        "metrics": metrics,
        "per_bearing": rows,
        "decomposition_note": "R = mean(1(e>0)) * mean(e | e>0); binary occupancy and positive excess magnitude are audited separately.",
    }


def leave_one_ims_experiment_out(streams: List[dict]) -> dict:
    bearings = build_bearings(streams, rate_score)
    groups = sorted({bearing["bearing_id"].split("_b")[0] for bearing in bearings if bearing["dataset"] == "ims"})
    details = []
    for held_group in groups:
        calibration = [bearing for bearing in bearings if bearing["dataset"] == "ims" and not bearing["failed"] and not bearing["bearing_id"].startswith(held_group)]
        held = [bearing for bearing in bearings if bearing["dataset"] == "ims" and bearing["bearing_id"].startswith(held_group)]
        threshold = 1.01 * max(float(np.max(score)) for bearing in calibration for score in bearing["axes"].values())
        false_alarms = sum(
            any(alarm_index(score, threshold) < score.size for score in bearing["axes"].values())
            for bearing in held
            if not bearing["failed"]
        )
        n_censored = sum(not bearing["failed"] for bearing in held)
        detections = sum(
            any(alarm_index(score, threshold) < score.size for score in bearing["axes"].values())
            for bearing in held
            if bearing["failed"]
        )
        n_failed = sum(bearing["failed"] for bearing in held)
        details.append(
            {
                "held_out_ims_experiment": held_group,
                "n_calibration_censored_bearings": len(calibration),
                "held_out_threshold": threshold,
                "held_out_false_alarms": false_alarms,
                "n_held_out_censored": n_censored,
                "held_out_failed_detected": detections,
                "n_held_out_failed": n_failed,
            }
        )
    return {
        "method": "leave-one-IMS-experiment-out threshold calibration",
        "total_held_out_false_alarms": sum(row["held_out_false_alarms"] for row in details),
        "total_held_out_censored": sum(row["n_held_out_censored"] for row in details),
        "details": details,
    }


def tuned_baseline_audit(streams: List[dict]) -> dict:
    variants: List[Tuple[str, dict]] = [("excess_rate", {"parameter": None, "fn": rate_score})]
    for delta in (0.0, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02):
        variants.append((f"normalized_cusum", {"parameter": delta, "fn": lambda rms, delta=delta: normalized_cusum_score(rms, delta)}))
    for decay in (0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.975):
        variants.append(("ewma_excess", {"parameter": decay, "fn": lambda rms, decay=decay: ewma_excess_score(rms, decay)}))
    for window in (3, 5, 10, 20, 50, 100):
        variants.append(("moving_average_excess", {"parameter": window, "fn": lambda rms, window=window: rolling_mean_score(score_arrays(rms)["excess"], window)}))
        variants.append(("binary_rolling_occupancy", {"parameter": window, "fn": lambda rms, window=window: rolling_mean_score((score_arrays(rms)["excess"] > 0).astype(float), window)}))
        variants.append(("run_rule", {"parameter": window, "fn": lambda rms, window=window: run_rule_score(rms, window)}))
    variants.append(("binary_lifetime_occupancy", {"parameter": None, "fn": binary_occupancy_score}))

    by_family: Dict[str, List[dict]] = {}
    for family, specification in variants:
        bearings = build_bearings(streams, specification["fn"])
        evaluation = evaluate_bearings(bearings)
        row = {
            "family": family,
            "parameter": specification["parameter"],
            **evaluation,
        }
        by_family.setdefault(family, []).append(row)
    selected = {}
    for family, candidates in by_family.items():
        if len(candidates) == 1:
            selected[family] = candidates[0]
        else:
            selected[family] = max(
                candidates,
                key=lambda row: (
                    row["per_dataset"]["ims"]["n_detected"],
                    row["per_dataset"]["ims"]["median_lead_pct_life"] if row["per_dataset"]["ims"]["median_lead_pct_life"] is not None else -1.0,
                    -float(row["parameter"]),
                ),
            )
    return {
        "status": "post-review baseline audit",
        "tuning_rule": "For each family, choose the parameter maximizing IMS detections, then IMS median lead; thresholds are always calibrated on all 8 IMS censored bearings. This favors baselines and is not independent validation.",
        "selection_tie_break": "If IMS detections and IMS median lead remain tied, v1 implicitly chooses the smaller numeric parameter. For EWMA-of-excess, decay=0.3 and 0.5 tie on IMS but have total detections 27 and 28, respectively; see results_sg1026_adverse_result_reaudit.json.",
        "selected": selected,
        "all_candidates": by_family,
    }


def fleet_sensitivity() -> dict:
    p = 10.0 / 54.0
    machines = 100
    rows = []
    for correlation in (-0.22, -0.10, 0.0, 0.25, 0.50, 0.75, 1.0):
        feasible = max(-p / (1.0 - p), -1.0) <= correlation <= 1.0
        p_alarm = 2.0 * p - p * p - correlation * p * (1.0 - p) if feasible else float("nan")
        rows.append(
            {
                "within_bearing_axis_correlation": correlation,
                "feasible": feasible,
                "probability_machine_alarms": p_alarm,
                "expected_machines_with_at_least_one_alarm": machines * p_alarm if feasible else None,
            }
        )
    return {
        "status": "model_based_sensitivity_not_field_measurement",
        "axis_level_early_life_exceedance_proxy_p": p,
        "machines": machines,
        "axes_per_machine": 2,
        "expected_axis_level_alarm_events_100_machines_under_fixed_marginal": 200.0 * p,
        "independent_axis_point_estimate_expected_alarms": machines * (2 * p - p * p),
        "axis_correlation_sensitivity": rows,
        "bearing_level_bootstrap_ci_from_existing_audit": [11.0, 67.0],
    }


def communication_budget() -> dict:
    frame_bytes = 4
    event_bytes = 64
    intervals = {"ims": 600.0, "xjtu": 60.0, "femto": 10.0}
    by_dataset: Dict[str, List[float]] = {"ims": [], "xjtu": [], "femto": []}
    feature_dir = ROOT / "data" / "features"
    import pandas as pd

    for filename in ("test1.csv", "test2.csv", "test3.csv"):
        frame = pd.read_csv(feature_dir / filename)
        for _, group in frame.groupby(["filename", "bearing"]):
            by_dataset["ims"].append(float(group["n_rows"].sum() * frame_bytes))
    for filename, dataset in (("xjtu_features.csv", "xjtu"), ("femto_features.csv", "femto")):
        frame = pd.read_csv(feature_dir / filename)
        for _, group in frame.groupby(["group", "bearing", "snapshot"]):
            if len(group) > 0:
                by_dataset[dataset].append(float(group["n_rows"].sum() * frame_bytes))
    return {
        "status": "deterministic_packet_accounting_no_network_measurement",
        "assumed_raw_sample_format": "float32",
        "assumed_alarm_event_packet_bytes": event_bytes,
        "per_dataset": {
            dataset: {
                "median_raw_bytes_per_snapshot_across_axes": float(np.median(values)),
                "snapshot_interval_seconds": interval,
                "median_raw_bytes_per_second": float(np.median(values) / interval,
                ),
                "one_event_packet_bytes": event_bytes,
                "median_raw_snapshot_to_event_packet_ratio": float(np.median(values) / event_bytes),
            }
            for dataset, interval, values in (
                ("ims", intervals["ims"], by_dataset["ims"]),
                ("xjtu", intervals["xjtu"], by_dataset["xjtu"]),
                ("femto", intervals["femto"], by_dataset["femto"]),
            )
        },
    }


def ldm_alarm_audit() -> dict:
    path = ROOT / "results" / "results_run_to_failure_public_locked_validation_ldm_upb.json"
    result = json.loads(path.read_text())
    per_bearing = result["per_bearing"]
    earliest = sum(row["historical_alarm_index"] == 29 for row in per_bearing)
    return {
        "status": "frozen_alarm_coverage_not_event_detection",
        "n_eligible": result["n_protocol_eligible"],
        "historical_alarm_coverage": result["historical_detected"],
        "median_monitor_segment_lead_pct": result["median_hierarchical_lead_pct_life"],
        "alarms_at_earliest_allowed_update": earliest,
        "fault_onset_available": False,
        "interpretation": "The metric is alarm coverage under a locked rule; without a prespecified fault onset it cannot establish early fault detection.",
    }


def main() -> None:
    data_dir = Path(os.environ.get("RTF_DATA_DIR", str(ROOT / "data" / "features")))
    streams = classical.load_all_streams(data_dir)
    summary = {
        "script_version": "sg1026_review_audit_v1_1",
        "occupancy_and_horizon": occupancy_and_horizon_audit(streams),
        "ims_leave_one_experiment_out": leave_one_ims_experiment_out(streams),
        "tuned_baselines": tuned_baseline_audit(streams),
        "fleet_sensitivity": fleet_sensitivity(),
        "communication_budget": communication_budget(),
        "ldm_alarm_audit": ldm_alarm_audit(),
    }
    destination = ROOT / "results" / "results_sg1026_review_audit.json"
    destination.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(f"saved {destination}")
    print(json.dumps({key: summary[key] for key in ("ims_leave_one_experiment_out", "ldm_alarm_audit")}, indent=2))
    print("selected baselines:")
    for family, row in summary["tuned_baselines"]["selected"].items():
        print(f"  {family:28s} parameter={row['parameter']} detected={row['n_detected']}/{row['total_n']}")


if __name__ == "__main__":
    main()
