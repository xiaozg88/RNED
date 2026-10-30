#!/usr/bin/env python3
"""Frozen SCA validation of block-wise anytime-valid rate detectors."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import argparse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FEATURE_CSV = ROOT / "data/features/sca_features.csv"
PREREGISTRATION = ROOT / "BLOCKWISE_ANYTIME_SCA_PREREGISTRATION.md"
OUTPUT = ROOT / "results/results_run_to_failure_blockwise_anytime_sca.json"

BASELINE_START = 10
BASELINE_END = 60
MIN_MONITOR = 30
EXCESS_RATIO = 1.2
HISTORICAL_THRESHOLD = 0.042407782315012556
BLOCK_SIZE = 10
EXCESS_BOUND = 0.20
NULL_MEAN = 0.05
ALPHA_CASE = 0.05
LAMBDAS = np.asarray([0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0])


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_streams(
    feature_csv: Path,
) -> dict[tuple[str, int], dict[str, dict[str, dict[str, object]]]]:
    streams: dict[tuple[str, int], dict[str, dict[str, dict[str, object]]]] = {}
    grouped: dict[tuple[str, int, str], list[dict[str, str]]] = defaultdict(list)
    with feature_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            grouped[(row["split"], int(row["case"]), row["axis"])].append(row)

    for (split, case, axis), rows in grouped.items():
        rows.sort(key=lambda row: int(row["snapshot"]))
        snapshots = [int(row["snapshot"]) for row in rows]
        if snapshots != list(range(1, len(rows) + 1)):
            raise ValueError(f"nonconsecutive snapshots: {split}/{case}/{axis}")
        streams.setdefault((split, case), {})[axis] = {
            "rms": np.asarray([float(row["rms"]) for row in rows], dtype=np.float64),
            "rpm": np.asarray([float(row["rpm"]) for row in rows], dtype=np.float64),
            "label": np.asarray([int(row["label"]) for row in rows], dtype=np.int64),
            "timestamp": [row["timestamp"] for row in rows],
            "n_snapshots": len(rows),
        }
    return streams


def condition_residual(values: np.ndarray, conditions: np.ndarray) -> tuple[np.ndarray, bool]:
    baseline_conditions = conditions[BASELINE_START:BASELINE_END]
    if float(np.std(baseline_conditions)) < 1.0:
        return values.copy(), False

    mean = float(np.mean(baseline_conditions))
    scale = float(np.std(baseline_conditions))
    standardized = (conditions - mean) / scale
    baseline_standardized = standardized[BASELINE_START:BASELINE_END]
    clipped = np.clip(
        standardized,
        float(np.min(baseline_standardized)),
        float(np.max(baseline_standardized)),
    )
    design = np.column_stack(
        [np.ones(values.size), clipped]
    )
    baseline_design = design[BASELINE_START:BASELINE_END]
    penalty = np.eye(baseline_design.shape[1]) * 0.01
    penalty[0, 0] = 0.0
    normal_matrix = baseline_design.T @ baseline_design + penalty
    if float(np.linalg.cond(normal_matrix)) > 1.0e6:
        return values.copy(), False

    baseline_log_values = np.log(
        np.maximum(values[BASELINE_START:BASELINE_END], np.finfo(float).tiny)
    )
    coefficients = np.linalg.solve(
        normal_matrix, baseline_design.T @ baseline_log_values
    )
    residual = np.log(np.maximum(values, np.finfo(float).tiny)) - design @ coefficients
    baseline_residual = float(np.median(residual[BASELINE_START:BASELINE_END]))
    normalized = np.exp(residual - baseline_residual)
    return normalized, True


def historical_alarm(excess: np.ndarray) -> int | None:
    rate = np.cumsum(excess) / np.arange(1, excess.size + 1)
    hits = np.flatnonzero(rate[MIN_MONITOR - 1 :] > HISTORICAL_THRESHOLD)
    if hits.size == 0:
        return None
    return int(hits[0] + MIN_MONITOR)


def blockwise_log_e_process(
    excess: np.ndarray,
    alpha_axis: float,
) -> tuple[float, list[float], int | None]:
    log_process = 0.0
    history: list[float] = []
    alarm_block: int | None = None
    for block_index, start in enumerate(range(0, excess.size, BLOCK_SIZE)):
        block = excess[start : start + BLOCK_SIZE]
        if block.size < BLOCK_SIZE:
            break
        block_sum = float(np.sum(block))
        exponents = (
            LAMBDAS * (block_sum - BLOCK_SIZE * NULL_MEAN)
            - BLOCK_SIZE * LAMBDAS ** 2 * EXCESS_BOUND ** 2 / 8.0
        )
        maximum = float(np.max(exponents))
        log_mixture = maximum + float(
            np.log(np.mean(np.exp(exponents - maximum)))
        )
        log_process += log_mixture
        history.append(log_process)
        if alarm_block is None and log_process > math.log(1.0 / alpha_axis):
            alarm_block = block_index
    return log_process, history, alarm_block


def axis_result(
    row: dict[str, object], use_residual: bool, alpha_axis: float
) -> dict[str, object]:
    values = np.asarray(row["rms"], dtype=np.float64)
    conditions = np.asarray(row["rpm"], dtype=np.float64)
    if use_residual:
        values, residual_used = condition_residual(values, conditions)
    else:
        residual_used = False

    baseline = float(np.median(values[BASELINE_START:BASELINE_END]))
    monitor_values = values[BASELINE_END:]
    raw_excess = np.maximum(monitor_values / baseline - EXCESS_RATIO, 0.0)
    bounded_excess = np.minimum(raw_excess, EXCESS_BOUND)
    rate_alarm = historical_alarm(raw_excess)
    log_e, log_e_history, alarm_block = blockwise_log_e_process(
        bounded_excess, alpha_axis
    )
    completed_blocks = [
        float(np.mean(bounded_excess[start : start + BLOCK_SIZE]))
        for start in range(0, bounded_excess.size, BLOCK_SIZE)
        if start + BLOCK_SIZE <= bounded_excess.size
    ]
    e_alarm = (
        (alarm_block + 1) * BLOCK_SIZE if alarm_block is not None else None
    )
    return {
        "n_monitor": int(monitor_values.size),
        "n_completed_blocks": len(log_e_history),
        "baseline": baseline,
        "condition_residual_used": residual_used,
        "historical_alarm_monitor_index": rate_alarm,
        "historical_alarm_snapshot": (
            BASELINE_END + rate_alarm + 1 if rate_alarm is not None else None
        ),
        "anytime_alarm_monitor_index": e_alarm,
        "anytime_alarm_snapshot": (
            BASELINE_END + e_alarm + 1 if e_alarm is not None else None
        ),
        "maximum_log_e": max(log_e_history) if log_e_history else None,
        "final_log_e": log_e,
        "bounded_excess_mean": (
            float(np.mean(bounded_excess)) if bounded_excess.size else None
        ),
        "completed_block_mean_max": (
            max(completed_blocks) if completed_blocks else None
        ),
        "n_completed_blocks_above_null_mean": (
            sum(value > NULL_MEAN for value in completed_blocks)
        ),
        "log_e_history": log_e_history,
    }


def machine_result(axes: dict[str, dict[str, object]], use_residual: bool) -> dict[str, object]:
    alpha_axis = ALPHA_CASE / len(axes)
    results = {
        axis: axis_result(row, use_residual, alpha_axis)
        for axis, row in axes.items()
    }
    historical_alarms = [
        result["historical_alarm_snapshot"]
        for result in results.values()
        if result["historical_alarm_snapshot"] is not None
    ]
    anytime_alarms = [
        result["anytime_alarm_snapshot"]
        for result in results.values()
        if result["anytime_alarm_snapshot"] is not None
    ]
    return {
        "historical_alarm_snapshot": min(historical_alarms) if historical_alarms else None,
        "anytime_alarm_snapshot": min(anytime_alarms) if anytime_alarms else None,
        "axes": results,
    }


def evaluate_case(
    split: str,
    case: int,
    axes: dict[str, dict[str, object]],
) -> dict[str, object] | None:
    if any(np.asarray(row["rms"]).size < BASELINE_END + MIN_MONITOR for row in axes.values()):
        return None
    labels = np.asarray(next(iter(axes.values()))["label"], dtype=np.int64)
    fault_indices = np.flatnonzero(labels != 0)
    first_fault_snapshot = (
        int(fault_indices[0] + 1) if fault_indices.size else None
    )
    if split == "test" and (first_fault_snapshot is None or first_fault_snapshot <= BASELINE_END):
        return None

    raw = machine_result(axes, use_residual=False)
    residual = machine_result(axes, use_residual=True)
    n_snapshots = int(next(iter(axes.values()))["n_snapshots"])
    return {
        "split": split,
        "case": case,
        "n_snapshots": n_snapshots,
        "first_fault_snapshot": first_fault_snapshot,
        "raw": raw,
        "residual": residual,
    }


def lead_time(result: dict[str, object], method: str) -> float | None:
    alarm = result[method]["anytime_alarm_snapshot"]
    first_fault = result["first_fault_snapshot"]
    if alarm is None or first_fault is None or alarm >= first_fault:
        return None
    return 100.0 * (first_fault - alarm) / first_fault


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-csv", type=Path, default=FEATURE_CSV)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    if not args.feature_csv.exists():
        raise FileNotFoundError(
            f"{args.feature_csv} is missing; run RNED/src/extract_sca.py first"
        )
    streams = load_streams(args.feature_csv)
    cases = [
        evaluate_case(split, case, axes)
        for (split, case), axes in sorted(streams.items())
    ]
    cases = [case for case in cases if case is not None]
    healthy = [case for case in cases if case["split"] == "train"]
    failures = [case for case in cases if case["split"] == "test"]

    methods = ("raw", "residual")
    summary: dict[str, object] = {
        "protocol": {
            "status": "frozen before outcome computation",
            "preregistration_sha256": sha256_file(PREREGISTRATION),
            "baseline": "median(feature[10:60])",
            "excess_ratio": EXCESS_RATIO,
            "minimum_monitor_snapshots": MIN_MONITOR,
            "historical_threshold": HISTORICAL_THRESHOLD,
            "block_size": BLOCK_SIZE,
            "excess_bound": EXCESS_BOUND,
            "null_mean": NULL_MEAN,
            "lambda_mixture": [float(value) for value in LAMBDAS],
            "alpha_case": ALPHA_CASE,
            "alpha_axis_rule": "alpha_case / n_available_positions",
            "post_hoc_tuning_allowed": False,
        },
        "n_healthy_censored_cases": len(healthy),
        "n_eligible_failure_cases": len(failures),
        "healthy_censored": {},
        "failure": {},
        "per_case": cases,
        "interpretation": (
            "This is an independent industrial healthy-censored validation. "
            "The e-process control remains conditional on bounded excesses, a "
            "correct null mean, and effectively independent blocks; the SCA "
            "channels are positions, not collocated axes on one bearing."
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    for method in methods:
        healthy_alarms = [
            case for case in healthy if case[method]["anytime_alarm_snapshot"] is not None
        ]
        historical_healthy_alarms = [
            case
            for case in healthy
            if case[method]["historical_alarm_snapshot"] is not None
        ]
        early = [
            case
            for case in failures
            if case[method]["anytime_alarm_snapshot"] is not None
            and case[method]["anytime_alarm_snapshot"] < case["first_fault_snapshot"]
        ]
        detected = [
            case
            for case in failures
            if case[method]["anytime_alarm_snapshot"] is not None
        ]
        leads = [lead_time(case, method) for case in early]
        summary["healthy_censored"][method] = {
            "anytime_false_alarms": len(healthy_alarms),
            "anytime_false_alarm_rate_pct": (
                100.0 * len(healthy_alarms) / len(healthy) if healthy else None
            ),
            "historical_false_alarms": len(historical_healthy_alarms),
            "historical_false_alarm_rate_pct": (
                100.0 * len(historical_healthy_alarms) / len(healthy)
                if healthy
                else None
            ),
        }
        summary["failure"][method] = {
            "early_warnings_before_label": len(early),
            "detected_by_trajectory_end": len(detected),
            "median_lead_before_label_pct": float(np.median(leads)) if leads else None,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    compact = {
        "n_healthy_censored_cases": len(healthy),
        "n_eligible_failure_cases": len(failures),
        "healthy_censored": summary["healthy_censored"],
        "failure": summary["failure"],
    }
    print(json.dumps(compact, indent=2))
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
