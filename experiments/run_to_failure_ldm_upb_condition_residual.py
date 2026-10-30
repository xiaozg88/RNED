#!/usr/bin/env python3
"""Post-hoc LDM_UPB condition-residual mechanism probe.

This experiment does not replace the frozen detector. It asks whether an
alarm coincides with an operating-condition transition that can be removed by
an interpretable per-axis regression fitted only on the protocol's baseline
window. The frozen threshold is reused without recalibration or tuning.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FEATURE_CSV = ROOT / "data" / "features" / "ldm_upb_features.csv"
RAW_DIR = ROOT / "data" / "raw_archives" / "ldm_upb_hf"
THRESHOLD = 0.042407782315012556
EXCESS_RATIO = 1.2
BASELINE_START = 10
BASELINE_END = 60
MIN_MONITOR = 30
RIDGE_LAMBDA = 0.01


def excess_rate(ratio: np.ndarray) -> np.ndarray:
    excess = np.maximum(ratio - EXCESS_RATIO, 0.0)
    return np.cumsum(excess) / np.arange(1, excess.size + 1)


def alarm_index(score: np.ndarray) -> int:
    hits = np.flatnonzero(score[MIN_MONITOR - 1 :] > THRESHOLD) + (
        MIN_MONITOR - 1
    )
    return int(hits[0]) if hits.size else int(score.size)


def normalized_rms(
    rms: np.ndarray, conditions: np.ndarray
) -> tuple[np.ndarray, dict]:
    baseline_rms = rms[BASELINE_START:BASELINE_END]
    baseline_conditions = conditions[BASELINE_START:BASELINE_END]
    log_rms = np.log(np.maximum(baseline_rms, np.finfo(float).tiny))
    means = np.mean(baseline_conditions, axis=0)
    scales = np.std(baseline_conditions, axis=0)
    scales[scales < 1e-12] = 1.0
    baseline_design = np.column_stack(
        [
            np.ones(baseline_conditions.shape[0]),
            (baseline_conditions - means) / scales,
        ]
    )
    penalty = np.eye(baseline_design.shape[1]) * RIDGE_LAMBDA
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        baseline_design.T @ baseline_design + penalty,
        baseline_design.T @ log_rms,
    )
    design = np.column_stack(
        [np.ones(conditions.shape[0]), (conditions - means) / scales]
    )
    residual = np.log(np.maximum(rms, np.finfo(float).tiny)) - design @ coefficients
    baseline_median = np.median(residual[BASELINE_START:BASELINE_END])
    normalized = np.exp(residual - baseline_median)
    diagnostics = {
        "baseline_condition_number": float(
            np.linalg.cond(baseline_design.T @ baseline_design + penalty)
        ),
        "coefficients_intercept_dyn_stat_speed": [
            float(value) for value in coefficients
        ],
        "baseline_residual_mae": float(
            np.mean(np.abs(residual[BASELINE_START:BASELINE_END]))
        ),
    }
    return normalized, diagnostics


def load_data() -> tuple[dict, dict]:
    rms: dict[tuple[str, str], list[float]] = defaultdict(list)
    with FEATURE_CSV.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["dataset"] == "ldm_upb":
                rms[(row["bearing"], row["axis"])].append(float(row["rms"]))

    condition_rows: dict[str, list[dict[str, str]]] = {}
    for (bearing, _axis) in rms:
        path = RAW_DIR / bearing / f"{bearing}_operatingConditions.csv"
        with path.open(newline="") as handle:
            condition_rows[bearing] = list(csv.DictReader(handle))
    return rms, condition_rows


def main() -> None:
    rms_rows, condition_rows = load_data()
    bearings = sorted({bearing for bearing, _axis in rms_rows})
    results = []

    for bearing in bearings:
        conditions = np.asarray(
            [
                [
                    float(row["setDynLoad / N"]),
                    float(row["setStatLoad / N"]),
                    float(row["setSpeed / rpm"]),
                ]
                for row in condition_rows[bearing]
            ],
            dtype=np.float64,
        )
        raw_scores = []
        residual_scores = []
        diagnostics = []
        for axis in sorted(
            axis for candidate, axis in rms_rows if candidate == bearing
        ):
            rms = np.asarray(rms_rows[(bearing, axis)], dtype=np.float64)
            if rms.size != conditions.shape[0]:
                raise ValueError(
                    f"{bearing}/{axis}: RMS and condition lengths differ"
                )
            baseline = np.median(rms[BASELINE_START:BASELINE_END])
            raw_scores.append(excess_rate(rms[BASELINE_END:] / baseline))
            normalized, diagnostic = normalized_rms(rms, conditions)
            residual_scores.append(excess_rate(normalized[BASELINE_END:]))
            diagnostics.append(
                {"axis": axis, **diagnostic}
            )

        raw_score = np.maximum.reduce(raw_scores)
        residual_score = np.maximum.reduce(residual_scores)
        raw_alarm = alarm_index(raw_score)
        residual_alarm = alarm_index(residual_score)
        n_monitor = raw_score.size
        residual_detected = residual_alarm < n_monitor
        alarm_snapshot = 60 + residual_alarm if residual_detected else None
        baseline_conditions = conditions[59]
        alarm_conditions = (
            conditions[alarm_snapshot] if alarm_snapshot is not None else None
        )
        condition_change = (
            {
                "dynamic_load": 100.0
                * (alarm_conditions[0] / baseline_conditions[0] - 1.0),
                "static_load": 100.0
                * (alarm_conditions[1] / baseline_conditions[1] - 1.0),
                "speed": 100.0
                * (alarm_conditions[2] / baseline_conditions[2] - 1.0),
            }
            if alarm_conditions is not None
            else None
        )
        results.append(
            {
                "bearing": bearing,
                "n_monitor": n_monitor,
                "raw_alarm_index": raw_alarm,
                "raw_detected": raw_alarm < n_monitor,
                "raw_lead_pct_life": (
                    100.0 * (n_monitor - raw_alarm) / n_monitor
                    if raw_alarm < n_monitor
                    else None
                ),
                "residual_alarm_index": residual_alarm,
                "residual_detected": residual_detected,
                "residual_lead_pct_life": (
                    100.0 * (n_monitor - residual_alarm) / n_monitor
                    if residual_alarm < n_monitor
                    else None
                ),
                "residual_alarm_snapshot": alarm_snapshot,
                "condition_relative_change_at_alarm_pct": condition_change,
                "axis_diagnostics": diagnostics,
            }
        )

    detected_raw = sum(row["raw_detected"] for row in results)
    detected_residual = sum(row["residual_detected"] for row in results)
    raw_leads = [
        row["raw_lead_pct_life"]
        for row in results
        if row["raw_lead_pct_life"] is not None
    ]
    residual_leads = [
        row["residual_lead_pct_life"]
        for row in results
        if row["residual_lead_pct_life"] is not None
    ]
    early_rows = [row for row in results if row["raw_alarm_index"] == 29]
    delayed = [
        row
        for row in early_rows
        if row["residual_alarm_index"] > row["raw_alarm_index"]
    ]

    summary = {
        "protocol": {
            "status": "post-hoc mechanism exploration; not a frozen validation",
            "feature": "per-axis RMS residual after condition regression",
            "regression": (
                "ridge on [dynamic load, static load, speed], fitted only on "
                "snapshots 10:60; log-RMS response; lambda=0.01"
            ),
            "threshold": THRESHOLD,
            "excess_ratio": EXCESS_RATIO,
            "minimum_monitor_snapshots": MIN_MONITOR,
            "post_hoc_tuning_allowed": False,
        },
        "n_bearings": len(results),
        "raw_detected": detected_raw,
        "residual_detected": detected_residual,
        "median_raw_lead_pct_life": float(np.median(raw_leads)),
        "median_residual_lead_pct_life": (
            float(np.median(residual_leads)) if residual_leads else None
        ),
        "n_earliest_raw_alarms": len(early_rows),
        "n_earliest_raw_alarms_delayed_by_residualization": len(delayed),
        "per_bearing": results,
        "interpretation": (
            "This is explanatory only: the same frozen threshold is applied to "
            "a different feature, no independent healthy censored fleet is "
            "available, and the regression was designed after observing LDM_UPB "
            "early alarms. It cannot upgrade the main method."
        ),
    }
    output = (
        ROOT
        / "results"
        / "results_run_to_failure_ldm_upb_condition_residual.json"
    )
    output.write_text(json.dumps(summary, indent=2) + "\n")
    compact = {
        key: summary[key]
        for key in (
            "n_bearings",
            "raw_detected",
            "residual_detected",
            "median_raw_lead_pct_life",
            "median_residual_lead_pct_life",
            "n_earliest_raw_alarms",
            "n_earliest_raw_alarms_delayed_by_residualization",
        )
    }
    print(json.dumps(compact, indent=2))
    print(f"saved {output}")


if __name__ == "__main__":
    main()
