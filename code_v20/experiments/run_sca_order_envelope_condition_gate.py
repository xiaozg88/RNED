#!/usr/bin/env python3
"""Test whether RPM-conditioned envelope residuals pass an e-process gate."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from run_sca_order_envelope_residual import (
    BASELINE_END,
    BASELINE_START,
    BLOCK_SIZE,
    TARGET_FAULT_CASES,
    aggregate_scores,
    load_axes,
    robust_scales,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data/raw_archives/sca"
DEFAULT_OUTPUT = ROOT / "results/results_sca_order_envelope_condition_gate.json"
DEFAULT_DETAIL = ROOT / "results/results_sca_order_envelope_condition_gate.csv"
RIDGE_PENALTY = 0.01
PREVIOUS_BLOCK_COUNT = 5


def condition_residual(
    values: np.ndarray, rpm: np.ndarray
) -> tuple[np.ndarray, bool]:
    baseline_rpm = rpm[BASELINE_START:BASELINE_END]
    if float(np.std(baseline_rpm)) < 1.0:
        return values.copy(), False

    center = float(np.mean(baseline_rpm))
    scale = float(np.std(baseline_rpm))
    standardized = (rpm - center) / scale
    baseline_standardized = standardized[BASELINE_START:BASELINE_END]
    clipped = np.clip(
        standardized,
        float(np.min(baseline_standardized)),
        float(np.max(baseline_standardized)),
    )
    design = np.column_stack([np.ones(values.size), clipped])
    baseline_design = design[BASELINE_START:BASELINE_END]
    penalty = np.eye(2) * RIDGE_PENALTY
    penalty[0, 0] = 0.0
    normal_matrix = baseline_design.T @ baseline_design + penalty
    if float(np.linalg.cond(normal_matrix)) > 1.0e8:
        return values.copy(), False

    coefficients = np.linalg.solve(
        normal_matrix, baseline_design.T @ values[BASELINE_START:BASELINE_END]
    )
    residual = values - design @ coefficients
    residual -= float(np.median(residual[BASELINE_START:BASELINE_END]))
    return residual, True


def stream_scores(stream: dict[str, object]) -> tuple[dict[str, np.ndarray], bool]:
    vectors = stream["vectors"]
    rpm = np.asarray(stream["rpm"], dtype=np.float64)
    center, scale = robust_scales(vectors[BASELINE_START:BASELINE_END])
    raw_rows = [aggregate_scores(vector, center, scale) for vector in vectors]
    fields = sorted(raw_rows[0])
    raw = {
        field: np.asarray([row[field] for row in raw_rows], dtype=np.float64)
        for field in fields
    }
    residual = {}
    used = False
    for field in fields:
        values, residual_used = condition_residual(raw[field], rpm)
        residual[field] = values
        used = used or residual_used
    return {**raw, **residual}, used


def block_values(values: dict[str, np.ndarray], stop: int) -> dict[str, list[float]]:
    output: dict[str, list[float]] = {}
    for field, array in values.items():
        blocks: list[float] = []
        for start in range(BASELINE_END, stop, BLOCK_SIZE):
            end = min(start + BLOCK_SIZE, stop)
            if end - start < math.ceil(BLOCK_SIZE / 2):
                continue
            valid = array[start:end]
            valid = valid[np.isfinite(valid)]
            blocks.append(float(np.median(valid)) if valid.size else math.nan)
        output[field] = blocks
    return output


def jump_blocks(blocks: list[float]) -> list[float]:
    output: list[float] = []
    for index, value in enumerate(blocks):
        history = blocks[max(0, index - PREVIOUS_BLOCK_COUNT) : index]
        if history:
            output.append(max(value - float(np.median(history)), 0.0))
        else:
            output.append(0.0)
    return output


def rank_auc(negative: list[float], positive: list[float]) -> float:
    values = np.asarray(negative + positive, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    begin = 0
    while begin < sorted_values.size:
        end = begin + 1
        while end < sorted_values.size and sorted_values[end] == sorted_values[begin]:
            end += 1
        ranks[order[begin:end]] = (begin + end + 1) / 2.0
        begin = end
    positive_rank_sum = float(ranks[len(negative) :].sum())
    return float(
        (positive_rank_sum - len(positive) * (len(positive) + 1) / 2.0)
        / (len(negative) * len(positive))
    )


def audit(streams: dict[tuple[str, int, str], dict[str, object]]) -> tuple[list[dict[str, object]], dict[str, object]]:
    health_keys = [
        key
        for key in streams
        if key[0] == "train"
        and float(streams[key]["order_resolution"]) <= 0.035
    ]
    fault_keys = [
        key for key in streams if key[0] == "test" and key[1] in TARGET_FAULT_CASES
    ]
    stream_maxima: dict[tuple[str, int, str], dict[str, float]] = {}
    residual_used: dict[str, bool] = {}
    detail_rows: list[dict[str, object]] = []

    for key in health_keys + fault_keys:
        stream = streams[key]
        raw_and_residual, used = stream_scores(stream)
        residual_used[f"{key[1]}/{key[2]}"] = used
        if key in health_keys:
            stop = len(stream["vectors"])
        elif stream["transition"] is not None:
            stop = int(stream["transition"])
        else:
            same_case_transitions = [
                int(other["transition"])
                for other_key, other in streams.items()
                if other_key[0] == "test"
                and other_key[1] == key[1]
                and other["transition"] is not None
            ]
            stop = min(same_case_transitions)

        blocks = block_values(raw_and_residual, stop)
        names: list[str] = []
        for field in sorted(raw_and_residual):
            names.extend([field, f"jump_{field}"])
        stream_maxima[key] = {}
        for block_index in range(max(len(values) for values in blocks.values())):
            row: dict[str, object] = {
                "label": "health" if key in health_keys else "fault",
                "case": key[1],
                "axis": key[2],
                "block_index": block_index,
                "snapshot": BASELINE_END + block_index * BLOCK_SIZE + 1,
            }
            for field in sorted(blocks):
                values = blocks[field]
                value = values[block_index] if block_index < len(values) else math.nan
                jumps = jump_blocks(values)
                jump = jumps[block_index] if block_index < len(jumps) else math.nan
                row[field] = value
                row[f"jump_{field}"] = jump
                stream_maxima[key][field] = max(
                    stream_maxima[key].get(field, -math.inf), value
                )
                stream_maxima[key][f"jump_{field}"] = max(
                    stream_maxima[key].get(f"jump_{field}", -math.inf), jump
                )
            detail_rows.append(row)

    fields = [
        field
        for field in detail_rows[0]
        if field not in {"label", "case", "axis", "block_index", "snapshot"}
    ]
    health_maxima = {field: [stream_maxima[key][field] for key in health_keys] for field in fields}
    fault_maxima = {field: [stream_maxima[key][field] for key in fault_keys] for field in fields}
    field_results: dict[str, dict[str, object]] = {}
    for field in fields:
        negative = health_maxima[field]
        positive = fault_maxima[field]
        thresholds = []
        for index, _ in enumerate(negative):
            thresholds.append(max(negative[:index] + negative[index + 1 :]))
        field_results[field] = {
            "separation_auc": rank_auc(negative, positive),
            "loo_health_false_alarms": sum(
                value > threshold for value, threshold in zip(negative, thresholds)
            ),
            "mean_loo_fault_detections": float(
                np.mean([sum(value > threshold for value in positive) for threshold in thresholds])
            ),
            "health_max_median": float(np.median(negative)),
            "fault_max_median": float(np.median(positive)),
            "per_stream": {
                f"{'health' if key in health_keys else 'fault'}/{key[1]}/{key[2]}": stream_maxima[key][field]
                for key in health_keys + fault_keys
            },
        }

    eligible = {
        field: result
        for field, result in field_results.items()
        if result["separation_auc"] >= 0.90
        and result["loo_health_false_alarms"] <= 1
        and result["mean_loo_fault_detections"] >= 0.60
    }
    return detail_rows, {
        "protocol": {
            "status": "exploratory mechanism gate only; not a frozen validation",
            "conditioning": "snapshot score regressed causally on RPM using baseline 10:60",
            "drift_statistic": "block median minus median of previous five blocks",
            "gate": "AUC >= 0.90, LOO health false alarms <= 1/16, mean LOO fault detections >= 0.60",
            "action_if_empty": "do not run block-wise e-process on independent data",
        },
        "condition_residual_used": residual_used,
        "n_health_streams": len(health_keys),
        "n_fault_streams": len(fault_keys),
        "passed_fields": sorted(eligible),
        "fields": field_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--detail", type=Path, default=DEFAULT_DETAIL)
    args = parser.parse_args()

    streams = load_axes(args.input)
    detail_rows, result = audit(streams)
    result["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True))
    with args.detail.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detail_rows[0]))
        writer.writeheader()
        writer.writerows(detail_rows)
    print(f"passed fields: {result['passed_fields']}")
    print(f"wrote {args.output}")
    print(f"wrote {args.detail}")


if __name__ == "__main__":
    main()
