#!/usr/bin/env python3
"""Exploratory SCA audit of speed-normalized fault-band residuals."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data/features/sca_fault_band_features.csv"
DEFAULT_OUTPUT = ROOT / "results/results_sca_fault_band_mechanism_audit.json"
DEFAULT_DETAIL = ROOT / "results/results_sca_fault_band_mechanism_audit.csv"

BASELINE_START = 10
BASELINE_END = 60
BLOCK_SIZE = 10
TARGET_FAULT_CASES = {5, 7, 8}

CANDIDATES = (
    "rms",
    "fault_family_log_snr",
    "fault_family_multiband_log_snr",
    "fault_family_multiband_top2_mean",
    "bpfo_multiband_log_snr",
    "bpfi_multiband_log_snr",
    "bsf_multiband_log_snr",
    "envelope_kurtosis",
)


def load_streams(path: Path) -> dict[tuple[str, int, str], dict[str, object]]:
    grouped: dict[tuple[str, int, str], list[dict[str, str]]] = defaultdict(list)
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            grouped[
                (row["split"], int(row["case"]), row["axis"])
            ].append(row)

    streams: dict[tuple[str, int, str], dict[str, object]] = {}
    for key, rows in grouped.items():
        rows.sort(key=lambda row: int(row["snapshot"]))
        snapshots = [int(row["snapshot"]) for row in rows]
        if snapshots != list(range(1, len(rows) + 1)):
            raise ValueError(f"nonconsecutive snapshots: {key}")
        labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
        transition = None
        positive = np.flatnonzero(labels > 0)
        if positive.size and labels[0] == 0:
            transition = int(positive[0])
        streams[key] = {
            "labels": labels,
            "transition": transition,
            "rpm": np.asarray([float(row["rpm"]) for row in rows]),
            **{
                field: np.asarray([float(row[field]) for row in rows])
                for field in CANDIDATES
            },
        }
    return streams


def condition_log_feature(
    values: np.ndarray, rpm: np.ndarray
) -> tuple[np.ndarray, bool]:
    values = np.log(np.maximum(values, np.finfo(float).tiny))
    baseline_rpm = rpm[BASELINE_START:BASELINE_END]
    if float(np.std(baseline_rpm)) < 1.0:
        return values, False

    mean = float(np.mean(baseline_rpm))
    scale = float(np.std(baseline_rpm))
    standardized = (rpm - mean) / scale
    baseline_standardized = standardized[BASELINE_START:BASELINE_END]
    clipped = np.clip(
        standardized,
        float(np.min(baseline_standardized)),
        float(np.max(baseline_standardized)),
    )
    design = np.column_stack([np.ones(values.size), clipped])
    baseline_design = design[BASELINE_START:BASELINE_END]
    penalty = np.eye(2) * 0.01
    penalty[0, 0] = 0.0
    normal = baseline_design.T @ baseline_design + penalty
    if float(np.linalg.cond(normal)) > 1.0e8:
        return values, False
    coefficients = np.linalg.solve(
        normal, baseline_design.T @ values[BASELINE_START:BASELINE_END]
    )
    residual = values - design @ coefficients
    residual -= float(np.median(residual[BASELINE_START:BASELINE_END]))
    return residual, True


def normalized_scores(values: np.ndarray) -> np.ndarray:
    baseline = values[BASELINE_START:BASELINE_END]
    finite = np.isfinite(values)
    result = np.full(values.shape, np.nan, dtype=np.float64)
    baseline_finite = baseline[np.isfinite(baseline)]
    if baseline_finite.size < 10:
        return result

    center = float(np.median(baseline_finite))
    mad = float(np.median(np.abs(baseline_finite - center)))
    scale = 1.4826 * mad
    if not math.isfinite(scale) or scale < 1.0e-6:
        scale = float(np.std(baseline_finite))
    if not math.isfinite(scale) or scale < 1.0e-6:
        result[finite] = values[finite] - center
    else:
        result[finite] = (values[finite] - center) / scale
    return result


def block_summary(scores: np.ndarray, stop: int | None = None) -> list[dict[str, float]]:
    limit = scores.size if stop is None else min(stop, scores.size)
    summary: list[dict[str, float]] = []
    for block_index, start in enumerate(
        range(BASELINE_END, limit, BLOCK_SIZE)
    ):
        block = scores[start : min(start + BLOCK_SIZE, limit)]
        valid = block[np.isfinite(block)]
        if valid.size < math.ceil(block.size / 2):
            continue
        summary.append(
            {
                "block_index": block_index,
                "snapshot": start + 1,
                "median": float(np.median(valid)),
                "q75": float(np.quantile(valid, 0.75)),
            }
        )
    return summary


def stream_scores(
    stream: dict[str, object], field: str, conditioned: bool
) -> np.ndarray:
    values = np.asarray(stream[field], dtype=np.float64)
    rpm = np.asarray(stream["rpm"], dtype=np.float64)
    if field == "rms" or conditioned:
        transformed, _ = condition_log_feature(values, rpm)
    else:
        transformed = values
    return normalized_scores(transformed)


def rank_auc(negative: list[float], positive: list[float]) -> float:
    all_values = np.asarray(negative + positive)
    order = all_values.argsort()
    ranks = np.empty(all_values.size)
    sorted_values = all_values[order]
    i = 0
    while i < sorted_values.size:
        j = i + 1
        while j < sorted_values.size and sorted_values[j] == sorted_values[i]:
            j += 1
        ranks[order[i:j]] = (i + j + 1) / 2.0
        i = j
    positive_rank_sum = float(
        ranks[len(negative) :].sum()
    )
    return float(
        (
            positive_rank_sum
            - len(positive) * (len(positive) + 1) / 2.0
        )
        / (len(negative) * len(positive))
    )


def threshold_metrics(
    health_maxima: list[float],
    fault_maxima: list[float],
    quantile: float,
) -> dict[str, object]:
    finite_health = [value for value in health_maxima if math.isfinite(value)]
    finite_fault = [value for value in fault_maxima if math.isfinite(value)]
    if not finite_health or not finite_fault:
        return {"valid": False}
    threshold = float(np.quantile(finite_health, quantile))
    return {
        "valid": True,
        "quantile": quantile,
        "threshold": threshold,
        "health_alarms": sum(value > threshold for value in finite_health),
        "n_health": len(finite_health),
        "fault_alarms": sum(value > threshold for value in finite_fault),
        "n_fault": len(finite_fault),
    }


def audit(streams: dict[tuple[str, int, str], dict[str, object]]) -> dict[str, object]:
    health_keys = [key for key in streams if key[0] == "train"]
    target_keys = [
        key
        for key in streams
        if key[0] == "test"
        and key[1] in TARGET_FAULT_CASES
        and streams[key]["transition"] is not None
    ]
    candidate_results: dict[str, dict[str, object]] = {}
    detail_rows: list[dict[str, object]] = []
    detail_by_feature: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)

    for field in CANDIDATES:
        for conditioned in (False, True):
            health_maxima: list[float] = []
            health_tail: list[float] = []
            fault_maxima: list[float] = []
            fault_tail: list[float] = []
            for key in health_keys + target_keys:
                scores = stream_scores(streams[key], field, conditioned)
                transition = (
                    None
                    if key in health_keys
                    else int(streams[key]["transition"])
                )
                summary = block_summary(scores, transition)
                if not summary:
                    continue
                maxima = [item["median"] for item in summary]
                max_score = max(maxima)
                tail = max(maxima[-2:])
                if key in health_keys:
                    health_maxima.append(max_score)
                    health_tail.append(tail)
                    label = "health"
                else:
                    fault_maxima.append(max_score)
                    fault_tail.append(tail)
                    label = "fault"
                detail_rows.append(
                    {
                        "feature": field,
                        "conditioned": int(conditioned),
                        "label": label,
                        "case": key[1],
                        "axis": key[2],
                        "n_blocks": len(summary),
                        "max_block_z": max_score,
                        "tail_max_block_z": tail,
                    }
                )
                detail_by_feature[(field, int(conditioned))].append(detail_rows[-1])

            finite_health = [
                value for value in health_maxima if math.isfinite(value)
            ]
            finite_fault = [
                value for value in fault_maxima if math.isfinite(value)
            ]
            name = f"{field}{'_rpm_residual' if conditioned else ''}"
            candidate_results[name] = {
                "n_health": len(finite_health),
                "n_fault": len(finite_fault),
                "health_max_median": float(np.median(finite_health)),
                "health_max_q95": float(np.quantile(finite_health, 0.95)),
                "fault_max_median": float(np.median(finite_fault)),
                "separation_auc": rank_auc(finite_health, finite_fault),
                "thresholds": [
                    threshold_metrics(health_maxima, fault_maxima, quantile)
                    for quantile in (0.90, 0.95, 0.975, 1.0)
                ],
                "per_stream": {
                    f"{row['label']}/{row['case']}/{row['axis']}": row["max_block_z"]
                    for row in detail_by_feature[(field, int(conditioned))]
                },
            }

    return detail_rows, {
        "protocol": {
            "status": "exploratory mechanism audit; not independent validation",
            "baseline_snapshots": [BASELINE_START, BASELINE_END],
            "block_size": BLOCK_SIZE,
            "health_source": "SCA train/censored trajectories",
            "fault_target": "test cases 5, 7, 8 before first positive label",
            "normalization": "per-axis baseline median/MAD of log RMS or envelope residual",
        },
        "candidates": candidate_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--detail", type=Path, default=DEFAULT_DETAIL)
    args = parser.parse_args()

    streams = load_streams(args.input)
    detail_rows, result = audit(streams)
    result["inputs"] = {
        "path": str(args.input),
        "n_rows": sum(
            int(np.asarray(stream["labels"]).size) for stream in streams.values()
        ),
        "n_streams": len(streams),
    }
    result["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True))
    with args.detail.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detail_rows[0]))
        writer.writeheader()
        writer.writerows(detail_rows)
    print(f"wrote {args.output}")
    print(f"wrote {args.detail}")


if __name__ == "__main__":
    main()
