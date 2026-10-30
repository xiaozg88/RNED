#!/usr/bin/env python3
"""Explore order-domain envelope-spectral template residuals on SCA."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import detrend, hilbert


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data/raw_archives/sca"
DEFAULT_OUTPUT = ROOT / "results/results_sca_order_envelope_residual.json"
DEFAULT_DETAIL = ROOT / "results/results_sca_order_envelope_residual.csv"

BASELINE_START = 10
BASELINE_END = 60
BLOCK_SIZE = 10
FAMILIES = ("bpfo", "bpfi", "bsf")
HARMONICS = tuple(range(1, 7))
ORDER_WIDTH = 0.035
CONTROL_OFFSETS = (-0.18, -0.12, 0.12, 0.18)
TARGET_FAULT_CASES = {5, 7, 8}


def order_power(
    spec: np.ndarray,
    shaft_hz: float,
    center_order: float,
    sampling_rate: float,
) -> float:
    frequencies = np.fft.rfftfreq(
        2 * (spec.size - 1), d=1.0 / sampling_rate
    )
    center_hz = center_order * shaft_hz
    width_hz = ORDER_WIDTH * shaft_hz
    mask = (frequencies >= center_hz - width_hz) & (
        frequencies <= center_hz + width_hz
    )
    return float(np.sum(spec[mask]))


def snapshot_features(
    signal: np.ndarray,
    rpm: float,
    sampling_rate: float,
    multipliers: dict[str, float],
) -> tuple[dict[str, float], float]:
    centered = detrend(np.asarray(signal, dtype=np.float64), type="constant")
    envelope = detrend(np.abs(hilbert(centered)), type="constant")
    window = np.hanning(envelope.size)
    spec = np.square(np.abs(np.fft.rfft(envelope * window)))
    spec[0] = 0.0
    shaft_hz = rpm / 60.0

    vector: dict[str, float] = {}
    for family, multiplier in multipliers.items():
        if family not in FAMILIES:
            continue
        for harmonic in HARMONICS:
            center = harmonic * multiplier
            fault_power = order_power(
                spec, shaft_hz, center, sampling_rate
            )
            controls = [
                order_power(
                    spec, shaft_hz, center + offset, sampling_rate
                )
                for offset in CONTROL_OFFSETS
            ]
            vector[f"{family}_{harmonic}"] = math.log2(
                (fault_power + 1.0e-20) / (float(np.mean(controls)) + 1.0e-20)
            )
    rms = float(np.sqrt(np.mean(np.square(centered))))
    return vector, rms


def aggregate_scores(
    vector: dict[str, float], baseline_center: dict[str, float], baseline_scale: dict[str, float]
) -> dict[str, float]:
    standardized = {
        key: (vector[key] - baseline_center[key]) / baseline_scale[key]
        for key in vector
    }
    scores: dict[str, float] = {}
    for family in FAMILIES:
        values = np.asarray(
            [standardized[f"{family}_{harmonic}"] for harmonic in HARMONICS]
        )
        scores[f"{family}_max"] = float(np.max(values))
        scores[f"{family}_top2"] = float(np.mean(np.sort(values)[-2:]))
        scores[f"{family}_median"] = float(np.median(values))
    for statistic in ("max", "top2", "median"):
        scores[f"family_{statistic}"] = max(
            scores[f"{family}_{statistic}"] for family in FAMILIES
        )
    return scores


def robust_scales(vectors: list[dict[str, float]]) -> tuple[dict[str, float], dict[str, float]]:
    keys = vectors[0].keys()
    center: dict[str, float] = {}
    scale: dict[str, float] = {}
    for key in keys:
        values = np.asarray([vector[key] for vector in vectors])
        center[key] = float(np.median(values))
        mad = float(np.median(np.abs(values - center[key]))) * 1.4826
        if not math.isfinite(mad) or mad < 0.05:
            mad = float(np.std(values))
        if not math.isfinite(mad) or mad < 0.05:
            mad = 1.0
        scale[key] = mad
    return center, scale


def block_summary(values: dict[str, np.ndarray], stop: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for block_index, start in enumerate(range(BASELINE_END, stop, BLOCK_SIZE)):
        end = min(start + BLOCK_SIZE, stop)
        if end - start < math.ceil(BLOCK_SIZE / 2):
            continue
        row: dict[str, object] = {
            "block_index": block_index,
            "snapshot": start + 1,
        }
        for field, array in values.items():
            valid = array[start:end]
            valid = valid[np.isfinite(valid)]
            if valid.size:
                row[field] = float(np.median(valid))
            else:
                row[field] = float("nan")
        rows.append(row)
    return rows


def rank_auc(negative: list[float], positive: list[float]) -> float:
    values = np.asarray(negative + positive, dtype=np.float64)
    order = values.argsort()
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


def load_axes(input_dir: Path) -> dict[tuple[str, int, str], dict[str, object]]:
    streams: dict[tuple[str, int, str], dict[str, object]] = {}
    for path in sorted(input_dir.glob("*_train.mat")) + sorted(
        input_dir.glob("*_test.mat")
    ):
        loaded = loadmat(path, squeeze_me=True, struct_as_record=False)
        split = path.stem.split("_", 1)[1]
        case = int(loaded["id"])
        for axis_name, axis_data in loaded.items():
            if (
                axis_name.startswith("__")
                or not hasattr(axis_data, "_fieldnames")
                or "rawData" not in axis_data._fieldnames
            ):
                continue
            raw = np.asarray(axis_data.rawData)
            if raw.dtype == object:
                signals = [
                    np.asarray(record, dtype=np.float64).ravel()
                    for record in raw
                ]
            else:
                signals = [
                    raw[index].astype(np.float64).ravel()
                    for index in range(raw.shape[0])
                ]
            rpm = np.asarray(axis_data.RPM, dtype=np.float64)
            labels = np.asarray(axis_data.label, dtype=np.int64)
            eligible = np.flatnonzero((labels != -1) & (rpm > 0))
            frequencies = {
                "bpfo": float(axis_data.faultFrequencies.BPFOMultiple),
                "bpfi": float(axis_data.faultFrequencies.BPFIMultiple),
                "bsf": float(axis_data.faultFrequencies.BPFMultiple),
            }
            vectors: list[dict[str, float]] = []
            rms_values: list[float] = []
            for index in eligible:
                vector, rms = snapshot_features(
                    signals[index],
                    float(rpm[index]),
                    float(axis_data.samplingRate[index]),
                    frequencies,
                )
                vectors.append(vector)
                rms_values.append(rms)
            transition = None
            positive = np.flatnonzero(labels[eligible] > 0)
            if positive.size and labels[eligible][0] == 0:
                transition = int(positive[0])
            baseline_indices = eligible[BASELINE_START:BASELINE_END]
            order_resolution = float(
                np.median(
                    np.asarray(axis_data.samplingRate[baseline_indices])
                    / np.asarray([signals[index].size for index in baseline_indices])
                    / (rpm[baseline_indices] / 60.0)
                )
            )
            streams[(split, case, axis_name)] = {
                "vectors": vectors,
                "rms": np.asarray(rms_values),
                "rpm": rpm[eligible],
                "labels": labels[eligible],
                "transition": transition,
                "order_resolution": order_resolution,
            }
        print(f"processed {path.name}", flush=True)
    return streams


def audit(streams: dict[tuple[str, int, str], dict[str, object]]) -> tuple[list[dict[str, object]], dict[str, object]]:
    health_keys = [
        key
        for key in streams
        if key[0] == "train"
        and float(streams[key]["order_resolution"]) <= ORDER_WIDTH
    ]
    fault_keys = [
        key
        for key in streams
        if key[0] == "test" and key[1] in TARGET_FAULT_CASES
    ]
    detail_rows: list[dict[str, object]] = []
    field_results: dict[str, dict[str, object]] = defaultdict(dict)

    for key in health_keys + fault_keys:
        stream = streams[key]
        vectors = stream["vectors"]
        center, scale = robust_scales(vectors[BASELINE_START:BASELINE_END])
        score_rows = [
            aggregate_scores(vector, center, scale) for vector in vectors
        ]
        score_fields = sorted(score_rows[0])
        scores = {
            field: np.asarray([row[field] for row in score_rows])
            for field in score_fields
        }
        baseline_rms = np.log(
            np.asarray(stream["rms"])[BASELINE_START:BASELINE_END]
        )
        rms_center = float(np.median(baseline_rms))
        rms_scale = float(np.median(np.abs(baseline_rms - rms_center)) * 1.4826)
        scores["rms"] = (
            np.log(np.asarray(stream["rms"])) - rms_center
        ) / max(rms_scale, 1.0e-6)
        scores["family_max_or_rms"] = np.maximum(
            scores["family_max"], scores["rms"]
        )
        scores["family_top2_or_rms"] = np.maximum(
            scores["family_top2"], scores["rms"]
        )

        if key in health_keys:
            stop = len(vectors)
        elif stream["transition"] is not None:
            stop = int(stream["transition"])
        else:
            same_case_transitions = [
                int(streams[other_key]["transition"])
                for other_key in streams
                if other_key[0] == "test"
                and other_key[1] == key[1]
                and streams[other_key]["transition"] is not None
            ]
            stop = min(same_case_transitions)
        blocks = block_summary(scores, stop)
        for block in blocks:
            row: dict[str, object] = {
                "label": "health" if key in health_keys else "fault",
                "case": key[1],
                "axis": key[2],
                "block_index": block["block_index"],
                "snapshot": block["snapshot"],
                **{
                    field: block[field]
                    for field in scores
                },
            }
            detail_rows.append(row)

    all_fields = [
        field
        for field in detail_rows[0]
        if field not in {"label", "case", "axis", "block_index", "snapshot"}
    ]
    for field in all_fields:
        health_maxima = [
            max(row[field] for row in detail_rows if row["label"] == "health" and (row["case"], row["axis"]) == (key[1], key[2]))
            for key in health_keys
        ]
        fault_maxima = [
            max(row[field] for row in detail_rows if row["label"] == "fault" and (row["case"], row["axis"]) == (key[1], key[2]))
            for key in fault_keys
        ]
        thresholds = {}
        for quantile in (0.90, 0.95, 0.975, 1.0):
            threshold = float(np.quantile(health_maxima, quantile))
            thresholds[f"q{quantile:g}"] = {
                "threshold": threshold,
                "health_alarms": sum(value > threshold for value in health_maxima),
                "n_health": len(health_maxima),
                "fault_alarms": sum(value > threshold for value in fault_maxima),
                "n_fault": len(fault_maxima),
            }
        field_results[field] = {
            "health_max_median": float(np.median(health_maxima)),
            "health_max_q95": float(np.quantile(health_maxima, 0.95)),
            "fault_max_median": float(np.median(fault_maxima)),
            "separation_auc": rank_auc(health_maxima, fault_maxima),
            "thresholds": thresholds,
            "per_stream": {
                f"{'health' if key in health_keys else 'fault'}/{key[1]}/{key[2]}": value
                for key, value in zip(health_keys + fault_keys, health_maxima + fault_maxima)
            },
        }

    return detail_rows, {
        "protocol": {
            "status": "exploratory order-domain envelope template residual; not frozen validation",
            "baseline_snapshots": [BASELINE_START, BASELINE_END],
            "block_size": BLOCK_SIZE,
            "harmonics": list(HARMONICS),
            "order_width": ORDER_WIDTH,
            "control_offsets": list(CONTROL_OFFSETS),
            "target_faults": sorted(TARGET_FAULT_CASES),
            "fault_time": "case-level first positive label; all axes in that case are audited",
            "physical_eligibility": "training order-resolution median <= ORDER_WIDTH",
        },
        "fields": dict(field_results),
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
    print(f"wrote {args.output}")
    print(f"wrote {args.detail}")


if __name__ == "__main__":
    main()
