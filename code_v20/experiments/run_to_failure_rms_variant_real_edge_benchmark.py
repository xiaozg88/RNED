#!/usr/bin/env python3
"""Benchmark RMS variants on representative real raw event windows."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.signal import hilbert


ROOT = Path(__file__).resolve().parents[1]
WINDOW_PATH = ROOT / "data/benchmark/representative_raw_windows.npz"
REPEATS = 100


def raw_rms(signal: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(signal))))


def envelope_rms(signal: np.ndarray) -> float:
    envelope = np.abs(hilbert(signal - float(np.mean(signal))))
    envelope -= float(np.mean(envelope))
    return float(np.sqrt(np.mean(np.square(envelope))))


def benchmark(function, repeats: int) -> dict:
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        samples.append((time.perf_counter() - start) * 1000.0)
    return {
        "mean_ms": float(np.mean(samples)),
        "median_ms": float(np.median(samples)),
        "p95_ms": float(np.percentile(samples, 95)),
    }


def main() -> None:
    with np.load(WINDOW_PATH, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        datasets = {
            dataset: archive[dataset].astype(np.float32, copy=False)
            for dataset in ("ims", "xjtu", "femto")
        }

    results = {
        "platform_note": "local Jetson-class aarch64 CPU; representative real raw event windows",
        "protocol": {
            "repeats_per_window": REPEATS,
            "envelope_stride": 5,
            "window_selection": "first and last complete local raw event for each source; no outcome-based tuning",
            "provenance": metadata,
        },
        "windows": {},
    }
    for dataset, signals in datasets.items():
        window_results = []
        for position, signal in zip(("first", "last"), signals):
            result = {
                "position": position,
                "samples": int(signal.size),
                "raw_rms": benchmark(lambda: raw_rms(signal), REPEATS),
                "envelope_rms": benchmark(lambda: envelope_rms(signal), REPEATS),
            }
            result["envelope_rms_stride5_amortized"] = {
                key: value / 5.0 for key, value in result["envelope_rms"].items()
            }
            window_results.append(result)
        results["windows"][dataset] = {
            "events": window_results,
            "aggregate_median_ms": {
                "raw_rms": float(
                    np.median([event["raw_rms"]["median_ms"] for event in window_results])
                ),
                "envelope_rms": float(
                    np.median([event["envelope_rms"]["median_ms"] for event in window_results])
                ),
            },
        }

    output_path = ROOT / "results/results_run_to_failure_rms_variant_real_edge_benchmark.json"
    output_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
