#!/usr/bin/env python3
"""Benchmark raw RMS and Hilbert-envelope RMS on the local edge CPU."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.signal import hilbert


ROOT = Path(__file__).resolve().parents[1]
WINDOWS = {"ims": 20480, "xjtu": 32768, "femto": 2560}


def raw_rms(signal: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(signal))))


def envelope_rms(signal: np.ndarray) -> float:
    envelope = np.abs(hilbert(signal - float(np.mean(signal))))
    envelope -= float(np.mean(envelope))
    return float(np.sqrt(np.mean(np.square(envelope))))


def benchmark(function, repeats: int = 100) -> dict:
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
    rng = np.random.default_rng(20261025)
    results = {
        "platform_note": "local Jetson-class aarch64 CPU; stationary synthetic windows",
        "repeats": 100,
        "windows": {},
    }
    for dataset, length in WINDOWS.items():
        signal = rng.standard_normal(length).astype(np.float32)
        results["windows"][dataset] = {
            "samples": length,
            "raw_rms": benchmark(lambda: raw_rms(signal)),
            "envelope_rms": benchmark(lambda: envelope_rms(signal)),
        }

    output = ROOT / "results/results_run_to_failure_rms_variant_edge_benchmark.json"
    output.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
