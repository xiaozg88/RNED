#!/usr/bin/env python3
"""Incremental on-device cost benchmark for the hierarchical cross-axis rule."""
from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WINDOWS = {"ims": 20480, "xjtu": 32768, "femto": 2560}
AXIS_COUNTS = (2, 4)


class PowerSampler:
    def __init__(self) -> None:
        from jtop import jtop

        self._jetson = jtop()
        self._samples: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._jetson.start()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            stats = self._jetson.stats
            if stats and "Power TOT" in stats:
                self._samples.append(float(stats["Power TOT"]))
            time.sleep(0.25)

    def mean_mw(self) -> float:
        return float(np.mean(self._samples)) if self._samples else float("nan")

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._jetson.close()


class IncrementalHierarchy:
    """Constant-time update for the frozen historical OR second-max rule."""

    def __init__(self, axes: int, baselines: np.ndarray, q_history: float, q_cross: float) -> None:
        if axes < 1 or baselines.shape != (axes,):
            raise ValueError("baselines must match a positive axis count")
        self.axes = axes
        self.baselines = baselines
        self.q_history = q_history
        self.q_cross = q_cross
        self.excess_sums = np.zeros(axes, dtype=np.float64)
        self.counts = np.zeros(axes, dtype=np.int64)
        self.recent = [deque(maxlen=10) for _ in range(axes)]
        self.history_rates = np.zeros(axes, dtype=np.float64)
        self.recent_rates = np.zeros(axes, dtype=np.float64)
        self.history_alarm = False
        self.cross_alarm = False

    def update(self, rms: np.ndarray) -> tuple[bool, bool]:
        excess = np.maximum(rms / self.baselines - 1.2, 0.0)
        self.excess_sums += excess
        self.counts += 1
        self.history_rates = self.excess_sums / self.counts
        for axis in range(self.axes):
            self.recent[axis].append(float(excess[axis]))
            self.recent_rates[axis] = sum(self.recent[axis]) / len(self.recent[axis])
        self.history_alarm = bool(np.any(self.history_rates > self.q_history))
        if self.axes >= 2:
            second_max = np.partition(self.recent_rates, -2)[-2]
            self.cross_alarm = bool(second_max > self.q_cross)
        else:
            self.cross_alarm = False
        return self.history_alarm or self.cross_alarm, self.history_alarm


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x * x)))


def timeit(function, repetitions: int) -> float:
    start = time.perf_counter()
    for _ in range(repetitions):
        function()
    return (time.perf_counter() - start) / repetitions


def platform_name() -> str:
    model_path = Path("/proc/device-tree/model")
    try:
        return model_path.read_text().rstrip("\x00")
    except OSError:
        return f"{os.uname().machine} Linux"


def detector_update_costs(rng: np.random.Generator) -> dict:
    rows = {}
    for axes in AXIS_COUNTS:
        detector = IncrementalHierarchy(
            axes=axes,
            baselines=np.full(axes, 0.1, dtype=np.float64),
            q_history=1.0,
            q_cross=1.0,
        )
        values = np.full(axes, 0.1, dtype=np.float64)

        def update() -> tuple[bool, bool]:
            return detector.update(values)

        rows[f"axes_{axes}"] = {
            "incremental_update_us": round(timeit(update, 50000) * 1e6, 4),
            "memory_bytes_state": 8 * (2 * axes + 2 * axes) + 8 * axes * 10 + 64 * axes,
        }
    return rows


def main() -> None:
    rng = np.random.default_rng(20261022)
    output = {
        "platform": platform_name(),
        "protocol": {
            "implementation": "stateful constant-time historical cumulative mean plus 10-snapshot rolling mean",
            "decision": "historical OR cross-axis",
            "axis_counts": list(AXIS_COUNTS),
            "rms_dtype": "float32",
        },
        "detector_incremental_update": detector_update_costs(rng),
        "windows": {},
    }

    for dataset, samples in WINDOWS.items():
        signal = rng.standard_normal(samples).astype(np.float32)
        rms_latency = timeit(lambda: rms(signal), 2000)
        detector = IncrementalHierarchy(
            axes=4,
            baselines=np.full(4, 0.1),
            q_history=1.0,
            q_cross=1.0,
        )
        values = np.full(4, 0.1)

        def pipeline() -> tuple[bool, bool]:
            values[0] = rms(signal)
            return detector.update(values)

        pipeline_latency = timeit(pipeline, 2000)
        output["windows"][dataset] = {
            "samples": samples,
            "rms_extraction_1_axis_ms": round(rms_latency * 1e3, 5),
            "rms_plus_incremental_hierarchy_k4_ms": round(pipeline_latency * 1e3, 5),
            "incremental_detector_share_k4_ms": round(
                max(0.0, pipeline_latency - rms_latency) * 1e3, 5
            ),
        }

    sustained_seconds = float(os.environ.get("SUSTAINED_SECONDS", "45"))
    sampler = PowerSampler()
    sampler.start()
    time.sleep(5)
    idle_power = sampler.mean_mw()
    signal = rng.standard_normal(WINDOWS["femto"]).astype(np.float32)
    detector = IncrementalHierarchy(
        axes=4,
        baselines=np.full(4, 0.1),
        q_history=1.0,
        q_cross=1.0,
    )
    values = np.full(4, 0.1)
    windows = 0
    start = time.perf_counter()
    while time.perf_counter() - start < sustained_seconds:
        values[0] = rms(signal)
        detector.update(values)
        windows += 1
    elapsed = time.perf_counter() - start
    load_power = sampler.mean_mw()
    sampler.stop()
    throughput = windows / elapsed
    output["sustained_rms_plus_hierarchy_k4"] = {
        "seconds": round(elapsed, 3),
        "window_samples": WINDOWS["femto"],
        "throughput_windows_per_s": round(throughput, 1),
        "power_idle_mw": round(idle_power, 1),
        "power_load_mw": round(load_power, 1),
        "power_delta_mw": round(load_power - idle_power, 1),
        "energy_per_window_mj": round((load_power - idle_power) / throughput, 5),
    }

    destination = ROOT / "results" / "results_run_to_failure_cross_axis_edge_benchmark.json"
    destination.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))
    print(f"saved {destination}")


if __name__ == "__main__":
    main()
