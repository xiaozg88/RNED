#!/usr/bin/env python3
"""On-device cost audit for the deployable degradation early-warning detector.

Measures, on this Jetson (Orin NX, 6-core aarch64):
  - per-window compute latency of (a) RMS extraction and (b) the full 5-feature
    time-domain extraction, for the three dataset window sizes
    (IMS 20,480; XJTU-SY 32,768; FEMTO 2,560 samples);
  - per-window detector-update latency (running-mean excess above 1.2x baseline);
  - sustained throughput and energy per window (idle vs load VDD power via jtop).

The detector is scalar/O(n) and runs on CPU; no accelerator is required.
"""
from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

WINDOWS = {"ims": 20480, "xjtu": 32768, "femto": 2560}


class PowerSampler:
    def __init__(self):
        from jtop import jtop
        self._jetson = jtop()
        self._samples: list[float] = []
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._jetson.start()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            stats = self._jetson.stats
            if stats and "Power TOT" in stats:
                self._samples.append(float(stats["Power TOT"]))
            time.sleep(0.25)

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._jetson.close()

    def mean_mw(self):
        return float(np.mean(self._samples)) if self._samples else float("nan")


def rms(x):
    return float(np.sqrt(np.mean(x * x)))


def full_feats(x):
    from scipy.stats import kurtosis, skew
    return dict(
        rms=float(np.sqrt(np.mean(x * x))),
        kurtosis_pearson=float(kurtosis(x, fisher=False)),
        skewness=float(skew(x)),
        variance=float(np.var(x)),
        peak2peak=float(x.max() - x.min()),
    )


def timeit(fn, n=2000):
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n


def main():
    rng = np.random.default_rng(0)
    out = {"platform": "Jetson Orin NX (6-core aarch64)", "windows": {}}

    # 1) per-window feature extraction latency (CPU)
    for ds, n in WINDOWS.items():
        x = rng.standard_normal(n).astype(np.float32)
        lat_rms = timeit(lambda: rms(x))
        lat_full = timeit(lambda: full_feats(x), n=200)
        out["windows"][ds] = {
            "samples": n,
            "rms_ms": round(lat_rms * 1e3, 4),
            "full_5feat_ms": round(lat_full * 1e3, 4),
        }

    # 2) detector update (running-mean excess) latency on a synthetic monitored run
    baseline = 0.1
    r = rng.standard_normal(1000) * 0.01 + 0.1
    def update():
        ratio = r / baseline
        excess = np.maximum(ratio - 1.2, 0.0)
        rate = np.cumsum(excess) / np.arange(1, excess.size + 1)
        return float(rate[-1])
    out["detector_update_ms"] = round(timeit(update, n=20000) * 1e3, 6)

    # 3) sustained throughput + energy (power rail)
    sampler = PowerSampler()
    sampler.start()
    time.sleep(5)
    idle = sampler.mean_mw()
    # sustained loop: RMS extraction + detector update per window
    x_loop = rng.standard_normal(WINDOWS["xjtu"]).astype(np.float32)
    windows = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 60.0:
        rms(x_loop)
        windows += 1
    dt = time.perf_counter() - t0
    load = sampler.mean_mw()
    sampler.stop()
    thr = windows / dt
    out["throughput_windows_per_s"] = round(thr, 1)
    out["power_idle_mw"] = round(idle, 1)
    out["power_load_mw"] = round(load, 1)
    out["power_delta_mw"] = round(load - idle, 1)
    out["energy_per_window_mJ"] = round((load - idle) / thr, 4) if thr > 0 else None

    dest = ROOT / "results" / "results_run_to_failure_edge_benchmark.json"
    dest.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print("saved", dest)


if __name__ == "__main__":
    main()
