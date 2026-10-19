#!/usr/bin/env python3
"""Classical online change-detection baselines vs the deployable excess-rate detector.

Same causal protocol as run_to_failure_early_warning.py: baseline snapshots
11-60, monitoring from 61, latching first alarm with a minimum of 30 monitored
snapshots, zero-false-alarm threshold calibrated on the 8 IMS censored
bearings. All detectors share the per-axis -> per-bearing (max) aggregation.

Detectors (score is computed per monitored snapshot):
  - excess_rate: running mean of max(rms/base - 1.2, 0)  [paper's detector]
  - shewhart:    rms/base ratio (single-sample exceedance)
  - cusum:       S_t = max(0, S_{t-1} + max(r_t - 1.2, 0))
  - ewma:        z_t = 0.2*max(r_t-1.2,0) + 0.8*z_{t-1}   (r_t = rms_t/base)
  - sprt:        one-sided Gaussian SPRT, H0: mu=1.0 vs H1: mu=1.2, reset at 0
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

IMS_FAILED = {("test1", 3), ("test1", 4), ("test2", 1), ("test3", 3)}
SECONDS_PER_SNAPSHOT = {"ims": 600.0, "xjtu": 60.0, "femto": 10.0}


def load_all_streams(here: Path) -> list[dict]:
    streams = []
    for test in ("test1", "test2", "test3"):
        df = pd.read_csv(here / f"{test}.csv")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        for bearing in sorted(df["bearing"].unique()):
            for channel in sorted(df[df["bearing"] == bearing]["channel"].unique()):
                sub = df[(df["bearing"] == bearing) & (df["channel"] == channel)].sort_values("timestamp")
                streams.append(
                    {
                        "dataset": "ims",
                        "bearing_id": f"ims_{test}_b{bearing}",
                        "axis": int(channel),
                        "failed": (test, int(bearing)) in IMS_FAILED,
                        "rms": sub["rms"].to_numpy(dtype=np.float64),
                    }
                )
    for csv, ds in (("xjtu_features.csv", "xjtu"), ("femto_features.csv", "femto")):
        df = pd.read_csv(here / csv)
        for (group, bearing) in sorted({(g, b) for g, b in zip(df["group"], df["bearing"])}):
            for axis in sorted(df[(df["group"] == group) & (df["bearing"] == bearing)]["axis"].unique()):
                sub = df[(df["group"] == group) & (df["bearing"] == bearing) & (df["axis"] == axis)].sort_values("snapshot")
                streams.append(
                    {
                        "dataset": ds,
                        "bearing_id": f"{ds}_{bearing}",
                        "axis": axis,
                        "failed": True,
                        "rms": sub["rms"].to_numpy(dtype=np.float64),
                    }
                )
    return streams


def _base(rms: np.ndarray) -> tuple[float, np.ndarray]:
    base = float(np.median(rms[10:60]))
    monitor = rms[60:]
    ratio = monitor / base
    return base, ratio


def detector_excess(rms):
    _, ratio = _base(rms)
    excess = np.maximum(ratio - 1.2, 0.0)
    return np.cumsum(excess) / np.arange(1, excess.size + 1)


def detector_shewhart(rms):
    _, ratio = _base(rms)
    return ratio


def detector_cusum(rms):
    _, ratio = _base(rms)
    inc = np.maximum(ratio - 1.2, 0.0)
    s = np.empty_like(inc)
    acc = 0.0
    for i, v in enumerate(inc):
        acc = max(0.0, acc + v)
        s[i] = acc
    return s


def detector_ewma(rms):
    _, ratio = _base(rms)
    inc = np.maximum(ratio - 1.2, 0.0)
    z = np.empty_like(inc)
    acc = 0.0
    for i, v in enumerate(inc):
        acc = 0.2 * v + 0.8 * acc
        z[i] = acc
    return z


def detector_sprt(rms):
    base, ratio = _base(rms)
    sigma = float(np.std(rms[10:60]) / base)
    sigma = max(sigma, 1e-9)
    mu0, mu1 = 1.0, 1.2
    # log-likelihood increment for a Gaussian observation ratio_t
    inc = (mu1 - mu0) / (sigma * sigma) * (ratio - (mu0 + mu1) / 2.0)
    s = np.empty_like(inc)
    acc = 0.0
    for i, v in enumerate(inc):
        acc = max(0.0, acc + v)
        s[i] = acc
    return s


DETECTORS = {
    "excess_rate": detector_excess,
    "shewhart": detector_shewhart,
    "cusum": detector_cusum,
    "ewma": detector_ewma,
    "sprt": detector_sprt,
}


def alarm_index(score: np.ndarray, threshold: float, min_window: int = 30) -> int:
    start = max(0, min_window - 1)
    hits = np.flatnonzero(score[start:] > threshold)
    hits = hits + start
    return int(hits[0]) if hits.size else int(score.size)


def evaluate(detector_name, detector_fn, streams):
    # per-axis scores
    axes = []
    for s in streams:
        score = detector_fn(s["rms"])
        if score.size < 10:
            continue
        axes.append(
            {
                "bearing_id": s["bearing_id"],
                "dataset": s["dataset"],
                "axis": s["axis"],
                "failed": s["failed"],
                "score": score,
                "n_monitor": int(score.size),
            }
        )
    # per-bearing max score across axes
    by_bearing: dict[str, dict] = {}
    for a in axes:
        b = by_bearing.setdefault(
            a["bearing_id"],
            {"bearing_id": a["bearing_id"], "dataset": a["dataset"], "failed": a["failed"], "n_monitor": a["n_monitor"], "axes": {}},
        )
        b["axes"][a["axis"]] = a["score"]
        b["n_monitor"] = max(b["n_monitor"], a["n_monitor"])
    censored = [b for b in by_bearing.values() if not b["failed"]]
    failed = [b for b in by_bearing.values() if b["failed"]]
    max_censored = max(np.max(a) for b in censored for a in b["axes"].values())
    t_zero = float(max_censored * 1.01)
    result = {"zero_fa_threshold": t_zero, "max_censored_final": float(max_censored)}
    for ds in sorted({b["dataset"] for b in failed}):
        sub = [b for b in failed if b["dataset"] == ds]
        det = 0
        leads = []
        for b in sub:
            first = min(alarm_index(a, t_zero) for a in b["axes"].values())
            if first < b["n_monitor"]:
                det += 1
                leads.append(
                    {
                        "lead_snapshots": int(b["n_monitor"] - first),
                        "lead_pct_life": 100.0 * (b["n_monitor"] - first) / b["n_monitor"],
                        "lead_seconds": (b["n_monitor"] - first) * SECONDS_PER_SNAPSHOT[ds],
                    }
                )
        result[ds] = {
            "n": len(sub),
            "n_detected": det,
            "detection_rate": 100.0 * det / len(sub),
            "median_lead_pct_life": float(np.median([x["lead_pct_life"] for x in leads])) if leads else None,
            "median_lead_seconds": float(np.median([x["lead_seconds"] for x in leads])) if leads else None,
        }
    return result


def main() -> None:
    here = Path(os.environ.get("RTF_DATA_DIR", str(ROOT / "data" / "features")))
    streams = load_all_streams(here)
    eligible = [s for s in streams if len(s["rms"]) - 60 >= 10]
    summary = {"detectors": {}}
    summary["n_failed_bearings"] = len({s["bearing_id"] for s in eligible if s["failed"]})
    summary["n_censored_bearings"] = len({s["bearing_id"] for s in eligible if not s["failed"]})
    for name, fn in DETECTORS.items():
        summary["detectors"][name] = evaluate(name, fn, streams)
    out = ROOT / "results" / "results_run_to_failure_classical_baselines.json"
    out.write_text(json.dumps(summary, indent=2))
    # compact print
    for name, r in summary["detectors"].items():
        det = "/".join(f"{ds}:{r[ds]['n_detected']}/{r[ds]['n']}" for ds in ("ims", "xjtu", "femto"))
        print(f"{name:12s} T0={r['zero_fa_threshold']:.4f}  {det}")
    print("saved", out)


if __name__ == "__main__":
    main()
