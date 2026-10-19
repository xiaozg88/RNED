#!/usr/bin/env python3
"""Protocol independence / sensitivity analysis for the deployable detector.

Answers: do the two headline conclusions depend on the protocol choices?
  1. the rate-normalised excess detector is the best zero-FA online detector;
  2. detection ~31/34 with the c = 1.2 operating point.

Variations swept (each applied one-at-a-time around the default):
  baseline_end in {40, 50, 60}      (default 60)
  c in [1.10 .. 3.00]               (default 1.20)
  runin in {5, 10, 15}              (default 10)
  min_monitor in {10, 30, 60}       (default 30)
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


def load_all_streams(here: Path):
    streams = []
    for test in ("test1", "test2", "test3"):
        df = pd.read_csv(here / f"{test}.csv")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        for bearing in sorted(df["bearing"].unique()):
            for channel in sorted(df[df["bearing"] == bearing]["channel"].unique()):
                sub = df[(df["bearing"] == bearing) & (df["channel"] == channel)].sort_values("timestamp")
                streams.append(dict(dataset="ims", bearing_id=f"ims_{test}_b{bearing}", axis=int(channel),
                                   failed=(test, int(bearing)) in IMS_FAILED,
                                   rms=sub["rms"].to_numpy(dtype=np.float64)))
    for csv, ds in (("xjtu_features.csv", "xjtu"), ("femto_features.csv", "femto")):
        df = pd.read_csv(here / csv)
        for (group, bearing) in sorted({(g, b) for g, b in zip(df["group"], df["bearing"])}):
            for axis in sorted(df[(df["group"] == group) & (df["bearing"] == bearing)]["axis"].unique()):
                sub = df[(df["group"] == group) & (df["bearing"] == bearing) & (df["axis"] == axis)].sort_values("snapshot")
                streams.append(dict(dataset=ds, bearing_id=f"{ds}_{bearing}", axis=axis, failed=True,
                                   rms=sub["rms"].to_numpy(dtype=np.float64)))
    return streams


def base_ratio(rms, runin=10, base_end=60):
    base = float(np.median(rms[runin:base_end]))
    return base, rms[base_end:] / base


def score_excess(rms, runin=10, base_end=60, c=1.2):
    _, ratio = base_ratio(rms, runin, base_end)
    inc = np.maximum(ratio - c, 0.0)
    return np.cumsum(inc) / np.arange(1, inc.size + 1)


def score_cusum(rms, runin=10, base_end=60, c=1.2):
    _, ratio = base_ratio(rms, runin, base_end)
    inc = np.maximum(ratio - c, 0.0)
    s = np.empty_like(inc)
    acc = 0.0
    for i, v in enumerate(inc):
        acc = max(0.0, acc + v)
        s[i] = acc
    return s


def score_shewhart(rms, runin=10, base_end=60, c=1.2):
    _, ratio = base_ratio(rms, runin, base_end)
    return ratio


def score_ewma(rms, runin=10, base_end=60, c=1.2):
    _, ratio = base_ratio(rms, runin, base_end)
    inc = np.maximum(ratio - c, 0.0)
    z = np.empty_like(inc)
    acc = 0.0
    for i, v in enumerate(inc):
        acc = 0.2 * v + 0.8 * acc
        z[i] = acc
    return z


def score_sprt(rms, runin=10, base_end=60, c=1.2):
    base, ratio = base_ratio(rms, runin, base_end)
    sigma = max(float(np.std(rms[runin:base_end]) / base), 1e-9)
    inc = (c - 1.0) / (sigma * sigma) * (ratio - (1.0 + c) / 2.0)
    s = np.empty_like(inc)
    acc = 0.0
    for i, v in enumerate(inc):
        acc = max(0.0, acc + v)
        s[i] = acc
    return s


def alarm_index(score, threshold, min_window=30):
    start = max(0, min_window - 1)
    hits = np.flatnonzero(score[start:] > threshold)
    hits = hits + start
    return int(hits[0]) if hits.size else int(score.size)


def evaluate(fn, streams, runin=10, base_end=60, c=1.2, min_window=30):
    by_bearing = {}
    for s in streams:
        score = fn(s["rms"], runin, base_end, c)
        if score.size < 10:
            continue
        b = by_bearing.setdefault(s["bearing_id"], dict(dataset=s["dataset"], failed=s["failed"],
                                                        n_monitor=int(score.size), axes={}))
        b["axes"][s["axis"]] = score
        b["n_monitor"] = max(b["n_monitor"], int(score.size))
    censored = [b for b in by_bearing.values() if not b["failed"]]
    failed = [b for b in by_bearing.values() if b["failed"]]
    t_zero = float(max(np.max(a) for b in censored for a in b["axes"].values()) * 1.01)
    det = sum(1 for b in failed
              if min(alarm_index(a, t_zero, min_window) for a in b["axes"].values()) < b["n_monitor"])
    return det, len(failed), t_zero


def main():
    here = Path(os.environ.get("RTF_DATA_DIR", str(ROOT / "data" / "features")))
    streams = load_all_streams(here)
    eligible = [s for s in streams if len(s["rms"]) - 60 >= 10]
    n_failed = len({s["bearing_id"] for s in eligible if s["failed"]})
    n_cens = len({s["bearing_id"] for s in eligible if not s["failed"]})
    out = {"n_failed": n_failed, "n_censored": n_cens, "default": {}}

    # c-sweep x baseline-end (headline: 1.2x choice and protocol robustness)
    out["c_sweep"] = {}
    for base_end in (40, 50, 60):
        for c in [round(x, 2) for x in np.arange(1.10, 3.001, 0.05)]:
            det, n, t0 = evaluate(score_excess, streams, base_end=base_end, c=c)
            out["c_sweep"][f"base{base_end}|c{c:.2f}"] = {"detected": det, "n": n, "threshold": round(t0, 4)}

    # run-in sensitivity (default c=1.2, base_end=60)
    out["runin"] = {f"runin{r}": evaluate(score_excess, streams, runin=r) for r in (5, 10, 15)}
    # min-monitor sensitivity
    out["min_monitor"] = {f"min{m}": evaluate(score_excess, streams, min_window=m) for m in (10, 30, 60)}

    # ranking robustness: all five detectors across baseline windows
    dets = {"excess": score_excess, "cusum": score_cusum, "shewhart": score_shewhart,
            "ewma": score_ewma, "sprt": score_sprt}
    out["ranking"] = {}
    for base_end in (40, 50, 60):
        out["ranking"][f"base{base_end}"] = {name: evaluate(fn, streams, base_end=base_end)[0]
                                             for name, fn in dets.items()}

    dest = ROOT / "results" / "results_run_to_failure_protocol_sensitivity.json"
    dest.write_text(json.dumps(out, indent=2, default=float))

    # compact report
    print(f"evaluated {n_failed} failed / {n_cens} censored bearings")
    print("\nranking (detected/34) across baseline windows:")
    for base_end in (40, 50, 60):
        row = out["ranking"][f"base{base_end}"]
        print(f"  base{base_end}: " + "  ".join(f"{k}={v}" for k, v in row.items()))
    print("\nrun-in:", {k: v[0] for k, v in out["runin"].items()})
    print("min-monitor:", {k: v[0] for k, v in out["min_monitor"].items()})
    # c-sweep headline at base60
    print("\nc-sweep (base60) detection rate:")
    for c in (1.1, 1.15, 1.2, 1.5, 2.0, 2.5, 3.0):
        e = out["c_sweep"][f"base60|c{c:.2f}"]
        print(f"  c={c:.2f}: {e['detected']}/{e['n']}  T0={e['threshold']}")
    print("saved", dest)


if __name__ == "__main__":
    main()
