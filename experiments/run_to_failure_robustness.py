#!/usr/bin/env python3
"""Robustness of the deployable detector to sensor noise and axis dropout.

The threshold T = 0.0424 is fixed at the clean-data calibration and is NOT
re-tuned under perturbation. Noise: additive Gaussian on the RMS stream with
std = a * median(rms[11:60]) for a in {0.01, 0.02, 0.05, 0.10}, applied to the
whole stream (baseline and monitoring, as in deployment). Axis dropout: for
each failed bearing each axis is removed in turn and the any-axis detection is
recomputed; the worst-case (min) detection over dropouts is reported.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import run_to_failure_early_warning as ew  # noqa: E402

T_FIXED = 0.0424
np.random.seed(0)


def detection(streams, threshold, seed_off=0):
    by_bearing = {}
    for s in streams:
        r = s["rms"]
        if r.size < 70:
            continue
        _, rate = ew.online_excess_rate(r)
        bd = by_bearing.setdefault(s["bearing_id"], {"failed": s["failed"], "n": int(rate.size), "axes": {}})
        bd["axes"][s["axis"]] = rate
    failed = {bid: b for bid, b in by_bearing.items() if b["failed"]}
    det = 0
    for b in failed.values():
        first = min(ew.alarm_index(a, threshold) for a in b["axes"].values())
        det += int(first < b["n"])
    return det, len(failed)


def main() -> None:
    here = Path(os.environ.get("RTF_DATA_DIR", str(ROOT / "data" / "features")))
    streams = ew.load_all_streams(here)
    base_det, n = detection(streams, T_FIXED)
    out = {"n_failed": n, "clean_detection": f"{base_det}/{n}", "threshold": T_FIXED}

    # sensor noise
    noise = {}
    for a in (0.01, 0.02, 0.05, 0.10):
        pert = []
        for s in streams:
            r = s["rms"].copy()
            base = float(np.median(r[10:60]))
            r = r + np.random.normal(0.0, a * base, size=r.shape)
            s2 = dict(s)
            s2["rms"] = r
            pert.append(s2)
        d, _ = detection(pert, T_FIXED)
        noise[f"noise_{a}"] = f"{d}/{n}"

    # axis dropout (failed bearings only): drop one axis of one bearing at a
    # time while keeping all other bearings intact; report worst/median over
    # the 68 (bearing, dropped-axis) trials, all against the full 34 failed set.
    failed_streams = [s for s in streams if s["failed"]]
    cens_streams = [s for s in streams if not s["failed"]]
    groups = {}
    for s in failed_streams:
        groups.setdefault(s["bearing_id"], []).append(s)
    dets = []
    for bid, gs in groups.items():
        others = [s for s in failed_streams if s["bearing_id"] != bid]
        for drop_ax in sorted({s["axis"] for s in gs}):
            kept = [s for s in gs if s["axis"] != drop_ax]
            if not kept:
                continue
            d, _ = detection(kept + others + cens_streams, T_FIXED)
            dets.append(d)
    dropout = {"n_bearing_axis_dropout_trials": len(dets),
               "min_detection": f"{min(dets)}/{n}" if dets else None,
               "median_detection": f"{int(np.median(dets))}/{n}" if dets else None}

    out["sensor_noise"] = noise
    out["axis_dropout"] = dropout
    dst = ROOT / "results" / "results_run_to_failure_robustness.json"
    dst.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print("saved", dst)


if __name__ == "__main__":
    main()
