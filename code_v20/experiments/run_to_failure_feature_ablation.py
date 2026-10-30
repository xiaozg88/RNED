#!/usr/bin/env python3
"""One-factor ablation: swap the scalar feature in the excess-rate detector.

Identical causal protocol and zero-false-alarm calibration as
run_to_failure_early_warning.py, but the monitored scalar is changed one at a
time (RMS / Pearson kurtosis / skewness / variance / peak-to-peak). The 1.2x
ratio rule is kept verbatim even though it was tuned on RMS; this is the whole
point of a one-factor ablation and is reported honestly.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IMS_FAILED = {("test1", 3), ("test1", 4), ("test2", 1), ("test3", 3)}
FEATURES = ["rms", "kurtosis_pearson", "skewness", "variance", "peak2peak"]
SECONDS_PER_SNAPSHOT = {"ims": 600.0, "xjtu": 60.0, "femto": 10.0}


def load_streams(here: Path) -> list[dict]:
    streams = []
    for test in ("test1", "test2", "test3"):
        df = pd.read_csv(here / f"{test}.csv")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        for b in sorted(df["bearing"].unique()):
            for ch in sorted(df[df["bearing"] == b]["channel"].unique()):
                sub = df[(df["bearing"] == b) & (df["channel"] == ch)].sort_values("timestamp")
                streams.append({"dataset": "ims", "bearing_id": f"ims_{test}_b{b}", "axis": int(ch),
                                "failed": (test, int(b)) in IMS_FAILED,
                                "x": sub[FEATURES].to_numpy(dtype=np.float64)})
    for csv, ds in (("xjtu_features.csv", "xjtu"), ("femto_features.csv", "femto")):
        df = pd.read_csv(here / csv)
        for (gr, b) in sorted({(g, b) for g, b in zip(df["group"], df["bearing"])}):
            for ax in sorted(df[(df["group"] == gr) & (df["bearing"] == b)]["axis"].unique()):
                sub = df[(df["group"] == gr) & (df["bearing"] == b) & (df["axis"] == ax)].sort_values("snapshot")
                streams.append({"dataset": ds, "bearing_id": f"{ds}_{b}", "axis": ax,
                                "failed": True, "x": sub[FEATURES].to_numpy(dtype=np.float64)})
    return streams


def online_rate(x: np.ndarray) -> tuple[float, np.ndarray]:
    base = float(np.median(x[10:60]))
    mon = x[60:]
    ex = np.maximum(mon / base - 1.2, 0.0)
    return base, np.cumsum(ex) / np.arange(1, ex.size + 1)


def alarm_index(rate: np.ndarray, thr: float, min_window: int = 30) -> int:
    hits = np.flatnonzero(rate[min_window - 1:] > thr)
    return int(hits[0] + min_window - 1) if hits.size else int(rate.size)


def evaluate(streams: list[dict], fi: int) -> dict:
    by_bearing: dict[str, dict] = {}
    for s in streams:
        if s["x"].shape[0] < 70:
            continue
        base, rate = online_rate(s["x"][:, fi])
        bd = by_bearing.setdefault(s["bearing_id"], {"dataset": s["dataset"], "failed": s["failed"],
                                                      "n_monitor": int(rate.size), "axes": {}})
        bd["axes"][s["axis"]] = rate
    censored = [b for b in by_bearing.values() if not b["failed"]]
    failed = [b for b in by_bearing.values() if b["failed"]]
    t_zero = float(max(np.max(a) for b in censored for a in b["axes"].values()) * 1.01)
    det = 0
    leads = []
    for b in failed:
        first = min(alarm_index(a, t_zero) for a in b["axes"].values())
        if first < b["n_monitor"]:
            det += 1
            leads.append(100.0 * (b["n_monitor"] - first) / b["n_monitor"])
    return {"n_failed": len(failed), "n_detected": det,
            "detection_rate": 100.0 * det / len(failed),
            "median_lead_pct_life": float(np.median(leads)) if leads else None,
            "zero_fa_threshold": t_zero}


def main() -> None:
    here = Path(os.environ.get("RTF_DATA_DIR", str(ROOT / "data" / "features")))
    streams = load_streams(here)
    out = {}
    for fi, name in enumerate(FEATURES):
        r = evaluate(streams, fi)
        out[name] = r
        print(f"{name:16s} det={r['n_detected']}/{r['n_failed']} ({r['detection_rate']:.1f}%)  "
              f"median_lead={r['median_lead_pct_life'] if r['median_lead_pct_life'] is not None else None}")
    dst = ROOT / "results" / "results_run_to_failure_feature_ablation.json"
    dst.write_text(json.dumps(out, indent=2))
    print("saved", dst)


if __name__ == "__main__":
    main()
