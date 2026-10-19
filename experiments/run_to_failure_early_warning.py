#!/usr/bin/env python3
"""Deployable degradation early-warning detector and lead-time analysis.

Online (causal) detector: after a 60-snapshot calibration period, maintain the
cumulative excess of RMS above 1.2x the baseline. A bearing alarms when any
axis crosses a fixed threshold. The threshold is swept, and at the zero-false-
alarm operating point (calibrated on the IMS censored bearings) the detection
rate and lead time are reported per dataset.
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

FEATURES = ["rms", "kurtosis_pearson", "skewness", "variance", "peak2peak"]
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


def online_excess_rate(rms: np.ndarray) -> tuple[float, np.ndarray]:
    """Return (baseline, running-mean excess from snapshot 61 onward)."""
    baseline = float(np.median(rms[10:60]))
    monitor = rms[60:]
    excess = np.maximum(monitor / baseline - 1.2, 0.0)
    rate = np.cumsum(excess) / np.arange(1, excess.size + 1)
    return baseline, rate


def alarm_index(rate: np.ndarray, threshold: float, min_window: int = 30) -> int:
    start = max(0, min_window - 1)
    hits = np.flatnonzero(rate[start:] > threshold)
    hits = hits + start
    return int(hits[0]) if hits.size else int(rate.size)


def main() -> None:
    here = Path(os.environ.get("RTF_DATA_DIR", str(ROOT / "data" / "features")))
    streams = load_all_streams(here)
    # Per-axis online scores; bearing = max (any-axis) excess rate.
    by_bearing: dict[str, dict] = {}
    for s in streams:
        base, rate = online_excess_rate(s["rms"])
        if rate.size < 10:
            continue
        b = by_bearing.setdefault(
            s["bearing_id"],
            {
                "bearing_id": s["bearing_id"],
                "dataset": s["dataset"],
                "failed": s["failed"],
                "axes": {},
                "n_monitor": int(rate.size),
            },
        )
        b["axes"][s["axis"]] = {"baseline": base, "rate": rate}

    # Threshold sweep: detection rate, false-alarm rate, median lead time.
    censored = [b for b in by_bearing.values() if not b["failed"]]
    failed = [b for b in by_bearing.values() if b["failed"]]
    all_final = [max(np.max(a["rate"]) for a in b["axes"].values()) for b in by_bearing.values()]
    thresholds = np.linspace(0.01, max(all_final) * 1.2, 200)
    sweep = []
    for t in thresholds:
        def first_alarm(b):
            return min(alarm_index(a["rate"], t) for a in b["axes"].values())

        det = [first_alarm(b) < b["n_monitor"] for b in failed]
        fa = [first_alarm(b) < b["n_monitor"] for b in censored]
        lead_pct = [
            100.0 * (b["n_monitor"] - first_alarm(b)) / b["n_monitor"]
            for b in failed
            if first_alarm(b) < b["n_monitor"]
        ]
        sweep.append(
            {
                "threshold": round(float(t), 3),
                "detection_rate": 100.0 * np.mean(det) if det else np.nan,
                "false_alarm_rate": 100.0 * np.mean(fa) if fa else np.nan,
                "median_lead_pct": float(np.median(lead_pct)) if lead_pct else np.nan,
            }
        )

    # Zero-false-alarm operating point on the IMS censored bearings.
    max_censored = max(
        np.max(a["rate"]) for b in censored for a in b["axes"].values()
    )
    T_zero_fa = float(max_censored * 1.01)
    zero_fa = {}
    for ds in sorted({b["dataset"] for b in failed}):
        sub = [b for b in failed if b["dataset"] == ds]
        detected = []
        leads = []
        for b in sub:
            fa = min(alarm_index(a["rate"], T_zero_fa) for a in b["axes"].values())
            if fa < b["n_monitor"]:
                detected.append(b["bearing_id"])
                leads.append(
                    {
                        "bearing": b["bearing_id"],
                        "lead_snapshots": int(b["n_monitor"] - fa),
                        "lead_pct_life": 100.0 * (b["n_monitor"] - fa) / b["n_monitor"],
                        "lead_seconds": (b["n_monitor"] - fa) * SECONDS_PER_SNAPSHOT[ds],
                    }
                )
        zero_fa[ds] = {
            "n": len(sub),
            "n_detected": len(detected),
            "detection_rate": 100.0 * len(detected) / len(sub),
            "median_lead_pct_life": float(np.median([x["lead_pct_life"] for x in leads])) if leads else np.nan,
            "median_lead_seconds": float(np.median([x["lead_seconds"] for x in leads])) if leads else np.nan,
            "per_bearing": leads,
        }

    summary = {
        "n_failed": len(failed),
        "n_censored": len(censored),
        "zero_fa_threshold": T_zero_fa,
        "max_censored_final": float(max_censored),
        "threshold_sweep": sweep,
        "zero_fa_operating_point": zero_fa,
    }
    out = ROOT / "results" / "results_run_to_failure_early_warning.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "threshold_sweep"}, indent=2))
    print("saved", out)


if __name__ == "__main__":
    main()
