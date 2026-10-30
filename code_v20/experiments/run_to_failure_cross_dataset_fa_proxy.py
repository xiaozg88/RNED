#!/usr/bin/env python3
"""Cross-dataset false-alarm proxy from early-life pseudo-health segments (P0-3).

The zero-false-alarm threshold T=0.0424 is calibrated on the 8 IMS censored
bearings only, so cross-rig false alarms are unmeasurable in the strict sense.
As a measurable proxy, we take the early-life segment of each XJTU-SY and
FEMTO bearing (all of which eventually fail but are healthy early in life),
run the same causal rate detector over snapshots 61-150, and report how often
it would raise a (spurious) alarm against T=0.0424. This exceedance rate is a
point estimate of residual false alarms, not a strict bound; we also report
its Clopper-Pearson 95% confidence interval.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

T_ZERO_FA = 0.0424
EARLY_SNAPSHOTS = 150  # early-life pseudo-health window length (inclusive)


def load_early_streams(here: Path) -> list[dict]:
    streams = []
    for csv, ds in (("xjtu_features.csv", "xjtu"), ("femto_features.csv", "femto")):
        df = pd.read_csv(here / csv)
        for (group, bearing) in sorted({(g, b) for g, b in zip(df["group"], df["bearing"])}):
            for axis in sorted(df[(df["group"] == group) & (df["bearing"] == bearing)]["axis"].unique()):
                sub = df[(df["group"] == group) & (df["bearing"] == bearing) & (df["axis"] == axis)]
                sub = sub.sort_values("snapshot")
                rms = sub["rms"].to_numpy(dtype=np.float64)
                if rms.size < 150:
                    continue
                streams.append(
                    {
                        "dataset": ds,
                        "bearing_id": f"{ds}_{bearing}",
                        "axis": axis,
                        "rms": rms[:EARLY_SNAPSHOTS],
                    }
                )
    return streams


def rate_series(rms: np.ndarray) -> np.ndarray:
    baseline = float(np.median(rms[10:60]))
    monitor = rms[60:]
    excess = np.maximum(monitor / baseline - 1.2, 0.0)
    return np.cumsum(excess) / np.arange(1, excess.size + 1)


def alarms_within(rate: np.ndarray, threshold: float, min_window: int = 30) -> bool:
    start = max(0, min_window - 1)
    hits = np.flatnonzero(rate[start:] > threshold)
    return bool(hits.size)


def clopper_pearson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact two-sided Clopper-Pearson 95% confidence interval for k/n."""
    if n <= 0:
        return (0.0, 1.0)
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1.0 - alpha / 2, k + 1, n - k))
    return (lo, hi)


def main() -> None:
    here = Path(os.environ.get("RTF_DATA_DIR", str(ROOT / "data" / "features")))
    streams = load_early_streams(here)
    per_axis = []
    for s in streams:
        rate = rate_series(s["rms"])
        per_axis.append(
            {
                "dataset": s["dataset"],
                "bearing_id": s["bearing_id"],
                "axis": s["axis"],
                "max_rate": float(np.max(rate)),
                "alarmed": alarms_within(rate, T_ZERO_FA),
            }
        )
    df = pd.DataFrame(per_axis)
    n_alarmed = int(df["alarmed"].sum())
    ci_lo, ci_hi = clopper_pearson_ci(n_alarmed, int(len(df)))
    out = {
        "threshold": T_ZERO_FA,
        "early_life_snapshots": EARLY_SNAPSHOTS,
        "n_pseudo_health_axes": int(len(df)),
        "n_pseudo_health_bearings": int(df["bearing_id"].nunique()),
        "n_alarmed_axes": n_alarmed,
        "axis_false_alarm_rate_pct": float(100.0 * df["alarmed"].mean()),
        "axis_false_alarm_ci95_pct": [round(100.0 * ci_lo, 1), round(100.0 * ci_hi, 1)],
        "per_dataset": {},
    }
    for ds, sub in df.groupby("dataset"):
        per_bearing = sub.groupby("bearing_id")["alarmed"].any()
        out["per_dataset"][ds] = {
            "n_axes": int(len(sub)),
            "n_bearings": int(per_bearing.size),
            "axis_false_alarm_rate_pct": float(100.0 * sub["alarmed"].mean()),
            "bearing_false_alarm_rate_pct": float(100.0 * per_bearing.mean()),
            "max_rate_over_all_axes": float(sub["max_rate"].max()),
        }
    dst = ROOT / "results" / "results_run_to_failure_cross_dataset_fa_proxy.json"
    dst.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print("saved", dst)


if __name__ == "__main__":
    main()
