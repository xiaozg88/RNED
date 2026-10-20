#!/usr/bin/env python3
"""Robustness checks: fleet independence, FEMTO misses, mixing-coefficient decay.

Three cheap, data-grounded checks that close remaining hand-waves:
  (2) fleet independence: bootstrap the 27 pseudo-health bearings (preserving
      within-bearing axis clustering) to bound the fleet false-alarm estimate,
      and contrast axis-independent vs bearing-clustered aggregation;
  (3) FEMTO misses: verify that the three missed FEMTO bearings are genuinely
      low-amplitude (max RMS ratio and cumulative excess vs detected bearings);
  (4) mixing decay: estimate the average sample autocorrelation of the censored
      excess sequences and fit a power law to check the Proposition-5 a>2
      plausibility.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "experiments"))
import run_to_failure_classical_baselines as cb  # noqa: E402

T = 0.0424
EARLY = 150


def load_pseudo_health(here: Path) -> list[dict]:
    out = []
    for csv, ds in (("xjtu_features.csv", "xjtu"), ("femto_features.csv", "femto")):
        df = pd.read_csv(here / csv)
        for (g, bearing) in sorted({(a, b) for a, b in zip(df["group"], df["bearing"])}):
            for axis in sorted(df[(df["group"] == g) & (df["bearing"] == bearing)]["axis"].unique()):
                sub = df[(df["group"] == g) & (df["bearing"] == bearing) & (df["axis"] == axis)]
                sub = sub.sort_values("snapshot")
                rms = sub["rms"].to_numpy(dtype=np.float64)
                if rms.size < EARLY:
                    continue
                out.append({"dataset": ds, "bearing": f"{ds}_{bearing}", "axis": axis, "rms": rms[:EARLY]})
    return out


def axis_alarms(stream: dict) -> bool:
    base = float(np.median(stream["rms"][10:60]))
    ratio = stream["rms"][60:] / base
    excess = np.maximum(ratio - 1.2, 0.0)
    rate = np.cumsum(excess) / np.arange(1, excess.size + 1)
    return bool(np.flatnonzero(rate[29:] > T).size)


def check_fleet(streams: list[dict]) -> dict:
    bear = {}
    for s in streams:
        bear.setdefault(s["bearing"], []).append(axis_alarms(s))
    # per-axis rate and per-bearing (any-axis) rate
    axes_alarm = [a for axs in bear.values() for a in axs]
    p_axis = float(np.mean(axes_alarm))
    p_bearing = float(np.mean([any(axs) for axs in bear.values()]))
    # within-bearing correlation: fraction of bearings with mixed (some axes alarm, some not)
    mixed = sum(1 for axs in bear.values() if any(axs) and not all(axs))
    n_alarm_bear = sum(1 for axs in bear.values() if any(axs))
    # bootstrap over bearings for the per-axis p and fleet numbers
    bearings = list(bear.values())
    rng = np.random.default_rng(0)
    fleet100_expected = []
    fleet100_pge1 = []
    for _ in range(2000):
        sample = [bearings[i] for i in rng.integers(0, len(bearings), len(bearings))]
        ax = [a for axs in sample for a in axs]
        p = float(np.mean(ax))
        fleet100_expected.append(200 * p)  # 100 machines x 2 axes
        fleet100_pge1.append(1.0 - (1.0 - p) ** 200)
    return {
        "n_bearings": len(bearings),
        "n_axes": len(axes_alarm),
        "p_axis": p_axis,
        "p_bearing": p_bearing,
        "mixed_alarm_bearings": mixed,
        "n_alarm_bearings": n_alarm_bear,
        "bootstrap_p_axis_ci": [float(np.percentile(fleet100_expected, q) / 200) for q in (2.5, 97.5)],
        "fleet100_expected_fa_per_epoch": {
            "point": 200 * p_axis,
            "bootstrap_ci": [float(np.percentile(fleet100_expected, 2.5)), float(np.percentile(fleet100_expected, 97.5))],
        },
        "fleet100_p_ge1": {
            "point": 1.0 - (1.0 - p_axis) ** 200,
            "bootstrap_ci": [float(np.percentile(fleet100_pge1, 2.5)), float(np.percentile(fleet100_pge1, 97.5))],
        },
    }


def check_femto_misses(here: Path) -> dict:
    df = pd.read_csv(here / "femto_features.csv")
    missed = {"Bearing1_5", "Bearing1_6", "Bearing2_4"}
    rows = []
    for bearing in sorted(df["bearing"].unique()):
        for axis in sorted(df[df["bearing"] == bearing]["axis"].unique()):
            sub = df[(df["bearing"] == bearing) & (df["axis"] == axis)].sort_values("snapshot")
            rms = sub["rms"].to_numpy(dtype=np.float64)
            base = float(np.median(rms[10:60]))
            ratio = rms[60:] / base
            ecum = float(np.sum(np.maximum(ratio - 1.2, 0.0)))
            rows.append(
                {
                    "bearing": bearing,
                    "axis": axis,
                    "missed": bearing in missed,
                    "max_rms_ratio": float(np.max(ratio)),
                    "cumulative_excess": ecum,
                }
            )
    rdf = pd.DataFrame(rows)
    per_bearing = rdf.groupby("bearing").agg(
        missed=("missed", "max"),
        max_rms_ratio=("max_rms_ratio", "max"),
        cumulative_excess=("cumulative_excess", "max"),
    ).reset_index()
    det = per_bearing[~per_bearing["missed"]]
    mis = per_bearing[per_bearing["missed"]]
    return {
        "detected_median_max_rms_ratio": float(det["max_rms_ratio"].median()),
        "missed_median_max_rms_ratio": float(mis["max_rms_ratio"].median()),
        "detected_median_cumulative_excess": float(det["cumulative_excess"].median()),
        "missed_median_cumulative_excess": float(mis["cumulative_excess"].median()),
        "missed_bearings": [
            {"bearing": r["bearing"], "max_rms_ratio": r["max_rms_ratio"], "cumulative_excess": r["cumulative_excess"]}
            for _, r in mis.iterrows()
        ],
    }


def check_mixing(streams: list[dict]) -> dict:
    # censored IMS streams -> per-axis excess sequence autocorrelation
    rng = np.random.default_rng(0)
    censored = [s for s in streams if not s["failed"] and s["dataset"] == "ims"]
    acf = []
    maxlag = 60
    for s in censored:
        _, ratio = cb._base(s["rms"])
        e = np.maximum(ratio - 1.2, 0.0)
        e = e - e.mean()
        if e.std() == 0:
            continue
        ac = [1.0]
        for k in range(1, maxlag + 1):
            ac.append(float(np.corrcoef(e[:-k], e[k:])[0, 1]))
        acf.append(ac)
    acf = np.array(acf)
    mean_acf = acf.mean(axis=0)
    # fit power law |rho(k)| ~ k^{-a} on lags 1..maxlag (positive values only)
    lags = np.arange(1, maxlag + 1)
    pos = mean_acf[1:] > 0
    a = None
    if pos.sum() >= 10:
        a = -float(np.polyfit(np.log(lags[pos]), np.log(np.clip(mean_acf[1:][pos], 1e-6, None)), 1)[0])
    return {
        "n_censored_axes": int(acf.shape[0]),
        "acf_lag1": float(np.mean(acf[:, 1])),
        "acf_lag5": float(np.mean(acf[:, 5])),
        "acf_lag20": float(np.mean(acf[:, 20])),
        "fitted_powerlaw_exponent_a": round(a, 2) if a is not None else None,
    }


def main() -> None:
    here = Path(os.environ.get("RTF_DATA_DIR", str(ROOT / "data" / "features")))
    ph = load_pseudo_health(here)
    all_streams = cb.load_all_streams(here)
    summary = {
        "fleet_independence": check_fleet(ph),
        "femto_misses": check_femto_misses(here),
        "mixing_decay": check_mixing(all_streams),
    }
    dst = ROOT / "results" / "results_run_to_failure_robustness_checks.json"
    dst.write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2))
    print("saved", dst)


if __name__ == "__main__":
    main()
