#!/usr/bin/env python3
"""Fig 5: operating curve - per-bearing detection rate vs median lead time across c.

For each RMS-ratio threshold c we:
  1. compute the online cumulative excess rate per axis (baseline = median of
     snapshots 10:60, monitored from snapshot 61 onward);
  2. set the alarm threshold T(c) at 1.01x the largest final rate among the
     censored (non-failing) IMS bearings -> zero false alarms by construction;
  3. declare a bearing detected when ANY axis crosses T(c) before end of life.
Detection is thus reported at the bearing level (matching the 31/34 headline).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "figures"))
from paper_plot_style import *  # noqa: E402,F401

IMS_FAILED = {("test1", 3), ("test1", 4), ("test2", 1), ("test3", 3)}


def load_bearings() -> list[dict]:
    bearings: dict[str, dict] = {}
    for t in ("test1", "test2", "test3"):
        d = pd.read_csv(ROOT / f"data/features/{t}.csv")
        d["ts"] = pd.to_datetime(d["timestamp"])
        for b, g in d.groupby("bearing"):
            bid = f"ims_{t}_b{b}"
            for ch, gg in g.groupby("channel"):
                gg = gg.sort_values("ts")
                r = gg["rms"].to_numpy(np.float64)
                if r.size < 70:
                    continue
                bd = bearings.setdefault(bid, {"bearing_id": bid, "dataset": "ims", "failed": (t, int(b)) in IMS_FAILED, "axes": {}})
                bd["axes"][int(ch)] = r
    for csv, ds in (("xjtu_features.csv", "xjtu"), ("femto_features.csv", "femto")):
        d = pd.read_csv(ROOT / f"data/features/{csv}")
        for (gr, b), g in d.groupby(["group", "bearing"]):
            bid = f"{ds}_{b}"
            for ax, gg in g.groupby("axis"):
                gg = gg.sort_values("snapshot")
                r = gg["rms"].to_numpy(np.float64)
                if r.size < 70:
                    continue
                bd = bearings.setdefault(bid, {"bearing_id": bid, "dataset": ds, "failed": True, "axes": {}})
                bd["axes"][ax] = r
    return list(bearings.values())


def rate(rms: np.ndarray, c: float) -> np.ndarray:
    base = float(np.median(rms[10:60]))
    mon = rms[60:]
    ex = np.maximum(mon / base - c, 0.0)
    return np.cumsum(ex) / np.arange(1, ex.size + 1)


def first_alarm(rates: list[np.ndarray], thr: float, min_window: int = 30) -> int:
    n = max(r.size for r in rates)
    idx = n
    start = min_window - 1
    for r in rates:
        hits = np.flatnonzero(r[start:] > thr)
        if hits.size:
            idx = min(idx, int(hits[0] + start))
    return idx


def main() -> None:
    bearings = load_bearings()
    censored = [b for b in bearings if not b["failed"]]
    failed = [b for b in bearings if b["failed"]]
    print(f"bearings: {len(bearings)} total, {len(failed)} failed, {len(censored)} censored")

    cs = [round(v, 2) for v in np.arange(1.10, 2.001, 0.05)]
    rows = []
    for c in cs:
        score = {b["bearing_id"]: {ax: rate(r, c) for ax, r in b["axes"].items()} for b in bearings}
        T = max(np.max(list(score[b["bearing_id"]].values())) for b in censored) * 1.01
        det = 0
        leads = []
        for b in failed:
            n = max(r.size for r in score[b["bearing_id"]].values())
            fa = first_alarm(list(score[b["bearing_id"]].values()), T)
            if fa < n:
                det += 1
                leads.append(100.0 * (n - fa) / n)
        dr = 100.0 * det / len(failed)
        ml = float(np.median(leads)) if leads else np.nan
        rows.append({"c": c, "detection_rate": dr, "median_lead_pct": ml})
        print(f"c={c:4.2f}  det={det}/{len(failed)} ({dr:5.1f}%)  median_lead={ml if np.isnan(ml) else round(ml, 1)}%")

    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(4.8, 3.4), constrained_layout=True)
    sc = ax.scatter(df["median_lead_pct"], df["detection_rate"], c=df["c"],
                    cmap="viridis", s=46, alpha=0.75, linewidths=0)
    ax.set_xlabel("median lead time (% of life)")
    ax.set_ylabel("detection rate (% of bearings)")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("c (RMS ratio threshold)")
    # annotate only the operating points that bracket the trade-off
    for c, dx, dy, va in ((1.1, 0, -8, "top"), (1.2, 0, 7, "bottom"), (2.0, 0, 7, "bottom")):
        r = df[df["c"] == c].iloc[0]
        ax.annotate(f"c={c}", (r["median_lead_pct"], r["detection_rate"]),
                    textcoords="offset points", xytext=(dx, dy), ha="center",
                    va=va, fontsize=FONT_SIZE - 1)
    ax.set_ylim(70, 100)
    ax.set_xlim(0, 50)
    save_fig(fig, "fig5_operating_curve")

    out = ROOT / "results" / "fig5_operating_curve.csv"
    df.to_csv(out, index=False)
    print("saved", out)


if __name__ == "__main__":
    main()
