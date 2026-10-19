#!/usr/bin/env python3
"""Paired significance test for the five online detectors (RQ4).

For each failed bearing (n=34) record whether each detector alarms before end
of life under its own zero-false-alarm threshold. Excess-rate is compared to
each baseline with McNemar's exact test (matched binary outcomes); p-values
are Holm-corrected across the four pre-declared comparisons.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
IMS_FAILED = {("test1", 3), ("test1", 4), ("test2", 1), ("test3", 3)}


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
                                "rms": sub["rms"].to_numpy(dtype=np.float64)})
    for csv, ds in (("xjtu_features.csv", "xjtu"), ("femto_features.csv", "femto")):
        df = pd.read_csv(here / csv)
        for (gr, b) in sorted({(g, b) for g, b in zip(df["group"], df["bearing"])}):
            for ax in sorted(df[(df["group"] == gr) & (df["bearing"] == b)]["axis"].unique()):
                sub = df[(df["group"] == gr) & (df["bearing"] == b) & (df["axis"] == ax)].sort_values("snapshot")
                streams.append({"dataset": ds, "bearing_id": f"{ds}_{b}", "axis": ax,
                                "failed": True, "rms": sub["rms"].to_numpy(dtype=np.float64)})
    return streams


def _base(rms):
    base = float(np.median(rms[10:60]))
    return base, rms[60:] / base


def det_excess(rms):
    _, ratio = _base(rms)
    ex = np.maximum(ratio - 1.2, 0.0)
    return np.cumsum(ex) / np.arange(1, ex.size + 1)


def det_shewhart(rms):
    _, ratio = _base(rms)
    return ratio


def det_cusum(rms):
    _, ratio = _base(rms)
    inc = np.maximum(ratio - 1.2, 0.0)
    s = np.empty_like(inc); acc = 0.0
    for i, v in enumerate(inc):
        acc = max(0.0, acc + v); s[i] = acc
    return s


def det_ewma(rms):
    _, ratio = _base(rms)
    inc = np.maximum(ratio - 1.2, 0.0)
    z = np.empty_like(inc); acc = 0.0
    for i, v in enumerate(inc):
        acc = 0.2 * v + 0.8 * acc; z[i] = acc
    return z


def det_sprt(rms):
    base, ratio = _base(rms)
    sigma = max(float(np.std(rms[10:60]) / base), 1e-9)
    inc = (0.2) / (sigma * sigma) * (ratio - 1.1)
    s = np.empty_like(inc); acc = 0.0
    for i, v in enumerate(inc):
        acc = max(0.0, acc + v); s[i] = acc
    return s


DETECTORS = {"excess_rate": det_excess, "shewhart": det_shewhart,
             "cusum": det_cusum, "ewma": det_ewma, "sprt": det_sprt}


def alarm_index(score, thr, min_window=30):
    hits = np.flatnonzero(score[min_window - 1:] > thr)
    return int(hits[0] + min_window - 1) if hits.size else int(score.size)


def main() -> None:
    here = Path(os.environ.get("RTF_DATA_DIR", str(ROOT / "data" / "features")))
    streams = load_streams(here)
    by_bearing = {}
    for s in streams:
        if len(s["rms"]) - 60 < 10:
            continue
        bd = by_bearing.setdefault(s["bearing_id"], {"failed": s["failed"], "n": 0, "axes": {}})
        bd["axes"][s["axis"]] = s["rms"]
        bd["n"] = max(bd["n"], len(s["rms"]) - 60)
    censored_bids = [bid for bid, b in by_bearing.items() if not b["failed"]]
    failed_bids = [bid for bid, b in by_bearing.items() if b["failed"]]

    # per-bearing detection bit per detector
    dets = {name: {} for name in DETECTORS}
    for name, fn in DETECTORS.items():
        axes = {bid: {ax: fn(r) for ax, r in b["axes"].items()} for bid, b in by_bearing.items()}
        t_zero = max(np.max(a) for bid in censored_bids for a in axes[bid].values()) * 1.01
        for bid in failed_bids:
            b = by_bearing[bid]
            first = min(alarm_index(a, t_zero) for a in axes[bid].values())
            dets[name][bid] = int(first < b["n"])

    # McNemar exact test: excess_rate vs each baseline (matched 34 bearings)
    base = "excess_rate"
    comparisons = []
    pvals = []
    for other in ("shewhart", "cusum", "ewma", "sprt"):
        a = [dets[base][bid] for bid in failed_bids]
        o = [dets[other][bid] for bid in failed_bids]
        # discordant pairs b/c
        b_ = sum(1 for x, y in zip(a, o) if x == 1 and y == 0)
        c_ = sum(1 for x, y in zip(a, o) if x == 0 and y == 1)
        p = float(binomtest(k=b_, n=b_ + c_, p=0.5).pvalue)
        pvals.append(p)
        comparisons.append({"vs": other, "discordant_excess_1_other_0": b_,
                            "discordant_excess_0_other_1": c_, "mcnemar_p": p})

    # Holm correction across 4 comparisons
    m = len(pvals)
    order = np.argsort(pvals)
    sorted_p = [pvals[i] for i in order]
    adjusted = [min(1.0, p * (m - rank)) for rank, p in enumerate(sorted_p)]
    for rank in range(1, m):  # enforce monotonicity
        adjusted[rank] = max(adjusted[rank], adjusted[rank - 1])
    for rank, ci in enumerate(order):
        comparisons[ci]["holm_corrected_p"] = float(adjusted[rank])

    summary = {
        "n_failed": len(failed_bids),
        "n_censored": len(censored_bids),
        "detected_per_detector": {k: {bid: int(v) for bid, v in dets[k].items()} for k in dets},
        "comparisons_vs_excess_rate": comparisons,
        "note": "McNemar exact test on matched 34 failed bearings; Holm correction over 4 pre-declared comparisons.",
    }
    dst = ROOT / "results" / "results_run_to_failure_detector_significance.json"
    dst.write_text(json.dumps(summary, indent=2))
    for c in comparisons:
        print(f"excess_rate vs {c['vs']:9s}  discordant {c['discordant_excess_1_other_0']}/{c['discordant_excess_0_other_1']}  "
              f"p={c['mcnemar_p']:.4f}  holm={c['holm_corrected_p']:.4f}")
    print("saved", dst)


if __name__ == "__main__":
    main()
