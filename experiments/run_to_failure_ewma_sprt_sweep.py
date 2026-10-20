#!/usr/bin/env python3
"""Parameter sweep for the EWMA and SPRT baselines (P2-10).

The main comparison uses EWMA lambda=0.2 and a one-sided Gaussian SPRT with
H0: mu=1.0 vs H1: mu=1.2. This experiment sweeps EWMA lambda in
{0.05, 0.1, 0.2, 0.3, 0.5} and the SPRT alternative mean mu1 in
{1.05, 1.1, 1.2, 1.3, 1.5} under the same frozen causal protocol and the same
zero-false-alarm calibration on the 8 IMS censored bearings, to show that the
reported ranking is not an artifact of unfavorable hyperparameter choices.
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

import run_to_failure_classical_baselines as cb  # noqa: E402

EWMA_LAMBDAS = (0.05, 0.1, 0.2, 0.3, 0.5)
SPRT_MU1 = (1.05, 1.1, 1.2, 1.3, 1.5)


def detector_ewma_lambda(rms, lam):
    _, ratio = cb._base(rms)
    inc = np.maximum(ratio - 1.2, 0.0)
    z = np.empty_like(inc)
    acc = 0.0
    for i, v in enumerate(inc):
        acc = lam * v + (1.0 - lam) * acc
        z[i] = acc
    return z


def detector_sprt_mu1(rms, mu1):
    base, ratio = cb._base(rms)
    sigma = float(np.std(rms[10:60]) / base)
    sigma = max(sigma, 1e-9)
    mu0 = 1.0
    inc = (mu1 - mu0) / (sigma * sigma) * (ratio - (mu0 + mu1) / 2.0)
    s = np.empty_like(inc)
    acc = 0.0
    for i, v in enumerate(inc):
        acc = max(0.0, acc + v)
        s[i] = acc
    return s


def evaluate(streams: list[dict], score_fn) -> dict:
    axes = []
    for s in streams:
        score = score_fn(s["rms"])
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
    by_bearing: dict[str, dict] = {}
    for a in axes:
        b = by_bearing.setdefault(
            a["bearing_id"],
            {"bearing_id": a["bearing_id"], "dataset": a["dataset"], "failed": a["failed"],
             "n_monitor": a["n_monitor"], "axes": {}},
        )
        b["axes"][a["axis"]] = a["score"]
        b["n_monitor"] = max(b["n_monitor"], a["n_monitor"])
    censored = [b for b in by_bearing.values() if not b["failed"]]
    failed = [b for b in by_bearing.values() if b["failed"]]
    max_censored = max(np.max(a) for b in censored for a in b["axes"].values())
    t_zero = float(max_censored * 1.01)
    result = {"zero_fa_threshold": t_zero}
    total_det = 0
    total_n = 0
    for ds in sorted({b["dataset"] for b in failed}):
        sub = [b for b in failed if b["dataset"] == ds]
        det = 0
        leads = []
        for b in sub:
            first = min(cb.alarm_index(a, t_zero) for a in b["axes"].values())
            if first < b["n_monitor"]:
                det += 1
                leads.append(100.0 * (b["n_monitor"] - first) / b["n_monitor"])
        total_det += det
        total_n += len(sub)
        result[ds] = {
            "n": len(sub),
            "n_detected": det,
            "detection_rate": 100.0 * det / len(sub),
            "median_lead_pct_life": float(np.median(leads)) if leads else None,
        }
    result["total_detected"] = total_det
    result["total_n"] = total_n
    return result


def main() -> None:
    here = Path(os.environ.get("RTF_DATA_DIR", str(ROOT / "data" / "features")))
    streams = cb.load_all_streams(here)
    summary = {"ewma": {}, "sprt": {}}
    for lam in EWMA_LAMBDAS:
        summary["ewma"][f"lambda={lam}"] = evaluate(
            streams, lambda r, l=lam: detector_ewma_lambda(r, l)
        )
    for mu1 in SPRT_MU1:
        summary["sprt"][f"mu1={mu1}"] = evaluate(
            streams, lambda r, m=mu1: detector_sprt_mu1(r, m)
        )
    out = ROOT / "results" / "results_run_to_failure_ewma_sprt_sweep.json"
    out.write_text(json.dumps(summary, indent=2))
    for name, sub in summary.items():
        for key, r in sub.items():
            print(f"{name:5s} {key:12s} total={r['total_detected']}/{r['total_n']}  "
                  f"T0={r['zero_fa_threshold']:.3f}")
    print("saved", out)


if __name__ == "__main__":
    main()
