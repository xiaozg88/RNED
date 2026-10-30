#!/usr/bin/env python3
"""Time-normalized accumulator comparisons (the decisive fairness check).

The paper's rate R_t = S_t/(t-60) with S_t = cumsum of e_i is, by construction,
a time-normalized no-reset CUSUM with zero drift compensation. A fair
comparator must therefore normalize the *reset* accumulators too. This
experiment evaluates, under the identical frozen protocol and zero-false-alarm
calibration:
  - CUSUM(delta)/t : U_t = max(0, U_{t-1} + e_t - delta), score U_t/(t-60),
                     delta in {0, 0.005, 0.01, mu_hat} (delta=0 reduces to R_t);
  - SPRT/t          : reflected Gaussian log-likelihood ratio divided by (t-60).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "experiments"))
import run_to_failure_classical_baselines as cb  # noqa: E402

SECONDS_PER_SNAPSHOT = cb.SECONDS_PER_SNAPSHOT


def rate_score(rms: np.ndarray) -> np.ndarray:
    return cb.detector_excess(rms)


def cusum_delta_over_t(rms: np.ndarray, delta: float) -> np.ndarray:
    _, ratio = cb._base(rms)
    inc = np.maximum(ratio - 1.2, 0.0) - delta
    n = inc.size
    u = np.empty(n)
    acc = 0.0
    for i, v in enumerate(inc):
        acc = max(0.0, acc + v)
        u[i] = acc
    return u / np.arange(1, n + 1)


def sprt_over_t(rms: np.ndarray) -> np.ndarray:
    base, ratio = cb._base(rms)
    sigma = float(np.std(rms[10:60]) / base)
    sigma = max(sigma, 1e-9)
    mu0, mu1 = 1.0, 1.2
    inc = (mu1 - mu0) / (sigma * sigma) * (ratio - (mu0 + mu1) / 2.0)
    n = inc.size
    lam = np.empty(n)
    acc = 0.0
    for i, v in enumerate(inc):
        acc = max(0.0, acc + v)
        lam[i] = acc
    return lam / np.arange(1, n + 1)


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
        b["axes"][len(b["axes"])] = a["score"]
        b["n_monitor"] = max(b["n_monitor"], a["n_monitor"])
    censored = [b for b in by_bearing.values() if not b["failed"]]
    failed = [b for b in by_bearing.values() if b["failed"]]
    max_censored = max(np.max(a) for b in censored for a in b["axes"].values())
    t_zero = float(max_censored * 1.01)
    total_det = 0
    per_ds = {}
    for ds in sorted({b["dataset"] for b in failed}):
        subd = [b for b in failed if b["dataset"] == ds]
        det = 0
        leads = []
        for b in subd:
            first = min(cb.alarm_index(a, t_zero) for a in b["axes"].values())
            if first < b["n_monitor"]:
                det += 1
                leads.append(100.0 * (b["n_monitor"] - first) / b["n_monitor"])
        total_det += det
        per_ds[ds] = {
            "n": len(subd),
            "n_detected": det,
            "median_lead_pct_life": float(np.median(leads)) if leads else None,
        }
    return {
        "zero_fa_threshold": t_zero,
        "total_detected": total_det,
        "total_n": int(len(failed)),
        "per_dataset": per_ds,
    }


def main() -> None:
    here = Path(os.environ.get("RTF_DATA_DIR", str(ROOT / "data" / "features")))
    streams = cb.load_all_streams(here)
    # estimate the censored mean drift mu_hat
    censored = [s for s in streams if not s["failed"] and s["dataset"] == "ims"]
    excess_all = []
    for s in censored:
        _, ratio = cb._base(s["rms"])
        excess_all.append(np.maximum(ratio - 1.2, 0.0))
    mu_hat = float(np.mean(np.concatenate(excess_all)))

    summary = {"mu_hat": mu_hat, "variants": {}}
    summary["variants"]["rate (paper)"] = evaluate(streams, rate_score)
    for delta in (0.0, 0.005, 0.01):
        summary["variants"][f"cusum_delta_over_t delta={delta}"] = evaluate(
            streams, lambda r, d=delta: cusum_delta_over_t(r, d)
        )
    summary["variants"][f"cusum_delta_over_t delta=mu_hat"] = evaluate(
        streams, lambda r: cusum_delta_over_t(r, mu_hat)
    )
    summary["variants"]["sprt_over_t"] = evaluate(streams, sprt_over_t)

    dst = ROOT / "results" / "results_run_to_failure_normalized_accumulators.json"
    dst.write_text(json.dumps(summary, indent=2))
    for name, r in summary["variants"].items():
        print(f"{name:36s} total={r['total_detected']}/{r['total_n']}  T0={r['zero_fa_threshold']:.4f}")
    print("saved", dst)


if __name__ == "__main__":
    main()
