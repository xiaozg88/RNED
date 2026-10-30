#!/usr/bin/env python3
"""Sensitivity of the zero-false-alarm threshold calibration factor (P2-11).

The online threshold is T = eps * max_rate_on_censored (Eq. (8) uses eps=1.01).
This experiment sweeps eps in {1.00, 1.01, 1.05, 1.10} and reports the
detection rate and median lead time under the same frozen causal protocol, to
show that the 1.01 factor is a conservative margin rather than an arbitrary
choice: eps=1.00 still yields 31/34 because the max is attained on a censored
bearing, while the margin matters only at the level of the zero-false-alarm
construction itself.
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

SECONDS_PER_SNAPSHOT = cb.SECONDS_PER_SNAPSHOT
EPSILONS = (1.00, 1.01, 1.05, 1.10)


def evaluate_epsilon(streams: list[dict], eps: float) -> dict:
    axes = []
    for s in streams:
        score = cb.detector_excess(s["rms"])
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
    t_zero = float(max_censored * eps)
    result = {
        "epsilon": eps,
        "zero_fa_threshold": t_zero,
        "max_censored_final": float(max_censored),
    }
    for ds in sorted({b["dataset"] for b in failed}):
        sub = [b for b in failed if b["dataset"] == ds]
        det = 0
        leads = []
        for b in sub:
            first = min(cb.alarm_index(a, t_zero) for a in b["axes"].values())
            if first < b["n_monitor"]:
                det += 1
                leads.append(100.0 * (b["n_monitor"] - first) / b["n_monitor"])
        result[ds] = {
            "n": len(sub),
            "n_detected": det,
            "detection_rate": 100.0 * det / len(sub),
            "median_lead_pct_life": float(np.median(leads)) if leads else None,
        }
    return result


def main() -> None:
    here = Path(os.environ.get("RTF_DATA_DIR", str(ROOT / "data" / "features")))
    streams = cb.load_all_streams(here)
    eligible = [s for s in streams if len(s["rms"]) - 60 >= 10]
    summary = {
        "n_failed_bearings": len({s["bearing_id"] for s in eligible if s["failed"]}),
        "n_censored_bearings": len({s["bearing_id"] for s in eligible if not s["failed"]}),
        "sweep": [evaluate_epsilon(streams, eps) for eps in EPSILONS],
    }
    out = ROOT / "results" / "results_run_to_failure_threshold_epsilon_sweep.json"
    out.write_text(json.dumps(summary, indent=2))
    for r in summary["sweep"]:
        det = "/".join(f"{ds}:{r[ds]['n_detected']}/{r[ds]['n']}" for ds in ("ims", "xjtu", "femto"))
        print(f"eps={r['epsilon']:.2f}  T0={r['zero_fa_threshold']:.4f}  {det}")
    print("saved", out)


if __name__ == "__main__":
    main()
