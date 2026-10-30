#!/usr/bin/env python3
"""Robustness of the failure-vs-drift discrimination (small sample: 4 failed vs 8 censored).

Recomputes the `auc_above_1p2x` per-bearing feature on IMS and reports, for the
AUROC = 100 separation:
  - exact one-sided Mann-Whitney / permutation p (1 / C(12,4))
  - bootstrap percentile 95% CI (resampling bearings)
  - leave-one-bearing-out AUROC (does one bearing drive the separation?)
  - effect size (median excess for failed vs censored)
"""
from __future__ import annotations

import json
import os
import sys
from math import comb
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "experiments"))

from run_to_failure_failure_vs_drift import load_ims_trajectories, trajectory_features  # noqa: E402


def main() -> None:
    here = Path(os.environ.get("RTF_DATA_DIR", str(ROOT / "data" / "features")))
    streams = load_ims_trajectories(here)
    rows = []
    for key, axes in sorted(streams.items()):
        test, bearing = key
        feats = [trajectory_features(a) for a in axes]
        auc = max(f["auc_above_1p2x"] for f in feats)
        rows.append(
            {
                "bearing_id": f"{test}_b{bearing}",
                "failed": key in {("test1", 3), ("test1", 4), ("test2", 1), ("test3", 3)},
                "auc_above_1p2x": float(auc),
            }
        )
    y = np.asarray([r["failed"] for r in rows], dtype=int)
    x = np.asarray([r["auc_above_1p2x"] for r in rows], dtype=float)
    failed, censored = x[y == 1], x[y == 0]

    auroc = float(roc_auc_score(y, x))
    u, p_onesided = mannwhitneyu(failed, censored, alternative="greater")
    # exact one-sided permutation p: the smallest Mann-Whitney p attainable with
    # 4 vs 8 is 1/C(12,4)
    exact_p = 1.0 / comb(12, 4)

    # bootstrap percentile CI (resample bearings within class)
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(20000):
        f = rng.choice(failed, size=failed.size, replace=True)
        c = rng.choice(censored, size=censored.size, replace=True)
        yy = np.concatenate([np.ones(f.size), np.zeros(c.size)])
        xx = np.concatenate([f, c])
        if yy.sum() and (yy.size - yy.sum()):
            boots.append(float(roc_auc_score(yy, xx)))
    lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

    # leave-one-bearing-out AUROC
    loo = {}
    for i in range(len(rows)):
        mask = np.ones(len(rows), dtype=bool)
        mask[i] = False
        if y[mask].sum() and (y[mask].size - y[mask].sum()):
            loo[rows[i]["bearing_id"]] = float(roc_auc_score(y[mask], x[mask]))

    out = {
        "n_failed": int(failed.size),
        "n_censored": int(censored.size),
        "auroc": auroc,
        "mann_whitney_u": float(u),
        "p_onesided_mw": p_onesided,
        "p_exact_permutation": exact_p,
        "bootstrap_ci95_auroc": [lo, hi],
        "bootstrap_share_auroc_100": float(np.mean(np.asarray(boots) >= 0.999999)),
        "median_auc_above_1p2x_failed": float(np.median(failed)),
        "median_auc_above_1p2x_censored": float(np.median(censored)),
        "per_bearing": {r["bearing_id"]: {"failed": r["failed"], "auc_above_1p2x": r["auc_above_1p2x"]} for r in rows},
        "leave_one_out_auroc": loo,
        "loo_min_auroc": float(min(loo.values())),
    }
    dest = ROOT / "results" / "results_run_to_failure_failure_vs_drift_robustness.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"n={failed.size} vs {censored.size}  AUROC={auroc:.3f}  p_exact={exact_p:.5f}  "
          f"bootstrap_CI=[{lo:.2f},{hi:.2f}]  LOO_min_AUROC={out['loo_min_auroc']:.3f}")
    print("saved", dest)


if __name__ == "__main__":
    main()
