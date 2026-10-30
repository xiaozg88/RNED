#!/usr/bin/env python3
"""Fleet-scale false-alarm budget (P3).

Per-machine zero-false-alarm calibration does not compose across a fleet: even
if one detector is calibrated to zero false alarms on 8 censored bearings, a
fleet of M machines each with k axes accumulates false alarms. This experiment
quantifies the budget using (a) the same-rig censored calibration (0/8; 95%
Clopper-Pearson CI 0--36.9%) and (b) the cross-rig early-life proxy (18.5%
per axis over a 150-snapshot segment; 95% CI 9.3--31.4%), and derives (c) the
fleet-level threshold rescaling needed to keep the fleet false-alarm
probability below a target, and (d) the union-bound composition of
Proposition 4's anytime-valid bound.

All numbers are probability arithmetic on frozen results; no new model is
trained.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "experiments"))
import run_to_failure_classical_baselines as cb  # noqa: E402

# Frozen inputs.
P_SEG_AXIS = 0.185            # cross-rig early-life FA proxy, per axis / 150 snapshots
P_SEG_BEARING = 0.22          # cross-rig early-life FA proxy, per bearing / 150 snapshots
SEGMENT_SNAPSHOTS = 150       # early-life pseudo-health segment length
T = 0.0424                    # zero-FA threshold
MU = 0.0153                   # per-snapshot excess mean over censored bearings (Fig. 7)


def data_driven_m() -> float:
    """Max per-snapshot excess over the IMS censored bearings (Hoeffding support)."""
    here = ROOT / "data" / "features"
    streams = cb.load_all_streams(here)
    censored = [s for s in streams if not s["failed"] and s["dataset"] == "ims"]
    m = 0.0
    for s in censored:
        base = float(np.median(s["rms"][10:60]))
        ratio = s["rms"][60:] / base
        m = max(m, float(np.max(np.maximum(ratio - 1.2, 0.0))))
    return m


def per_snapshot_rate(p_seg: float) -> float:
    # P(no alarm in segment) = 1 - p_seg = (1 - q)^L  =>  q = 1 - (1-p_seg)^(1/L)
    return float(1.0 - (1.0 - p_seg) ** (1.0 / SEGMENT_SNAPSHOTS))


def clopper_pearson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact two-sided Clopper-Pearson 95% confidence interval for k/n."""
    from scipy.stats import beta

    if n <= 0:
        return (0.0, 1.0)
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1.0 - alpha / 2, k + 1, n - k))
    return (lo, hi)


def main() -> None:
    m_bound = data_driven_m()
    q_axis = per_snapshot_rate(P_SEG_AXIS)
    same_rig_lo, same_rig_hi = clopper_pearson_ci(0, 8)
    cross_rig_lo, cross_rig_hi = clopper_pearson_ci(round(P_SEG_AXIS * 54), 54)  # 10/54 alarms

    fleets = [1, 5, 10, 50, 100, 500, 1000]
    k_axes = 2
    rows = []
    for m in fleets:
        n_axes = m * k_axes
        # Probability one machine raises >=1 FA in one segment (cross-rig, per-axis p).
        p_machine = 1.0 - (1.0 - P_SEG_AXIS) ** k_axes
        # Probability at least one FA somewhere in the fleet in one segment.
        p_fleet = 1.0 - (1.0 - P_SEG_AXIS) ** n_axes
        expected = n_axes * P_SEG_AXIS
        rows.append(
            {
                "fleet_size_machines": m,
                "axes": n_axes,
                "p_machine_false_alarm_one_segment": round(float(p_machine), 4),
                "p_fleet_ge1_false_alarm_one_segment": round(float(p_fleet), 4),
                "expected_false_alarms_per_segment": round(float(expected), 3),
            }
        )

    # Fleet-level rescaling: to keep P(fleet >=1 FA in one segment) <= alpha,
    # require per-axis segment FA p <= 1 - (1-alpha)^(1/(M*k)).
    alpha = 0.05
    rescale = []
    for m in fleets:
        n_axes = m * k_axes
        p_req = 1.0 - (1.0 - alpha) ** (1.0 / n_axes)
        rescale.append(
            {
                "fleet_size_machines": m,
                "required_per_axis_segment_fa_for_5pct_fleet": round(float(p_req), 6),
                "tightening_factor_vs_18.5pct": round(float(P_SEG_AXIS / p_req), 1),
            }
        )

    # Proposition 4 composition (union bound over fleet axes): the per-axis
    # anytime-valid bound exp(-2 n0 (T-mu)^2 / M^2) multiplies by M*k.
    def fleet_union_bound(n0: int, m: int) -> float:
        per_axis = float(np.exp(-2 * n0 * (T - MU) ** 2 / m_bound**2))
        return float(min(1.0, m * k_axes * per_axis))

    union_rows = [
        {
            "monitoring_start_n0": n0,
            "per_axis_anytime_bound": float(np.exp(-2 * n0 * (T - MU) ** 2 / m_bound**2)),
            "fleet_union_bound_m1000": fleet_union_bound(n0, 1000),
        }
        for n0 in (30, 60, 120, 240, 480)
    ]

    summary = {
        "inputs": {
            "p_seg_axis": P_SEG_AXIS,
            "p_seg_bearing": P_SEG_BEARING,
            "segment_snapshots": SEGMENT_SNAPSHOTS,
            "data_driven_excess_bound_M": round(m_bound, 3),
            "per_snapshot_rate_axis_est": round(q_axis, 5),
            "same_rig_censored_fa_0_of_8_cp_ci95_pct": [round(100.0 * same_rig_lo, 1), round(100.0 * same_rig_hi, 1)],
            "cross_rig_proxy_cp_ci95_pct": [round(100.0 * cross_rig_lo, 1), round(100.0 * cross_rig_hi, 1)],
        },
        "fleet_budget_cross_rig": rows,
        "fleet_rescaling_for_5pct": rescale,
        "proposition4_fleet_union_bound": union_rows,
    }
    dst = ROOT / "results" / "results_run_to_failure_fleet_fa_budget.json"
    dst.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("saved", dst)


if __name__ == "__main__":
    main()
