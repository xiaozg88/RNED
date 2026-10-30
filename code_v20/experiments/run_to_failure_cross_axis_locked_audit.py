#!/usr/bin/env python3
"""Locked evidence audit for the pre-registered cross-axis hierarchy."""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def conformal_requirements() -> list[dict]:
    rows = []
    for alpha in (0.20, 0.10, 0.05):
        per_layer = alpha / 2.0
        minimum_n = math.ceil(1.0 / per_layer - 1.0)
        rows.append(
            {
                "total_false_alarm_level_alpha": alpha,
                "bonferroni_alpha_per_layer": per_layer,
                "minimum_healthy_calibration_bearings_per_layer": minimum_n,
                "order_statistic_index": math.ceil((minimum_n + 1) * (1 - per_layer)),
            }
        )
    return rows


def main() -> None:
    exploratory_path = ROOT / "results" / "results_run_to_failure_cross_axis_persistence.json"
    exploratory = json.loads(exploratory_path.read_text())
    edge_path = ROOT / "results" / "results_run_to_failure_cross_axis_edge_benchmark.json"
    edge = json.loads(edge_path.read_text()) if edge_path.exists() else None
    variants = exploratory["variants"]

    result = {
        "protocol_locked_at": "Before FEMTO Learning Set extraction/evaluation and before new multi-rig testing",
        "frozen_rule": {
            "feature": "per-axis RMS",
            "baseline": "median(rms[10:60])",
            "excess_ratio_threshold": 1.2,
            "historical_statistic": "any-axis cumulative excess mean",
            "recent_window_snapshots": 10,
            "cross_axis_statistic": "second-largest axis recent excess mean",
            "decision": "historical OR cross-axis threshold crossing",
            "minimum_monitor_snapshots": 30,
            "post_lock_tuning_allowed": False,
        },
        "current_exploratory_point_estimate": {
            name: {
                key: variant[key]
                for key in (
                    "n_detected",
                    "n_failed",
                    "n_false_alarms",
                    "n_censored",
                    "median_lead_pct_life",
                )
            }
            for name, variant in variants.items()
        },
        "independent_validation_status": {
            "available_local_rigs": ["IMS", "XJTU-SY", "FEMTO/PRONOSTIA"],
            "all_available_rigs_were_used_in_rule_design": True,
            "genuinely_independent_multi_rig_test_available": False,
            "status": "blocked pending new data",
            "interpretation": "A 34/34 result on already-design-involved data cannot by itself justify upgrade to the main method.",
        },
        "joint_false_alarm_theory": {
            "unit": "independent healthy bearing trajectory",
            "calibration_statistic": "full-trajectory maximum of each layer",
            "guarantee": "P(max_R_new > q_R OR max_J_new > q_J) <= alpha_R + alpha_J",
            "alpha_allocation": "alpha_R = alpha_J = alpha/2",
            "required_assumptions": [
                "healthy calibration and deployment bearings are exchangeable",
                "bearings, not snapshots, are independent experimental units",
                "thresholds are split before testing",
            ],
            "finite_sample_requirements": conformal_requirements(),
        },
        "current_calibration_capacity": {
            "multi_axis_healthy_calibration_bearings": exploratory["protocol"][
                "multi_axis_calibration_bearings"
            ],
            "all_healthy_ims_calibration_bearings": variants["historical_rate"][
                "n_censored"
            ],
            "sufficient_for_alpha_0_20_per_layer": exploratory["protocol"][
                "multi_axis_calibration_bearings"
            ]
            >= 9,
            "sufficient_for_alpha_0_10_per_layer": exploratory["protocol"][
                "multi_axis_calibration_bearings"
            ]
            >= 19,
            "sufficient_for_alpha_0_05_per_layer": exploratory["protocol"][
                "multi_axis_calibration_bearings"
            ]
            >= 39,
            "leave_one_censored_out_hierarchical_false_alarms": exploratory[
                "leave_one_censored_out"
            ]["hierarchical_false_alarms"],
            "leave_one_censored_out_hierarchical_false_alarm_rate_pct": exploratory[
                "leave_one_censored_out"
            ]["hierarchical_false_alarm_rate_pct"],
        },
        "statistical_increment": exploratory["baseline_vs_hierarchical_mcnemar"],
        "edge_cost": (
            {
                "measured": True,
                "platform": edge["platform"],
                "incremental_update_us": edge["detector_incremental_update"],
                "rms_plus_hierarchy_k4_ms": {
                    dataset: values["rms_plus_incremental_hierarchy_k4_ms"]
                    for dataset, values in edge["windows"].items()
                },
                "sustained_energy_per_window_mj": edge["sustained_rms_plus_hierarchy_k4"][
                    "energy_per_window_mj"
                ],
                "passes_deployment_cost_check": True,
            }
            if edge is not None
            else {"measured": False, "passes_deployment_cost_check": False}
        ),
        "upgrade_decision": {
            "passed": False,
            "decision": "do_not_upgrade_to_main_method",
            "reasons": [
                "no genuinely independent multi-rig run-to-failure test is available",
                "healthy multi-axis calibration capacity is below all finite conformal requirements",
                "the three-bearing increment over 31/34 is not significant (exact McNemar p=0.25)",
                "joint false-alarm control has not been demonstrated on a sufficiently large held-out healthy fleet",
            ],
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    destination = ROOT / "results" / "results_run_to_failure_cross_axis_locked_audit.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["upgrade_decision"], indent=2))
    print(f"saved {destination}")


if __name__ == "__main__":
    main()
