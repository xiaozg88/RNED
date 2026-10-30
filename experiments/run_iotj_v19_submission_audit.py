#!/usr/bin/env python3
"""Build the deterministic IoTJ v19 evidence and submission-readiness audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List

from scipy.stats import beta


ROOT = Path(__file__).resolve().parents[1]
RESULT_FILES = [
    "results/results_run_to_failure_early_warning.json",
    "results/results_run_to_failure_classical_baselines.json",
    "results/results_run_to_failure_detector_significance.json",
    "results/results_sg1026_review_audit.json",
    "results/results_sg1026_adverse_result_reaudit.json",
    "results/results_sg1026_1_closest_rms_ratio_baseline.json",
    "results/results_sg1026_1_public_d1_rms_protocol_replication.json",
    "results/results_run_to_failure_transformer_baselines.json",
    "results/results_run_to_failure_edge_benchmark.json",
    "results/results_run_to_failure_deployment_value_experiments.json",
    "results/results_run_to_failure_fleet_fa_budget.json",
    "results/results_run_to_failure_public_external_validation_summary.json",
]


def load_json(relative_path: str) -> Dict[str, Any]:
    with (ROOT / relative_path).open(encoding="utf-8") as handle:
        return json.load(handle)


def sha256(relative_path: str) -> str:
    digest = hashlib.sha256()
    with (ROOT / relative_path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_ci(successes: int, total: int, alpha: float = 0.05) -> List[float]:
    if total == 0:
        return [0.0, 1.0]
    lower = 0.0 if successes == 0 else beta.ppf(alpha / 2, successes, total - successes + 1)
    upper = 1.0 if successes == total else beta.ppf(1 - alpha / 2, successes + 1, total - successes)
    return [float(lower), float(upper)]


def detector_total(detector: Dict[str, Any], datasets: Iterable[str] = ("ims", "xjtu", "femto")) -> int:
    return int(sum(detector[dataset]["n_detected"] for dataset in datasets))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="results/results_iotj_v19_submission_audit.json",
        help="output path relative to the repository root",
    )
    args = parser.parse_args()

    evidence = {path: load_json(path) for path in RESULT_FILES}
    early = evidence[RESULT_FILES[0]]
    classical = evidence[RESULT_FILES[1]]
    significance = evidence[RESULT_FILES[2]]
    review = evidence[RESULT_FILES[3]]
    adverse = evidence[RESULT_FILES[4]]
    rms = evidence[RESULT_FILES[5]]
    d1 = evidence[RESULT_FILES[6]]
    deep = evidence[RESULT_FILES[7]]
    edge = evidence[RESULT_FILES[8]]
    deployment = evidence[RESULT_FILES[9]]
    fleet = evidence[RESULT_FILES[10]]
    external = evidence[RESULT_FILES[11]]

    early_by_dataset = {
        dataset: {
            "n": int(values["n"]),
            "detected": int(values["n_detected"]),
            "median_lead_pct_life": float(values["median_lead_pct_life"]),
        }
        for dataset, values in early["zero_fa_operating_point"].items()
    }
    early_detected = sum(item["detected"] for item in early_by_dataset.values())
    early_total = sum(item["n"] for item in early_by_dataset.values())

    zero_fa_candidates = [
        row for row in early["threshold_sweep"] if row["false_alarm_rate"] == 0.0
    ]
    first_zero_fa = zero_fa_candidates[0] if zero_fa_candidates else None
    threshold_margin_pct = 100.0 * (
        float(early["zero_fa_threshold"]) / float(early["max_censored_final"]) - 1.0
    )

    loeo = review["ims_leave_one_experiment_out"]
    d1_paper = d1["paper_34_failure_cohort"]
    d1_healthy = d1_paper["ims_censored_healthy_reference_false_alarms"]
    deep_seed_counts = [
        int(run["total_detected"])
        for model_runs in deep["models"].values()
        for run in model_runs
    ]

    baseline_table = {
        "RNED": early_detected,
        "Shewhart": detector_total(classical["detectors"]["shewhart"]),
        "CUSUM": detector_total(classical["detectors"]["cusum"]),
        "EWMA": detector_total(classical["detectors"]["ewma"]),
        "SPRT": detector_total(classical["detectors"]["sprt"]),
        "single_sample_RMS_ratio": int(rms["total_detected"]),
        "D1_endpoint_persistence": int(d1_paper["n_detected"]),
        "deep_reconstruction_median": int(sorted(deep_seed_counts)[len(deep_seed_counts) // 2]),
    }

    rubric = [
        {
            "dimension": "IoTJ_scope_alignment",
            "weight": 0.15,
            "score": 0.82,
            "basis": "Edge detector, event communication, deployment cost, and fleet-scale risk are central IoTJ concerns.",
        },
        {
            "dimension": "differentiated_contribution",
            "weight": 0.15,
            "score": 0.72,
            "basis": "The latching event semantic, causal rate normalization, frozen threshold, and explicit event budget differ from accuracy-oriented diagnosis and RUL work.",
        },
        {
            "dimension": "methodological_clarity",
            "weight": 0.15,
            "score": 0.78,
            "basis": "The detector has a deterministic update and explicit propositions, but it is intentionally simple rather than algorithmically novel.",
        },
        {
            "dimension": "empirical_breadth",
            "weight": 0.15,
            "score": 0.70,
            "basis": "Three public run-to-failure rigs, 34 failures, classical/deep/D1 baselines, edge hardware, and external secondary replication are covered.",
        },
        {
            "dimension": "claim_evidence_alignment",
            "weight": 0.15,
            "score": 0.88,
            "basis": "v19 removes finite-population false-alarm guarantees and states healthy-censored and network-test limitations.",
        },
        {
            "dimension": "reproducibility",
            "weight": 0.10,
            "score": 0.90,
            "basis": "Public datasets, deterministic scripts, result hashes, and exact counts support reproduction.",
        },
        {
            "dimension": "engineering_value",
            "weight": 0.10,
            "score": 0.80,
            "basis": "Measured Jetson update and energy costs plus communication ratios support constrained-node deployment.",
        },
        {
            "dimension": "residual_external_risk_control",
            "weight": 0.05,
            "score": 0.48,
            "basis": "Healthy-censored evidence remains small and real multi-node network validation is absent.",
        },
    ]
    portfolio_score = sum(item["weight"] * item["score"] for item in rubric)
    independence_discount = 0.80
    venue_competition_discount = 0.94
    provisional_acceptance_estimate = portfolio_score * independence_discount * venue_competition_discount

    evidence_matrix = {
        "supported": {
            "online_detection_and_lead_time": "early_warning plus per-dataset trajectories",
            "classical_baseline_comparison": "matched detector results and McNemar/Holm audit",
            "single_sample_and_D1_semantic_comparison": "frozen baseline result files",
            "deep_baseline_limitation": "six deterministic reconstruction runs",
            "held_out_threshold_sensitivity": "leave-one-IMS-experiment-out audit",
            "edge_cost": "Jetson Orin NX benchmark",
            "event_budget": "fleet bootstrap/modeling, explicitly not field validation",
        },
        "not_supported_as_validation": [
            "a finite-population or distribution-free false-alarm guarantee",
            "large-fleet empirical zero-false-alarm performance",
            "real multi-node broker/network deployment",
            "condition-invariant degradation detection",
        ],
    }

    output = {
        "schema_version": "iotj_v19_submission_audit_v1",
        "audit_type": "deterministic evidence aggregation; no data were modified",
        "inputs": [
            {"path": path, "sha256": sha256(path)} for path in RESULT_FILES
        ],
        "primary": {
            "datasets": early_by_dataset,
            "detected": early_detected,
            "failed": early_total,
            "detection_ci95_exact": exact_ci(early_detected, early_total),
            "censored": int(early["n_censored"]),
            "in_sample_false_alarms": 0,
            "frozen_threshold": float(early["zero_fa_threshold"]),
            "max_censored_score": float(early["max_censored_final"]),
            "threshold_margin_pct": threshold_margin_pct,
            "first_zero_fa_grid_candidate": first_zero_fa,
        },
        "threshold_and_held_out": {
            "first_zero_fa_grid_interpretation": (
                "The public sweep is coarse; the frozen threshold lies between the maximum censored score "
                "and the next displayed zero-false-alarm grid point. This is stability evidence, not a margin guarantee."
            ),
            "leave_one_ims_experiment_out": loeo,
            "loeo_false_alarms": int(loeo["total_held_out_false_alarms"]),
            "loeo_censored": int(loeo["total_held_out_censored"]),
            "loeo_false_alarm_ci95_exact": exact_ci(
                int(loeo["total_held_out_false_alarms"]), int(loeo["total_held_out_censored"])
            ),
        },
        "baseline_counts": baseline_table,
        "statistical_tests": significance["comparisons_vs_excess_rate"],
        "D1_semantics": {
            "paper_cohort_detected": int(d1_paper["n_detected"]),
            "paper_cohort_failed": int(d1_paper["n_failures"]),
            "healthy_reference_bearings_with_episode": int(d1_healthy["n_bearings_with_episode"]),
            "healthy_reference_bearings": int(d1_healthy["n_bearings"]),
            "interpretation": (
                "D1 is competitive on endpoint detection but changes the task to persistence confirmed at the endpoint; "
                "its reference-zone episodes are not the same online latching-event objective."
            ),
        },
        "edge": {
            "platform": edge["platform"],
            "detector_update_ms": float(edge["detector_update_ms"]),
            "throughput_windows_per_s": float(edge["throughput_windows_per_s"]),
            "energy_per_window_mJ": float(edge["energy_per_window_mJ"]),
            "quantization_8bit": deployment["low_power_robustness"]["rms_quantization"][0],
        },
        "communication": {
            "event_packet_bytes": int(adverse["communication_reaudit"]["assumptions"]["event_packet_bytes"]),
            "ims_event_packet_ratio": float(
                adverse["communication_reaudit"]["ims_pooled_event_packet_ratio"]
            ),
            "xjtu_event_packet_ratio": float(
                adverse["communication_reaudit"]["other_datasets"]["xjtu"]["event_packet_ratio"]
            ),
            "femto_event_packet_ratio": float(
                adverse["communication_reaudit"]["other_datasets"]["femto"]["event_packet_ratio"]
            ),
            "scope": "idealized byte accounting; excludes retransmission, cryptography, authentication, and broker overhead",
        },
        "fleet": fleet["fleet_budget_cross_rig"],
        "external_secondary_replication": {
            "protocol_status": external["protocol_status"],
            "public_failed": int(external["n_public_failed"]),
            "protocol_eligible": int(external["n_protocol_eligible"]),
            "historical_detected": int(external["historical_detected"]),
            "main_method_upgrade": external["main_method_upgrade"],
        },
        "evidence_matrix": evidence_matrix,
        "submission_readiness": {
            "status": "provisional",
            "review_independence": "same-family executor; not an external IoTJ review",
            "rubric": rubric,
            "portfolio_score": portfolio_score,
            "independence_discount": independence_discount,
            "venue_competition_discount": venue_competition_discount,
            "provisional_acceptance_estimate": provisional_acceptance_estimate,
            "uncertainty_interval": [0.45, 0.70],
            "decision_rule": "Continue only if the estimate is <= 0.55; stop when both language versions are synchronized and the estimate is > 0.55.",
            "interpretation": (
                "This is a transparent pre-submission risk score, not a prediction of an editorial decision. "
                "The interval reflects unobserved reviewer priorities and the remaining independent-validation gap."
            ),
        },
    }

    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    print(f"Wrote {output_path.relative_to(ROOT)}")
    print(
        "Primary: {}/{}; held-out false alarms {}/{}; provisional estimate {:.3f}".format(
            early_detected,
            early_total,
            output["threshold_and_held_out"]["loeo_false_alarms"],
            output["threshold_and_held_out"]["loeo_censored"],
            provisional_acceptance_estimate,
        )
    )


if __name__ == "__main__":
    main()
