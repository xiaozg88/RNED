#!/usr/bin/env python3
"""Run the deterministic evidence chain for manuscript v20.

The ``core`` mode reproduces the principal detector, statistical, boundary,
and deployment results without optional deep-learning baselines. The
``baselines`` and ``all`` modes progressively add slower PyTorch experiments.
Every command is executed from the release root so relative data/result paths
remain stable.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CORE_SCRIPTS = [
    "run_to_failure_early_warning.py",
    "run_to_failure_failure_vs_drift.py",
    "run_to_failure_classical_baselines.py",
    "run_to_failure_detector_significance.py",
    "run_to_failure_feature_ablation.py",
    "run_to_failure_cross_dataset_fa_proxy.py",
    "run_to_failure_protocol_sensitivity.py",
    "run_to_failure_robustness.py",
    "run_to_failure_robustness_checks.py",
    "run_to_failure_threshold_epsilon_sweep.py",
    "run_to_failure_ewma_sprt_sweep.py",
    "run_to_failure_normalized_accumulators.py",
    "run_to_failure_rate_acceleration.py",
    "run_to_failure_cross_axis_persistence.py",
    "run_sg1026_1_public_d1_rms_protocol_replication.py",
    "run_sg1026_1_closest_rms_ratio_baseline.py",
    "run_to_failure_cwru_normal_stress.py",
    "run_to_failure_fleet_fa_budget.py",
    "run_bcsy_iot_validation_simulation.py",
    "run_to_failure_deployment_value_experiments.py",
    "run_iotj_v20_risk_resolution_audit.py",
]

BASELINE_SCRIPTS = [
    "run_to_failure_ood.py",
    "run_to_failure_ae_baseline.py",
    "run_to_failure_lstmae_baseline.py",
    "run_to_failure_modern_baselines.py",
    "run_to_failure_transformer_baselines.py",
    "run_to_failure_rul_baseline.py",
]

EXTERNAL_SCRIPTS = [
    "run_to_failure_modern_signal_features.py",
    "run_to_failure_public_locked_validation.py",
    "run_to_failure_ldm_upb_condition_residual.py",
]

EXPLORATORY_SCRIPTS = [
    "run_to_failure_cross_axis_locked_audit.py",
    "run_to_failure_cross_axis_edge_benchmark.py",
    "run_to_failure_feature_complementarity.py",
    "run_to_failure_mechanism_reaudit.py",
    "run_to_failure_mechanism_value_paths.py",
    "run_to_failure_two_layer_rescue.py",
]


def run_script(name: str, continue_on_error: bool) -> int:
    command = [sys.executable, str(ROOT / "experiments" / name)]
    print(f"\n=== {name} ===", flush=True)
    completed = subprocess.run(command, cwd=ROOT)
    if completed.returncode and not continue_on_error:
        raise SystemExit(completed.returncode)
    return completed.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    levels = {"core", "baselines", "external", "all", "list"}
    parser.add_argument("level", nargs="?", choices=levels)
    parser.add_argument("--list", action="store_true", help="print scripts without running them")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="run all selected scripts and report failures at the end",
    )
    args = parser.parse_args()

    if args.list and args.level:
        parser.error("use either --list or a reproduction level, not both")
    if args.list or args.level == "list":
        for index, name in enumerate(CORE_SCRIPTS, start=1):
            print(f"{index:02d} core          {name}")
        for name in BASELINE_SCRIPTS:
            print(f"   baselines     {name}")
        for name in EXTERNAL_SCRIPTS:
            print(f"   external      {name}")
        for name in EXPLORATORY_SCRIPTS:
            print(f"   exploratory   {name}")
        return

    level = args.level or "core"
    if level not in {"core", "baselines", "external", "all"}:
        parser.error(f"invalid reproduction level: {level}")

    if level == "core":
        selected = CORE_SCRIPTS
    elif level == "baselines":
        selected = CORE_SCRIPTS + BASELINE_SCRIPTS
    elif level == "external":
        selected = CORE_SCRIPTS + BASELINE_SCRIPTS + EXTERNAL_SCRIPTS
    else:
        selected = CORE_SCRIPTS + BASELINE_SCRIPTS + EXTERNAL_SCRIPTS + EXPLORATORY_SCRIPTS

    failures: list[tuple[str, int]] = []
    for name in selected:
        return_code = run_script(name, args.continue_on_error)
        if return_code:
            failures.append((name, return_code))

    if failures:
        print("\nFailed scripts:")
        for name, return_code in failures:
            print(f"- {name}: exit {return_code}")
        raise SystemExit(1)
    print(f"\nCompleted {len(selected)} scripts successfully.")


if __name__ == "__main__":
    main()
