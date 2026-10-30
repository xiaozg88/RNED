#!/usr/bin/env python3
"""Deterministic IoT validation-gap simulations for the BCSY protocol plan.

The experiments are synthetic protocol stress tests. They do not retune the
historical RMS-excess-rate detector and must not be reported as real multi-rig,
healthy-censored, network, or multi-node validation.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from statistics import NormalDist
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = ROOT / "results" / "results_bcsy_iot_validation_simulation.json"

SEED = 20261026
RAW_THRESHOLD = 0.042407782315012556
BASELINE_SLICE = slice(10, 60)
MONITOR_START = 60
EARLIEST_UPDATE = 30

N_RIGS = 8
HEALTHY_MACHINES_PER_RIG = 3
N_AXES_PER_MACHINE = 2
FAILURES_PER_RIG = 1

N_NETWORK_MACHINES = 100
N_NETWORK_CYCLES = 5000
LOSS_RATES = (0.00, 0.01, 0.05, 0.10)
RETRIES = (0, 1, 2, 3)
EVENT_PACKET_BYTES = 64
PERIODIC_STATUS_BYTES = 32
LATENCY_LOG_MEAN_MS = 3.4
LATENCY_LOG_SIGMA_MS = 0.65
STALE_EVENT_MS = 100.0
COMPLETION_CENSOR_MS = 1000.0

FLEET_PROBABILITIES = (0.001, 0.005, 0.010, 0.020)
FLEET_CORRELATIONS = (0.0, 0.2, 0.5, 0.8)


def wilson_interval(successes: int, total: int) -> Optional[List[float]]:
    if total == 0:
        return None
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    half = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return [100.0 * max(0.0, center - half), 100.0 * min(1.0, center + half)]


def percentile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    return float(np.percentile(np.asarray(values), q))


def rate_score(rms: np.ndarray) -> np.ndarray:
    baseline = float(np.median(rms[BASELINE_SLICE]))
    excess = np.maximum(rms[MONITOR_START:] / baseline - 1.2, 0.0)
    return np.cumsum(excess) / np.arange(1, excess.size + 1)


def first_alarm(rms: np.ndarray) -> Optional[int]:
    score = rate_score(rms)
    if score.size < EARLIEST_UPDATE:
        return None
    eligible = np.flatnonzero(score[EARLIEST_UPDATE - 1 :] > RAW_THRESHOLD)
    return int(eligible[0] + EARLIEST_UPDATE) if eligible.size else None


def shared_condition(
    rng: np.random.Generator,
    length: int,
    amplitude: float,
    frequency: float,
    walk_sigma: float,
) -> np.ndarray:
    time = np.arange(length, dtype=float)
    phase = rng.uniform(0.0, 2.0 * math.pi)
    walk = np.cumsum(rng.normal(0.0, walk_sigma, length))
    walk -= walk[:50].mean()
    return 1.0 + amplitude * np.sin(2.0 * math.pi * frequency * time + phase) + walk


def axis_stream(
    rng: np.random.Generator,
    length: int,
    condition: np.ndarray,
    noise_sigma: float,
) -> np.ndarray:
    return np.exp(np.log(condition) + rng.normal(0.0, noise_sigma, length))


def simulate_multi_rig_cohort(rng: np.random.Generator) -> dict:
    rig_results: List[dict] = []
    failure_rows: List[dict] = []
    healthy_machine_rows: List[dict] = []
    healthy_axis_rows: List[dict] = []

    for rig_index in range(N_RIGS):
        condition_scale = rng.uniform(0.85, 1.15)
        condition_amplitude = rng.uniform(0.05, 0.11)
        noise_sigma = rng.uniform(0.06, 0.11)
        walk_sigma = rng.uniform(0.00012, 0.00045)
        rig_failures = []
        rig_healthy_machines = 0
        rig_healthy_axes = 0

        for machine_index in range(HEALTHY_MACHINES_PER_RIG):
            length = int(rng.integers(220, 501))
            condition = condition_scale * shared_condition(
                rng, length, condition_amplitude, rng.uniform(0.002, 0.010), walk_sigma
            )
            axis_alarms = []
            axis_false = 0
            for _axis in range(N_AXES_PER_MACHINE):
                rms = axis_stream(rng, length, condition, noise_sigma)
                alarm = first_alarm(rms)
                axis_alarms.append(alarm)
                axis_false += int(alarm is not None)
            machine_false = any(alarm is not None for alarm in axis_alarms)
            healthy_machine_rows.append(
                {
                    "rig": f"R{rig_index + 1}",
                    "machine": f"H{machine_index + 1}",
                    "length": length,
                    "false_alarm": machine_false,
                    "false_axes": axis_false,
                }
            )
            healthy_axis_rows.extend(
                {
                    "rig": f"R{rig_index + 1}",
                    "machine": f"H{machine_index + 1}",
                    "axis": axis_index + 1,
                    "false_alarm": alarm is not None,
                }
                for axis_index, alarm in enumerate(axis_alarms)
            )
            rig_healthy_machines += int(machine_false)
            rig_healthy_axes += axis_false

        for failure_index in range(FAILURES_PER_RIG):
            length = int(rng.integers(520, 901))
            fault_onset = int(rng.integers(180, 381))
            condition = condition_scale * shared_condition(
                rng, length, condition_amplitude, rng.uniform(0.002, 0.010), walk_sigma
            )
            final_gain = rng.uniform(1.55, 3.20)
            onset_gain = rng.uniform(1.03, 1.12)
            progress_power = rng.uniform(1.7, 4.2)
            fault_axis = int(rng.integers(0, N_AXES_PER_MACHINE))
            machine_alarm = None
            per_axis = []
            max_rates = []
            for axis_index in range(N_AXES_PER_MACHINE):
                healthy_rms = axis_stream(rng, length, condition, noise_sigma)
                if axis_index == fault_axis:
                    progress = np.zeros(length)
                    active = np.arange(fault_onset, length)
                    fraction = (active - fault_onset) / max(1, length - 1 - fault_onset)
                    gain = onset_gain + (final_gain - onset_gain) * np.power(fraction, progress_power)
                    progress[active] = np.maximum.accumulate(gain - 1.0)
                    rms = healthy_rms * (1.0 + progress)
                else:
                    rms = healthy_rms
                alarm = first_alarm(rms)
                if alarm is not None:
                    alarm_snapshot = MONITOR_START + alarm
                    machine_alarm = alarm_snapshot if machine_alarm is None else min(machine_alarm, alarm_snapshot)
                per_axis.append(None if alarm is None else MONITOR_START + alarm)
                max_rates.append(float(np.max(rate_score(rms))))
            detected = machine_alarm is not None
            endpoint_lead_pct = None if machine_alarm is None else 100.0 * (length - machine_alarm) / length
            detection_lag = None if machine_alarm is None else machine_alarm - fault_onset
            failure_rows.append(
                {
                    "rig": f"R{rig_index + 1}",
                    "machine": f"F{failure_index + 1}",
                    "length": length,
                    "fault_onset": fault_onset,
                    "fault_axis": fault_axis + 1,
                    "alarm_snapshot": machine_alarm,
                    "detected": detected,
                    "endpoint_lead_pct": endpoint_lead_pct,
                    "detection_lag_after_onset": detection_lag,
                    "axis_alarm_snapshots": per_axis,
                    "max_rate": max(max_rates),
                    "final_gain": final_gain,
                }
            )
            rig_failures.append(detected)

        rig_results.append(
            {
                "rig": f"R{rig_index + 1}",
                "detected_failures": sum(rig_failures),
                "n_failures": len(rig_failures),
                "false_healthy_machines": rig_healthy_machines,
                "n_healthy_machines": HEALTHY_MACHINES_PER_RIG,
                "false_healthy_axes": rig_healthy_axes,
                "n_healthy_axes": HEALTHY_MACHINES_PER_RIG * N_AXES_PER_MACHINE,
            }
        )

    n_failures = len(failure_rows)
    detected = sum(row["detected"] for row in failure_rows)
    leads = [row["endpoint_lead_pct"] for row in failure_rows if row["endpoint_lead_pct"] is not None]
    lags = [row["detection_lag_after_onset"] for row in failure_rows if row["detection_lag_after_onset"] is not None]
    false_machines = sum(row["false_alarm"] for row in healthy_machine_rows)
    false_axes = sum(row["false_alarm"] for row in healthy_axis_rows)
    per_rig_detection = [row["detected_failures"] / row["n_failures"] for row in rig_results]
    per_rig_machine_false = [row["false_healthy_machines"] / row["n_healthy_machines"] for row in rig_results]

    return {
        "design": {
            "n_rigs": N_RIGS,
            "healthy_machines_per_rig": HEALTHY_MACHINES_PER_RIG,
            "failures_per_rig": FAILURES_PER_RIG,
            "axes_per_machine": N_AXES_PER_MACHINE,
            "healthy_lengths_snapshots": [220, 500],
            "failure_lengths_snapshots": [520, 900],
            "fault_onset_snapshots": [180, 380],
            "condition": "shared low-frequency sinusoid plus bounded random walk; axes have independent lognormal noise",
            "failure": "nonlinear monotonic RMS gain after latent fault onset on one randomly selected axis",
            "detector": "historical frozen RMS-excess-rate rule; no parameter changed",
        },
        "totals": {
            "n_failed_machines": n_failures,
            "detected": detected,
            "detection_rate_pct": 100.0 * detected / n_failures,
            "detection_wilson_95_pct": wilson_interval(detected, n_failures),
            "n_healthy_machines": len(healthy_machine_rows),
            "false_alarm_machines": false_machines,
            "machine_false_alarm_rate_pct": 100.0 * false_machines / len(healthy_machine_rows),
            "machine_false_alarm_wilson_95_pct": wilson_interval(false_machines, len(healthy_machine_rows)),
            "n_healthy_axes": len(healthy_axis_rows),
            "false_alarm_axes": false_axes,
            "axis_false_alarm_rate_pct": 100.0 * false_axes / len(healthy_axis_rows),
            "axis_false_alarm_wilson_95_pct": wilson_interval(false_axes, len(healthy_axis_rows)),
            "median_endpoint_lead_pct": percentile(leads, 50),
            "endpoint_lead_p25_pct": percentile(leads, 25),
            "endpoint_lead_p75_pct": percentile(leads, 75),
            "median_detection_lag_after_onset_snapshots": percentile(lags, 50),
            "detection_lag_p25_snapshots": percentile(lags, 25),
            "detection_lag_p75_snapshots": percentile(lags, 75),
            "detection_rate_range_across_rigs_pct": [100.0 * min(per_rig_detection), 100.0 * max(per_rig_detection)]
            if per_rig_detection
            else None,
            "machine_false_alarm_range_across_rigs_pct": [100.0 * min(per_rig_machine_false), 100.0 * max(per_rig_machine_false)]
            if per_rig_machine_false
            else None,
        },
        "per_rig": rig_results,
        "failure_trajectories": failure_rows,
        "healthy_machines": healthy_machine_rows,
    }


def correlated_axis_event_matrix(
    rng: np.random.Generator, rows: int, probability: float, correlation: float
) -> np.ndarray:
    common = rng.standard_normal((rows, 1))
    independent = rng.standard_normal((rows, N_AXES_PER_MACHINE))
    latent = math.sqrt(correlation) * common + math.sqrt(1.0 - correlation) * independent
    threshold = NormalDist().inv_cdf(1.0 - probability)
    return latent > threshold


def transmit_event(
    rng: np.random.Generator, loss_rate: float, retries: int
) -> Tuple[bool, int, float, float, int]:
    delivered = 0
    first_latency = None
    attempts = 0
    for _attempt in range(retries + 1):
        attempts += 1
        data_delivered = rng.random() >= loss_rate
        if data_delivered:
            delivered += 1
            latency = float(rng.lognormal(LATENCY_LOG_MEAN_MS, LATENCY_LOG_SIGMA_MS))
            first_latency = latency if first_latency is None else min(first_latency, latency)
            if rng.random() >= loss_rate:
                break
    completion_latency = COMPLETION_CENSOR_MS if first_latency is None else first_latency
    return delivered > 0, delivered - 1, first_latency, completion_latency, attempts


def simulate_network(rng: np.random.Generator) -> dict:
    scenarios = []
    event_probability = 0.01
    periodic_bytes = N_NETWORK_MACHINES * N_AXES_PER_MACHINE * N_NETWORK_CYCLES * PERIODIC_STATUS_BYTES

    for loss_rate in LOSS_RATES:
        for retries in RETRIES:
            delivered_events = 0
            duplicate_events = 0
            stale_events = 0
            latencies: List[float] = []
            completion_latencies: List[float] = []
            transmitted_packets = 0
            delivered_packets = 0
            active = correlated_axis_event_matrix(
                rng, N_NETWORK_CYCLES * N_NETWORK_MACHINES, event_probability, 0.2
            )
            event_count = int(np.sum(np.any(active, axis=1)))
            for _ in range(event_count):
                delivered, duplicates, latency, completion_latency, attempts = transmit_event(
                    rng, loss_rate, retries
                )
                transmitted_packets += attempts
                completion_latencies.append(completion_latency)
                if delivered:
                    delivered_events += 1
                    delivered_packets += 1 + duplicates
                duplicate_events += int(duplicates > 0)
                if latency is not None:
                    latencies.append(latency)
                    stale_events += int(latency > STALE_EVENT_MS)
            event_bytes = transmitted_packets * EVENT_PACKET_BYTES
            scenarios.append(
                {
                    "loss_rate_pct": 100.0 * loss_rate,
                    "retries": retries,
                    "n_events": event_count,
                    "delivery_ratio_pct": 100.0 * delivered_events / event_count,
                    "duplicate_ratio_pct": 100.0 * duplicate_events / event_count,
                    "stale_event_ratio_pct": 100.0 * stale_events / event_count,
                    "latency_p50_ms": percentile(latencies, 50),
                    "latency_p95_ms": percentile(latencies, 95),
                    "latency_p99_ms": percentile(latencies, 99),
                    "completion_latency_p50_ms_censored_at_1000": percentile(completion_latencies, 50),
                    "completion_latency_p95_ms_censored_at_1000": percentile(completion_latencies, 95),
                    "completion_latency_p99_ms_censored_at_1000": percentile(completion_latencies, 99),
                    "event_packet_bytes": event_bytes,
                    "periodic_status_bytes": periodic_bytes,
                    "byte_compression_ratio_pct": 100.0 * (1.0 - event_bytes / periodic_bytes),
                }
            )

    return {
        "design": {
            "machines": N_NETWORK_MACHINES,
            "axes_per_machine": N_AXES_PER_MACHINE,
            "cycles": N_NETWORK_CYCLES,
            "axis_event_probability_per_cycle": event_probability,
            "axis_correlation": 0.2,
            "event_packet_bytes": EVENT_PACKET_BYTES,
            "periodic_status_packet_bytes": PERIODIC_STATUS_BYTES,
            "latency_ms": "lognormal(mu=3.4, sigma=0.65)",
            "latency_median_mean_variance_ms": [30.0, 37.0, 720.3],
            "stale_event_threshold_ms": STALE_EVENT_MS,
            "retry_wait_ms": 0.0,
            "completion_time_rule": "first delivered packet latency; unavailable events are censored at 1000 ms",
            "deduplication": "receiver keeps the first event ID and discards later duplicate payloads",
            "acknowledgement": "each successful data or ACK transmission is lost independently",
        },
        "scenarios": scenarios,
        "interpretation": "Network results are packet-level simulations, not measurements on a real MQTT/CoAP deployment.",
    }


def simulate_fleet_budget(rng: np.random.Generator) -> dict:
    results = []
    for probability in FLEET_PROBABILITIES:
        for correlation in FLEET_CORRELATIONS:
            active = correlated_axis_event_matrix(
                rng, N_NETWORK_CYCLES * N_NETWORK_MACHINES, probability, correlation
            ).reshape(N_NETWORK_CYCLES, N_NETWORK_MACHINES, N_AXES_PER_MACHINE)
            axis_array = np.sum(active, axis=(1, 2))
            any_array = np.sum(np.any(active, axis=2), axis=1)
            both_array = np.sum(np.all(active, axis=2), axis=1)
            results.append(
                {
                    "axis_event_probability": probability,
                    "axis_correlation": correlation,
                    "mean_axis_events_per_cycle": float(axis_array.mean()),
                    "p95_axis_events_per_cycle": float(np.percentile(axis_array, 95)),
                    "mean_any_axis_machines_per_cycle": float(any_array.mean()),
                    "p95_any_axis_machines_per_cycle": float(np.percentile(any_array, 95)),
                    "probability_any_axis_machines_ge_5pct": float(np.mean(any_array >= 5)),
                    "mean_cross_axis_confirmed_machines_per_cycle": float(both_array.mean()),
                    "p95_cross_axis_confirmed_machines_per_cycle": float(np.percentile(both_array, 95)),
                    "event_reduction_by_cross_axis_confirmation_pct": 100.0
                    * (1.0 - both_array.mean() / max(any_array.mean(), 1e-12)),
                }
            )
    return {
        "design": {
            "machines": N_NETWORK_MACHINES,
            "axes_per_machine": N_AXES_PER_MACHINE,
            "cycles": N_NETWORK_CYCLES,
            "axis_event_probabilities": list(FLEET_PROBABILITIES),
            "axis_correlations": list(FLEET_CORRELATIONS),
            "correlation_model": "bivariate Gaussian copula with common correlation",
            "budget_rule": "report load and the probability that at least 5% of machines emit an any-axis event in a cycle",
        },
        "scenarios": results,
    }


def result_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def main() -> None:
    rng = np.random.default_rng(SEED)
    output = {
        "schema_version": "1.0",
        "simulation_only": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "frozen_detector": {
            "baseline_slice": [10, 60],
            "monitor_start_snapshot": MONITOR_START,
            "earliest_update": EARLIEST_UPDATE,
            "raw_threshold": RAW_THRESHOLD,
            "rms_threshold_multiplier": 1.2,
            "retuning_allowed": False,
        },
        "multi_rig_healthy_censored_simulation": simulate_multi_rig_cohort(rng),
        "network_multi_node_simulation": simulate_network(rng),
        "fleet_event_budget_simulation": simulate_fleet_budget(rng),
        "evidence_boundary": {
            "supports": "protocol-level sensitivity, false-alarm, packet-load, and budget stress testing under stated synthetic mechanisms",
            "does_not_support": [
                "completion of independent multi-rig frozen validation",
                "real healthy-censored fleet validation",
                "real MQTT/CoAP or multi-hop network measurement",
                "MCU energy, ADC, gateway, cloud, or maintenance-workflow validation",
            ],
            "required_real_validation": "Pre-register the protocol, obtain an independent multi-rig multi-axis cohort with healthy censoring and trustworthy fault onset, run once without tuning, and record all alarms and network/edge costs.",
        },
        "script_sha256": result_sha256(),
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {RESULT_PATH}")


if __name__ == "__main__":
    main()
