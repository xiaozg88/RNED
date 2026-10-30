#!/usr/bin/env python3
"""Evaluate periodic envelope updates for the impact-aware RMS feature."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "experiments") not in sys.path:
    sys.path.insert(str(ROOT / "experiments"))

from run_to_failure_rms_variants import evaluate, load_axis_streams


STRIDES = (1, 5, 10, 20, 50)
IMPACT_WEIGHT = 0.10


def periodic_impact_aware_rms(values: dict, stride: int) -> np.ndarray:
    raw = values["raw"]
    envelope = values["env"]
    raw_ratio = raw / float(np.median(raw[10:60]))
    envelope_ratio = envelope / float(np.median(envelope[10:60]))
    sampled_ratio = np.empty_like(envelope_ratio)
    held_ratio = envelope_ratio[0]
    for index in range(len(envelope_ratio)):
        if index < 60 or index == 60 or (index - 60) % stride == 0:
            held_ratio = envelope_ratio[index]
        sampled_ratio[index] = held_ratio
    return np.sqrt(
        (1.0 - IMPACT_WEIGHT) * np.square(raw_ratio)
        + IMPACT_WEIGHT * np.square(sampled_ratio)
    )


def main() -> None:
    bearings = {}
    for (dataset, _group, bearing, axis), values in load_axis_streams().items():
        bearing_key = f"{dataset}_{bearing}"
        bearings.setdefault(bearing_key, {"dataset": dataset, "axes": {}})
        variants = {}
        for stride in STRIDES:
            name = f"periodic_envelope_w10_stride{stride}"
            variants[name] = periodic_impact_aware_rms(values, stride)
        bearings[bearing_key]["axes"][axis] = variants

    variant_names = list(next(iter(bearings.values()))["axes"].values())[0]
    results = {variant: evaluate(bearings, variant) for variant in variant_names}
    output = {
        "protocol": {
            "status": "exploratory post-hoc duty-cycle screen; not a locked validation",
            "feature": "sqrt(0.9*(RMS/b_RMS)^2+0.1*(envRMS/b_envRMS)^2)",
            "envelope_update": "snapshot 60 and every stride thereafter; last envelope value is held between updates",
            "baseline": "per-axis median over snapshots 11-60; envelope baseline uses the same early-life snapshots",
            "excess_ratio_threshold": 1.2,
            "calibration_set": "eight IMS censored bearings",
            "strides": list(STRIDES),
        },
        "results": results,
    }
    output_path = ROOT / "results/results_run_to_failure_duty_cycle_rms.json"
    output_path.write_text(json.dumps(output, indent=2))

    csv_path = ROOT / "results/results_run_to_failure_duty_cycle_rms.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "stride",
                "envelope_duty_cycle_pct",
                "detected",
                "median_lead_pct_life",
                "n_false_alarm_axes",
                "n_false_alarm_bearings",
            ],
        )
        writer.writeheader()
        for stride in STRIDES:
            result = results[f"periodic_envelope_w10_stride{stride}"]
            writer.writerow(
                {
                    "stride": stride,
                    "envelope_duty_cycle_pct": 100.0 / stride,
                    "detected": result["detected"],
                    "median_lead_pct_life": result["median_lead_pct_life"],
                    "n_false_alarm_axes": result["n_false_alarm_axes"],
                    "n_false_alarm_bearings": result["n_false_alarm_bearings"],
                }
            )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
