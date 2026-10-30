#!/usr/bin/env python3
"""Frozen steady-state false-alarm stress test on CWRU normal baselines."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "experiments"))
import run_to_failure_rate_acceleration as probe  # noqa: E402
from run_to_failure_cross_axis_persistence import cross_axis_score  # noqa: E402


SELECTED_FILES = ("1730_Normal.npz", "1750_Normal.npz", "1772_Normal.npz")
SNAPSHOT_SAMPLES = 2048
BASELINE_SNAPSHOTS = 60
MINIMUM_MONITOR_SNAPSHOTS = 30
RECENT_WINDOW = 10
FROZEN_HISTORICAL_THRESHOLD = 0.042407782315012556
FROZEN_CROSS_AXIS_THRESHOLD = 0.24602807750741573


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protocol_sha256() -> str:
    return hashlib.sha256(
        (ROOT / "CWRU_NORMAL_STRESS_PREREGISTRATION.md").read_bytes()
    ).hexdigest()


def load_channel(path: Path, channel: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        if channel not in archive.files:
            raise ValueError(f"missing {channel} in {path}")
        values = np.asarray(archive[channel])
    if values.ndim == 2 and 1 in values.shape:
        values = values.reshape(-1)
    if values.ndim != 1:
        raise ValueError(f"{channel} in {path} is not one-dimensional: {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"non-finite values in {channel} in {path}")
    return values.astype(np.float64, copy=False)


def segment_rms(values: np.ndarray, snapshot_samples: int) -> np.ndarray:
    n_snapshots = values.size // snapshot_samples
    if n_snapshots < BASELINE_SNAPSHOTS + 90:
        raise ValueError(
            f"only {n_snapshots} snapshots; expected at least "
            f"{BASELINE_SNAPSHOTS + 90}"
        )
    trimmed = values[: n_snapshots * snapshot_samples]
    segments = trimmed.reshape(n_snapshots, snapshot_samples)
    return np.sqrt(np.mean(np.square(segments), axis=1))


def wilson_interval(successes: int, total: int) -> list[float]:
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half_width = (
        z
        * np.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total**2))
        / denominator
    )
    return [
        round(100.0 * max(0.0, center - half_width), 2),
        round(100.0 * min(1.0, center + half_width), 2),
    ]


def evaluate_machine(path: Path) -> tuple[dict, list[dict]]:
    axis_results = []
    historical_rates = []
    ratios = []
    for channel in ("DE", "FE"):
        values = load_channel(path, channel)
        rms = segment_rms(values, SNAPSHOT_SAMPLES)
        baseline = float(np.median(rms[10:60]))
        _, historical_rate, _ = probe.excess_statistics(
            rms, recent_window=RECENT_WINDOW
        )
        alarm = probe.alarm_index(
            historical_rate,
            FROZEN_HISTORICAL_THRESHOLD,
            min_window=MINIMUM_MONITOR_SNAPSHOTS,
        )
        axis_results.append(
            {
                "axis": channel,
                "n_samples": int(values.size),
                "n_snapshots": int(rms.size),
                "n_monitor": int(historical_rate.size),
                "baseline_rms": baseline,
                "historical_max": float(np.max(historical_rate)),
                "historical_alarm_index": int(alarm),
                "historical_false_alarm": bool(alarm < historical_rate.size),
            }
        )
        historical_rates.append(historical_rate)
        ratios.append(rms[BASELINE_SNAPSHOTS:] / baseline)

    machine_historical = np.maximum.reduce(historical_rates)
    historical_alarm = probe.alarm_index(
        machine_historical,
        FROZEN_HISTORICAL_THRESHOLD,
        min_window=MINIMUM_MONITOR_SNAPSHOTS,
    )
    cross_axis = cross_axis_score(ratios, window=RECENT_WINDOW)
    cross_axis_alarm = probe.alarm_index(
        cross_axis,
        FROZEN_CROSS_AXIS_THRESHOLD,
        min_window=MINIMUM_MONITOR_SNAPSHOTS,
    )
    hierarchical_alarm = min(historical_alarm, cross_axis_alarm)
    n_monitor = int(machine_historical.size)

    machine = {
        "file": path.name,
        "sha256": file_sha256(path),
        "n_axes": len(axis_results),
        "n_monitor": n_monitor,
        "historical_max": float(np.max(machine_historical)),
        "historical_alarm_index": int(historical_alarm),
        "historical_false_alarm": bool(historical_alarm < n_monitor),
        "cross_axis_max": float(np.max(cross_axis)),
        "cross_axis_alarm_index": int(cross_axis_alarm),
        "cross_axis_false_alarm": bool(cross_axis_alarm < n_monitor),
        "hierarchical_alarm_index": int(hierarchical_alarm),
        "hierarchical_false_alarm": bool(hierarchical_alarm < n_monitor),
        "axes": axis_results,
    }
    return machine, axis_results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="directory containing CWRU .npz files")
    parser.add_argument("output", type=Path, help="output JSON")
    parser.add_argument(
        "--features-output",
        type=Path,
        help="optional CSV containing the frozen pseudo-snapshot RMS streams",
    )
    args = parser.parse_args()

    machines = []
    feature_rows = []
    for filename in SELECTED_FILES:
        path = args.root / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        print(f"evaluating {filename}", flush=True)
        machine, axis_results = evaluate_machine(path)
        machines.append(machine)
        rpm = filename.split("_", 1)[0]
        for axis_result in axis_results:
            values = load_channel(path, axis_result["axis"])
            rms = segment_rms(values, SNAPSHOT_SAMPLES)
            for snapshot, value in enumerate(rms, start=1):
                feature_rows.append(
                    {
                        "dataset": "cwru_normal",
                        "group": f"{rpm}rpm",
                        "bearing": f"normal_{rpm}rpm",
                        "snapshot": snapshot,
                        "file": filename,
                        "axis": axis_result["axis"],
                        "rms": float(value),
                        "n_rows": SNAPSHOT_SAMPLES,
                        "sampling_frequency_hz": 12000,
                    }
                )

    n_machines = len(machines)
    n_axes = sum(machine["n_axes"] for machine in machines)
    axis_false_alarms = sum(
        axis["historical_false_alarm"]
        for machine in machines
        for axis in machine["axes"]
    )
    machine_historical = sum(
        machine["historical_false_alarm"] for machine in machines
    )
    machine_cross_axis = sum(
        machine["cross_axis_false_alarm"] for machine in machines
    )
    machine_hierarchical = sum(
        machine["hierarchical_false_alarm"] for machine in machines
    )

    result = {
        "protocol": {
            "status": "frozen before outcome computation",
            "protocol_sha256": protocol_sha256(),
            "selected_files": list(SELECTED_FILES),
            "snapshot_samples": SNAPSHOT_SAMPLES,
            "baseline": "median(rms[10:60])",
            "excess_ratio_threshold": 1.2,
            "minimum_monitor_snapshots": MINIMUM_MONITOR_SNAPSHOTS,
            "recent_window": RECENT_WINDOW,
            "frozen_historical_threshold": FROZEN_HISTORICAL_THRESHOLD,
            "frozen_cross_axis_threshold": FROZEN_CROSS_AXIS_THRESHOLD,
            "post_hoc_tuning_allowed": False,
        },
        "axis_level": {
            "n_axes": n_axes,
            "historical_false_alarms": axis_false_alarms,
            "false_alarm_rate_pct": 100.0 * axis_false_alarms / n_axes,
            "wilson_95_pct": wilson_interval(axis_false_alarms, n_axes),
        },
        "machine_level": {
            "n_machines": n_machines,
            "historical_false_alarms": machine_historical,
            "historical_false_alarm_rate_pct": 100.0 * machine_historical / n_machines,
            "cross_axis_false_alarms": machine_cross_axis,
            "cross_axis_false_alarm_rate_pct": 100.0 * machine_cross_axis / n_machines,
            "hierarchical_false_alarms": machine_hierarchical,
            "hierarchical_false_alarm_rate_pct": 100.0 * machine_hierarchical / n_machines,
            "historical_wilson_95_pct": wilson_interval(machine_historical, n_machines),
            "hierarchical_wilson_95_pct": wilson_interval(machine_hierarchical, n_machines),
        },
        "per_machine": machines,
        "interpretation": (
            "steady-state normal-condition stress test only; not a healthy "
            "censored-fleet validation and not evidence of finite conformal "
            "false-alarm control"
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")

    if args.features_output is not None:
        fields = [
            "dataset",
            "group",
            "bearing",
            "snapshot",
            "file",
            "axis",
            "rms",
            "n_rows",
            "sampling_frequency_hz",
        ]
        args.features_output.parent.mkdir(parents=True, exist_ok=True)
        with args.features_output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(feature_rows)

    print(f"wrote {args.output}")
    print(
        "axis historical false alarms: "
        f"{axis_false_alarms}/{n_axes}; machine hierarchy: "
        f"{machine_hierarchical}/{n_machines}"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise
