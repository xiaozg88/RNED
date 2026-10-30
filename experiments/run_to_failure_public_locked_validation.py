#!/usr/bin/env python3
"""Run the frozen hierarchy on public external feature tables."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "experiments"))
import run_to_failure_rate_acceleration as probe  # noqa: E402
from run_to_failure_cross_axis_persistence import cross_axis_score  # noqa: E402


FROZEN_HISTORICAL_THRESHOLD = 0.042407782315012556
FROZEN_CROSS_AXIS_THRESHOLD = 0.24602807750741573
PROTOCOL_SHA256 = hashlib.sha256(
    (ROOT / "CROSS_AXIS_PREREGISTRATION.md").read_bytes()
).hexdigest()


def load_bearings(paths: list[Path]) -> list[dict]:
    grouped: dict[
        tuple[str, str, str], dict[str, list[tuple[int, float]]]
    ] = defaultdict(dict)
    for path in paths:
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            key = (row["dataset"], row["group"], row["bearing"])
            axis = row["axis"]
            grouped[key].setdefault(axis, []).append(
                (int(row["snapshot"]), float(row["rms"]))
            )

    for key, axes in grouped.items():
        for axis in list(axes):
            axes[axis] = np.asarray(
                [value for _, value in sorted(axes[axis])], dtype=np.float64
            )
            if not np.all(np.isfinite(axes[axis])) or axes[axis].size < 60:
                raise ValueError(f"invalid or short {key}/{axis} stream")

    bearings = []
    for key, axes in sorted(grouped.items()):
        bearings.append(
            {
                "dataset": key[0],
                "group": key[1],
                "bearing_id": key[2],
                "rms": [axes[axis] for axis in sorted(axes)],
                "axis_names": sorted(axes),
                "failed": True,
            }
        )
    return bearings


def evaluate_bearing(bearing: dict) -> dict:
    ratios = []
    historical_rates = []
    for rms in bearing["rms"]:
        _, historical_rate, _ = probe.excess_statistics(rms, recent_window=10)
        baseline = float(np.median(rms[10:60]))
        ratios.append(rms[60:] / baseline)
        historical_rates.append(historical_rate)

    historical_score = np.maximum.reduce(historical_rates)
    historical_alarm = probe.alarm_index(
        historical_score, FROZEN_HISTORICAL_THRESHOLD
    )
    result = {
        "bearing_id": bearing["bearing_id"],
        "dataset": bearing["dataset"],
        "group": bearing["group"],
        "n_axes": len(bearing["rms"]),
        "n_monitor": int(historical_score.size),
        "protocol_eligible": bool(historical_score.size >= 30),
        "historical_max": float(np.max(historical_score)),
        "historical_alarm_index": int(historical_alarm),
        "historical_detected": bool(historical_alarm < historical_score.size),
        "historical_lead_pct_life": (
            100.0 * (historical_score.size - historical_alarm) / historical_score.size
            if historical_alarm < historical_score.size
            else None
        ),
        "cross_axis_evaluable": len(ratios) >= 2,
        "cross_axis_max": None,
        "cross_axis_alarm_index": None,
        "cross_axis_detected": False,
        "cross_axis_lead_pct_life": None,
    }

    if len(ratios) >= 2:
        cross_axis = cross_axis_score(ratios, window=10)
        alarm = probe.alarm_index(cross_axis, FROZEN_CROSS_AXIS_THRESHOLD)
        result.update(
            {
                "cross_axis_max": float(np.max(cross_axis)),
                "cross_axis_alarm_index": int(alarm),
                "cross_axis_detected": bool(alarm < cross_axis.size),
                "cross_axis_lead_pct_life": (
                    100.0 * (cross_axis.size - alarm) / cross_axis.size
                    if alarm < cross_axis.size
                    else None
                ),
            }
        )

    hierarchical_alarm = min(
        result["historical_alarm_index"],
        result["cross_axis_alarm_index"]
        if result["cross_axis_alarm_index"] is not None
        else result["historical_alarm_index"],
    )
    result["hierarchical_alarm_index"] = int(hierarchical_alarm)
    result["hierarchy_evaluable"] = len(ratios) >= 2
    result["hierarchical_detected"] = bool(
        result["hierarchy_evaluable"] and hierarchical_alarm < result["n_monitor"]
    )
    result["hierarchical_lead_pct_life"] = (
        100.0 * (result["n_monitor"] - hierarchical_alarm) / result["n_monitor"]
        if result["hierarchical_detected"]
        else None
    )
    return result


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "feature_csv", nargs="+", type=Path, help="public per-axis RMS feature CSV(s)"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    bearings = load_bearings(args.feature_csv)
    if not bearings:
        raise ValueError("no public bearings found")
    rows = [evaluate_bearing(bearing) for bearing in bearings]
    eligible_rows = [row for row in rows if row["protocol_eligible"]]
    historical_detected = sum(row["historical_detected"] for row in eligible_rows)
    hierarchy_rows = [
        row for row in eligible_rows if row["hierarchy_evaluable"]
    ]
    hierarchical_detected = sum(row["hierarchical_detected"] for row in hierarchy_rows)
    hierarchical_leads = [row["hierarchical_lead_pct_life"] for row in rows if row["hierarchical_lead_pct_life"] is not None]

    result = {
        "protocol": {
            "status": "frozen before public external testing",
            "protocol_sha256": PROTOCOL_SHA256,
            "feature": "per-axis RMS",
            "baseline": "median(rms[10:60])",
            "excess_ratio_threshold": 1.2,
            "window": 10,
            "cross_axis_statistic": "second-largest axis recent excess mean",
            "decision": "historical OR cross-axis threshold crossing",
            "minimum_monitor_snapshots": 30,
            "frozen_historical_threshold": FROZEN_HISTORICAL_THRESHOLD,
            "frozen_cross_axis_threshold": FROZEN_CROSS_AXIS_THRESHOLD,
            "post_hoc_tuning_allowed": False,
        },
        "inputs": [
            {"path": str(path), "sha256": file_sha256(path)} for path in args.feature_csv
        ],
        "n_failed": len(rows),
        "n_protocol_eligible": len(eligible_rows),
        "historical_detected": int(historical_detected),
        "hierarchical_n_evaluable": len(hierarchy_rows),
        "hierarchical_detected": int(hierarchical_detected),
        "median_hierarchical_lead_pct_life": (
            float(np.median(hierarchical_leads)) if hierarchical_leads else None
        ),
        "per_bearing": rows,
        "false_alarm_endpoint": {
            "evaluable": False,
            "reason": "public run-to-failure inputs contain no independent healthy censored fleet",
        },
        "interpretation": (
            "locked external failure-only replication; cross-axis hierarchy is evaluable only for bearings with at least two axes; not sufficient for main-method upgrade"
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    destination = args.output or (
        ROOT / "results" / "results_run_to_failure_public_locked_validation.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("n_failed", "n_protocol_eligible", "historical_detected", "hierarchical_n_evaluable", "hierarchical_detected", "median_hierarchical_lead_pct_life")}, indent=2))
    print(f"saved {destination}")


if __name__ == "__main__":
    main()
