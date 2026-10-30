#!/usr/bin/env python3
"""Exact in-paper replication of the closest raw RMS-ratio rule.

The rule is placed under the manuscript's frozen 34-failure/8-censored
protocol: snapshots 11--60 calibrate each axis, monitoring starts at 61, an
alarm is eligible after 30 updates, any axis may alarm, and the threshold is
1.01 times the largest eligible censored-axis score.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "experiments"))
import run_to_failure_classical_baselines as classical  # noqa: E402

EARLIEST_UPDATE = 30
THRESHOLD_MARGIN = 1.01


def ratio_score(rms: np.ndarray) -> np.ndarray:
    baseline = float(np.median(rms[10:60]))
    return rms[60:] / baseline


def first_alarm(score: np.ndarray, threshold: float) -> int:
    eligible = np.flatnonzero(score[EARLIEST_UPDATE - 1 :] > threshold)
    return int(eligible[0] + EARLIEST_UPDATE - 1) if eligible.size else int(score.size)


def main() -> None:
    feature_dir = ROOT / "data" / "features"
    streams = classical.load_all_streams(feature_dir)
    bearings: Dict[str, dict] = {}
    for stream in streams:
        score = ratio_score(stream["rms"])
        if score.size < EARLIEST_UPDATE:
            continue
        bearing = bearings.setdefault(
            stream["bearing_id"],
            {
                "bearing_id": stream["bearing_id"],
                "dataset": stream["dataset"],
                "failed": bool(stream["failed"]),
                "axes": {},
                "length": int(len(stream["rms"])),
            },
        )
        bearing["axes"][str(stream["axis"])] = score

    healthy = [row for row in bearings.values() if not row["failed"]]
    failed = [row for row in bearings.values() if row["failed"]]
    max_censored = max(float(np.max(score[EARLIEST_UPDATE - 1 :])) for row in healthy for score in row["axes"].values())
    threshold = THRESHOLD_MARGIN * max_censored

    per_bearing = []
    per_dataset = {}
    for dataset in ("ims", "xjtu", "femto"):
        rows = [row for row in failed if row["dataset"] == dataset]
        detected = 0
        leads = []
        for row in rows:
            first = min(first_alarm(score, threshold) for score in row["axes"].values())
            is_detected = first < row["length"] - 60
            lead = 100.0 * (row["length"] - (60 + first)) / row["length"] if is_detected else None
            detected += int(is_detected)
            if lead is not None:
                leads.append(lead)
            per_bearing.append(
                {
                    "dataset": dataset,
                    "bearing_id": row["bearing_id"],
                    "detected": is_detected,
                    "alarm_snapshot": 60 + first if is_detected else None,
                    "endpoint_lead_pct_life": lead,
                    "axis_alarm_updates": {
                        axis: first_alarm(score, threshold) if first_alarm(score, threshold) < score.size else None
                        for axis, score in row["axes"].items()
                    },
                }
            )
        per_dataset[dataset] = {
            "n": len(rows),
            "n_detected": detected,
            "median_lead_pct_life": float(np.median(leads)) if leads else None,
            "missed_bearing_ids": [row["bearing_id"] for row in rows if not any(item["bearing_id"] == row["bearing_id"] and item["detected"] for item in per_bearing)],
        }

    output = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "baseline_slice": [10, 60],
            "monitor_start_snapshot": 61,
            "earliest_update": EARLIEST_UPDATE,
            "bearing_rule": "any axis exceeds threshold",
            "threshold_rule": "1.01 times maximum eligible censored-axis raw RMS ratio",
            "lead_definition": "(endpoint snapshot - alarm snapshot) / endpoint snapshot",
        },
        "n_failed_bearings": len(failed),
        "n_censored_bearings": len(healthy),
        "threshold": threshold,
        "max_censored_score": max_censored,
        "total_detected": sum(row["n_detected"] for row in per_dataset.values()),
        "per_dataset": per_dataset,
        "per_bearing": per_bearing,
    }
    destination = ROOT / "results" / "results_sg1026_1_closest_rms_ratio_baseline.json"
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
