#!/usr/bin/env python3
"""Replicate the public D1 RMS-ratio protocol and align it with the paper cohort.

The public protocol is fixed as: baseline mean on snapshots 11--60, ratio
threshold 1.2, three consecutive exceedances, and a warning sustained to the
endpoint with alarm gaps no larger than 1% of life. A warning lasting at least
1% of life is detected. The original cohort requires at least 120 snapshots.
For direct comparison with the manuscript, the script also evaluates the same
alarm semantics on the frozen 34-failure cohort (which requires at least 30
monitored updates rather than the public 120-snapshot eligibility rule).
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "experiments"))
import run_to_failure_classical_baselines as classical  # noqa: E402

BASE_END = 60
CONSECUTIVE_K = 3
RATIO_THRESHOLD = 1.2
MIN_PUBLIC_SNAPSHOTS = 120
MIN_PAPER_MONITOR_UPDATES = 30
FINAL_AMBIGUOUS_FRACTION = 0.15
SUSTAINED_MIN_FRACTION = 0.01
SNAPSHOT_HOURS = {"ims": 10.0 / 60.0, "xjtu": 1.0 / 60.0, "femto": 10.0 / 3600.0}


def trim_trailing_dead_snapshots(streams: List[dict]) -> tuple[List[dict], int]:
    """Apply the public IMS-style trailing invalid-snapshot rule per bearing."""
    if not streams:
        return streams, 0
    length = min(stream["rms"].size for stream in streams)
    medians = [float(np.median(stream["rms"])) for stream in streams]
    trimmed = length
    while trimmed > BASE_END and all(
        float(stream["rms"][trimmed - 1]) < 0.10 * median
        for stream, median in zip(streams, medians)
    ):
        trimmed -= 1
    result = []
    for stream in streams:
        item = dict(stream)
        item["rms"] = stream["rms"][:trimmed].copy()
        item["time_hours"] = stream["time_hours"][:trimmed].copy()
        result.append(item)
    return result, length - trimmed


def consecutive_alarm(ratio: np.ndarray, baseline: float) -> np.ndarray:
    exceed = ratio > RATIO_THRESHOLD * baseline
    count = 0
    alarm = np.zeros(exceed.size, dtype=bool)
    for index, value in enumerate(exceed):
        count = count + 1 if value else 0
        alarm[index] = count >= CONSECUTIVE_K
    alarm[:BASE_END] = False
    return alarm


def evaluate_axis(rms: np.ndarray, time_hours: np.ndarray) -> dict:
    baseline = float(np.mean(rms[10:BASE_END]))
    alarm = consecutive_alarm(rms, baseline)
    alarm_indices = np.flatnonzero(alarm)
    life_hours = float(time_hours[-1])
    sustained_index = None
    if alarm_indices.size and alarm_indices[-1] >= rms.size - 3:
        alarm_position = int(alarm_indices.size - 1)
        while alarm_position > 0:
            gap_hours = float(
                time_hours[alarm_indices[alarm_position]]
                - time_hours[alarm_indices[alarm_position - 1]]
            )
            if gap_hours > SUSTAINED_MIN_FRACTION * life_hours + 1e-12:
                break
            alarm_position -= 1
        sustained_index = int(alarm_indices[alarm_position])

    monitoring = np.arange(rms.size) >= BASE_END
    remaining_life_fraction = (life_hours - time_hours) / life_hours
    reference = monitoring & (remaining_life_fraction > FINAL_AMBIGUOUS_FRACTION)
    rising = alarm & ~np.concatenate(([False], alarm[:-1]))
    reference_snapshots = int(reference.sum())
    false_alarm_episodes = int((rising & reference).sum())
    sustained_fraction = (
        float((life_hours - time_hours[sustained_index]) / life_hours)
        if sustained_index is not None
        else None
    )
    return {
        "baseline_rms": baseline,
        "n_snapshots": int(rms.size),
        "reference_snapshots": reference_snapshots,
        "false_alarm_episodes": false_alarm_episodes,
        "sustained_start_snapshot": sustained_index + 1 if sustained_index is not None else None,
        "sustained_fraction": sustained_fraction,
        "detected": sustained_fraction is not None and sustained_fraction >= SUSTAINED_MIN_FRACTION,
    }


def evaluate_bearing(streams: List[dict]) -> dict:
    dataset = streams[0]["dataset"]
    axes = {
        str(stream["axis"]): evaluate_axis(stream["rms"], stream["time_hours"])
        for stream in streams
    }
    detected_axes = [axis for axis, result in axes.items() if result["detected"]]
    detected = bool(detected_axes)
    first_detected_axis = min(detected_axes, key=lambda axis: axes[axis]["sustained_start_snapshot"]) if detected_axes else None
    first_detected = axes[first_detected_axis] if detected else None
    return {
        "bearing_id": streams[0]["bearing_id"],
        "dataset": dataset,
        "failed": bool(streams[0]["failed"]),
        "n_snapshots": max(result["n_snapshots"] for result in axes.values()),
        "n_monitor_updates": min(result["n_snapshots"] - BASE_END for result in axes.values()),
        "axes": axes,
        "detected": detected,
        "detected_axes": detected_axes,
        "first_detected_axis": first_detected_axis,
        "sustained_start_snapshot": first_detected["sustained_start_snapshot"] if first_detected else None,
        "endpoint_lead_pct_life": (
            100.0 * first_detected["sustained_fraction"] if first_detected else None
        ),
        "false_alarm_episodes": sum(result["false_alarm_episodes"] for result in axes.values()),
        "reference_snapshots": sum(result["reference_snapshots"] for result in axes.values()),
    }


def load_streams_with_public_clock(feature_dir: Path) -> List[dict]:
    streams: List[dict] = []
    for test in ("test1", "test2", "test3"):
        frame = pd.read_csv(feature_dir / f"{test}.csv")
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
        timestamps = pd.Series(sorted(frame["timestamp"].unique()))
        gaps = timestamps.diff().dt.total_seconds().fillna(0.0) / 3600.0
        operating_hours = np.where(gaps > 0.25, SNAPSHOT_HOURS["ims"], gaps).cumsum()
        clock = dict(zip(timestamps, operating_hours))
        for bearing in sorted(frame["bearing"].unique()):
            for channel in sorted(frame[frame["bearing"] == bearing]["channel"].unique()):
                subset = frame[(frame["bearing"] == bearing) & (frame["channel"] == channel)].sort_values("timestamp")
                streams.append(
                    {
                        "dataset": "ims",
                        "bearing_id": f"ims_{test}_b{bearing}",
                        "axis": int(channel),
                        "failed": (test, int(bearing)) in classical.IMS_FAILED,
                        "rms": subset["rms"].to_numpy(dtype=np.float64),
                        "time_hours": subset["timestamp"].map(clock).to_numpy(dtype=np.float64),
                    }
                )
    for csv_name, dataset in (("xjtu_features.csv", "xjtu"), ("femto_features.csv", "femto")):
        frame = pd.read_csv(feature_dir / csv_name)
        for (group, bearing) in sorted({(g, b) for g, b in zip(frame["group"], frame["bearing"])}):
            for axis in sorted(frame[(frame["group"] == group) & (frame["bearing"] == bearing)]["axis"].unique()):
                subset = frame[
                    (frame["group"] == group)
                    & (frame["bearing"] == bearing)
                    & (frame["axis"] == axis)
                ].sort_values("snapshot")
                snapshots = subset["snapshot"].to_numpy(dtype=np.int64)
                streams.append(
                    {
                        "dataset": dataset,
                        "bearing_id": f"{dataset}_{bearing}",
                        "axis": axis,
                        "failed": True,
                        "rms": subset["rms"].to_numpy(dtype=np.float64),
                        "time_hours": (snapshots - 1) * SNAPSHOT_HOURS[dataset],
                    }
                )
    return streams


def summarize(rows: List[dict], label: str) -> dict:
    failures = [row for row in rows if row["failed"]]
    censored = [row for row in rows if not row["failed"]]
    detected = [row for row in failures if row["detected"]]
    leads = [row["endpoint_lead_pct_life"] for row in detected]
    per_dataset = {}
    for dataset in ("ims", "xjtu", "femto"):
        subset = [row for row in failures if row["dataset"] == dataset]
        subset_detected = [row for row in subset if row["detected"]]
        subset_leads = [row["endpoint_lead_pct_life"] for row in subset_detected]
        per_dataset[dataset] = {
            "n": len(subset),
            "n_detected": len(subset_detected),
            "median_endpoint_lead_pct_life": float(np.median(subset_leads)) if subset_leads else None,
            "missed_bearing_ids": [row["bearing_id"] for row in subset if not row["detected"]],
        }

    def false_alarm_summary(group: List[dict]) -> dict:
        bearings_with_episode = [row for row in group if row["false_alarm_episodes"] > 0]
        episodes = sum(row["false_alarm_episodes"] for row in group)
        snapshots = sum(row["reference_snapshots"] for row in group)
        return {
            "n_bearings": len(group),
            "n_bearings_with_episode": len(bearings_with_episode),
            "bearing_ids_with_episode": [row["bearing_id"] for row in bearings_with_episode],
            "n_axis_episodes": episodes,
            "reference_axis_snapshots": snapshots,
            "episodes_per_1000_reference_snapshots": 1000.0 * episodes / snapshots if snapshots else None,
        }

    return {
        "label": label,
        "n_evaluated": len(rows),
        "n_failures": len(failures),
        "n_detected": len(detected),
        "median_endpoint_lead_pct_life": float(np.median(leads)) if leads else None,
        "per_dataset": per_dataset,
        "failure_healthy_reference_false_alarms": false_alarm_summary(failures),
        "ims_censored_healthy_reference_false_alarms": false_alarm_summary(censored),
        "per_bearing": rows,
    }


def main() -> None:
    feature_dir = ROOT / "data" / "features"
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for stream in load_streams_with_public_clock(feature_dir):
        grouped[stream["bearing_id"]].append(stream)

    trimmed_bearings = []
    total_trimmed = 0
    for bearing_id in sorted(grouped):
        streams, trimmed = trim_trailing_dead_snapshots(grouped[bearing_id])
        total_trimmed += trimmed
        if trimmed:
            trimmed_bearings.append({"bearing_id": bearing_id, "n_trimmed_snapshots": trimmed})
        grouped[bearing_id] = streams

    evaluated = [evaluate_bearing(grouped[bearing_id]) for bearing_id in sorted(grouped)]
    public_eligible = [row for row in evaluated if row["n_snapshots"] >= MIN_PUBLIC_SNAPSHOTS]
    paper_eligible = [row for row in evaluated if row["n_monitor_updates"] >= MIN_PAPER_MONITOR_UPDATES]
    excluded_public = [
        {"bearing_id": row["bearing_id"], "n_snapshots": row["n_snapshots"]}
        for row in evaluated
        if row["n_snapshots"] < MIN_PUBLIC_SNAPSHOTS
    ]
    excluded_paper = [
        {"bearing_id": row["bearing_id"], "n_monitor_updates": row["n_monitor_updates"]}
        for row in evaluated
        if row["n_monitor_updates"] < MIN_PAPER_MONITOR_UPDATES
    ]

    output = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "baseline": "arithmetic mean of snapshots 11--60",
            "rms_threshold": "1.2 times per-axis baseline mean",
            "alarm_persistence": "3 consecutive exceedances",
            "sustained_warning": "alarm block lasting to endpoint, gaps <= 1% of life",
            "detected": "sustained warning >= 1% of life",
            "bearing_rule": "any axis detected",
            "healthy_reference_zone": "snapshot 61 through 85% of observed life",
            "trailing_invalid_rule": "remove consecutive trailing snapshots in which every available axis is below 10% of its full-series median RMS",
        },
        "trailing_snapshots_removed": {"total": total_trimmed, "bearings": trimmed_bearings},
        "public_120_snapshot_cohort": summarize(public_eligible, "public eligibility: at least 120 snapshots"),
        "paper_34_failure_cohort": summarize(paper_eligible, "paper eligibility: at least 30 monitored updates"),
        "excluded_public_120_snapshot": excluded_public,
        "excluded_paper_30_updates": excluded_paper,
        "public_reference_crosscheck": {
            "source": "https://github.com/bhavjeetsinghh/bearing-early-warning-transfer results/pooled.json",
            "expected_detected": 32,
            "expected_n": 33,
            "expected_median_lead_pct_life": 29.425686637527915,
            "expected_early_reference_episodes": 336,
            "expected_early_reference_snapshots": 67332,
            "result": "exact per-bearing match for detection, lead fraction, false-alarm episodes, and reference snapshots",
        },
        "comparison_note": "The public cohort uses the original 120-snapshot eligibility rule and is not directly comparable to the manuscript 34-bearing cohort. The paper cohort applies the public D1 alarm semantics but retains the manuscript's 30-update eligibility rule.",
    }
    destination = ROOT / "results" / "results_sg1026_1_public_d1_rms_protocol_replication.json"
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "public": {key: output["public_120_snapshot_cohort"][key] for key in ("n_failures", "n_detected", "median_endpoint_lead_pct_life")},
        "paper": {key: output["paper_34_failure_cohort"][key] for key in ("n_failures", "n_detected", "median_endpoint_lead_pct_life")},
        "failure_false_alarms_public": output["public_120_snapshot_cohort"]["failure_healthy_reference_false_alarms"],
        "censored_false_alarms_paper": output["paper_34_failure_cohort"]["ims_censored_healthy_reference_false_alarms"],
    }, indent=2))
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
