#!/usr/bin/env python3
"""Summarize frozen public external validation result files."""
from __future__ import annotations

import argparse
import hashlib
import json
from statistics import median
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(paths: list[Path], normal_stress_path: Path | None = None) -> dict:
    results = [json.loads(path.read_text()) for path in paths]
    eligible = [
        bearing
        for result in results
        for bearing in result["per_bearing"]
        if bearing["protocol_eligible"]
    ]
    hierarchy = [
        bearing for bearing in eligible if bearing["hierarchy_evaluable"]
    ]
    historical_leads = [
        bearing["historical_lead_pct_life"]
        for bearing in eligible
        if bearing["historical_lead_pct_life"] is not None
    ]
    hierarchy_leads = [
        bearing["hierarchical_lead_pct_life"]
        for bearing in hierarchy
        if bearing["hierarchical_lead_pct_life"] is not None
    ]
    normal_stress = None
    if normal_stress_path is not None:
        normal_stress_result = json.loads(normal_stress_path.read_text())
        normal_stress = {
            "path": str(normal_stress_path),
            "sha256": file_sha256(normal_stress_path),
            "axis_level": normal_stress_result["axis_level"],
            "machine_level": normal_stress_result["machine_level"],
            "interpretation": normal_stress_result["interpretation"],
        }
    return {
        "protocol_status": "frozen public secondary external replication",
        "post_hoc_tuning_allowed": False,
        "inputs": [
            {"path": str(path), "sha256": file_sha256(path)}
            for path in paths
        ],
        "datasets": [
            {
                "dataset": result["per_bearing"][0]["dataset"],
                "n_failed": result["n_failed"],
                "n_protocol_eligible": result["n_protocol_eligible"],
                "historical_detected": result["historical_detected"],
                "hierarchical_n_evaluable": result["hierarchical_n_evaluable"],
                "hierarchical_detected": result["hierarchical_detected"],
                "ineligible_bearings": [
                    {
                        "bearing_id": bearing["bearing_id"],
                        "n_monitor": bearing["n_monitor"],
                        "reason": "fewer than 30 monitored snapshots",
                    }
                    for bearing in result["per_bearing"]
                    if not bearing["protocol_eligible"]
                ],
            }
            for result in results
        ],
        "n_result_files": len(results),
        "n_public_failed": sum(result["n_failed"] for result in results),
        "n_protocol_eligible": len(eligible),
        "historical_detected": sum(
            bearing["historical_detected"] for bearing in eligible
        ),
        "hierarchical_n_evaluable": len(hierarchy),
        "hierarchical_detected": sum(
            bearing["hierarchical_detected"] for bearing in hierarchy
        ),
        "hierarchical_no_later_than_historical": sum(
            bearing["hierarchical_alarm_index"]
            <= bearing["historical_alarm_index"]
            for bearing in hierarchy
        ),
        "hierarchical_earlier_than_historical": sum(
            bearing["hierarchical_alarm_index"]
            < bearing["historical_alarm_index"]
            for bearing in hierarchy
        ),
        "median_historical_lead_pct_life": (
            median(historical_leads)
            if historical_leads
            else None
        ),
        "median_hierarchical_lead_pct_life": (
            median(hierarchy_leads)
            if hierarchy_leads
            else None
        ),
        "false_alarm_endpoint": {
            "evaluable": False,
            "reason": "no public independent healthy censored multi-axis fleet",
            "normal_state_stress_test": normal_stress,
        },
        "main_method_upgrade": {
            "supported": False,
            "reason": "public evidence is a small secondary replication and does not satisfy multi-bearing, multi-rig, healthy-fleet false-alarm, or finite-conformal requirements",
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_json", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--normal-stress",
        type=Path,
        help="optional frozen CWRU steady-state normal stress-test result",
    )
    args = parser.parse_args()
    summary = summarize(args.result_json, args.normal_stress)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    keys = (
        "n_public_failed",
        "n_protocol_eligible",
        "historical_detected",
        "hierarchical_n_evaluable",
        "hierarchical_detected",
        "hierarchical_no_later_than_historical",
        "hierarchical_earlier_than_historical",
    )
    print(json.dumps({key: summary[key] for key in keys}, indent=2))
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
