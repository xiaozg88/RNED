#!/usr/bin/env python3
"""Independent re-audit of adverse sg1026 results.

This script does not replace the frozen v1 audit.  It checks whether the
unfavorable findings are sensitive to axis aggregation, horizon choice,
threshold margin, bounded score calibration, or baseline tie-breaking.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "experiments"))

import run_sg1026_review_audit as review_audit  # noqa: E402
import run_to_failure_classical_baselines as classical  # noqa: E402


HORIZONS = (120, 180, 300, 600, 924)
METRIC_NAMES = ("rate", "cumulative", "occupancy", "positive_mean")


def axis_metrics(stream: dict, horizon: int) -> dict:
    excess = review_audit.score_arrays(stream["rms"])["excess"][:horizon]
    active = excess > 0
    return {
        "rate": float(excess.mean()),
        "cumulative": float(excess.sum()),
        "occupancy": float(active.mean()),
        "positive_mean": float(excess[active].mean()) if active.any() else 0.0,
    }


def rolling_max_rate(stream: dict, horizon: int, window: int) -> float:
    excess = review_audit.score_arrays(stream["rms"])["excess"][:horizon]
    if excess.size < window:
        return 0.0
    prefix = np.concatenate(([0.0], np.cumsum(excess)))
    return float(np.max((prefix[window:] - prefix[:-window]) / window))


def metric_summary(labels: List[bool], scores: List[float]) -> dict:
    return {
        "auc_failed_higher": review_audit.directional_auc(labels, scores),
        "exact_two_sided_p": review_audit.exact_two_sided_auc_p(labels, scores),
        "failed_median": float(np.median([score for score, failed in zip(scores, labels) if failed])),
        "censored_median": float(np.median([score for score, failed in zip(scores, labels) if not failed])),
    }


def horizon_audit(streams: List[dict]) -> dict:
    ims_streams = [stream for stream in streams if stream["dataset"] == "ims"]
    rows = []
    for horizon in HORIZONS:
        by_bearing: Dict[str, dict] = {}
        for stream in ims_streams:
            metrics = axis_metrics(stream, horizon)
            entry = by_bearing.setdefault(
                stream["bearing_id"], {"failed": bool(stream["failed"]), "axes": []}
            )
            entry["axes"].append(metrics)

        horizon_row = {"horizon_monitor_snapshots": horizon}
        for aggregation in ("max", "mean"):
            labels = [entry["failed"] for entry in by_bearing.values()]
            metrics = {
                name: [
                    float(getattr(np, aggregation)([axis[name] for axis in entry["axes"]]))
                    for entry in by_bearing.values()
                ]
                for name in METRIC_NAMES
            }
            horizon_row[f"bearing_{aggregation}"] = {
                name: metric_summary(labels, scores) for name, scores in metrics.items()
            }

        rolling = {}
        for window in (30, 60, 120, 300):
            if window > horizon:
                continue
            labels = [entry["failed"] for entry in by_bearing.values()]
            scores = [
                max(
                    rolling_max_rate(stream, horizon, window)
                    for stream in ims_streams
                    if stream["bearing_id"] == bearing_id
                )
                for bearing_id in by_bearing
            ]
            rolling[f"window_{window}"] = metric_summary(labels, scores)
        horizon_row["max_rolling_excess_rate_bearing_max"] = rolling
        rows.append(horizon_row)

    return {
        "status": "post_review_exploratory_sensitivity",
        "purpose": "Check whether the weak fixed-horizon result depends on the single 924-snapshot horizon or max-axis aggregation.",
        "aggregation_note": "Bearing score is first computed per axis, then aggregated by max or mean; exact permutation tests treat bearings, not axes, as units.",
        "rows": rows,
    }


def loeo_margin_sensitivity(streams: List[dict]) -> dict:
    bearings = review_audit.build_bearings(streams, review_audit.rate_score)
    ims = [bearing for bearing in bearings if bearing["dataset"] == "ims"]
    groups = sorted({bearing["bearing_id"].split("_b")[0] for bearing in ims})
    rows = []
    for margin in (1.0, 1.001, 1.005, 1.01, 1.02, 1.05, 1.10):
        details = []
        for group in groups:
            calibration = [
                bearing
                for bearing in ims
                if not bearing["failed"] and not bearing["bearing_id"].startswith(group)
            ]
            held = [bearing for bearing in ims if bearing["bearing_id"].startswith(group)]
            base = max(
                float(np.max(score))
                for bearing in calibration
                for score in bearing["axes"].values()
            )
            threshold = margin * base
            false_alarms = sum(
                any(review_audit.alarm_index(score, threshold) < score.size for score in bearing["axes"].values())
                for bearing in held
                if not bearing["failed"]
            )
            detections = sum(
                any(review_audit.alarm_index(score, threshold) < score.size for score in bearing["axes"].values())
                for bearing in held
                if bearing["failed"]
            )
            details.append(
                {
                    "held_group": group,
                    "threshold": threshold,
                    "held_out_false_alarms": false_alarms,
                    "held_out_detected_failed": detections,
                }
            )
        rows.append(
            {
                "threshold_margin": margin,
                "total_held_out_false_alarms": sum(row["held_out_false_alarms"] for row in details),
                "total_held_out_failed_detected": sum(row["held_out_detected_failed"] for row in details),
                "details": details,
            }
        )
    return {
        "status": "post_review_margin_sensitivity_not_a_new_protocol",
        "rows": rows,
        "interpretation": "The prespecified 1.01 margin gives 1/8. The false alarm disappears at 1.05, but selecting that margin after seeing the held-out result would be post hoc.",
    }


def bounded_rule_reaudit(streams: List[dict]) -> dict:
    result = json.loads((ROOT / "results" / "results_sg1026_review_audit.json").read_text())
    candidates = result["tuned_baselines"]["all_candidates"]
    rows = []
    for family in ("binary_rolling_occupancy", "run_rule"):
        for row in candidates[family]:
            if family == "binary_rolling_occupancy":
                score_fn = lambda rms, window=row["parameter"]: review_audit.rolling_mean_score(
                    (review_audit.score_arrays(rms)["excess"] > 0).astype(float), window
                )
            else:
                score_fn = lambda rms, window=row["parameter"]: review_audit.run_rule_score(rms, window)
            bearings = review_audit.build_bearings(streams, score_fn)
            censored = [bearing for bearing in bearings if not bearing["failed"]]
            calibration_max = max(
                float(np.max(score)) for bearing in censored for score in bearing["axes"].values()
            )
            exact_threshold_detections = sum(
                any(
                    review_audit.alarm_index(score, calibration_max) < score.size
                    for score in bearing["axes"].values()
                )
                for bearing in bearings
                if bearing["failed"]
            )
            rows.append(
                {
                    "family": family,
                    "parameter": row["parameter"],
                    "reported_threshold": row["zero_fa_threshold"],
                    "calibration_max": calibration_max,
                    "detections_with_exact_max_and_strict_greater": exact_threshold_detections,
                }
            )
    return {
        "status": "bounded_rule_calibration_audit",
        "rows": rows,
        "interpretation": "For rolling occupancy, censored maxima reach the discrete ceiling 1; under strict > and a zero-false-alarm constraint, no failed bearing can exceed that ceiling. The 0/34 result is therefore reproducible, but it reflects incompatibility of this bounded rule with the calibration rule rather than an arithmetic failure. Run-rule requires a longer run than every censored axis and also detects 0/34.",
    }


def baseline_selection_reaudit() -> dict:
    result = json.loads((ROOT / "results" / "results_sg1026_review_audit.json").read_text())
    rows = {}
    for family, candidates in result["tuned_baselines"]["all_candidates"].items():
        best_total = max(candidates, key=lambda row: (row["n_detected"], row["median_lead_pct_life"] or -1.0))
        rows[family] = {
            "protocol_selected": result["tuned_baselines"]["selected"][family],
            "best_total_post_hoc": best_total,
        }
    return {
        "status": "tie_break_and_post_hoc_diagnostic",
        "important_correction": "EWMA-of-excess with decay=0.5 has the same IMS detection count and median lead as decay=0.3 but detects 28/34 overall versus 27/34. The v1 tie-break implicitly selected the smaller decay; the manuscript must report the grid range or use a predeclared family-specific tie-break.",
        "families": rows,
    }


def communication_reaudit() -> dict:
    feature_dir = ROOT / "data" / "features"
    frame_bytes = 4
    event_bytes = 64
    ims_experiment_rows = []
    ims_bearing_snapshot_bytes = []
    for filename in ("test1.csv", "test2.csv", "test3.csv"):
        frame = pd.read_csv(feature_dir / filename)
        values = []
        for _, group in frame.groupby(["filename", "bearing"], sort=False):
            value = float(group["n_rows"].sum() * frame_bytes)
            values.append(value)
            ims_bearing_snapshot_bytes.append(value)
        machine_values = []
        for _, group in frame.groupby("filename", sort=False):
            machine_values.append(float(group["n_rows"].sum() * frame_bytes))
        ims_experiment_rows.append(
            {
                "experiment": filename.replace(".csv", ""),
                "median_bytes_per_bearing_snapshot": float(np.median(values)),
                "distinct_bytes_per_bearing_snapshot": sorted(set(values)),
                "distinct_bytes_per_machine_snapshot": sorted(set(machine_values)),
            }
        )

    by_dataset = {}
    for filename, dataset in (("xjtu_features.csv", "xjtu"), ("femto_features.csv", "femto")):
        frame = pd.read_csv(feature_dir / filename)
        values = [
            float(group["n_rows"].sum() * frame_bytes)
            for _, group in frame.groupby(["group", "bearing", "snapshot"], sort=False)
        ]
        by_dataset[dataset] = {
            "median_bytes_per_bearing_snapshot": float(np.median(values)),
            "event_packet_ratio": float(np.median(values) / event_bytes),
        }
    return {
        "status": "independent_byte_accounting",
        "assumptions": {"float32_samples": True, "event_packet_bytes": event_bytes},
        "ims_by_experiment": ims_experiment_rows,
        "ims_pooled_median_bytes_per_bearing_snapshot": float(np.median(ims_bearing_snapshot_bytes)),
        "ims_pooled_event_packet_ratio": float(np.median(ims_bearing_snapshot_bytes) / event_bytes),
        "other_datasets": by_dataset,
        "interpretation": "The pooled IMS median 81,920 B is a median over heterogeneous bearing snapshots: test1 has two axes per bearing (163,840 B), whereas test2/test3 have one (81,920 B). Machine-level IMS snapshots are 655,360 B in test1 and 327,680 B in test2/test3. The manuscript should state the bearing-level pooling rule rather than imply a single IMS channel layout.",
    }


def main() -> None:
    streams = classical.load_all_streams(ROOT / "data" / "features")
    output = {
        "script_version": "sg1026_adverse_result_reaudit_v1",
        "fixed_horizon_sensitivity": horizon_audit(streams),
        "loeo_margin_sensitivity": loeo_margin_sensitivity(streams),
        "bounded_rule_reaudit": bounded_rule_reaudit(streams),
        "baseline_selection_reaudit": baseline_selection_reaudit(),
        "communication_reaudit": communication_reaudit(),
    }
    destination = ROOT / "results" / "results_sg1026_adverse_result_reaudit.json"
    destination.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    print(f"saved {destination}")
    print(
        json.dumps(
            {
                "loeo_margins": [
                    {
                        "margin": row["threshold_margin"],
                        "false_alarms": row["total_held_out_false_alarms"],
                        "failed_detected": row["total_held_out_failed_detected"],
                    }
                    for row in output["loeo_margin_sensitivity"]["rows"]
                ],
                "ewma_best_total": output["baseline_selection_reaudit"]["families"]["ewma_excess"]["best_total_post_hoc"]["n_detected"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
