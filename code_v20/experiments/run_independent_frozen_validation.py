#!/usr/bin/env python3
"""Run the pre-registered independent RMS validation without retuning."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "INDEPENDENT_FROZEN_VALIDATION_PREREGISTRATION.md"
RAW_THRESHOLD = 0.042407782315012556
IMPACT_THRESHOLD = 0.04256682923039607
BASELINE_SLICE = slice(10, 60)
MONITOR_START = 60
EARLIEST_UPDATE = 30
FALSE_ALARM_LAST_UPDATE = 90
MIN_SNAPSHOTS = 150
MIN_FAILED = 20
MIN_HEALTHY = 20
MIN_RIGS = 2
MIN_AXES = 2
BOOTSTRAP_RESAMPLES = 20000
BOOTSTRAP_SEED = 20261025
FORBIDDEN_DATASETS = {
    "ims",
    "xjtu",
    "xjtu-sy",
    "femto",
    "pronostia",
    "ldm_upb",
    "ldm-upb",
    "cwru",
    "case western reserve",
    "sca",
    "ferrara",
    "kaist",
    "unsw",
    "unicamp54",
    "unicamp-54",
    "eosc54",
    "eosc-54",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def exact_sign_p(wins: int, losses: int, one_sided: bool = True) -> Optional[float]:
    total = wins + losses
    if total == 0:
        return None
    tail = sum(math.comb(total, count) for count in range(wins, total + 1)) / (
        2.0 ** total
    )
    return min(1.0, tail) if one_sided else min(1.0, 2.0 * min(tail, 1.0 - tail))


def exact_mcnemar_p(first_only: int, second_only: int) -> Optional[float]:
    total = first_only + second_only
    if total == 0:
        return None
    smaller = min(first_only, second_only)
    tail = sum(math.comb(total, count) for count in range(smaller + 1)) / (
        2.0 ** total
    )
    return min(1.0, 2.0 * tail)


def wilson_interval(successes: int, total: int) -> Optional[dict]:
    if total == 0:
        return None
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return {"estimate": proportion, "lower": max(0.0, center - half), "upper": min(1.0, center + half)}


def median(values: list[float]) -> Optional[float]:
    return float(np.median(values)) if values else None


def bootstrap_median_difference(
    values: list[float], rigs: list[str], resamples: int = BOOTSTRAP_RESAMPLES
) -> Optional[dict]:
    if not values:
        return None
    by_rig = defaultdict(list)
    for value, rig in zip(values, rigs):
        by_rig[rig].append(value)
    unique_rigs = sorted(by_rig)
    rng = random.Random(BOOTSTRAP_SEED)
    samples = []
    for _ in range(resamples):
        sampled = []
        for _ in unique_rigs:
            sampled.extend(by_rig[rng.choice(unique_rigs)])
        samples.append(float(np.median(sampled)))
    return {
        "lower": float(np.percentile(samples, 2.5)),
        "median": float(np.percentile(samples, 50.0)),
        "upper": float(np.percentile(samples, 97.5)),
        "resamples": resamples,
        "seed": BOOTSTRAP_SEED,
        "clustering": "rig",
    }


def load_manifest(manifest_path: Path) -> tuple[dict, list[dict]]:
    manifest = read_json(manifest_path)
    if manifest.get("protocol") != "independent_frozen_rms_v1":
        raise ValueError("unsupported manifest protocol")
    entries = manifest.get("bearings")
    if not isinstance(entries, list) or not entries:
        raise ValueError("manifest contains no bearings")
    seen = set()
    for entry in entries:
        required = {
            "dataset", "group", "rig", "bearing_id", "endpoint",
            "endpoint_snapshot", "path", "sha256",
        }
        missing = required.difference(entry)
        if missing:
            raise ValueError(f"manifest entry missing fields: {sorted(missing)}")
        key = (entry["dataset"], entry["group"], entry["rig"], entry["bearing_id"])
        if key in seen:
            raise ValueError(f"duplicate bearing key: {key}")
        seen.add(key)
        if entry["endpoint"] not in {"failed", "healthy_censored"}:
            raise ValueError(f"invalid endpoint for {key}")
        if int(entry["endpoint_snapshot"]) < MIN_SNAPSHOTS:
            raise ValueError(f"trajectory shorter than {MIN_SNAPSHOTS}: {key}")
        dataset_name = str(entry["dataset"]).lower()
        if dataset_name in FORBIDDEN_DATASETS:
            raise ValueError(f"non-independent dataset in manifest: {entry['dataset']}")
    return manifest, entries


def load_bearing(entry: dict, manifest_path: Path) -> dict:
    path = manifest_path.parent / str(entry["path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    if sha256(path) != str(entry["sha256"]):
        raise ValueError(f"SHA-256 mismatch: {path}")
    grouped: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"dataset", "group", "rig", "bearing", "axis", "snapshot", "rms", "envelope_rms"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"feature CSV lacks required columns: {path}")
        for row in reader:
            if (
                row["dataset"] != entry["dataset"]
                or row["group"] != entry["group"]
                or row["rig"] != entry["rig"]
                or row["bearing"] != entry["bearing_id"]
            ):
                raise ValueError(f"feature metadata disagree with manifest: {path}")
            grouped[row["axis"]].append(
                (int(row["snapshot"]), float(row["rms"]), float(row["envelope_rms"]))
            )
    if len(grouped) < MIN_AXES:
        raise ValueError(f"fewer than {MIN_AXES} axes: {path}")
    axes = {}
    for axis, rows in grouped.items():
        rows.sort(key=lambda row: row[0])
        snapshots = [row[0] for row in rows]
        if snapshots != list(range(1, len(rows) + 1)):
            raise ValueError(f"snapshots are not consecutive and one-based: {path}/{axis}")
        if len(rows) != int(entry["endpoint_snapshot"]):
            raise ValueError(f"endpoint and row count disagree: {path}/{axis}")
        raw = np.asarray([row[1] for row in rows], dtype=np.float64)
        envelope = np.asarray([row[2] for row in rows], dtype=np.float64)
        if raw.size < MIN_SNAPSHOTS or envelope.size != raw.size:
            raise ValueError(f"invalid stream length: {path}/{axis}")
        if not np.all(np.isfinite(raw)) or not np.all(raw > 0.0):
            raise ValueError(f"invalid raw RMS values: {path}/{axis}")
        if not np.all(np.isfinite(envelope)) or not np.all(envelope > 0.0):
            raise ValueError(f"invalid envelope RMS values: {path}/{axis}")
        axes[axis] = {"raw": raw, "envelope": envelope}
    lengths = {axis: values["raw"].size for axis, values in axes.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"axes have unequal lengths: {path}")
    return {
        **entry,
        "path": str(path),
        "axes": axes,
    }


def raw_rate(values: np.ndarray) -> np.ndarray:
    baseline = float(np.median(values[BASELINE_SLICE]))
    excess = np.maximum(values[MONITOR_START:] / baseline - 1.2, 0.0)
    return np.cumsum(excess) / np.arange(1, excess.size + 1)


def impact_rate(raw: np.ndarray, envelope: np.ndarray) -> np.ndarray:
    raw_ratio = raw / float(np.median(raw[BASELINE_SLICE]))
    envelope_ratio = envelope / float(np.median(envelope[BASELINE_SLICE]))
    sampled_envelope = np.empty_like(envelope_ratio)
    held = envelope_ratio[0]
    for index in range(envelope_ratio.size):
        if index < MONITOR_START or (index - MONITOR_START) % 5 == 0:
            held = envelope_ratio[index]
        sampled_envelope[index] = held
    feature = np.sqrt(0.9 * raw_ratio ** 2 + 0.1 * sampled_envelope ** 2)
    return raw_rate(feature)


def first_alarm(rate: np.ndarray, threshold: float) -> Optional[int]:
    eligible = rate[EARLIEST_UPDATE - 1 :]
    hits = np.flatnonzero(eligible > threshold)
    return int(hits[0] + EARLIEST_UPDATE - 1) if hits.size else None


def horizon_alarm(rate: np.ndarray, threshold: float, last_update: int) -> bool:
    if rate.size < EARLIEST_UPDATE:
        return False
    stop = min(last_update, rate.size)
    return bool(np.any(rate[EARLIEST_UPDATE - 1 : stop] > threshold))


def evaluate_axis(raw: np.ndarray, envelope: np.ndarray) -> dict:
    raw_statistics = raw_rate(raw)
    impact_statistics = impact_rate(raw, envelope)
    raw_alarm = first_alarm(raw_statistics, RAW_THRESHOLD)
    impact_alarm = first_alarm(impact_statistics, IMPACT_THRESHOLD)
    return {
        "raw": {
            "rate": raw_statistics,
            "alarm_update": raw_alarm,
            "horizon_false_alarm": horizon_alarm(
                raw_statistics, RAW_THRESHOLD, FALSE_ALARM_LAST_UPDATE
            ),
        },
        "impact": {
            "rate": impact_statistics,
            "alarm_update": impact_alarm,
            "horizon_false_alarm": horizon_alarm(
                impact_statistics, IMPACT_THRESHOLD, FALSE_ALARM_LAST_UPDATE
            ),
        },
    }


def evaluate_bearing(bearing: dict) -> dict:
    axis_results = {axis: evaluate_axis(**values) for axis, values in bearing["axes"].items()}
    result = {
        "dataset": bearing["dataset"],
        "group": bearing["group"],
        "rig": bearing["rig"],
        "bearing_id": bearing["bearing_id"],
        "endpoint": bearing["endpoint"],
        "endpoint_snapshot": int(bearing["endpoint_snapshot"]),
        "n_axes": len(axis_results),
        "axes": {},
    }
    for method in ("raw", "impact"):
        alarms = [
            axis_result[method]["alarm_update"]
            for axis_result in axis_results.values()
            if axis_result[method]["alarm_update"] is not None
        ]
        false_axes = sum(
            axis_result[method]["horizon_false_alarm"]
            for axis_result in axis_results.values()
        )
        alarm_update = min(alarms) if alarms else None
        detected = alarm_update is not None
        alarm_snapshot = 61 + alarm_update if alarm_update is not None else None
        result["axes"] = {
            axis: {
                "raw_alarm_update": values["raw"]["alarm_update"],
                "impact_alarm_update": values["impact"]["alarm_update"],
                "raw_horizon_false_alarm": values["raw"]["horizon_false_alarm"],
                "impact_horizon_false_alarm": values["impact"]["horizon_false_alarm"],
                "raw_max_rate": float(np.max(values["raw"]["rate"])),
                "impact_max_rate": float(np.max(values["impact"]["rate"])),
            }
            for axis, values in axis_results.items()
        }
        result[method] = {
            "alarm_update": alarm_update,
            "alarm_snapshot": alarm_snapshot,
            "detected": bool(detected),
            "horizon_false_alarm_bearing": bool(false_axes > 0),
            "horizon_false_alarm_axes": int(false_axes),
            "lead_pct_life": (
                100.0 * (result["endpoint_snapshot"] - alarm_snapshot) / result["endpoint_snapshot"]
                if detected
                else None
            ),
        }
    return result


def eligibility_audit(
    manifest: dict, entries: list[dict], strict: bool = True
) -> dict:
    failed = [entry for entry in entries if entry["endpoint"] == "failed"]
    healthy = [entry for entry in entries if entry["endpoint"] == "healthy_censored"]
    rigs = sorted({str(entry["rig"]) for entry in entries})
    failed_rigs = sorted({str(entry["rig"]) for entry in failed})
    healthy_rigs = sorted({str(entry["rig"]) for entry in healthy})
    minimums = {
        "failed": MIN_FAILED if strict else 1,
        "healthy_censored": MIN_HEALTHY if strict else 1,
        "rigs": MIN_RIGS if strict else 1,
    }
    checks = {
        "protocol_version": manifest.get("protocol") == "independent_frozen_rms_v1",
        "independence_statement_present": bool(manifest.get("independence_statement")),
        "prospective_endpoint_definition_present": bool(
            manifest.get("prospective_endpoint_definition")
        ),
        "minimum_failed": len(failed) >= minimums["failed"],
        "minimum_healthy_censored": len(healthy) >= minimums["healthy_censored"],
        "minimum_rigs": len(rigs) >= minimums["rigs"],
        "failed_in_multiple_rigs": len(failed_rigs) >= minimums["rigs"],
        "healthy_in_multiple_rigs": len(healthy_rigs) >= minimums["rigs"],
        "minimum_snapshots": all(
            int(entry["endpoint_snapshot"]) >= MIN_SNAPSHOTS for entry in entries
        ),
    }
    return {
        "n_failed": len(failed),
        "n_healthy_censored": len(healthy),
        "rigs": rigs,
        "failed_rigs": failed_rigs,
        "healthy_rigs": healthy_rigs,
        "minimums": minimums,
        "checks": checks,
        "eligible": all(checks.values()),
    }


def summarize_method(results: list[dict], method: str) -> dict:
    failures = [result for result in results if result["endpoint"] == "failed"]
    healthy = [result for result in results if result["endpoint"] == "healthy_censored"]
    detected = sum(result[method]["detected"] for result in failures)
    false_bearings = sum(result[method]["horizon_false_alarm_bearing"] for result in healthy)
    false_axes = sum(result[method]["horizon_false_alarm_axes"] for result in healthy)
    total_axes = sum(result["n_axes"] for result in healthy)
    leads = [result[method]["lead_pct_life"] for result in failures if result[method]["detected"]]
    by_rig = {}
    for rig in sorted({result["rig"] for result in results}):
        rig_results = [result for result in results if result["rig"] == rig]
        rig_failures = [result for result in rig_results if result["endpoint"] == "failed"]
        rig_healthy = [result for result in rig_results if result["endpoint"] == "healthy_censored"]
        rig_leads = [
            result[method]["lead_pct_life"]
            for result in rig_failures
            if result[method]["detected"]
        ]
        by_rig[rig] = {
            "n_failed": len(rig_failures),
            "n_detected": sum(result[method]["detected"] for result in rig_failures),
            "n_healthy_censored": len(rig_healthy),
            "n_false_alarm_bearings": sum(
                result[method]["horizon_false_alarm_bearing"] for result in rig_healthy
            ),
            "median_lead_pct_life": median(rig_leads),
        }
    return {
        "detection": {
            "successes": detected,
            "total": len(failures),
            "rate": detected / len(failures) if failures else None,
            "wilson_95": wilson_interval(detected, len(failures)),
        },
        "healthy_false_alarm": {
            "bearing_successes": false_bearings,
            "bearing_total": len(healthy),
            "bearing_rate": false_bearings / len(healthy) if healthy else None,
            "bearing_wilson_95": wilson_interval(false_bearings, len(healthy)),
            "axis_successes": false_axes,
            "axis_total": total_axes,
            "axis_rate": false_axes / total_axes if total_axes else None,
            "axis_wilson_95": wilson_interval(false_axes, total_axes),
            "false_alarm_rigs": sorted(
                {
                    result["rig"]
                    for result in healthy
                    if result[method]["horizon_false_alarm_bearing"]
                }
            ),
        },
        "median_lead_pct_life": median(leads),
        "by_rig": by_rig,
    }


def paired_analysis(results: list[dict]) -> dict:
    failures = [result for result in results if result["endpoint"] == "failed"]
    healthy = [result for result in results if result["endpoint"] == "healthy_censored"]
    raw_detect = [result["raw"]["detected"] for result in failures]
    impact_detect = [result["impact"]["detected"] for result in failures]
    detection_first_only = sum(raw and not impact for raw, impact in zip(raw_detect, impact_detect))
    detection_second_only = sum(impact and not raw for raw, impact in zip(raw_detect, impact_detect))
    raw_false = [result["raw"]["horizon_false_alarm_bearing"] for result in healthy]
    impact_false = [result["impact"]["horizon_false_alarm_bearing"] for result in healthy]
    false_first_only = sum(raw and not impact for raw, impact in zip(raw_false, impact_false))
    false_second_only = sum(impact and not raw for raw, impact in zip(raw_false, impact_false))
    paired = [
        {
            "rig": result["rig"],
            "bearing_id": result["bearing_id"],
            "raw": result["raw"]["lead_pct_life"],
            "impact": result["impact"]["lead_pct_life"],
            "difference": (
                result["impact"]["lead_pct_life"] - result["raw"]["lead_pct_life"]
            ),
        }
        for result in failures
        if result["raw"]["detected"] and result["impact"]["detected"]
    ]
    differences = [item["difference"] for item in paired]
    wins = sum(value > 1e-12 for value in differences)
    losses = sum(value < -1e-12 for value in differences)
    ties = len(differences) - wins - losses
    by_rig = {}
    for rig in sorted({item["rig"] for item in paired}):
        values = [item["difference"] for item in paired if item["rig"] == rig]
        by_rig[rig] = {
            "n_pairs": len(values),
            "wins": sum(value > 1e-12 for value in values),
            "losses": sum(value < -1e-12 for value in values),
            "median_difference": median(values),
        }
    median_difference = median(differences)
    bootstrap = bootstrap_median_difference(
        differences, [item["rig"] for item in paired]
    )
    sign_p = exact_sign_p(wins, losses)
    conditions = {
        "axis_false_alarms_do_not_increase": sum(
            result["impact"]["horizon_false_alarm_axes"]
            for result in healthy
        ) <= sum(result["raw"]["horizon_false_alarm_axes"] for result in healthy),
        "bearing_false_alarms_do_not_increase": sum(impact_false) <= sum(raw_false),
        "detection_does_not_decrease": sum(impact_detect) >= sum(raw_detect),
        "lead_wins_exceed_losses": wins > losses,
        "median_lead_improvement_at_least_3pp": (
            median_difference is not None and median_difference >= 3.0
        ),
        "sign_test_p_below_0_10": sign_p is not None and sign_p < 0.10,
        "nonnegative_direction_in_two_rigs": (
            sum(
                value.get("median_difference", 0.0) >= 0.0
                for value in by_rig.values()
            )
            >= 2
        ),
        "no_rig_median_loss_exceeds_10pp": all(
            value.get("median_difference", 0.0) >= -10.0
            for value in by_rig.values()
        ),
    }
    return {
        "detection": {
            "raw_only": detection_first_only,
            "impact_only": detection_second_only,
            "exact_mcnemar_p": exact_mcnemar_p(
                detection_first_only, detection_second_only
            ),
        },
        "healthy_false_alarm": {
            "raw_only": false_first_only,
            "impact_only": false_second_only,
            "exact_mcnemar_p": exact_mcnemar_p(false_first_only, false_second_only),
        },
        "lead": {
            "n_common_detections": len(paired),
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "median_difference_pct_life": median_difference,
            "cluster_bootstrap_95": bootstrap,
            "exact_one_sided_sign_p": sign_p,
            "by_rig": by_rig,
        },
        "promotion_conditions": conditions,
        "promotion_pass": all(conditions.values()),
    }


def raw_validation_decision(raw_summary: dict) -> dict:
    detection = raw_summary["detection"]
    false_alarm = raw_summary["healthy_false_alarm"]
    eligible_rigs = [
        value for value in raw_summary["by_rig"].values() if value["n_failed"] > 0
    ]
    conditions = {
        "healthy_false_alarm_rate_at_most_15pct": (
            false_alarm["bearing_rate"] is not None
            and false_alarm["bearing_rate"] <= 0.15
        ),
        "failure_detection_at_least_75pct": (
            detection["rate"] is not None and detection["rate"] >= 0.75
        ),
        "eligible_failures_in_two_rigs": len(eligible_rigs) >= 2,
        "false_alarms_absent_or_in_multiple_rigs": (
            not false_alarm["false_alarm_rigs"]
            or len(false_alarm["false_alarm_rigs"]) >= 2
        ),
        "threshold_not_adjusted": True,
    }
    return {"conditions": conditions, "pass": all(conditions.values())}


def lock_metadata(manifest_path: Path, manifest: dict) -> dict:
    return {
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "script_sha256": sha256(Path(__file__).resolve()),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "manifest_protocol": manifest.get("protocol"),
        "executed_utc": datetime.now(timezone.utc).isoformat(),
        "thresholds": {
            "raw_rms": RAW_THRESHOLD,
            "impact_aware_rms": IMPACT_THRESHOLD,
        },
    }


def write_new_json(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def validate_manifest_only(
    manifest_path: Path, audit_output: Optional[Path]
) -> dict:
    manifest, entries = load_manifest(manifest_path)
    audit = {
        "mode": "manifest_and_eligibility_only_no_detector_outcomes",
        "lock": lock_metadata(manifest_path, manifest),
        "eligibility": eligibility_audit(manifest, entries, strict=True),
        "input_hashes": [
            {
                "bearing_id": entry["bearing_id"],
                "rig": entry["rig"],
                "endpoint": entry["endpoint"],
                "path": entry["path"],
                "sha256": sha256(manifest_path.parent / str(entry["path"])),
            }
            for entry in entries
        ],
    }
    if audit_output is not None:
        write_new_json(audit_output, audit)
    return audit


def run_validation(manifest_path: Path, output: Path) -> dict:
    manifest, entries = load_manifest(manifest_path)
    eligibility = eligibility_audit(manifest, entries, strict=True)
    if not eligibility["eligible"]:
        raise ValueError(f"manifest does not satisfy the locked cohort: {eligibility}")
    bearings = [load_bearing(entry, manifest_path) for entry in entries]
    results = [evaluate_bearing(bearing) for bearing in bearings]
    raw_summary = summarize_method(results, "raw")
    impact_summary = summarize_method(results, "impact")
    paired = paired_analysis(results)
    return {
        "mode": "official_one_shot_independent_frozen_validation",
        "lock": lock_metadata(manifest_path, manifest),
        "eligibility": eligibility,
        "results_by_bearing": results,
        "raw_rms": raw_summary,
        "impact_aware_rms": impact_summary,
        "paired_comparison": paired,
        "raw_validation_decision": raw_validation_decision(raw_summary),
    }


def generate_self_test(root: Path) -> Path:
    manifest_path = root / "manifest.json"
    entries = []
    for rig_index, rig in enumerate(("A", "B")):
        for bearing_index in range(4):
            endpoint = "failed" if bearing_index < 2 else "healthy_censored"
            bearing_id = f"{rig}-{bearing_index + 1:03d}"
            path = root / f"{bearing_id}.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    ["dataset", "group", "rig", "bearing", "axis", "snapshot", "rms", "envelope_rms"]
                )
                for snapshot in range(1, 181):
                    for axis in ("H", "V"):
                        level = 1.0
                        envelope_level = 1.0
                        if endpoint == "failed" and snapshot > 100:
                            level = 1.4
                            envelope_level = 1.5
                        writer.writerow(
                            [
                                "self_test", "self_test", rig, bearing_id, axis,
                                snapshot, level, envelope_level,
                            ]
                        )
            entries.append(
                {
                    "dataset": "self_test", "group": "self_test", "rig": rig,
                    "bearing_id": bearing_id, "endpoint": endpoint,
                    "endpoint_snapshot": 180, "endpoint_evidence": "synthetic",
                    "path": path.name, "sha256": sha256(path),
                }
            )
    manifest = {
        "protocol": "independent_frozen_rms_v1",
        "independence_statement": "synthetic self-test",
        "prospective_endpoint_definition": "synthetic self-test",
        "bearings": entries,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="rms_frozen_self_test_") as temporary:
        root = Path(temporary)
        manifest_path = generate_self_test(root)
        manifest, entries = load_manifest(manifest_path)
        eligibility = eligibility_audit(manifest, entries, strict=False)
        if not eligibility["eligible"]:
            raise AssertionError(eligibility)
        results = [evaluate_bearing(load_bearing(entry, manifest_path)) for entry in entries]
        if len(results) != 8:
            raise AssertionError("unexpected self-test bearing count")
        paired = paired_analysis(results)
        if "promotion_conditions" not in paired:
            raise AssertionError("paired analysis incomplete")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("self-test passed")
        return
    if args.manifest is None:
        parser.error("--manifest is required unless --self-test is used")
    if args.validate_only:
        if args.audit_output is None:
            parser.error("--audit-output is required with --validate-only")
        print(json.dumps(validate_manifest_only(args.manifest, args.audit_output), indent=2))
        return
    if args.output is None:
        parser.error("--output is required for the official run")
    write_new_json(args.output, run_validation(args.manifest, args.output))


if __name__ == "__main__":
    main()
