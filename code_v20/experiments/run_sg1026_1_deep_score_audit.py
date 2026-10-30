#!/usr/bin/env python3
"""Export and audit raw scores from the five deep online baselines.

Each bearing-axis/model/seed combination is evaluated in a short-lived spawned
worker. This keeps PyTorch CPU allocations from accumulating across streams in
one long-lived process, which is necessary on the memory-constrained host.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "experiments"))

import run_to_failure_modern_baselines as modern  # noqa: E402
import run_to_failure_ood as ood  # noqa: E402
import run_to_failure_transformer_baselines as transformer  # noqa: E402

SEEDS = (0, 1, 2)
EARLIEST_UPDATE = 30
MODELS = ("mlp_ae", "deep_svdd", "lstm_ae", "transformer_ae", "tranad")
WORKERS = min(3, os.cpu_count() or 1)

torch.set_num_threads(1)
transformer.DEVICE = "cpu"


def score_array(model: str, features: np.ndarray, seed: int) -> np.ndarray:
    functions = {
        "mlp_ae": modern.mlp_ae_score,
        "deep_svdd": modern.deep_svdd_score,
        "lstm_ae": modern.lstm_ae_score,
        "transformer_ae": transformer.transformer_ae_score,
        "tranad": transformer.tranad_score,
    }
    return np.asarray(functions[model](features, seed), dtype=np.float32)


def score_stream(stream: dict) -> dict:
    baseline_mask, ood_mask = ood.ids_ood_masks(stream["x"].shape[0])
    if (
        stream["x"].shape[0] < 70
        or not np.all(np.isfinite(stream["x"]))
        or baseline_mask.sum() < 20
        or ood_mask.sum() < 5
    ):
        return {"eligible": False}

    raw_bearing_id = stream_raw_id(stream)
    bearing_id = f"{stream['dataset']}_{raw_bearing_id}"
    stem = f"{stream['dataset']}__{raw_bearing_id}__{stream['axis']}"
    scores = {
        f"{stem}__{model}__seed{seed}": score_array(model, stream["x"], seed)[60:]
        for model in MODELS
        for seed in SEEDS
    }
    return {
        "eligible": True,
        "bearing_id": bearing_id,
        "dataset": stream["dataset"],
        "failed": bool(stream["failed"]),
        "axis": str(stream["axis"]),
        "length": int(stream["x"].shape[0]),
        "scores": scores,
    }


def score_hash(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values, dtype=np.float32)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def stream_raw_id(stream: dict) -> str:
    if stream["dataset"] == "ims":
        return f"{stream['test']}_b{stream['bearing']}"
    return str(stream.get("bearing_id", stream.get("bearing")))


def first_alarm(score: np.ndarray, threshold: float) -> int:
    eligible = np.flatnonzero(score[EARLIEST_UPDATE - 1 :] > threshold)
    return int(eligible[0] + EARLIEST_UPDATE - 1) if eligible.size else int(score.size)


def load_eligible_streams(feature_dir: Path) -> List[dict]:
    streams = (
        ood.load_ims_streams(feature_dir)
        + ood.load_grouped_streams(feature_dir / "xjtu_features.csv", "xjtu", True)
        + ood.load_grouped_streams(feature_dir / "femto_features.csv", "femto", True)
    )
    eligible = []
    for stream in streams:
        baseline_mask, ood_mask = ood.ids_ood_masks(stream["x"].shape[0])
        if (
            stream["x"].shape[0] >= 70
            and np.all(np.isfinite(stream["x"]))
            and baseline_mask.sum() >= 20
            and ood_mask.sum() >= 5
        ):
            eligible.append(stream)
    return eligible, len(streams)


def stream_stem(stream: dict) -> str:
    return f"{stream['dataset']}__{stream_raw_id(stream)}__{stream['axis']}"


def stream_metadata(stream: dict) -> dict:
    raw_bearing_id = stream_raw_id(stream)
    return {
        "eligible": True,
        "bearing_id": f"{stream['dataset']}_{raw_bearing_id}",
        "dataset": stream["dataset"],
        "failed": bool(stream["failed"]),
        "axis": str(stream["axis"]),
        "length": int(stream["x"].shape[0]),
    }


def migrate_legacy_ims_keys(scores: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    migrated = {}
    for key, values in scores.items():
        fields = key.split("__")
        if (
            fields[0] == "ims"
            and len(fields) == 5
            and not fields[1].startswith("test")
        ):
            full_length = values.size + 60
            test = {2156: "test1", 984: "test2", 6324: "test3"}.get(full_length)
            if test is None:
                raise ValueError(f"cannot migrate legacy IMS key {key} with {values.size} scores")
            fields[1] = f"{test}_b{fields[1]}"
            key = "__".join(fields)
        migrated[key] = values
    return migrated


def analyze(raw_scores: Dict[str, np.ndarray], stream_results: List[dict]) -> dict:
    hashes = []
    bearings: Dict[str, dict] = {}
    seed0_suffix = "__seed0"
    for result in stream_results:
        bearing = bearings.setdefault(
            result["bearing_id"],
            {
                "bearing_id": result["bearing_id"],
                "dataset": result["dataset"],
                "failed": result["failed"],
                "axes": {},
                "length": result["length"],
            },
        )
        bearing["axes"].setdefault(result["axis"], {})
        prefix = f"{result['dataset']}__{result['bearing_id'].split('_', 1)[1]}__{result['axis']}"
        for key, values in raw_scores.items():
            if key.startswith(prefix + "__") and key.endswith(seed0_suffix):
                model = key[len(prefix) + 2 : -len(seed0_suffix)]
                bearing["axes"][result["axis"]].setdefault(model, {})

    for key, values in raw_scores.items():
        fields = key.split("__")
        model = fields[3]
        seed = fields[4].replace("seed", "")
        bearing_id = f"{fields[0]}_{fields[1]}"
        axis = fields[2]
        if bearing_id in bearings:
            bearings[bearing_id]["axes"][axis][model][seed] = values

    for key, values in raw_scores.items():
        fields = key.split("__")
        hashes.append(
            {
                "key": key,
                "model": fields[3],
                "seed": int(fields[4].replace("seed", "")),
                "dataset": fields[0],
                "bearing_id": f"{fields[0]}_{fields[1]}",
                "axis": fields[2],
                "n_score": int(values.size),
                "score_sha256": score_hash(values),
            }
        )

    model_results = {}
    for model in MODELS:
        seed_runs = []
        for seed in SEEDS:
            healthy = [row for row in bearings.values() if not row["failed"]]
            failed = [row for row in bearings.values() if row["failed"]]
            max_censored = max(
                float(np.max(axis[model][str(seed)][EARLIEST_UPDATE - 1 :]))
                for row in healthy
                for axis in row["axes"].values()
            )
            threshold = 1.01 * max_censored
            outcomes = []
            for row in failed:
                first = min(
                    first_alarm(axis[model][str(seed)], threshold)
                    for axis in row["axes"].values()
                )
                detected = first < row["length"] - 60
                outcomes.append(
                    {
                        "dataset": row["dataset"],
                        "bearing_id": row["bearing_id"],
                        "detected": detected,
                        "alarm_snapshot": 60 + first if detected else None,
                    }
                )
            per_dataset = {}
            for dataset in ("ims", "xjtu", "femto"):
                rows = [row for row in outcomes if row["dataset"] == dataset]
                per_dataset[dataset] = {
                    "n": len(rows),
                    "n_detected": sum(row["detected"] for row in rows),
                }
            seed_runs.append(
                {
                    "seed": seed,
                    "zero_false_alarm_threshold": threshold,
                    "max_censored_score": max_censored,
                    "total_detected": sum(row["detected"] for row in outcomes),
                    "per_dataset": per_dataset,
                    "per_bearing": outcomes,
                }
            )
        model_results[model] = {
            "total_detected_by_seed": [row["total_detected"] for row in seed_runs],
            "median_total_detected": float(np.median([row["total_detected"] for row in seed_runs])),
            "threshold_by_seed": [row["zero_false_alarm_threshold"] for row in seed_runs],
            "seed_runs": seed_runs,
        }

    pairwise = []
    for left_index, left in enumerate(MODELS):
        for right in MODELS[left_index + 1 :]:
            correlations = []
            exact_equal = True
            for row in bearings.values():
                for axis in row["axes"].values():
                    left_values = axis[left]["0"]
                    right_values = axis[right]["0"]
                    exact_equal &= bool(np.array_equal(left_values, right_values))
                    if np.std(left_values) > 0 and np.std(right_values) > 0:
                        correlations.append(float(np.corrcoef(left_values, right_values)[0, 1]))
            pairwise.append(
                {
                    "models": [left, right],
                    "median_pearson_correlation_seed0": float(np.median(correlations)),
                    "any_exact_array_equality_seed0": bool(exact_equal),
                }
            )

    return {
        "n_stream_arrays": len(hashes),
        "all_score_hashes_distinct": len({row["score_sha256"] for row in hashes}) == len(hashes),
        "pairwise_score_audit_seed0": pairwise,
        "model_decision_audit_seed0": model_results,
        "score_sha256": hashes,
    }


def main() -> None:
    feature_dir = ROOT / "data" / "features"
    eligible_streams, n_input_streams = load_eligible_streams(feature_dir)
    npz_path = ROOT / "results" / "results_sg1026_1_deep_score_audit.npz"
    append_missing = npz_path.exists() and os.environ.get("AUDIT_APPEND_MISSING") == "1"
    raw_scores = migrate_legacy_ims_keys(dict(np.load(npz_path))) if append_missing else {}
    known_stems = {"__".join(key.split("__")[:3]) for key in raw_scores}
    streams_to_score = [
        stream for stream in eligible_streams if stream_stem(stream) not in known_stems
    ]

    import multiprocessing

    context = multiprocessing.get_context("spawn")
    with context.Pool(processes=WORKERS, maxtasksperchild=1) as pool:
        for index, result in enumerate(pool.imap_unordered(score_stream, streams_to_score), start=1):
            if not result["eligible"]:
                continue
            raw_scores.update(result["scores"])
            print(f"scored {index}/{len(streams_to_score)} newly scored streams", flush=True)

    stream_by_stem = {stream_stem(stream): stream for stream in eligible_streams}
    all_stems = known_stems | {stream_stem(stream) for stream in streams_to_score}
    stream_results = [stream_metadata(stream_by_stem[stem]) for stem in sorted(all_stems)]
    np.savez_compressed(npz_path, **raw_scores)
    audit = analyze(raw_scores, stream_results)
    output = {
        "schema_version": "1.1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "baseline_slice": [10, 60],
            "monitor_start_snapshot": 61,
            "earliest_update": EARLIEST_UPDATE,
            "seeds": list(SEEDS),
            "device": "cpu",
            "workers": WORKERS,
            "worker_lifetime": "one eligible stream per spawned process",
            "score_slice": "causal monitoring snapshots 61 onward",
            "minimum_stream_snapshots": 70,
            "raw_score_export": str(npz_path.relative_to(ROOT)),
        },
        "n_input_streams": n_input_streams,
        "n_eligible_streams": len(eligible_streams),
        "n_newly_scored_streams": len(streams_to_score),
        **audit,
        "interpretation": "Different arrays and thresholds with identical alarm counts indicate score-level implementations are distinct; shared decision outcomes are not evidence of shared scores.",
    }
    destination = ROOT / "results" / "results_sg1026_1_deep_score_audit.json"
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
