#!/usr/bin/env python3
"""Extract per-axis RMS features from the LDM_UPB run-to-failure archive.

The Hugging Face mirror stores each single-volume experiment as ``meta.json`` and
``vibration.npy``. The NumPy array has shape ``(records, channels, samples)``
and stores int16 sensor codes. This script decodes them to acceleration in g
using the metadata formula ``10 * (offset[c] + step[c] * code)`` and emits the
feature schema expected by the locked external-validation script.

Usage:
    python3 extract_ldm_upb.py <ldm_upb_root> <output.csv>
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


FIELDS = [
    "dataset",
    "group",
    "bearing",
    "snapshot",
    "file",
    "axis",
    "rms",
    "n_rows",
    "sampling_frequency_hz",
    "records",
]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def axis_name(channel: str) -> str:
    suffix = channel.rsplit("_", 1)[-1]
    return suffix if suffix else channel


def extract_experiment(directory: Path) -> list[dict[str, object]]:
    meta_path = directory / "meta.json"
    vibration_path = directory / "vibration.npy"
    if not meta_path.is_file() or not vibration_path.is_file():
        raise FileNotFoundError(f"missing meta.json or vibration.npy in {directory}")

    with meta_path.open() as handle:
        meta = json.load(handle)
    if meta.get("dtype") != "int16":
        raise ValueError(f"unsupported dtype in {directory}: {meta.get('dtype')}")
    if meta.get("volts") != "offset[c] + step[c] * code":
        raise ValueError(f"unsupported decoding formula in {directory}")

    expected_sha = meta.get("sha256", {}).get("vibration.npy")
    if expected_sha and file_sha256(vibration_path) != expected_sha:
        raise ValueError(f"SHA-256 mismatch for {vibration_path}")

    vibration = np.load(vibration_path, mmap_mode="r")
    if vibration.ndim != 3 or vibration.shape[1] != len(meta["channels"]):
        raise ValueError(
            f"unexpected vibration shape in {directory}: {vibration.shape}"
        )
    records, channels, samples = vibration.shape
    if records != int(meta["records"]) or samples != int(meta["samples_per_record"]):
        raise ValueError(f"metadata and array dimensions disagree in {directory}")

    offsets = np.asarray(meta["offset_volts"], dtype=np.float64)
    steps = np.asarray(meta["step_volts"], dtype=np.float64)
    g_per_volt = float(meta["g_per_volt"])
    rows: list[dict[str, object]] = []
    for record_index in range(records):
        for channel_index, channel in enumerate(meta["channels"]):
            code = np.asarray(vibration[record_index, channel_index], dtype=np.int16)
            acceleration_g = g_per_volt * (
                offsets[channel_index] + steps[channel_index] * code.astype(np.float64)
            )
            rms = float(np.sqrt(np.mean(np.square(acceleration_g))))
            rows.append(
                {
                    "dataset": "ldm_upb",
                    "group": "run_to_failure",
                    "bearing": meta["bearing"],
                    "snapshot": record_index + 1,
                    "file": "vibration.npy",
                    "axis": axis_name(channel),
                    "rms": rms,
                    "n_rows": samples,
                    "sampling_frequency_hz": int(meta["sampling_rate_hz"]),
                    "records": records,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="directory containing B01/, B02/, ...")
    parser.add_argument("output", type=Path, help="output feature CSV")
    parser.add_argument(
        "--bearing",
        action="append",
        help="experiment directory to include; may be repeated (default: all B* directories)",
    )
    args = parser.parse_args()

    if args.bearing:
        directories = [args.root / bearing for bearing in args.bearing]
    else:
        directories = sorted(path for path in args.root.glob("B*") if path.is_dir())
    if not directories:
        raise SystemExit(f"no experiment directories found under {args.root}")

    rows: list[dict[str, object]] = []
    for directory in directories:
        print(f"extracting {directory.name}", flush=True)
        rows.extend(extract_experiment(directory))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    bearings = sorted({str(row["bearing"]) for row in rows})
    print(f"wrote {len(rows)} rows for {len(bearings)} bearings to {args.output}")
    print("bearings: " + ",".join(bearings))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise
