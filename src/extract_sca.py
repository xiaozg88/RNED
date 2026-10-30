#!/usr/bin/env python3
"""Extract per-snapshot RMS, speed, and labels from the SCA dataset."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from scipy.io import loadmat


FIELDS = (
    "dataset",
    "split",
    "case",
    "bearing",
    "snapshot",
    "timestamp",
    "axis",
    "rms",
    "rpm",
    "label",
    "n_rows",
    "n_rows_min",
    "n_rows_max",
    "sampling_frequency_hz",
    "from_date",
    "to_date",
    "fault_type",
    "fixed_speed",
    "asset_name",
    "fault_origin",
)


def finite_float(value: object) -> float:
    result = float(value)
    return result if math.isfinite(result) else float("nan")


def extract_file(path: Path) -> list[dict[str, object]]:
    data = loadmat(path, squeeze_me=True, struct_as_record=False)
    case = int(data["id"])
    rows: list[dict[str, object]] = []
    axis_names = [
        key
        for key, value in data.items()
        if not key.startswith("__")
        and hasattr(value, "_fieldnames")
        and "rawData" in value._fieldnames
    ]
    if not axis_names:
        raise ValueError(f"{path}: no vibration channel structures found")
    for axis_name in axis_names:
        axis_data = data[axis_name]
        raw_object = np.asarray(axis_data.rawData)
        rpm = np.asarray(axis_data.RPM, dtype=np.float64)
        labels = np.asarray(axis_data.label, dtype=np.int64)
        timestamps = np.asarray(axis_data.time).astype(str)
        sampling_rates = np.asarray(axis_data.samplingRate, dtype=np.float64)
        if not (raw_object.shape[0] == rpm.size == labels.size == timestamps.size == sampling_rates.size):
            raise ValueError(f"{path}/{axis_name}: inconsistent array lengths")
        if raw_object.dtype == object:
            record_arrays = [
                np.asarray(record, dtype=np.float64).ravel()
                for record in raw_object
            ]
            rms = np.asarray(
                [
                    np.sqrt(np.mean(np.square(record)))
                    for record in record_arrays
                ],
                dtype=np.float64,
            )
            row_counts = [record.size for record in record_arrays]
            n_rows = int(np.median(row_counts))
            n_rows_min = int(min(row_counts))
            n_rows_max = int(max(row_counts))
        else:
            raw = raw_object.astype(np.float64)
            rms = np.sqrt(np.mean(np.square(raw), axis=1))
            n_rows = int(raw.shape[1])
            n_rows_min = n_rows
            n_rows_max = n_rows
        eligible = np.flatnonzero((labels != -1) & np.isfinite(rpm) & (rpm > 0))
        for snapshot, index in enumerate(eligible, start=1):
            rows.append(
                {
                    "dataset": "sca",
                    "split": path.stem.split("_", 1)[1],
                    "case": case,
                    "bearing": f"case_{case:02d}",
                    "snapshot": snapshot,
                    "timestamp": timestamps[index],
                    "axis": axis_name,
                    "rms": finite_float(rms[index]),
                    "rpm": finite_float(rpm[index]),
                    "label": int(labels[index]),
                    "n_rows": n_rows,
                    "n_rows_min": n_rows_min,
                    "n_rows_max": n_rows_max,
                    "sampling_frequency_hz": finite_float(sampling_rates[index]),
                    "from_date": str(data["fromDate"]),
                    "to_date": str(data["toDate"]),
                    "fault_type": int(data["faultType"]),
                    "fixed_speed": int(data["fixedSpeed"]),
                    "asset_name": str(axis_data.assetName),
                    "fault_origin": str(data["faultOrigin"]),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data/raw_archives/sca",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data/features/sca_features.csv",
    )
    args = parser.parse_args()

    paths = sorted(args.input_dir.glob("*_train.mat")) + sorted(
        args.input_dir.glob("*_test.mat")
    )
    if not paths:
        raise ValueError(f"no SCA MAT files found in {args.input_dir}")
    rows: list[dict[str, object]] = []
    for path in paths:
        rows.extend(extract_file(path))
        print(f"extracted {path.name}")
    rows.sort(key=lambda row: (row["split"], row["case"], row["axis"], row["snapshot"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
