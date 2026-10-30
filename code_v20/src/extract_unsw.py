#!/usr/bin/env python3
"""Extract per-cycle horizontal/vertical RMS from an UNSW 6 Hz run."""
from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import numpy as np
from scipy.io import loadmat


FILENAME_RE = re.compile(r"^vib_(\d+)_(\d+)\.mat$", re.IGNORECASE)
FIELDS = (
    "dataset",
    "group",
    "bearing",
    "snapshot",
    "shaft_cycle",
    "file",
    "axis",
    "rms",
    "n_rows",
    "sampling_frequency_hz",
)


def clean_matrix(value: np.ndarray) -> np.ndarray:
    return np.asarray(value, dtype=np.float64).squeeze()


def extract_file(path: Path, dataset: str, group: str, bearing: str) -> list[dict]:
    match = FILENAME_RE.match(path.name)
    if match is None:
        raise ValueError(f"unexpected UNSW filename: {path.name}")
    shaft_cycle = int(match.group(1))
    shaft_speed_hz = int(match.group(2))
    if shaft_speed_hz != 6:
        raise ValueError(f"expected the continuous 6 Hz stream, got {shaft_speed_hz} Hz")

    payload = loadmat(path)
    rows = []
    for axis, variable in (("H", "accH"), ("V", "accV")):
        signal = clean_matrix(payload[variable])
        finite = signal[np.isfinite(signal)]
        if finite.size == 0:
            raise ValueError(f"no finite samples in {path.name}:{variable}")
        sampling_frequency = float(np.asarray(payload["Fs"]).squeeze())
        rows.append(
            {
                "dataset": dataset,
                "group": group,
                "bearing": bearing,
                "snapshot": 0,
                "shaft_cycle": shaft_cycle,
                "file": path.name,
                "axis": axis,
                "rms": float(np.sqrt(np.mean(finite ** 2))),
                "n_rows": int(finite.size),
                "sampling_frequency_hz": sampling_frequency,
            }
        )
    return rows


def extract(root: Path, output: Path, dataset: str) -> None:
    bearing = root.parent.name if root.name.lower() == "6hz" else root.name
    files = sorted(
        path for path in root.rglob("*.mat") if FILENAME_RE.match(path.name)
    )
    if not files:
        raise FileNotFoundError(f"no vib_*_06.mat files under {root}")

    grouped: dict[int, list[dict]] = {}
    for path in files:
        for row in extract_file(path, dataset, path.parent.name, bearing):
            grouped.setdefault(row["shaft_cycle"], []).append(row)

    rows = []
    for snapshot, (_, cycle_rows) in enumerate(sorted(grouped.items()), start=1):
        axes = {row["axis"]: row for row in cycle_rows}
        if set(axes) != {"H", "V"}:
            raise ValueError(f"cycle {cycle_rows[0]['shaft_cycle']} lacks one H/V axis")
        if len(cycle_rows) != 2:
            raise ValueError(f"duplicate H/V records at cycle {cycle_rows[0]['shaft_cycle']}")
        for row in cycle_rows:
            row["snapshot"] = snapshot
            if not math.isfinite(row["rms"]):
                raise ValueError(f"non-finite RMS at cycle {row['shaft_cycle']}")
            rows.append(row)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"wrote {len(rows)} rows, {len(grouped)} cycles, "
        f"1 bearing to {output}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="directory containing one UNSW 6 Hz stream")
    parser.add_argument("output", type=Path, help="output feature CSV")
    parser.add_argument("--dataset", default="unsw")
    args = parser.parse_args()
    extract(args.root, args.output, args.dataset)


if __name__ == "__main__":
    main()
