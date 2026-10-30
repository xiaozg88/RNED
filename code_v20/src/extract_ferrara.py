#!/usr/bin/env python3
"""Extract sequential RMS from a University of Ferrara run."""
from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import numpy as np
from scipy.io import loadmat


FILENAME_RE = re.compile(r"^E\d+_(\d+)\.mat$", re.IGNORECASE)
FIELDS = (
    "dataset",
    "group",
    "bearing",
    "snapshot",
    "file",
    "axis",
    "rms",
    "n_rows",
    "sampling_frequency_hz",
)


def extract(root: Path, output: Path, dataset: str) -> None:
    matches = []
    for path in root.glob("*.mat"):
        match = FILENAME_RE.match(path.name)
        if match is not None:
            matches.append((int(match.group(1)), path))
    if not matches:
        raise FileNotFoundError(f"no E*_<index>.mat files under {root}")

    bearing = root.name
    rows = []
    for snapshot, (_, path) in enumerate(sorted(matches), start=1):
        payload = loadmat(path)
        signal = np.asarray(payload["y"], dtype=np.float64).squeeze()
        finite = signal[np.isfinite(signal)]
        if finite.size == 0:
            raise ValueError(f"no finite samples in {path}")
        rms = float(np.sqrt(np.mean(finite ** 2)))
        if not math.isfinite(rms):
            raise ValueError(f"non-finite RMS in {path}")
        rows.append(
            {
                "dataset": dataset,
                "group": bearing,
                "bearing": bearing,
                "snapshot": snapshot,
                "file": path.name,
                "axis": "Y",
                "rms": rms,
                "n_rows": int(finite.size),
                "sampling_frequency_hz": float(
                    np.asarray(payload["Fs"]).squeeze()
                ),
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows, 1 bearing to {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dataset", default="ferrara_e6")
    args = parser.parse_args()
    extract(args.root, args.output, args.dataset)


if __name__ == "__main__":
    main()
