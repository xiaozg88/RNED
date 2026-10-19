#!/usr/bin/env python3
"""Extract per-snapshot time-domain features from raw NASA IMS bearing files.

The IMS raw snapshot files are whitespace/tab-separated text with one column per
accelerometer channel (Test 1: 8 channels, two per bearing; Tests 2 and 3: 4
channels, one per bearing) and 20,480 rows per snapshot. File names encode the
timestamp (YYYY.MM.DD.hh.mm.ss).

Feature definitions reproduce the label-free early-warning pipeline
(bhavjeetsinghh/ims-bearing-label-free-early-warning, MIT):
    RMS, Pearson kurtosis (scipy kurtosis fisher=False), sample skewness,
    population variance, peak-to-peak.

Usage:
    python3 extract_ims.py <IMS root with 1st_test/2nd_test/3rd_test> <out_dir>
Output: <out_dir>/test1.csv, test2.csv, test3.csv
"""
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

SNAP_RE = re.compile(r"^\d{4}\.\d{2}\.\d{2}\.\d{2}\.\d{2}\.\d{2}$")


def find_test_dirs(root):
    """Return {test_name: [dir, ...]} for directories that contain snapshot files."""
    tests = {}
    for dirpath, dirs, files in os.walk(root):
        dirs.sort()
        snaps = sorted(f for f in files if SNAP_RE.match(f))
        if snaps:
            # The IMS archive unpacks test 1/2 under 1st_test/2nd_test and test 3
            # under 3rd_test.rar -> 4th_test/txt.
            key = ("test1" if "1st" in dirpath else
                   "test2" if "2nd" in dirpath else
                   "test3" if ("3rd" in dirpath or "4th" in dirpath) else
                   os.path.basename(dirpath))
            tests.setdefault(key, []).append((dirpath, snaps))
    return tests


def feats(s):
    return dict(
        n_rows=len(s),
        rms=float(np.sqrt(np.mean(s ** 2))),
        kurtosis_pearson=float(kurtosis(s, fisher=False)),
        skewness=float(skew(s)),
        variance=float(np.var(s)),
        peak2peak=float(s.max() - s.min()),
    )


def extract_dir(dirpath, snaps, out_dir, test):
    rows = []
    nch = None
    for i, fn in enumerate(snaps):
        p = os.path.join(dirpath, fn)
        try:
            x = np.loadtxt(p, dtype=float)
        except Exception as e:
            print(f"  WARN: skip {fn}: {e}", file=sys.stderr)
            continue
        if x.ndim == 1:
            x = x[:, None]
        nch = x.shape[1] if nch is None else nch
        for ch in range(nch):
            s = x[:, ch]
            bearing = (ch // 2 + 1) if nch == 8 else ch + 1
            rows.append(dict(filename=fn, channel=ch + 1, bearing=bearing, **feats(s)))
        if (i + 1) % 500 == 0:
            print(f"  [{test}] {i + 1}/{len(snaps)} files", flush=True)
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["filename"], format="%Y.%m.%d.%H.%M.%S")
    out = os.path.join(out_dir, f"{test}.csv")
    df.to_csv(out, index=False)
    print(f"  wrote {out}: {len(df)} rows, {nch} channels", flush=True)


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    root, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    tests = find_test_dirs(root)
    if not tests:
        sys.exit(f"No IMS snapshot files found under {root}")
    for test in ("test1", "test2", "test3"):
        if test not in tests:
            print(f"WARN: {test} not found", file=sys.stderr)
            continue
        for dirpath, snaps in tests[test]:
            out = os.path.join(out_dir, f"{test}.csv")
            if os.path.exists(out):
                print(f"  skip {test}: {out} already exists")
                continue
            extract_dir(dirpath, snaps, out_dir, test)


if __name__ == "__main__":
    main()
