#!/usr/bin/env python3
"""Compare self-extracted feature tables against the Hora-derived reference tables.

Usage:
    python3 compare_features.py <mine_dir> <reference_dir>
    mine_dir: contains test1.csv, test2.csv, test3.csv, xjtu_features.csv,
              femto_features.csv (outputs of the extract_*.py scripts)
    reference_dir: data/hora_reference
"""
import os
import sys

import numpy as np
import pandas as pd

FEAT_COLS = ["rms", "kurtosis_pearson", "skewness", "variance", "peak2peak"]


def compare(name, mine, ref, keys):
    if not os.path.exists(mine):
        print(f"[{name}] MISSING mine file: {mine}")
        return
    a = pd.read_csv(mine)
    b = pd.read_csv(ref)
    print(f"[{name}] rows mine={len(a)} ref={len(b)}")
    a = a.set_index(keys)
    b = b.set_index(keys)
    common = sorted(set(a.index) & set(b.index))
    print(f"[{name}] common keys={len(common)} only-mine={len(set(a.index) - set(b.index))} "
          f"only-ref={len(set(b.index) - set(a.index))}")
    if not common:
        print(f"[{name}] NO COMMON KEYS — cannot compare")
        return
    am = a.loc[common, FEAT_COLS].to_numpy(dtype=float)
    bm = b.loc[common, FEAT_COLS].to_numpy(dtype=float)
    absdiff = np.abs(am - bm)
    reldiff = absdiff / (np.abs(bm) + 1e-12)
    for j, c in enumerate(FEAT_COLS):
        print(f"[{name}] {c:16s} max_abs={absdiff[:, j].max():.3e} "
              f"max_rel={reldiff[:, j].max():.3e} allclose={np.allclose(am[:, j], bm[:, j], rtol=1e-6, atol=1e-8)}")


def main():
    mine, ref = sys.argv[1], sys.argv[2]
    for t in ("test1", "test2", "test3"):
        compare(t, f"{mine}/{t}.csv", f"{ref}/{t}.csv", ["filename", "channel"])
    compare("xjtu", f"{mine}/xjtu_features.csv", f"{ref}/xjtu_features.csv",
            ["group", "bearing", "file", "axis"])
    compare("femto", f"{mine}/femto_features.csv", f"{ref}/femto_features.csv",
            ["group", "bearing", "file", "axis"])


if __name__ == "__main__":
    main()
