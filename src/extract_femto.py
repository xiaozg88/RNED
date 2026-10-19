#!/usr/bin/env python3
"""Extract per-snapshot features from raw FEMTO/PRONOSTIA (PHM 2012) acc CSVs.

Raw layout: <Learning_set | Full_Test_Set>/<BearingX_Y>/acc_00001.csv ... with
columns hour, minute, second, microsecond, horizontal, vertical (no header; ',' or
';'). Feature definitions reproduce Hora's MIT extractor.

Usage:
    python3 extract_femto.py <FEMTO root> <out.csv>
"""
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew


def feats(s):
    return dict(
        rms=float(np.sqrt(np.mean(s ** 2))),
        kurtosis_pearson=float(kurtosis(s, fisher=False)),
        skewness=float(skew(s)),
        variance=float(np.var(s)),
        peak2peak=float(s.max() - s.min()),
        n_rows=len(s),
    )


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    root, out = sys.argv[1], sys.argv[2]
    rows = []
    for dirpath, dirs, files in sorted(os.walk(root, followlinks=True)):
        dirs.sort()
        name = os.path.basename(dirpath)
        if not re.match(r"Bearing\d_\d+$", name):
            continue
        parent = os.path.basename(os.path.dirname(dirpath))
        if parent == "Test_set":
            continue  # truncated copies of Full_Test_Set bearings
        snaps = sorted(f for f in files if re.match(r"acc_\d+\.csv$", f))
        if not snaps:
            continue
        print(f"{parent}/{name}: {len(snaps)} snapshots", flush=True)
        for i, f in enumerate(snaps):
            x = pd.read_csv(os.path.join(dirpath, f), header=None, sep=None,
                            engine="python").to_numpy(dtype=float)
            hv = (x[:, 4], x[:, 5])
            t_min = (x[0, 0] * 3600 + x[0, 1] * 60 + x[0, 2] + x[0, 3] * 1e-6) / 60.0
            for axis, s in zip(("H", "V"), hv):
                rows.append(dict(dataset="femto", group=parent, bearing=name,
                                 snapshot=i + 1, file=f, axis=axis, t_min_raw=t_min,
                                 **feats(s)))
    df = pd.DataFrame(rows)
    if df.empty:
        sys.exit(f"No bearing folders found under {root}")
    # clock time wraps at midnight: unwrap per bearing
    for (g, b), d in df.groupby(["group", "bearing"]):
        t = d.drop_duplicates("snapshot").set_index("snapshot").t_min_raw
        add = (t.diff() < -600).cumsum() * 1440
        df.loc[d.index, "t_min_raw"] = d.snapshot.map(t + add)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows for {df.bearing.nunique()} bearings to {out}")


if __name__ == "__main__":
    main()
