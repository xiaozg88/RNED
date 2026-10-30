#!/usr/bin/env python3
"""Extract per-snapshot features from raw XJTU-SY bearing CSVs.

Raw layout (official data packet): <condition>/<BearingX_Y>/<n>.csv where each CSV
has a header row and two columns (horizontal, vertical vibration), 32768 rows,
one file per minute. Feature definitions reproduce Hora's MIT extractor
(bhavjeetsinghh/bearing-early-warning-transfer, src/extract_runs.py).

Usage:
    python3 extract_xjtu.py <XJTU-SY root> <out.csv>
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
        snaps = sorted((f for f in files if re.match(r"\d+\.csv$", f)),
                       key=lambda f: int(f[:-4]))
        if not snaps:
            continue
        print(f"{parent}/{name}: {len(snaps)} snapshots", flush=True)
        for i, f in enumerate(snaps):
            x = pd.read_csv(os.path.join(dirpath, f)).to_numpy(dtype=float)
            hv = (x[:, 0], x[:, 1])
            t_min = float(int(f[:-4]))
            for axis, s in zip(("H", "V"), hv):
                rows.append(dict(dataset="xjtu", group=parent, bearing=name,
                                 snapshot=i + 1, file=f, axis=axis, t_min_raw=t_min,
                                 **feats(s)))
    df = pd.DataFrame(rows)
    if df.empty:
        sys.exit(f"No bearing folders found under {root}")
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows for {df.bearing.nunique()} bearings to {out}")


if __name__ == "__main__":
    main()
