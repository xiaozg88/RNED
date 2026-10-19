#!/usr/bin/env python3
"""Extract envelope-spectrum fault-frequency indicators from the raw signals.

For every snapshot and axis, compute:
  - envelope = |Hilbert(x - mean(x))|
  - envelope spectrum via FFT
  - ff_ratio  : power in bands around BPFO/BPFI/BSF harmonics, divided by the
                total envelope-spectrum power (10 Hz .. Nyquist/2)
  - env_kurtosis : kurtosis of the envelope (impulsiveness)

Bearing geometry and shaft speeds are hard-coded from the dataset documentation
(see per-dataset GEOM dicts). Output is one row per (bearing, axis, snapshot).

Usage:
    python3 extract_spectral.py <ims|xjtu|femto> <raw_root> <out.csv>
"""
from __future__ import annotations

import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.signal import hilbert
from scipy.stats import kurtosis


def fault_freqs(N, d, D, phi_deg, rpm):
    fr = rpm / 60.0
    cos = np.cos(np.deg2rad(phi_deg))
    bpfo = (N / 2) * fr * (1 - d / D * cos)
    bpfi = (N / 2) * fr * (1 + d / D * cos)
    bsf = (D / (2 * d)) * fr * (1 - (d / D * cos) ** 2)
    ftf = (fr / 2) * (1 - d / D * cos)
    return bpfo, bpfi, bsf, ftf


# geometry: (N, d, D, contact angle deg)
GEOM = {
    "ims": dict(geom=(16, 0.331, 2.815, 15.17), fs=20000, rpm=2000),
    "xjtu": dict(geom=(8, 7.92, 34.55, 0.0), fs=25600, rpm={"35Hz12kN": 2100, "37.5Hz11kN": 2250, "40Hz10kN": 2400}),
    "femto": dict(geom=(13, 3.5, 25.6, 0.0), fs=25600, rpm={"1": 1800, "2": 1650, "3": 1500}),
}


def spectral_indicators(x, bpfo, bpfi, bsf, fs, n_harm=5, band=2.0):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    env = np.abs(hilbert(x) if x.size > 0 else x)
    env = env - env.mean()
    spec = np.abs(np.fft.rfft(env))
    freqs = np.fft.rfftfreq(env.size, d=1.0 / fs)
    lo = max(10.0, freqs[1])
    hi = freqs[-1]
    keep = (freqs >= lo) & (freqs <= hi)
    total = float(np.sum(spec[keep] ** 2))
    if total <= 0:
        return np.nan, float(kurtosis(env, fisher=False))
    fault = 0.0
    for f0 in (bpfo, bpfi, bsf):
        for h in range(1, n_harm + 1):
            fc = f0 * h
            if fc + band >= hi:
                break
            band_mask = (freqs >= fc - band) & (freqs <= fc + band)
            fault += float(np.sum(spec[band_mask] ** 2))
    return float(fault / total), float(kurtosis(env, fisher=False))


def read_ims_file(path, nch):
    x = np.loadtxt(path, dtype=np.float64)
    return x[:, :nch]


def process_ims(root, out):
    fs, rpm = GEOM["ims"]["fs"], GEOM["ims"]["rpm"]
    bpfo, bpfi, bsf, _ = fault_freqs(*GEOM["ims"]["geom"], rpm)
    rows = []
    for dirpath, dirs, files in os.walk(root):
        dirs.sort()
        snaps = sorted(f for f in files if re.match(r"^\d{4}\.\d{2}\.\d{2}\.\d{2}\.\d{2}\.\d{2}$", f))
        if not snaps:
            continue
        test = "test1" if "1st" in dirpath else ("test2" if "2nd" in dirpath else "test3")
        for fi, fn in enumerate(snaps):
            x = np.loadtxt(os.path.join(dirpath, fn), dtype=np.float64)
            nch = x.shape[1] if x.ndim > 1 else 1
            for ch in range(nch):
                bearing = (ch // 2 + 1) if nch == 8 else ch + 1
                ff, ek = spectral_indicators(x[:, ch], bpfo, bpfi, bsf, fs)
                rows.append(dict(dataset="ims", bearing_id=f"{test}_b{bearing}", axis=ch + 1,
                                 snapshot=fi + 1, ff_ratio=ff, env_kurtosis=ek))
            if (fi + 1) % 500 == 0:
                print(f"  [{test}] {fi + 1}/{len(snaps)}", flush=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out}: {len(rows)} rows")


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    kind, root, out = sys.argv[1], sys.argv[2], sys.argv[3]
    if kind != "ims":
        sys.exit("currently only 'ims' is implemented")
    process_ims(root, out)


if __name__ == "__main__":
    main()
