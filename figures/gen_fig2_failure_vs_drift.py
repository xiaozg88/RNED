#!/usr/bin/env python3
"""Fig 2: persistence > magnitude. Real IMS trajectories, failed vs censored."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "figures"))
from paper_plot_style import *  # noqa: E402,F401

df = pd.read_csv(ROOT / "data/features/test1.csv")
df["ts"] = pd.to_datetime(df["timestamp"])


def series(bearing, channel):
    s = df[(df.bearing == bearing) & (df.channel == channel)].sort_values("ts")
    return s["rms"].to_numpy(dtype=np.float64)


def ratio_rate(rms, c=1.2, runin=10, be=60):
    base = float(np.median(rms[runin:be]))
    mon = rms[be:]
    ratio = mon / base
    excess = np.maximum(ratio - c, 0.0)
    rate = np.cumsum(excess) / np.arange(1, excess.size + 1)
    return ratio, rate


T = json.load(open(ROOT / "full_supplement/results_run_to_failure_early_warning.json"))["zero_fa_threshold"]

fig, axes = plt.subplots(2, 2, figsize=(6.5, 4.6))

for col, (bearing, channel, label) in enumerate(
    [(3, 5, "failed (inner race, test1 B3)"), (1, 1, "censored (test1 B1)")]
):
    rms = series(bearing, channel)
    ratio, rate = ratio_rate(rms)
    x = np.arange(1, ratio.size + 1)
    # top: RMS ratio with shaded excess
    ax = axes[0, col]
    ax.plot(x, ratio, color=COLORS[0], lw=1.0)
    ax.axhline(1.2, color="0.35", lw=0.8, ls="--")
    ax.fill_between(x, 1.2, np.maximum(ratio, 1.2), color=COLORS[1], alpha=0.35, lw=0)
    ax.set_title(label, fontsize=FONT_SIZE)
    ax.set_ylabel("RMS ratio r_t")
    ax.set_xlabel("")
    # bottom: running-mean rate
    ax = axes[1, col]
    ax.plot(x, rate, color=COLORS[2], lw=1.0)
    ax.axhline(T, color="0.35", lw=0.8, ls="--", label="threshold T")
    ax.set_ylabel("excess rate R_t")
    ax.set_xlabel("monitored snapshots")
    if col == 1:
        ax.legend(frameon=False, loc="upper left")

fig.tight_layout()
save_fig(fig, "fig4_failure_vs_drift")
