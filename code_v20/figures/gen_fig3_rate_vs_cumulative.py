#!/usr/bin/env python3
"""Fig 3: rate normalization. Unbounded accumulators vs the rate statistic."""
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


def scores(rms, c=1.2, runin=10, be=60):
    base = float(np.median(rms[runin:be]))
    sigma = max(float(np.std(rms[runin:be]) / base), 1e-9)
    ratio = rms[be:] / base
    excess = np.maximum(ratio - c, 0.0)
    rate = np.cumsum(excess) / np.arange(1, excess.size + 1)
    cusum = np.empty_like(excess)
    acc = 0.0
    for i, v in enumerate(excess):
        acc = max(0.0, acc + v)
        cusum[i] = acc
    inc = (c - 1.0) / (sigma * sigma) * (ratio - (1.0 + c) / 2.0)
    sprt = np.empty_like(inc)
    acc = 0.0
    for i, v in enumerate(inc):
        acc = max(0.0, acc + v)
        sprt[i] = acc
    return rate, cusum, sprt


cb = json.load(open(ROOT / "full_supplement/results_run_to_failure_classical_baselines.json"))["detectors"]
T_rate = cb["excess_rate"]["zero_fa_threshold"]
T_cusum = cb["cusum"]["zero_fa_threshold"]
T_sprt = cb["sprt"]["zero_fa_threshold"]

fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.9))

# (a) benign drift is integrated by accumulators, not by the rate
rms_cens = series(1, 1)
rate_c, cusum_c, sprt_c = scores(rms_cens)
x = np.arange(1, rate_c.size + 1)
ax = axes[0]
ax.plot(x, cusum_c, color=COLORS[1], lw=1.0, label="CUSUM (cumulative)")
ax.plot(x, sprt_c, color=COLORS[3], lw=1.0, label="SPRT (cumulative)")
ax.set_xlabel("monitored snapshots")
ax.set_ylabel("cumulative score", color="0.2")
ax2 = ax.twinx()
ax2.plot(x, rate_c, color=COLORS[0], lw=1.2, ls="--", label="excess rate")
ax2.set_ylabel("excess rate R_t", color=COLORS[0])
ax2.tick_params(axis="y", colors=COLORS[0])
ax.set_title("censored bearing: benign drift", fontsize=FONT_SIZE)

# (b) normalized to each detector's zero-FA threshold, only the rate crosses 1
rms_fail = series(3, 5)
rate_f, cusum_f, sprt_f = scores(rms_fail)
ax = axes[1]
ax.plot(x[: rate_f.size], rate_f / T_rate, color=COLORS[0], lw=1.2, label="rate / T")
ax.plot(x[: cusum_f.size], cusum_f / T_cusum, color=COLORS[1], lw=1.0, label="CUSUM / T")
ax.plot(x[: sprt_f.size], sprt_f / T_sprt, color=COLORS[3], lw=1.0, label="SPRT / T")
ax.axhline(1.0, color="0.35", lw=0.8, ls="--")
ax.set_xlabel("monitored snapshots")
ax.set_ylabel("score / threshold")
ax.set_ylim(-0.05, 1.6)
ax.set_title("failed bearing: alarm when > 1", fontsize=FONT_SIZE)
ax.legend(frameon=False, loc="upper left")

fig.tight_layout()
save_fig(fig, "fig6_rate_vs_cumulative")
