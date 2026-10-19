#!/usr/bin/env python3
"""Fig 6: negative transfer - raw ID FPR by held condition (MCC5)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "figures"))
from paper_plot_style import *  # noqa: E402,F401

d = pd.read_csv(ROOT / "pi_caag/results_mcc5_streaming_unified_audit.csv")
raw = d[d.detector == "raw"]
g = raw.groupby("held_group").id_fpr.mean().sort_index()

fig, ax = plt.subplots(figsize=(4.6, 2.8), constrained_layout=True)
bars = ax.bar([f"group {i}" for i in g.index], g.values, color=COLORS[0])
for b, v in zip(bars, g.values):
    ax.text(b.get_x() + b.get_width() / 2, v + 1.0, f"{v:.1f}%", ha="center", va="bottom", fontsize=FONT_SIZE - 1)
ax.set_ylabel("raw ID false-positive rate (%)")
ax.set_xlabel("held-out operating condition")
ax.set_ylim(0, 100)
save_fig(fig, "fig3_negative_transfer")
