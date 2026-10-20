#!/usr/bin/env python3
"""Fig 1: overall architecture - offline discrimination + online deployment + IIoT layering."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "figures"))
from paper_plot_style import *  # noqa: E402,F401

fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6), gridspec_kw={"width_ratios": [1.05, 1.35, 1.0]})

# left: offline discrimination (RQ2)
ax = axes[0]
ax.axis("off")
steps = ["raw\nvibration", "RMS\nper snapshot", "cumulative excess\nsum max(r/1.2-1,0)",
         "failure-vs-drift\nAUROC / permutation"]
for i, s in enumerate(steps):
    x = i / (len(steps) - 1)
    ax.add_patch(plt.Rectangle((x - 0.10, 0.55), 0.20, 0.30, fc="none", ec="0.25", lw=0.8))
    ax.text(x, 0.70, s, ha="center", va="center", fontsize=FONT_SIZE - 2)
    if i < len(steps) - 1:
        ax.annotate("", xy=(x + 0.10, 0.70), xytext=(x + 0.185, 0.70),
                    arrowprops=dict(arrowstyle="-", color="0.4", lw=0.8))
ax.text(0.5, 0.28, "offline: identifies the\ndiscriminative signal (RQ2)", ha="center",
        fontsize=FONT_SIZE - 2, color="0.3")
ax.set_title("(a) offline discrimination", fontsize=FONT_SIZE)

# middle: online deployment (RQ3)
ax = axes[1]
ax.axis("off")
steps = ["raw\nvibration", "RMS\nfeature", "baseline\nmedian(11-60)", "excess rate\nR_t",
         "threshold\nT", "alarm\n(latching)"]
for i, s in enumerate(steps):
    x = i / (len(steps) - 1)
    ax.add_patch(plt.Rectangle((x - 0.055, 0.55), 0.11, 0.24, fc="none", ec="0.25", lw=0.8))
    ax.text(x, 0.67, s, ha="center", va="center", fontsize=FONT_SIZE - 2)
    if i < len(steps) - 1:
        ax.annotate("", xy=(x + 0.055, 0.67), xytext=(x + 0.125, 0.67),
                    arrowprops=dict(arrowstyle="-", color="0.4", lw=0.8))
ax.annotate("", xy=(1.0, 0.22), xytext=(0.0, 0.22), arrowprops=dict(arrowstyle="<->", color="0.3", lw=0.8))
ax.text(0.0, 0.10, "monitoring\nstart", ha="center", va="center", fontsize=FONT_SIZE - 2)
ax.text(0.68, 0.10, "first alarm", ha="center", va="center", fontsize=FONT_SIZE - 2)
ax.text(1.0, 0.10, "failure", ha="center", va="center", fontsize=FONT_SIZE - 2)
ax.annotate("lead time", xy=(0.68, 0.30), xytext=(0.84, 0.44),
            arrowprops=dict(arrowstyle="-", color=COLORS[0], lw=0.8))
ax.text(0.84, 0.46, "lead time\n14.5-46.3% of life", ha="center", fontsize=FONT_SIZE - 2, color=COLORS[0])
ax.text(0.5, -0.02, "0.032 ms update / ~0.038 mJ per window / no accelerator", ha="center",
        fontsize=FONT_SIZE - 2, color="0.3")
ax.set_title("(b) online deployment (causal)", fontsize=FONT_SIZE)

# right: three-layer IIoT deployment
ax = axes[2]
ax.axis("off")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

def box(y0, h, label, sub, fc):
    ax.add_patch(plt.Rectangle((0.06, y0), 0.88, h, fc=fc, ec="0.25", lw=0.8))
    ax.text(0.5, y0 + h * 0.62, label, ha="center", va="center", fontsize=FONT_SIZE - 1)
    ax.text(0.5, y0 + h * 0.22, sub, ha="center", va="center", fontsize=FONT_SIZE - 2, color="0.25")

box(0.72, 0.22, "sensing node", "RMS extraction + rate R_t\n(raw waveform stays local)", "none")
box(0.39, 0.22, "edge gateway", "threshold T + latching alarm\n0.032 ms / 0.038 mJ per window", "#eef4fb")
box(0.06, 0.22, "cloud / fleet", "batch baseline+T calibration\nalarm aggregation (Prop. 4 bound)", "#eafaf1")

ax.annotate("", xy=(0.5, 0.72), xytext=(0.5, 0.615),
            arrowprops=dict(arrowstyle="-|>", color="0.35", lw=1.0))
ax.text(0.53, 0.66, "alarm event only\n(narrow uplink)", ha="left", va="center", fontsize=FONT_SIZE - 2, color="0.3")
ax.annotate("", xy=(0.5, 0.39), xytext=(0.5, 0.285),
            arrowprops=dict(arrowstyle="-|>", color="0.35", lw=1.0))
ax.text(0.53, 0.33, "aggregated\nevents", ha="left", va="center", fontsize=FONT_SIZE - 2, color="0.3")
ax.set_title("(c) three-layer IIoT deployment", fontsize=FONT_SIZE)

fig.tight_layout()
save_fig(fig, "fig1_architecture")
