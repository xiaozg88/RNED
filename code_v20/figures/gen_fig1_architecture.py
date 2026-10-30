#!/usr/bin/env python3
"""Publication figure for the offline, online, and fleet architecture."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "figures"))
from paper_plot_style import *  # noqa: E402,F401


BLUE = "#E8F1FB"
GREEN = "#EAF7EF"
ORANGE = "#FDF0E5"
GRAY = "#F4F4F4"


def rounded_box(ax, x, y, w, h, title, subtitle="", fc=GRAY, title_size=8.4,
                subtitle_size=7.2):
    ax.add_patch(plt.Rectangle(
        (x, y), w, h, fc=fc, ec="#4A4A4A", lw=0.75,
        joinstyle="round",
    ))
    if subtitle:
        ax.text(x + w / 2, y + h * 0.65, title, ha="center", va="center",
                fontsize=title_size, fontweight="bold", linespacing=1.15)
        ax.text(x + w / 2, y + h * 0.29, subtitle, ha="center", va="center",
                fontsize=subtitle_size, color="#3B3B3B", linespacing=1.15)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=title_size, fontweight="bold", linespacing=1.15)


def arrow(ax, x0, y0, x1, y1, text=None, dx=0.0, dy=0.0, color="#4A4A4A"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=0.85, shrinkA=1, shrinkB=1))
    if text:
        ax.text((x0 + x1) / 2 + dx, (y0 + y1) / 2 + dy, text,
                ha="center", va="center", fontsize=6.9, color="#4A4A4A")


fig = plt.figure(figsize=(7.1, 4.9))
gs = fig.add_gridspec(2, 2, width_ratios=[0.94, 1.06],
                      height_ratios=[1.10, 0.90], left=0.035,
                      right=0.985, top=0.91, bottom=0.055,
                      wspace=0.16, hspace=0.30)

# (a) Offline mechanism audit.
ax = fig.add_subplot(gs[0, 0])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
ax.set_title("(a) Offline mechanism audit", loc="left", fontsize=9,
             fontweight="bold")
items = [
    ("Run-to-failure\nvibration", "public rigs"),
    ("RMS trajectory", "per bearing axis"),
    ("Cumulative excess", r"$\sum\max(f_t/b-1.2,0)$"),
    ("Failure vs drift", "persistence evidence"),
]
positions = [(0.03, 0.50), (0.54, 0.50), (0.03, 0.05), (0.54, 0.05)]
for (title, sub), (x, y) in zip(items, positions):
    rounded_box(ax, x, y, 0.40, 0.31, title, sub,
                fc=BLUE if title.startswith("RMS") else GRAY)
arrow(ax, 0.43, 0.655, 0.54, 0.655)
arrow(ax, 0.74, 0.50, 0.74, 0.36)
arrow(ax, 0.54, 0.205, 0.43, 0.205)
ax.text(0.50, 0.90, "Full-life audit only; not the online detector",
        ha="center", va="center", fontsize=7.2, color="#4A4A4A")

# (b) Online detector.
ax = fig.add_subplot(gs[0, 1])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
ax.set_title("(b) Causal online detector", loc="left", fontsize=9,
             fontweight="bold")
online = [
    ("Raw\nwindow", "sensor node", 0.02, 0.50),
    ("RMS\n$f_t$", "one scalar", 0.35, 0.50),
    ("Baseline\n$b$", "median 11--60", 0.68, 0.50),
    ("Excess\n$e_t$", r"$\max(f_t/b-1.2,0)$", 0.02, 0.05),
    ("Rate\n$R_t$", r"$S_t/(t-60)$", 0.35, 0.05),
    ("Alarm", "latch if $R_t>T$", 0.68, 0.05),
]
for title, sub, x, y in online:
    fc = GREEN if title == "Rate\n$R_t$" else (ORANGE if title == "Alarm" else GRAY)
    rounded_box(ax, x, y, 0.25, 0.30, title, sub, fc=fc)
arrow(ax, 0.27, 0.65, 0.35, 0.65)
arrow(ax, 0.60, 0.65, 0.68, 0.65)
arrow(ax, 0.805, 0.50, 0.145, 0.35,
      text="online from snapshot 61", dx=0.0, dy=0.03)
arrow(ax, 0.27, 0.20, 0.35, 0.20)
arrow(ax, 0.60, 0.20, 0.68, 0.20)
ax.text(0.50, 0.91, r"$O(1)$ update: 0.032 ms; full pipeline: 0.038 mJ/window",
        ha="center", va="center", fontsize=7.4, color="#4A4A4A")

# (c) Three-layer IIoT deployment.
ax = fig.add_subplot(gs[1, :])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
ax.set_title("(c) Three-layer IIoT deployment", loc="left", fontsize=9,
             fontweight="bold")
layers = [
    ("Sensing node", "RMS extraction + incremental\nrate update; raw waveform stays local", GRAY),
    ("Edge gateway", "threshold comparison + latching\nalarm; local calibration cache", BLUE),
    ("Cloud / fleet", "baseline and threshold calibration;\nevent aggregation and false-alarm budget", GREEN),
]
for i, (title, sub, fc) in enumerate(layers):
    x = 0.035 + i * 0.325
    rounded_box(ax, x, 0.20, 0.25, 0.49, title, sub, fc=fc,
                title_size=8.8, subtitle_size=7.2)
    if i < 2:
        arrow(ax, x + 0.25, 0.445, x + 0.325, 0.445,
              text="alarm event", dy=0.085)
ax.add_patch(plt.Rectangle((0.035, 0.03), 0.885, 0.11, fc="#FFFFFF",
                           ec="#B9C6D2", lw=0.7, linestyle="--"))
ax.text(0.50, 0.085, "Deployment constraints: narrow uplink, low power, accelerator-free execution, fleet-scale alarm management",
        ha="center", va="center", fontsize=7.4, color="#4A4A4A")

fig.savefig(ROOT / "figures" / "fig1_architecture.pdf")
fig.savefig(ROOT / "figures" / "fig1_architecture.png")
print(f"saved {ROOT / 'figures' / 'fig1_architecture.pdf'}")
print(f"saved {ROOT / 'figures' / 'fig1_architecture.png'}")
