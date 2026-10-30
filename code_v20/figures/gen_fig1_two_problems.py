#!/usr/bin/env python3
"""Fig 1 (hero): two OOD problems, two root causes. Schematic, data-grounded."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "figures"))
from paper_plot_style import *  # noqa: E402,F401

fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.6))

# left: degradation OOD
ax = axes[0]
x = np.arange(0, 100)
base = 0.05 * np.sin(x / 9) + 1.0
late = np.where(x > 70, 0.35 * ((x - 70) / 30) ** 2, 0.0)
ax.plot(x, base + late, color=COLORS[0], lw=1.2)
ax.axhline(1.0, color="0.35", lw=0.8, ls="--")
ax.fill_between(x, 1.0, base + late, where=base + late > 1.0, color=COLORS[0], alpha=0.2)
ax.set_xlabel("snapshot")
ax.set_ylabel("RMS ratio")
ax.set_title("degradation OOD (easy)", fontsize=FONT_SIZE)
ax.text(0.5, 0.06, "one machine drifts from its own\nearly-life baseline\n(AUROC 91.0)", transform=ax.transAxes, ha="center", fontsize=FONT_SIZE - 1)
ax.set_ylim(0.7, 2.2)

# right: cross-condition fault OOD
ax = axes[1]
xx = np.linspace(-3, 4, 300)
ax.plot(xx, 0.8 * np.exp(-0.5 * (xx - 0.0) ** 2), color=COLORS[0], lw=1.2, label="cond. A")
ax.plot(xx, 0.8 * np.exp(-0.5 * (xx - 1.4) ** 2), color=COLORS[1], lw=1.2, label="cond. B")
ax.fill_between(xx, 0.8 * np.exp(-0.5 * (xx - 0.0) ** 2), color=COLORS[0], alpha=0.18)
ax.fill_between(xx, 0.8 * np.exp(-0.5 * (xx - 1.4) ** 2), color=COLORS[1], alpha=0.18)
ax.set_xlabel("OOD score")
ax.set_ylabel("density")
ax.set_title("MCC5 cross-condition fault OOD (compound fault)", fontsize=FONT_SIZE)
ax.text(0.5, 0.06, "score distributions overlap\nacross conditions\n(AUROC 49.9-53.8)", transform=ax.transAxes, ha="center", fontsize=FONT_SIZE - 1)
ax.set_ylim(0, 1.0)
ax.legend(frameon=False, loc="upper right")

fig.tight_layout()
save_fig(fig, "fig2_two_problems")
