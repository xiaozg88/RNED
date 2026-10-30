#!/usr/bin/env python3
"""Fig 8: fleet-scale false-alarm budget from frozen results."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "figures"))
from paper_plot_style import *  # noqa: E402,F401

RESULT = ROOT / "RNED" / "results" / "results_run_to_failure_fleet_fa_budget.json"
import json
data = json.loads(RESULT.read_text())
rows = data["fleet_budget_cross_rig"]
rescale = data["fleet_rescaling_for_5pct"]

m = np.array([r["fleet_size_machines"] for r in rows], dtype=float)
p_fleet = np.array([r["p_fleet_ge1_false_alarm_one_segment"] for r in rows])
expected = np.array([r["expected_false_alarms_per_segment"] for r in rows])
tight = np.array([r["tightening_factor_vs_18.5pct"] for r in rescale])

fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6))

ax = axes[0]
ax.semilogx(m, p_fleet, "o-", color=COLORS[0], lw=1.2, ms=3.5)
ax.axhline(0.05, color="0.4", lw=0.8, ls="--")
ax.text(1.1, 0.09, "5% fleet budget", fontsize=FONT_SIZE - 2, color="0.3")
ax.set_xlabel("fleet size (machines)")
ax.set_ylabel("P($\\geq$1 false alarm / epoch)")
ax.set_ylim(0, 1.05)
ax.set_title("(a) fleet false-alarm probability", fontsize=FONT_SIZE)

ax = axes[1]
ax.loglog(m, expected, "s-", color=COLORS[1], lw=1.2, ms=3.5, label="expected FAs / epoch")
ax.loglog(m, tight, "d--", color=COLORS[0], lw=1.2, ms=3.5, label="required tightening $\\times$")
ax.set_xlabel("fleet size (machines)")
ax.set_ylabel("count / factor")
ax.legend(frameon=False, fontsize=FONT_SIZE - 2)
ax.set_title("(b) false-alarm load and tightening", fontsize=FONT_SIZE)

fig.tight_layout()
save_fig(fig, "fig8_fleet_false_alarm")
