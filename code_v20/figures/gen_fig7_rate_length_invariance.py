#!/usr/bin/env python3
"""Fig 7: length-invariance of the rate statistic (Propositions 1-3).

(a) Unbounded accumulators: the running sum S_t of the RMS excess grows
    linearly with monitoring time on the 8 censored (benign-drift) IMS
    bearings, matching the strong-law-of-large-numbers reference line
    mu * (t - 60). Any fixed threshold is therefore crossed eventually.
(b) The rate R_t (running mean) is length-invariant: censored-bearing rates
    converge to a bounded band, while failed-bearing rates escape the fixed
    zero-false-alarm threshold T.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "figures"))
from paper_plot_style import *  # noqa: E402,F401

IMS_FAILED = {("test1", 3), ("test1", 4), ("test2", 1), ("test3", 3)}
T = 0.0424  # zero-false-alarm threshold on the 8 IMS censored bearings


def bearing_series(data_dir: Path) -> dict[str, list[np.ndarray]]:
    """Return {bearing_id: [per-axis rms arrays]} for all IMS bearings."""
    out: dict[str, list[np.ndarray]] = {}
    for test in ("test1", "test2", "test3"):
        df = pd.read_csv(data_dir / f"{test}.csv")
        df["ts"] = pd.to_datetime(df["timestamp"])
        for bearing in sorted(df.bearing.unique()):
            bid = f"{test}_b{bearing}"
            for channel in sorted(df[df.bearing == bearing].channel.unique()):
                s = df[(df.bearing == bearing) & (df.channel == channel)].sort_values("ts")
                out.setdefault(bid, []).append(s["rms"].to_numpy(dtype=np.float64))
    return out


def axis_scores(rms: np.ndarray, c: float = 1.2, runin: int = 10, be: int = 60):
    base = float(np.median(rms[runin:be]))
    excess = np.maximum(rms[be:] / base - c, 0.0)
    cumulative = np.cumsum(excess)
    rate = cumulative / np.arange(1, excess.size + 1)
    return cumulative, rate, excess


def main() -> None:
    here = Path(__file__).resolve().parents[1]
    data_dir = here / "data" / "features"
    streams = bearing_series(data_dir)

    cens_cum, cens_rate, fail_cum, fail_rate = [], [], [], []
    all_cens_excess = []
    for bid, axes in streams.items():
        test, b = bid.split("_b")
        failed = (test, int(b)) in IMS_FAILED
        cum = [axis_scores(a)[0] for a in axes]
        rate = [axis_scores(a)[1] for a in axes]
        exc = [axis_scores(a)[2] for a in axes]
        n = min(len(c) for c in cum)
        cum = np.max(np.vstack([c[:n] for c in cum]), axis=0)
        rate = np.max(np.vstack([r[:n] for r in rate]), axis=0)
        if failed:
            fail_cum.append(cum)
            fail_rate.append(rate)
        else:
            cens_cum.append(cum)
            cens_rate.append(rate)
            all_cens_excess.extend(np.concatenate(exc).tolist())

    mu = float(np.mean(all_cens_excess))  # empirical benign-drift excess rate
    markov_bound = mu / T

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.9))

    # (a) SLLN: cumulative excess grows linearly on censored bearings
    ax = axes[0]
    xmax = 0
    for cum in cens_cum:
        x = np.arange(1, cum.size + 1)
        ax.plot(x, cum, color="0.75", lw=0.8)
        xmax = max(xmax, cum.size)
    xref = np.arange(1, xmax + 1)
    ax.plot(xref, mu * xref, color=COLORS[1], lw=1.4, ls="--",
            label=f"SLLN: $\\mu t$ ($\\mu$={mu:.3f})")
    ax.set_xlabel("monitored snapshots")
    ax.set_ylabel("cumulative excess $S_t$")
    ax.set_title("censored bearings: $S_t$ grows without bound", fontsize=FONT_SIZE)
    ax.legend(frameon=False, loc="upper left")

    # (b) rate is length-invariant and bounded
    ax = axes[1]
    for rate in cens_rate:
        ax.plot(np.arange(1, rate.size + 1), rate, color="0.75", lw=0.8)
    for rate in fail_rate:
        ax.plot(np.arange(1, rate.size + 1), rate, color=COLORS[0], lw=1.2)
    ax.axhline(T, color="0.35", lw=0.9, ls="--", label=f"threshold $T$={T}")
    ax.set_xlabel("monitored snapshots")
    ax.set_ylabel("excess rate $R_t$")
    ax.set_title("rate is bounded (censored) vs escaping (failed)", fontsize=FONT_SIZE)
    ax.legend(frameon=False, loc="upper left")

    fig.tight_layout()
    save_fig(fig, "fig7_rate_length_invariance")

    out = {
        "mu_hat": mu,
        "zero_fa_threshold": T,
        "markov_bound_mu_over_T": markov_bound,
        "n_censored": len(cens_rate),
        "n_failed": len(fail_rate),
        "note": "P(R_t > T) <= mu/T by Markov; mu = mean per-snapshot excess over censored axes.",
    }
    dst = here / "results" / "results_fig7_rate_length_invariance.json"
    dst.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
