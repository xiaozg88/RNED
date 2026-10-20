#!/usr/bin/env python3
"""Supervised RUL-thresholded alarm baseline (P2, GPU).

Answers the reviewer's ``why not optimize end-to-end'': a supervised RUL
regressor (unidirectional LSTM) is trained on the per-snapshot fraction of
life remaining of all failed bearings, then converted into an alarm rule by
``predicted remaining life < threshold''. The threshold is calibrated to zero
false alarms on the 8 IMS censored bearings under the same frozen causal
protocol (monitor from snapshot 61, latching first crossing, >=30 monitored
snapshots), so detection rate and lead time are directly comparable with the
rate-normalized detector. Note: the regressor is trained in-sample on all
failed bearings and evaluated on the same bearings, an upper-bound
(favorable) treatment for a baseline; it is still seed-unstable.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "experiments"))
import run_to_failure_ood as ood  # noqa: E402

FEATURES = ood.FEATURES
SEEDS = (0, 1, 2)
EPOCHS_GLOBAL = 8
LR = 0.005
HIDDEN = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class RULLSTM(nn.Module):
    def __init__(self, d_in: int):
        super().__init__()
        self.lstm = nn.LSTM(d_in, HIDDEN, batch_first=True)
        self.head = nn.Linear(HIDDEN, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)  # (1, L, H), causal (unidirectional)
        return self.head(out).squeeze(-1)  # (1, L)


def zscore(x: np.ndarray) -> np.ndarray:
    base = x[10:60]
    mu = base.mean(0)
    sd = base.std(0) + 1e-8
    return (x - mu[None, :]) / sd[None, :]


def rul_label(n: int) -> np.ndarray:
    frac = np.ones(n, dtype=np.float64)
    denom = max(1, n - 60)
    for t in range(60, n):
        frac[t] = (n - t) / denom
    return frac


def train_model(failed_streams, seed: int) -> RULLSTM:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = RULLSTM(len(FEATURES)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for ep in range(EPOCHS_GLOBAL):
        for s in failed_streams:
            x = s["x"]
            z = zscore(x)
            zt = torch.tensor(z, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            y = torch.tensor(rul_label(x.shape[0]), dtype=torch.float32, device=DEVICE).unsqueeze(0)
            opt.zero_grad()
            pred = model(zt)
            loss = ((pred[:, 60:] - y[:, 60:]) ** 2).mean()
            loss.backward()
            opt.step()
    return model


def alarm_below(score: np.ndarray, threshold: float, min_window: int = 30) -> int:
    start = max(0, min_window - 1)
    hits = np.flatnonzero(score[start:] < threshold)
    hits = hits + start
    return int(hits[0]) if hits.size else int(score.size)


def main() -> None:
    here = Path(os.environ.get("RTF_DATA_DIR", str(ROOT / "data" / "features")))
    streams = (
        ood.load_ims_streams(here)
        + ood.load_grouped_streams(here / "xjtu_features.csv", "xjtu", True)
        + ood.load_grouped_streams(here / "femto_features.csv", "femto", True)
    )
    streams = [s for s in streams if s["x"].shape[0] >= 70 and np.all(np.isfinite(s["x"]))]
    failed_streams = [s for s in streams if s["failed"]]
    print(f"device={DEVICE} streams={len(streams)} failed_streams={len(failed_streams)}", flush=True)

    summary = {"seeds": {}}
    for seed in SEEDS:
        model = train_model(failed_streams, seed)
        # per-axis predicted remaining-life score on the monitor segment
        axes = []
        for s in streams:
            x = s["x"]
            z = zscore(x)
            with torch.no_grad():
                pred = model(torch.tensor(z, dtype=torch.float32, device=DEVICE).unsqueeze(0))[0]
            pred = pred.cpu().numpy()[60:]
            axes.append(
                {
                    "bearing_id": (
                        f"ims_{s['test']}_b{s['bearing']}" if s["dataset"] == "ims"
                        else f"{s['dataset']}_{s['bearing']}"
                    ),
                    "dataset": s["dataset"],
                    "axis": s["axis"],
                    "failed": s["failed"],
                    "score": pred,
                    "n_monitor": int(pred.size),
                }
            )
        by_bearing: dict[str, dict] = {}
        for a in axes:
            b = by_bearing.setdefault(
                a["bearing_id"],
                {"bearing_id": a["bearing_id"], "dataset": a["dataset"], "failed": a["failed"],
                 "n_monitor": a["n_monitor"], "axes": {}},
            )
            b["axes"][a["axis"]] = a["score"]
            b["n_monitor"] = max(b["n_monitor"], a["n_monitor"])
        censored = [b for b in by_bearing.values() if not b["failed"]]
        failed = [b for b in by_bearing.values() if b["failed"]]
        min_censored = min(np.min(a) for b in censored for a in b["axes"].values())
        theta = float(min_censored)  # alarm when predicted RUL < theta -> 0 FA by construction
        total_det = 0
        per_ds = {}
        for ds in sorted({b["dataset"] for b in failed}):
            subd = [b for b in failed if b["dataset"] == ds]
            det = 0
            leads = []
            for b in subd:
                first = min(alarm_below(a, theta) for a in b["axes"].values())
                if first < b["n_monitor"]:
                    det += 1
                    leads.append(100.0 * (b["n_monitor"] - first) / b["n_monitor"])
            total_det += det
            per_ds[ds] = {
                "n": len(subd),
                "n_detected": det,
                "detection_rate": 100.0 * det / len(subd),
                "median_lead_pct_life": float(np.median(leads)) if leads else None,
            }
        summary["seeds"][str(seed)] = {
            "rul_threshold_theta": theta,
            "total_detected": total_det,
            "total_n": int(len(failed)),
            "per_dataset": per_ds,
        }
        print(f"seed={seed} total_detected={total_det}/{len(failed)} theta={theta:.4f}", flush=True)

    dst = ROOT / "results" / "results_run_to_failure_rul_baseline.json"
    dst.write_text(json.dumps(summary, indent=2, default=float))
    print("saved", dst)


if __name__ == "__main__":
    main()
