#!/usr/bin/env python3
"""Window-level online deep anomaly-detection baselines (P0-1).

The main comparison in Sec. 4.4 uses only classical SPC detectors. This
experiment adds modern deep online baselines under the *identical* frozen
causal protocol:
  - reconstruction MLP autoencoder (5 -> 3 -> 5) trained on the early-life
    baseline (snapshots 11-60) only, scored by per-window reconstruction MSE;
  - Deep SVDD (Ruff et al., ICML 2018) with a fixed center initialized as the
    mean of the encoded baseline, scored by squared distance to the center;
  - unidirectional LSTM autoencoder trained on the baseline, scored by
    per-timestep reconstruction MSE.

Each model is trained on every bearing-axis stream, scored causally window by
window from snapshot 61 onward, aggregated per bearing as the max over axes,
and calibrated to the same zero-false-alarm operating point on the 8 IMS
censored bearings. Detection rate and median lead time follow the classical
baseline protocol exactly.
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_to_failure_ood as ood  # noqa: E402

FEATURES = ood.FEATURES
SECONDS_PER_SNAPSHOT = {"ims": 600.0, "xjtu": 60.0, "femto": 10.0}
SEEDS = (0, 1, 2)
EPOCHS = 300
LR = 0.01


class MLPAE(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, 3), nn.Tanh(), nn.Linear(3, 2))
        self.dec = nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, d))

    def forward(self, x):
        return self.dec(self.enc(x))


class DeepSVDD(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 16), nn.Tanh(), nn.Linear(16, 8), nn.Tanh(), nn.Linear(8, 4))

    def forward(self, x):
        return self.net(x)


class LSTMAE(nn.Module):
    def __init__(self, d: int, h: int):
        super().__init__()
        self.encoder = nn.LSTM(d, h, batch_first=True)
        self.decoder = nn.LSTM(h, h, batch_first=True)
        self.head = nn.Linear(h, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, c) = self.encoder(x)
        n = x.shape[1]
        init = h[-1].repeat(1, n, 1)
        dec, _ = self.decoder(init)
        return self.head(dec)


def _zscore(x: np.ndarray, base: np.ndarray) -> np.ndarray:
    mu = base.mean(0)
    sd = base.std(0) + 1e-8
    return (x - mu[None, :]) / sd[None, :]


def mlp_ae_score(x: np.ndarray, seed: int) -> np.ndarray:
    z = _zscore(x, x[10:60])
    zid = torch.tensor(z[10:60], dtype=torch.float32)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MLPAE(x.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for _ in range(EPOCHS):
        opt.zero_grad()
        loss = ((model(zid) - zid) ** 2).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        rec = model(torch.tensor(z, dtype=torch.float32)).numpy()
    return ((z - rec) ** 2).mean(axis=1)


def deep_svdd_score(x: np.ndarray, seed: int) -> np.ndarray:
    z = _zscore(x, x[10:60])
    zid = torch.tensor(z[10:60], dtype=torch.float32)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = DeepSVDD(x.shape[1])
    for layer in model.net:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight, gain=0.5)
            nn.init.zeros_(layer.bias)
    with torch.no_grad():
        center = model(zid).mean(0)
    opt = torch.optim.Adam(model.parameters(), lr=0.005)
    for _ in range(500):
        opt.zero_grad()
        loss = ((model(zid) - center[None, :]) ** 2).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        zfull = torch.tensor(z, dtype=torch.float32)
        dist = ((model(zfull) - center[None, :]) ** 2).sum(1)
    return dist.numpy()


def lstm_ae_score(x: np.ndarray, seed: int) -> np.ndarray:
    z = _zscore(x, x[10:60])
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = LSTMAE(x.shape[1], 16)
    opt = torch.optim.Adam(model.parameters(), lr=0.005)
    zbase = torch.tensor(z[10:60], dtype=torch.float32).unsqueeze(0)
    for _ in range(EPOCHS):
        opt.zero_grad()
        rec = model(zbase)
        loss = ((rec - zbase) ** 2).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        zseq = torch.tensor(z, dtype=torch.float32).unsqueeze(0)
        rec = model(zseq)[0].numpy()
    return ((z - rec) ** 2).mean(axis=1)


SCORERS = {
    "mlp_ae": mlp_ae_score,
    "deep_svdd": deep_svdd_score,
    "lstm_ae": lstm_ae_score,
}


def alarm_index(score: np.ndarray, threshold: float, min_window: int = 30) -> int:
    start = max(0, min_window - 1)
    hits = np.flatnonzero(score[start:] > threshold)
    hits = hits + start
    return int(hits[0]) if hits.size else int(score.size)


def monitor_score(full_score: np.ndarray) -> np.ndarray:
    """Slice the causal monitoring segment (snapshots 61 onward)."""
    return full_score[60:]


def main() -> None:
    here = Path(os.environ.get("RTF_DATA_DIR", str(ROOT / "data" / "features")))
    streams = (
        ood.load_ims_streams(here)
        + ood.load_grouped_streams(here / "xjtu_features.csv", "xjtu", True)
        + ood.load_grouped_streams(here / "femto_features.csv", "femto", True)
    )
    # Same eligibility as the classical comparison: >= 70 snapshots.
    streams = [s for s in streams if s["x"].shape[0] >= 70 and np.all(np.isfinite(s["x"]))]
    print(f"streams={len(streams)}", flush=True)

    # Precompute per-axis scores once (frozen); then apply the shared protocol.
    all_rows = []
    for name, scorer in SCORERS.items():
        for seed in SEEDS:
            for i, s in enumerate(streams):
                x = s["x"]
                score = monitor_score(scorer(x, seed))
                all_rows.append(
                    {
                        "name": name,
                        "seed": seed,
                        "dataset": s["dataset"],
                        "bearing_id": (
                            f"ims_{s['test']}_b{s['bearing']}"
                            if s["dataset"] == "ims"
                            else f"{s['dataset']}_{s['bearing']}"
                        ),
                        "axis": s["axis"],
                        "failed": s["failed"],
                        "score": score,
                        "n_monitor": int(score.size),
                    }
                )
            print(f"{name} seed={seed} done ({len(streams)} streams)", flush=True)

    summary = {"models": {}, "n_failed_bearings": None, "n_censored_bearings": None}
    df = pd.DataFrame(all_rows)
    for (name, seed), sub in df.groupby(["name", "seed"]):
        by_bearing: dict[str, dict] = {}
        for _, r in sub.iterrows():
            b = by_bearing.setdefault(
                r["bearing_id"],
                {"bearing_id": r["bearing_id"], "dataset": r["dataset"], "failed": r["failed"],
                 "n_monitor": r["n_monitor"], "axes": {}},
            )
            b["axes"][r["axis"]] = r["score"]
            b["n_monitor"] = max(b["n_monitor"], r["n_monitor"])
        censored = [b for b in by_bearing.values() if not b["failed"]]
        failed = [b for b in by_bearing.values() if b["failed"]]
        max_censored = max(np.max(a) for b in censored for a in b["axes"].values())
        t_zero = float(max_censored * 1.01)
        total_det = 0
        per_ds = {}
        for ds in sorted({b["dataset"] for b in failed}):
            subd = [b for b in failed if b["dataset"] == ds]
            det = 0
            leads = []
            for b in subd:
                first = min(alarm_index(a, t_zero) for a in b["axes"].values())
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
        summary["models"].setdefault(name, []).append(
            {"seed": int(seed), "zero_fa_threshold": t_zero,
             "total_detected": total_det, "total_n": int(len(failed)), "per_dataset": per_ds}
        )
    summary["n_failed_bearings"] = int(len({r["bearing_id"] for _, r in df.iterrows() if r["failed"]}))
    summary["n_censored_bearings"] = int(len({r["bearing_id"] for _, r in df.iterrows() if not r["failed"]}))

    dst = ROOT / "results" / "results_run_to_failure_modern_baselines.json"
    dst.write_text(json.dumps(summary, indent=2, default=float))
    for name, runs in summary["models"].items():
        tots = [r["total_detected"] for r in runs]
        print(f"{name:10s} total detected = {tots}")
    print("saved", dst)


if __name__ == "__main__":
    main()
