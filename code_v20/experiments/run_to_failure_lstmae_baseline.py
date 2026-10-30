#!/usr/bin/env python3
"""LSTM autoencoder baseline for the degradation OOD task.

Mirrors run_to_failure_ae_baseline.py (same stream loading, ID/OOD masking, and
aggregation) but replaces the per-snapshot MLP AE with a sequence LSTM-AE:
  - the per-stream feature sequence is z-scored by the ID (snapshots 11-60)
    mean/std;
  - a unidirectional LSTM encoder maps the sequence to a hidden state, and a
    LSTM decoder reconstructs the sequence from that state;
  - training minimizes reconstruction MSE on the ID window only;
  - the per-snapshot OOD score is the per-timestep reconstruction MSE
    (mean over features).

This is the standard "train on healthy, score by reconstruction error" deep
time-series baseline that the degradation OOD benchmark previously lacked.
Three seeds are averaged; seed 0 is reported separately for table insertion.
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
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import run_to_failure_ood as ood  # noqa: E402
from evaluation_metrics import fpr_at_95_tpr  # noqa: E402

FEATURES = ood.FEATURES
SEEDS = (0, 1, 2)
HIDDEN = 16
EPOCHS = 300
LR = 0.005
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class LSTMAE(nn.Module):
    """Unidirectional LSTM autoencoder over a (1, n, d) sequence."""

    def __init__(self, d: int, h: int):
        super().__init__()
        self.encoder = nn.LSTM(d, h, batch_first=True)
        self.decoder = nn.LSTM(h, h, batch_first=True)
        self.head = nn.Linear(h, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (1, n, d)
        _, (h, c) = self.encoder(x)          # h, c: (1, 1, h)
        n = x.shape[1]
        init = h[-1].repeat(1, n, 1)         # (1, n, h)
        dec, _ = self.decoder(init)
        return self.head(dec)                 # (1, n, d)


def lstmae_score(x: np.ndarray, seed: int) -> np.ndarray:
    baseline, _ = ood.ids_ood_masks(x.shape[0])
    id_x = x[baseline]
    mu = id_x.mean(0)
    sd = id_x.std(0) + 1e-8
    z = (x - mu[None, :]) / sd[None, :]
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = LSTMAE(x.shape[1], HIDDEN).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    # Train on the ID baseline window only (snapshots 11-60); this is both
    # cheaper (O(50) per epoch instead of O(n)) and protocol-correct: the
    # detector never sees future/OOD data during fitting.
    zbase = torch.tensor(z[baseline], dtype=torch.float32, device=DEVICE).unsqueeze(0)
    for _ in range(EPOCHS):
        opt.zero_grad()
        rec = model(zbase)
        loss = ((rec - zbase) ** 2).mean()
        loss.backward()
        opt.step()
    # Inference: reconstruct the full sequence once; per-timestep MSE is the score.
    with torch.no_grad():
        zseq = torch.tensor(z, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        rec = model(zseq)[0].cpu().numpy()
    return ((z - rec) ** 2).mean(axis=1)


def main() -> None:
    here = Path(os.environ.get("RTF_DATA_DIR", str(ROOT / "data" / "features")))
    streams = (ood.load_ims_streams(here)
               + ood.load_grouped_streams(here / "xjtu_features.csv", "xjtu", True)
               + ood.load_grouped_streams(here / "femto_features.csv", "femto", True))
    streams = [
        s for s in streams
        if s["x"].shape[0] >= 120 and np.all(np.isfinite(s["x"]))
    ]
    streams = [
        s for s in streams
        if (m := ood.ids_ood_masks(s["x"].shape[0]))[0].sum() >= 20 and m[1].sum() >= 5
    ]
    rows = []
    for seed in SEEDS:
        for i, s in enumerate(streams):
            x = s["x"]
            baseline, ood_mask = ood.ids_ood_masks(x.shape[0])
            score = lstmae_score(x, seed)
            y = np.concatenate([np.zeros(baseline.sum()), np.ones(ood_mask.sum())])
            sc = np.concatenate([score[baseline], score[ood_mask]])
            rows.append({
                "seed": seed,
                "name": "lstm_ae",
                **ood.s_meta(s),
                "auroc": 100.0 * roc_auc_score(y, sc),
                "fpr95": 100.0 * fpr_at_95_tpr(y, sc),
            })
            if (i + 1) % 20 == 0:
                print(f"seed={seed} stream={i+1}/{len(streams)}", flush=True)
    df = pd.DataFrame(rows)

    def macro(sub: pd.DataFrame) -> tuple[float, float]:
        a = float(sub.groupby(["dataset", "bearing_id"])["auroc"].mean().mean())
        f = float(sub.groupby(["dataset", "bearing_id"])["fpr95"].mean().mean())
        return a, f

    per_seed = {
        int(seed): {
            "macro_auroc": macro(df[df.seed == seed])[0],
            "macro_fpr95": macro(df[df.seed == seed])[1],
        }
        for seed in SEEDS
    }
    auroc_across = [per_seed[s]["macro_auroc"] for s in SEEDS]
    fpr95_across = [per_seed[s]["macro_fpr95"] for s in SEEDS]
    failed = df[df.failed]
    per_dataset = {}
    for ds, sub in failed.groupby("dataset"):
        per_dataset[ds] = {
            "macro_auroc": macro(sub)[0],
            "macro_fpr95": macro(sub)[1],
        }
    out = {
        "n_streams": int(len(df) // len(SEEDS)),
        "seeds": SEEDS,
        "per_seed": per_seed,
        "macro_auroc_mean": float(np.mean(auroc_across)),
        "macro_auroc_std": float(np.std(auroc_across)),
        "macro_fpr95_mean": float(np.mean(fpr95_across)),
        "macro_fpr95_std": float(np.std(fpr95_across)),
        "seed0": per_seed[0],
        "per_dataset_auroc": {ds: v["macro_auroc"] for ds, v in per_dataset.items()},
        "per_method": {
            "auroc": float(failed.auroc.mean()),
            "fpr95": float(failed.fpr95.mean()),
        },
    }
    dst = ROOT / "results" / "results_run_to_failure_lstmae_baseline.json"
    dst.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print("saved", dst)


if __name__ == "__main__":
    main()
