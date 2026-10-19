#!/usr/bin/env python3
"""Reconstruction autoencoder (AE) baseline for the degradation OOD task.

Reuses the exact stream loading / ID-vs-OOD masking of run_to_failure_ood.py.
For each bearing-axis stream an MLP autoencoder (5 -> 3 -> 5, tanh hidden) is
trained on the ID (snapshots 11-60) features only; the reconstruction MSE is
used as the OOD score on all snapshots. Aggregation is identical to the OOD
benchmark (per-bearing mean of per-axis AUROC for the macro, per-stream mean
for per-method).
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
torch.manual_seed(0)
np.random.seed(0)


class AE(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, 3), nn.Tanh(), nn.Linear(3, 2))
        self.dec = nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, d))

    def forward(self, x):
        return self.dec(self.enc(x))


def ae_score(x: np.ndarray) -> np.ndarray:
    baseline, ood_mask = ood.ids_ood_masks(x.shape[0])
    id_x = x[baseline]
    mu = id_x.mean(0)
    sd = id_x.std(0) + 1e-8
    z = (x - mu[None, :]) / sd[None, :]
    zid = torch.tensor(z[baseline], dtype=torch.float32)
    model = AE(x.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(300):
        opt.zero_grad()
        loss = ((model(zid) - zid) ** 2).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        rec = model(torch.tensor(z, dtype=torch.float32)).numpy()
    return ((z - rec) ** 2).mean(axis=1)


def main() -> None:
    here = Path(os.environ.get("RTF_DATA_DIR", str(ROOT / "data" / "features")))
    streams = (ood.load_ims_streams(here)
               + ood.load_grouped_streams(here / "xjtu_features.csv", "xjtu", True)
               + ood.load_grouped_streams(here / "femto_features.csv", "femto", True))
    rows = []
    for s in streams:
        x = s["x"]
        if x.shape[0] < 120 or not np.all(np.isfinite(x)):
            continue
        baseline, ood_mask = ood.ids_ood_masks(x.shape[0])
        if baseline.sum() < 20 or ood_mask.sum() < 5:
            continue
        score = ae_score(x)
        y = np.concatenate([np.zeros(baseline.sum()), np.ones(ood_mask.sum())])
        sc = np.concatenate([score[baseline], score[ood_mask]])
        rows.append({
            "name": "reconstruction_ae",
            **ood.s_meta(s),
            "auroc": 100.0 * roc_auc_score(y, sc),
            "fpr95": 100.0 * fpr_at_95_tpr(y, sc),
        })
    df = pd.DataFrame(rows)
    failed = df[df["failed"]]
    macro_auroc = float(failed.groupby(["dataset", "bearing_id"])["auroc"].mean().mean())
    macro_fpr95 = float(failed.groupby(["dataset", "bearing_id"])["fpr95"].mean().mean())
    per_dataset = {ds: float(sub.groupby("bearing_id")["auroc"].mean().mean())
                   for ds, sub in failed.groupby("dataset")}
    out = {"n_streams": int(len(df)), "macro_auroc": macro_auroc, "macro_fpr95": macro_fpr95,
           "per_dataset_auroc": per_dataset,
           "per_method": {"auroc": float(failed.auroc.mean()), "fpr95": float(failed.fpr95.mean())}}
    dst = ROOT / "results" / "results_run_to_failure_ae_baseline.json"
    dst.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print("saved", dst)


if __name__ == "__main__":
    main()
