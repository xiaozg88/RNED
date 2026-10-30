#!/usr/bin/env python3
"""Transformer-based online anomaly-detection baselines (P1, GPU).

Closes the reviewer gap on transformer-class detectors under the *same* frozen
causal protocol as the other online baselines: per bearing-axis stream, train
only on snapshots 11-60, score causally from snapshot 61, aggregate per bearing
as the max over axes, and calibrate the zero-false-alarm threshold on the 8 IMS
censored bearings.

Two transformer variants:
  - causal Transformer-AE: a causally-masked transformer encoder with a learned
    start token, trained to reconstruct each snapshot from its strict prefix;
  - TranAD: two-encoder (focus/backbone) + decoder with the two-phase
    reconstruction/adversarial objective of Tuli et al. (VLDB 2022), with
    causal masking so scoring stays online.

Both are scored by per-snapshot reconstruction MSE. Run with the iot-env CUDA
torch build; falls back to CPU if CUDA is unavailable.
"""
from __future__ import annotations

import json
import math
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "experiments"))
import run_to_failure_ood as ood  # noqa: E402

warnings.filterwarnings("ignore", message="Converting mask without torch.bool dtype to bool")

FEATURES = ood.FEATURES
SEEDS = (0, 1, 2)
EPOCHS = 200
LR = 0.005
D_MODEL = 16
NHEAD = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def causal_mask(length: int) -> torch.Tensor:
    return torch.triu(
        torch.ones((length, length), dtype=torch.bool, device=DEVICE), diagonal=1
    )


class CausalTransformerAE(nn.Module):
    def __init__(self, d_in: int):
        super().__init__()
        self.embed = nn.Linear(d_in, D_MODEL)
        self.start = nn.Parameter(torch.randn(1, 1, D_MODEL) * 0.1)
        layer = nn.TransformerEncoderLayer(D_MODEL, NHEAD, dim_feedforward=32, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.head = nn.Linear(D_MODEL, d_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (1, L, d_in); predict token t from tokens [START, 0..t-1].
        e = self.embed(x)
        e = torch.cat([self.start, e[:, :-1, :]], dim=1)
        h = self.encoder(e, causal_mask(e.shape[1]))
        return self.head(h)


class TranAD(nn.Module):
    def __init__(self, d_in: int):
        super().__init__()
        self.embed = nn.Linear(d_in, D_MODEL)
        self.start = nn.Parameter(torch.randn(1, 1, D_MODEL) * 0.1)
        layer = nn.TransformerEncoderLayer(D_MODEL, NHEAD, dim_feedforward=32, batch_first=True)
        self.focus = nn.TransformerEncoder(layer, num_layers=1)
        self.backbone = nn.TransformerEncoder(nn.TransformerEncoderLayer(D_MODEL, NHEAD, dim_feedforward=32, batch_first=True), num_layers=1)
        self.decoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(D_MODEL, NHEAD, dim_feedforward=32, batch_first=True), num_layers=1)
        self.head = nn.Linear(D_MODEL, d_in)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        e = self.embed(x)
        e = torch.cat([self.start, e[:, :-1, :]], dim=1)
        return e

    def reconstruct_focus(self, e: torch.Tensor) -> torch.Tensor:
        m = causal_mask(e.shape[1])
        f = self.focus(e, m)
        return self.head(self.decoder(f, m))

    def reconstruct_backbone(self, e: torch.Tensor) -> torch.Tensor:
        m = causal_mask(e.shape[1])
        b = self.backbone(e, m)
        return self.head(self.decoder(b, m))


def _zscore(x: np.ndarray, base: np.ndarray) -> np.ndarray:
    mu = base.mean(0)
    sd = base.std(0) + 1e-8
    return (x - mu[None, :]) / sd[None, :]


def _causal_chunked_forward(forward, z: np.ndarray, block: int = 128) -> np.ndarray:
    """Evaluate a strictly causal model with bounded prefix attention."""
    outputs = []
    for start in range(0, z.shape[0], block):
        end = min(z.shape[0], start + block)
        prefix = torch.tensor(z[:end], dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            prediction = forward(prefix)[0, -end + start :].cpu().numpy()
        outputs.append(prediction)
    return np.concatenate(outputs, axis=0)


def transformer_ae_score(x: np.ndarray, seed: int) -> np.ndarray:
    z = _zscore(x, x[10:60])
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = CausalTransformerAE(x.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    zid = torch.tensor(z[10:60], dtype=torch.float32, device=DEVICE).unsqueeze(0)
    for _ in range(EPOCHS):
        opt.zero_grad()
        rec = model(zid)
        loss = ((rec - zid) ** 2).mean()
        loss.backward()
        opt.step()
    model.eval()
    rec = _causal_chunked_forward(model, z)
    return ((z - rec) ** 2).mean(axis=1)


def tranad_score(x: np.ndarray, seed: int) -> np.ndarray:
    z = _zscore(x, x[10:60])
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = TranAD(x.shape[1]).to(DEVICE)
    zid = torch.tensor(z[10:60], dtype=torch.float32, device=DEVICE).unsqueeze(0)
    # Phase 1: plain reconstruction for both encoders.
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for _ in range(EPOCHS // 2):
        opt.zero_grad()
        e = model.encode(zid)
        loss = ((model.reconstruct_focus(e) - zid) ** 2).mean() + ((model.reconstruct_backbone(e) - zid) ** 2).mean()
        loss.backward()
        opt.step()
    # Phase 2: adversarial TranAD objective (freeze backbone, focus amplifies errors).
    for p in model.backbone.parameters():
        p.requires_grad_(False)
    dec_params = list(model.embed.parameters()) + list(model.decoder.parameters()) + list(model.head.parameters())
    opt_dec = torch.optim.Adam(dec_params, lr=LR)
    opt_focus = torch.optim.Adam(model.focus.parameters(), lr=LR)
    for _ in range(EPOCHS // 2):
        e = model.encode(zid)
        rec_f = model.reconstruct_focus(e)
        rec_b = model.reconstruct_backbone(e).detach()
        opt_dec.zero_grad()
        loss_dec = ((rec_f - zid) ** 2).mean() + ((rec_b - zid) ** 2).mean()
        loss_dec.backward()
        opt_dec.step()
        # focus adversarial step
        e = model.encode(zid)
        rec_f2 = model.reconstruct_focus(e)
        opt_focus.zero_grad()
        loss_focus = -((rec_f2 - zid) ** 2).mean()
        loss_focus.backward()
        opt_focus.step()
    model.eval()
    rec = _causal_chunked_forward(
        lambda prefix: model.reconstruct_focus(model.encode(prefix)), z
    )
    return ((z - rec) ** 2).mean(axis=1)


SCORERS = {
    "transformer_ae": transformer_ae_score,
    "tranad": tranad_score,
}


def monitor_score(full_score: np.ndarray) -> np.ndarray:
    return full_score[60:]


def alarm_index(score: np.ndarray, threshold: float, min_window: int = 30) -> int:
    start = max(0, min_window - 1)
    hits = np.flatnonzero(score[start:] > threshold)
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
    print(f"device={DEVICE} streams={len(streams)}", flush=True)

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
            print(f"{name} seed={seed} done", flush=True)

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

    dst = ROOT / "results" / "results_run_to_failure_transformer_baselines.json"
    dst.write_text(json.dumps(summary, indent=2, default=float))
    for name, runs in summary["models"].items():
        print(f"{name:14s} total detected = {[r['total_detected'] for r in runs]}")
    print("saved", dst)


if __name__ == "__main__":
    main()
