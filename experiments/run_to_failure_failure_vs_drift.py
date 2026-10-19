#!/usr/bin/env python3
"""Failure-vs-drift discrimination on the IMS run-to-failure data.

The degradation benchmark showed that both failed and censored bearings move
away from their early-life baseline. This experiment asks whether trajectory
statistics separate failure-bound degradation from benign drift at the bearing
level (4 failed vs 8 censored IMS bearings).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from evaluation_metrics import fpr_at_95_tpr

FEATURES = ["rms", "kurtosis_pearson", "skewness", "variance", "peak2peak"]
IMS_FAILED = {("test1", 3), ("test1", 4), ("test2", 1), ("test3", 3)}


def load_ims_trajectories(csv_dir: Path) -> dict:
    streams = {}
    for test in ("test1", "test2", "test3"):
        df = pd.read_csv(csv_dir / f"{test}.csv")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        for bearing in sorted(df["bearing"].unique()):
            key = (test, int(bearing))
            for channel in sorted(df[df["bearing"] == bearing]["channel"].unique()):
                sub = df[(df["bearing"] == bearing) & (df["channel"] == channel)]
                sub = sub.sort_values("timestamp")
                streams.setdefault(key, []).append(sub[FEATURES].to_numpy(dtype=np.float64))
    return streams


def trajectory_features(x: np.ndarray) -> dict:
    n = x.shape[0]
    base = x[10:60]
    late = x[max(61, int(np.ceil(0.85 * n))):]
    rms = x[:, 0]
    rms_base = float(np.median(base[:, 0]))
    rms_late = float(np.median(late[:, 0])) if late.size else np.nan
    monitor = np.arange(60, n)
    t = monitor - monitor[0] + 1
    slope = float(np.polyfit(t, rms[monitor], 1)[0]) / (rms_base + 1e-12)
    above = rms[monitor] > 1.2 * rms_base
    episodes = 0
    run = 0
    episode_count = 0
    prev = False
    for a in above:
        run = run + 1 if a else 0
        episodes = max(episodes, run)
        if a and not prev:
            episode_count += 1
        prev = a
    frac_above = float(np.mean(above))
    excess = np.maximum(rms[monitor] - 1.2 * rms_base, 0.0)
    auc_above = float(np.sum(excess)) / (rms_base * max(1, monitor.size))
    first_above = float(above.argmax() + 1) / monitor.size if above.any() else 1.0
    peak_time = float(np.argmax(rms[monitor]) + 1) / monitor.size

    def ratio(col):
        b = float(np.median(base[:, col]))
        l = float(np.median(late[:, col])) if late.size else np.nan
        return l / (b + 1e-12)

    return {
        "rms_ratio_final": rms_late / (rms_base + 1e-12),
        "rms_ratio_max": float(np.max(rms)) / (rms_base + 1e-12),
        "rms_slope_per_snapshot": slope,
        "kurtosis_delta": float(np.median(late[:, 1]) - np.median(base[:, 1])) if late.size else np.nan,
        "variance_ratio": ratio(3),
        "peak2peak_ratio": ratio(4),
        "longest_episode_1p2x": episodes,
        "episode_count_1p2x": episode_count,
        "fraction_above_1p2x": frac_above,
        "auc_above_1p2x": auc_above,
        "first_exceedance_time": first_above,
        "peak_time": peak_time,
    }


def main() -> None:
    here = Path(os.environ.get("RTF_DATA_DIR", str(ROOT / "data" / "features")))
    streams = load_ims_trajectories(here)
    rows = []
    for key, axes in sorted(streams.items()):
        test, bearing = key
        per_axis = [trajectory_features(a) for a in axes]
        feats = {}
        for k in per_axis[0]:
            vals = [p[k] for p in per_axis]
            feats[k] = float(np.nanmax(vals))
            feats[f"{k}_min"] = float(np.nanmin(vals))
        rows.append(
            {
                "bearing_id": f"{test}_b{bearing}",
                "failed": key in IMS_FAILED,
                **feats,
            }
        )
    df = pd.DataFrame(rows)
    y = df["failed"].to_numpy(dtype=np.int64)
    feature_cols = [c for c in df.columns if c not in ("bearing_id", "failed")]

    univariate = {}
    for c in feature_cols:
        s = df[c].to_numpy(dtype=np.float64)
        if np.all(np.isfinite(s)) and np.unique(s).size > 1:
            auroc = 100.0 * roc_auc_score(y, s)
            fpr95 = 100.0 * fpr_at_95_tpr(y, s)
            univariate[c] = {"auroc": round(float(auroc), 2), "fpr95": round(float(fpr95), 2)}

    X = df[feature_cols].to_numpy(dtype=np.float64)
    X = np.nan_to_num(X)
    loo = LeaveOneOut()
    for name, model in (
        ("logistic_regression", LogisticRegression(max_iter=1000, C=1.0)),
        ("random_forest", RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=0)),
    ):
        proba = np.zeros(len(df))
        for tr, te in loo.split(X):
            scaler = StandardScaler().fit(X[tr])
            model.fit(scaler.transform(X[tr]), y[tr])
            proba[te] = model.predict_proba(scaler.transform(X[te]))[:, 1]
        auroc = 100.0 * roc_auc_score(y, proba)
        fpr95 = 100.0 * fpr_at_95_tpr(y, proba)
        univariate[name] = {"auroc": round(float(auroc), 2), "fpr95": round(float(fpr95), 2)}

    summary = {
        "n_failed": int(y.sum()),
        "n_censored": int((1 - y).sum()),
        "per_feature_and_model": univariate,
        "per_bearing": {
            r["bearing_id"]: {"failed": bool(r["failed"]), "rms_ratio_final": round(float(r["rms_ratio_final"]), 3)}
            for _, r in df.iterrows()
        },
    }
    out = ROOT / "results" / "results_run_to_failure_failure_vs_drift.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("saved", out)


if __name__ == "__main__":
    main()
