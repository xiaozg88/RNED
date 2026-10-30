#!/usr/bin/env python3
"""Run-to-failure degradation OOD benchmark (IMS + XJTU-SY).

Switches the data regime from fixed-condition fault classes to run-to-failure
degradation. For every failed bearing and axis, the early-life baseline
(snapshots 11-60) is ID and the final 15% of life is OOD. Unsupervised
detectors are fitted on ID only and scored on all snapshots.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import OneClassSVM

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from evaluation_metrics import fpr_at_95_tpr

FEATURES = ["rms", "kurtosis_pearson", "skewness", "variance", "peak2peak"]
IMS_FAILED = {("test1", 3), ("test1", 4), ("test2", 1), ("test3", 3)}


def load_ims_streams(csv_dir: Path) -> list[dict]:
    streams = []
    for test in ("test1", "test2", "test3"):
        df = pd.read_csv(csv_dir / f"{test}.csv")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        for bearing in sorted(df["bearing"].unique()):
            for channel in sorted(df[df["bearing"] == bearing]["channel"].unique()):
                sub = df[(df["bearing"] == bearing) & (df["channel"] == channel)]
                sub = sub.sort_values("timestamp")
                streams.append(
                    {
                        "dataset": "ims",
                        "test": test,
                        "bearing": int(bearing),
                        "axis": int(channel),
                        "failed": (test, int(bearing)) in IMS_FAILED,
                        "x": sub[FEATURES].to_numpy(dtype=np.float64),
                    }
                )
    return streams


def load_grouped_streams(csv_path: Path, dataset: str, failed: bool) -> list[dict]:
    df = pd.read_csv(csv_path)
    streams = []
    for (group, bearing) in sorted(
        {(g, b) for g, b in zip(df["group"], df["bearing"])}
    ):
        for axis in sorted(df[(df["group"] == group) & (df["bearing"] == bearing)]["axis"].unique()):
            sub = df[(df["group"] == group) & (df["bearing"] == bearing) & (df["axis"] == axis)]
            sub = sub.sort_values("snapshot")
            streams.append(
                {
                    "dataset": dataset,
                    "group": group,
                    "bearing": bearing,
                    "axis": axis,
                    "failed": failed,
                    "x": sub[FEATURES].to_numpy(dtype=np.float64),
                }
            )
    return streams


def ids_ood_masks(n: int) -> tuple[np.ndarray, np.ndarray]:
    # run-in 1-10 discarded; baseline 11-60 is ID; final 15% of life is OOD.
    baseline = np.zeros(n, dtype=bool)
    baseline[10:60] = True
    ood_start = max(61, int(np.ceil(n * 0.85)))
    ood = np.zeros(n, dtype=bool)
    ood[ood_start:] = True
    return baseline, ood


def score_streams(streams: list[dict]) -> list[dict]:
    rows = []
    for s in streams:
        x = s["x"]
        n = x.shape[0]
        if n < 120 or not np.all(np.isfinite(x)):
            continue
        baseline, ood = ids_ood_masks(n)
        if baseline.sum() < 20 or ood.sum() < 5:
            continue
        id_x = x[baseline]
        ood_x = x[ood]
        y = np.concatenate([np.zeros(id_x.shape[0]), np.ones(ood_x.shape[0])])

        def auc_fpr(score):
            sc = np.concatenate([score[baseline], score[ood]])
            auroc = 100.0 * roc_auc_score(y, sc)
            fpr95 = 100.0 * fpr_at_95_tpr(y, sc)
            return auroc, fpr95

        rms = x[:, 0]
        rms_ratio = rms / np.median(id_x[:, 0])
        a, f = auc_fpr(rms_ratio)
        rows.append({"name": "rms_ratio", **s_meta(s), "auroc": a, "fpr95": f})

        mu = id_x.mean(0)
        cov = np.cov(id_x, rowvar=False)
        cov = (1 - 0.1) * cov + 0.1 * np.trace(cov) / max(1, cov.shape[0]) * np.eye(cov.shape[0])
        inv = np.linalg.pinv(cov)
        diff = x - mu[None, :]
        maha = np.einsum("ni,ij,nj->n", diff, inv, diff)
        a, f = auc_fpr(maha)
        rows.append({"name": "mahalanobis", **s_meta(s), "auroc": a, "fpr95": f})

        centroid = np.linalg.norm(x - mu[None, :], axis=1)
        a, f = auc_fpr(centroid)
        rows.append({"name": "centroid_dist", **s_meta(s), "auroc": a, "fpr95": f})

        knn = NearestNeighbors(n_neighbors=min(5, id_x.shape[0])).fit(id_x)
        knn_dist = knn.kneighbors(x)[0].mean(axis=1)
        a, f = auc_fpr(knn_dist)
        rows.append({"name": "knn5_dist", **s_meta(s), "auroc": a, "fpr95": f})

        for name, model in (
            ("ocsvm", OneClassSVM(kernel="rbf", nu=0.1, gamma="scale")),
            ("isolation_forest", IsolationForest(n_estimators=300, contamination=0.1, random_state=0)),
        ):
            try:
                model.fit(id_x)
                sc = -model.score_samples(x)
                a, f = auc_fpr(sc)
            except Exception:
                a, f = float("nan"), float("nan")
            rows.append({"name": name, **s_meta(s), "auroc": a, "fpr95": f})
    return rows


def s_meta(s: dict) -> dict:
    if s["dataset"] == "ims":
        bearing_id = f"{s['test']}_b{s['bearing']}"
        extra = {"test": s["test"]}
    else:
        bearing_id = str(s["bearing"])
        extra = {"group": s["group"]}
    return {
        "dataset": s["dataset"],
        "bearing": s["bearing"],
        "bearing_id": bearing_id,
        "axis": s["axis"],
        "failed": s["failed"],
        "n": int(s["x"].shape[0]),
        **extra,
    }


def main() -> None:
    here = Path(os.environ.get("RTF_DATA_DIR", str(ROOT / "data" / "features")))
    streams = (
        load_ims_streams(here)
        + load_grouped_streams(here / "xjtu_features.csv", "xjtu", True)
        + load_grouped_streams(here / "femto_features.csv", "femto", True)
    )
    rows = score_streams(streams)
    df = pd.DataFrame(rows)
    out = ROOT / "results" / "results_run_to_failure_ood.csv"
    df.to_csv(out, index=False)

    failed = df[df["failed"]]
    censored = df[~df["failed"]]
    summary = {}
    for label, sub in (("failed_bearings", failed), ("censored_control", censored)):
        by_bearing = sub.groupby(["dataset", "bearing_id"])["auroc"].mean()
        summary[label] = {
            "n_bearings": int(by_bearing.size),
            "macro_auroc": float(by_bearing.mean()),
            "macro_fpr95": float(sub.groupby(["dataset", "bearing_id"])["fpr95"].mean().mean()),
        }
    summary["per_dataset"] = {
        ds: {
            "n_bearings": int(sub.groupby("bearing_id")["auroc"].mean().size),
            "macro_auroc": float(sub.groupby("bearing_id")["auroc"].mean().mean()),
            "macro_fpr95": float(sub.groupby("bearing_id")["fpr95"].mean().mean()),
        }
        for ds, sub in failed.groupby("dataset")
    }
    summary["per_method"] = {
        name: {"auroc": float(sub.auroc.mean()), "fpr95": float(sub.fpr95.mean())}
        for name, sub in failed.groupby("name")
    }
    meta = ROOT / "results" / "results_run_to_failure_ood_summary.json"
    meta.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("saved", out, meta)


if __name__ == "__main__":
    main()
