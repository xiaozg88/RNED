#!/usr/bin/env python3
"""Shared, dependency-light OOD metrics used by all experiment scripts.

The key convention in this repository is that higher scores mean "more OOD".
In particular, ``fpr_at_95_tpr`` is the false-positive rate at the threshold
where recall first reaches 95%.  It is *not* the recall at a fixed 5% ID FPR.
"""

from __future__ import annotations

import math

import numpy as np


def _validate_binary_scores(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=bool).reshape(-1)
    score = np.asarray(scores, dtype=np.float64).reshape(-1)
    if y.shape != score.shape:
        raise ValueError(f"y_true and scores must have the same shape, got {y.shape} and {score.shape}")
    if not np.all(np.isfinite(score)):
        raise ValueError("scores contain NaN or infinity")
    return y, score


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based AUROC in [0, 1], with average ranks for ties."""
    y, score = _validate_binary_scores(y_true, scores)
    positives = int(y.sum())
    negatives = int((~y).sum())
    if positives == 0 or negatives == 0:
        return float("nan")

    order = np.argsort(score, kind="mergesort")
    sorted_scores = score[order]
    ranks = np.empty(score.size, dtype=np.float64)
    start = 0
    while start < score.size:
        end = start + 1
        while end < score.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    positive_rank_sum = float(ranks[y].sum())
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (
        positives * negatives
    )


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Step-wise average precision in [0, 1]."""
    y, score = _validate_binary_scores(y_true, scores)
    positives = int(y.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    precision = tp / np.arange(1, y_sorted.size + 1)
    recall = tp / positives
    previous_recall = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - previous_recall) * precision))


def fpr_at_95_tpr(y_true: np.ndarray, scores: np.ndarray) -> float:
    """False-positive rate at the threshold where TPR first reaches 95%.

    The implementation includes all samples tied with the threshold score.
    Higher scores are treated as positive (OOD) decisions.
    """
    y, score = _validate_binary_scores(y_true, scores)
    positives = int(y.sum())
    negatives = int((~y).sum())
    if positives == 0 or negatives == 0:
        return float("nan")

    order = np.argsort(-score, kind="mergesort")
    sorted_scores = score[order]
    sorted_y = y[order]
    tp = np.cumsum(sorted_y)
    target = int(math.ceil(0.95 * positives))
    first = int(np.searchsorted(tp, target, side="left"))
    if first >= score.size:
        return 1.0

    threshold = sorted_scores[first]
    # Include every sample tied with the threshold so the metric is deterministic.
    last = int(np.searchsorted(-sorted_scores, -threshold, side="right"))
    false_positives = int(np.sum(~sorted_y[:last]))
    return false_positives / negatives


def tpr_at_fpr(y_true: np.ndarray, scores: np.ndarray, target_fpr: float = 0.05) -> float:
    """Recall at a fixed ID false-positive rate.

    This is an operational metric and must not be labelled FPR95.
    """
    y, score = _validate_binary_scores(y_true, scores)
    positives = int(y.sum())
    negatives = int((~y).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    threshold = float(np.quantile(score[~y], 1.0 - target_fpr, method="higher"))
    return float(np.mean(score[y] > threshold))


def ood_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """Return metrics in percentage units, matching the existing CSV schema."""
    return {
        "AUROC": 100.0 * roc_auc(y_true, scores),
        "AUPR": 100.0 * average_precision(y_true, scores),
        "FPR95": 100.0 * fpr_at_95_tpr(y_true, scores),
    }
