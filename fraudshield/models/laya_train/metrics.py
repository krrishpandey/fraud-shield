"""Weighted evaluation metrics for the Laya vs GBM comparison (legit rows are subsampled and carry
weights back to the true prevalence; fraud rows have weight 1)."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def _ok(y) -> bool:
    y = np.asarray(y).astype(bool)
    return 0 < y.sum() < len(y)


def pr_auc_w(y, s, w=None) -> float:
    if not _ok(y):
        return float("nan")
    return float(average_precision_score(np.asarray(y).astype(bool), np.asarray(s, float), sample_weight=w))


def roc_auc_w(y, s, w=None) -> float:
    if not _ok(y):
        return float("nan")
    return float(roc_auc_score(np.asarray(y).astype(bool), np.asarray(s, float), sample_weight=w))


def flag_top_per_day_w(ts, score, w, frac: float = 0.01) -> np.ndarray:
    """Flag the top `frac` of each day's booking WEIGHT by score (at least one booking per day).
    With unit weights this equals fraudshield.data.metrics.flag_top_per_day."""
    day = pd.to_datetime(pd.Series(ts)).dt.floor("D").to_numpy()
    score = np.asarray(score, float)
    w = np.asarray(w, float)
    flags = np.zeros(len(score), dtype=bool)
    order = np.lexsort((-score, day))
    d_sorted = day[order]
    starts = np.r_[0, np.flatnonzero(d_sorted[1:] != d_sorted[:-1]) + 1, len(order)]
    for a, b in zip(starts[:-1], starts[1:]):
        idx = order[a:b]
        budget = max(1.0, round(frac * w[idx].sum()))
        cum_before = np.r_[0.0, np.cumsum(w[idx])[:-1]]
        sel = cum_before < budget - 1e-9
        sel[0] = True
        flags[idx[sel]] = True
    return flags


def threshold_at_friction(s_legit, w_legit, fpr: float) -> float:
    """Score threshold so that the weighted share of legit bookings with s >= thr is about fpr."""
    s = np.asarray(s_legit, float)
    w = np.asarray(w_legit, float)
    o = np.argsort(-s, kind="mergesort")
    cw = np.cumsum(w[o]) / w.sum()
    k = int(np.searchsorted(cw, fpr - 1e-12, side="left"))
    k = min(k, len(s) - 1)
    return float(s[o][k])


def recall_at_fpr(y, s, w, fpr: float) -> float:
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, float)
    w = np.asarray(w, float)
    if not y.any():
        return float("nan")
    thr = threshold_at_friction(s[~y], w[~y], fpr)
    return float((s[y] >= thr).mean())


def bootstrap_ci(stat: Callable[[np.ndarray], float], y, clusters, B: int = 1000, seed: int = 0,
                 level: float = 0.95) -> tuple[float, float]:
    """Percentile CI. Fraud rows are resampled by cluster (campaign); legit rows (y == 0) row-wise."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(bool)
    clusters = np.asarray(clusters)
    legit = np.flatnonzero(~y)
    groups = [np.flatnonzero(y & (clusters == c)) for c in pd.unique(clusters[y])]
    vals = []
    for _ in range(B):
        pick = rng.integers(0, len(groups), len(groups)) if groups else []
        f = np.concatenate([groups[i] for i in pick]) if len(pick) else np.array([], int)
        idx = np.concatenate([legit[rng.integers(0, len(legit), len(legit))], f])
        v = stat(idx)
        if not np.isnan(v):
            vals.append(v)
    a = (1 - level) / 2
    return float(np.quantile(vals, a)), float(np.quantile(vals, 1 - a))
