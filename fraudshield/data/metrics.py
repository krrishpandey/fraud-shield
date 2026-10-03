"""Evaluation metrics shared by GBM results and sanity checks."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def pr_auc(y, s) -> float:
    y = np.asarray(y).astype(bool)
    if y.sum() == 0 or y.all():
        return float("nan")
    return float(average_precision_score(y, np.asarray(s, dtype=float)))


def roc_auc(y, s) -> float:
    y = np.asarray(y).astype(bool)
    if y.sum() == 0 or y.all():
        return float("nan")
    return float(roc_auc_score(y, np.asarray(s, dtype=float)))


def flag_top_per_day(ts, score, frac: float = 0.01) -> np.ndarray:
    """Flag the top `frac` of bookings by score within each calendar day (at least 1 per day)."""
    day = pd.to_datetime(pd.Series(ts)).dt.floor("D").to_numpy()
    score = np.asarray(score, dtype=float)
    flags = np.zeros(len(score), dtype=bool)
    order = np.lexsort((-score, day))
    d_sorted = day[order]
    starts = np.r_[0, np.flatnonzero(d_sorted[1:] != d_sorted[:-1]) + 1, len(order)]
    for a, b in zip(starts[:-1], starts[1:]):
        k = max(1, int(round(frac * (b - a))))
        flags[order[a:a + k]] = True
    return flags


def precision_recall_at_top_per_day(ts, score, y, frac: float = 0.01) -> tuple[float, float]:
    f = flag_top_per_day(ts, score, frac)
    y = np.asarray(y).astype(bool)
    return float(y[f].mean()) if f.any() else float("nan"), float(f[y].mean()) if y.any() else float("nan")
