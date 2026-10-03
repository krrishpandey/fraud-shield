"""Label-free performance estimate from calibrated probabilities (CBPE style: Kivimaki et al., "Performance
Estimation in Binary Classification Using Calibrated Confidence", arXiv 2505.05295).

If p is calibrated, P(fraud | p) = p, so before any label arrives:
- expected frauds among the stopped bookings = sum of p over them; expected precision of the stops = mean p
- expected frauds missed among the bookings let through = sum of p over them
- estimated recall = expected caught / (expected caught + expected missed)
This holds only while the calibration holds. A fraud type the model never learned gets a low p, so the estimate
cannot see it: proxy monitors catch covariate drift but miss concept drift that leaves the features unchanged
(Solozobov, "Evidence Sufficiency Under Delayed Ground Truth", arXiv 2604.15740). `psi` is the input-drift
signal: it fires when the score distribution moves, not when the meaning of a score moves.

"Stopped" = owner_confirm, review, hold, block (as in sim/stream.py and docs/LEARNING_GATE.md); everything else
(allow, allow_scan_gated) is "let through". numpy only.
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np

STOPPED = ("owner_confirm", "review", "hold", "block")
NAN = float("nan")


def stopped_mask(actions: Iterable[str]) -> np.ndarray:
    return np.array([a in STOPPED for a in actions], dtype=bool)


def _ratio(a: float, b: float) -> float:
    return float(a / b) if b > 0 else NAN


def estimate(p, stopped) -> dict[str, Any]:
    """Expected precision of the stops, expected frauds caught and missed, estimated recall. No labels used."""
    p, s = np.asarray(p, dtype=float), np.asarray(stopped, dtype=bool)
    caught, missed = float(p[s].sum()), float(p[~s].sum())
    return {"n": int(len(p)), "n_stopped": int(s.sum()), "n_let_through": int((~s).sum()),
            "precision_stopped": float(p[s].mean()) if s.any() else NAN,
            "fraud_caught": caught, "fraud_missed": missed, "recall": _ratio(caught, caught + missed)}


def realized(y, stopped) -> dict[str, Any]:
    """The same quantities from ground truth (only known once labels arrive; in the simulation, from the dataset)."""
    y, s = np.asarray(y, dtype=bool), np.asarray(stopped, dtype=bool)
    caught, missed = int((y & s).sum()), int((y & ~s).sum())
    return {"n": int(len(y)), "n_stopped": int(s.sum()), "n_let_through": int((~s).sum()),
            "precision_stopped": float(y[s].mean()) if s.any() else NAN,
            "fraud_caught": caught, "fraud_missed": missed, "recall": _ratio(caught, caught + missed)}


def period_ids(ts, start, days: int) -> np.ndarray:
    """Index of the `days`-long period (counted from `start`) each timestamp falls in."""
    t = np.asarray(ts, dtype="datetime64[ns]")
    step = np.timedelta64(int(days), "D").astype("timedelta64[ns]")
    return ((t - np.datetime64(start, "ns")) // step).astype(int)


def per_period(p, stopped, ids, y=None) -> list[dict[str, Any]]:
    """estimate (and realized, when y is given) for each period id, in order."""
    p, s, ids = np.asarray(p, dtype=float), np.asarray(stopped, dtype=bool), np.asarray(ids)
    yy = None if y is None else np.asarray(y, dtype=bool)
    out = []
    for k in np.unique(ids):
        m = ids == k
        row = {"period": int(k), "estimated": estimate(p[m], s[m])}
        if yy is not None:
            row["realized"] = realized(yy[m], s[m])
        out.append(row)
    return out


# ---------- input drift: population stability index of the score distribution ----------
def psi_edges(ref, n_bins: int = 10) -> np.ndarray:
    """Interior bin edges at the reference quantiles (duplicates dropped: many scores tie near 0)."""
    q = np.quantile(np.asarray(ref, dtype=float), np.linspace(0, 1, n_bins + 1)[1:-1])
    return np.unique(q)


def shares(x, edges) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    counts = np.bincount(np.searchsorted(edges, x, side="right"), minlength=len(edges) + 1)
    return counts / max(len(x), 1)


def psi_from_shares(ref_share, cur_share, eps: float = 1e-4) -> float:
    r = np.clip(np.asarray(ref_share, dtype=float), eps, None)
    c = np.clip(np.asarray(cur_share, dtype=float), eps, None)
    return float(np.sum((c - r) * np.log(c / r)))


def psi(ref, cur, edges=None, n_bins: int = 10) -> float:
    """PSI of `cur` against `ref`. Rule of thumb: < 0.1 stable, 0.1-0.25 moderate, > 0.25 large shift."""
    e = psi_edges(ref, n_bins) if edges is None else np.asarray(edges, dtype=float)
    return psi_from_shares(shares(ref, e), shares(cur, e))
