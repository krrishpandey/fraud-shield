"""Fit calibration.json for Laya answers (schema in fraudshield/contracts.py; applied by
fraudshield/models/calibration.py).

- temperature per question: softmax(log p / T), fitted by weighted NLL on the calibration split;
- Platt on misuse: sigmoid(a * logit(p_T) + b), weighted (training was enriched to ~20% fraud);
- conformal risk control (Angelopoulos et al., ICLR 2024) on calibrated misuse scores of held-out
  calibration fraud: allow only if p <= lambda_allow so that E[P(allow | fraud)] <= alpha;
- ECE (equal-mass bins) and Brier, weighted, for the before/after report.
numpy/scipy only (no torch).
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize, minimize_scalar

EPS = 1e-6


def sha12(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def temper(probs: np.ndarray, T: float) -> np.ndarray:
    z = np.log(np.clip(probs, EPS, 1.0)) / float(T)
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_temperature(probs: np.ndarray, target: np.ndarray, weights: np.ndarray | None = None,
                    lo: float = 0.05, hi: float = 20.0) -> float:
    """probs, target: [n, k]. Minimises weighted NLL of softmax(log p / T) over log T."""
    probs, target = np.asarray(probs, float), np.asarray(target, float)
    w = np.ones(len(probs)) if weights is None else np.asarray(weights, float)
    w = w / w.sum()
    logp = np.log(np.clip(probs, EPS, 1.0))

    def nll(lt):
        z = logp / np.exp(lt)
        z = z - z.max(1, keepdims=True)
        ls = z - np.log(np.exp(z).sum(1, keepdims=True))
        return -(w * (target * ls).sum(1)).sum()

    r = minimize_scalar(nll, bounds=(np.log(lo), np.log(hi)), method="bounded", options={"xatol": 1e-5})
    return float(np.exp(r.x))


def _logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def platt(p: np.ndarray, a: float, b: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(a * _logit(p) + b)))


def fit_platt(p: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None) -> tuple[float, float]:
    x = _logit(p)
    y = np.asarray(y, float)
    w = np.ones(len(x)) if weights is None else np.asarray(weights, float)
    w = w / w.sum()

    def f(ab):
        z = ab[0] * x + ab[1]
        nll = (w * (np.logaddexp(0, z) - y * z)).sum()
        s = 1 / (1 + np.exp(-z))
        g = w * (s - y)
        return nll, np.array([(g * x).sum(), g.sum()])

    r = minimize(f, np.array([1.0, 0.0]), jac=True, method="L-BFGS-B")
    return float(r.x[0]), float(r.x[1])


def conformal_lambda(fraud_scores: np.ndarray, alpha: float = 0.05) -> float:
    """Largest lambda with (n/(n+1)) * mean(1{s <= lambda}) + 1/(n+1) <= alpha (allow iff p <= lambda).
    Returns 0.0 when even allowing none fails the bound (too few fraud examples)."""
    s = np.sort(np.asarray(fraud_scores, float))
    n = len(s)
    if n == 0:
        return 0.0
    c = int(np.floor(alpha * (n + 1) - 1 + 1e-12))     # max number of allowed fraud
    if c < 1:
        return 0.0
    c = min(c, n)
    lam = s[c - 1]
    while (s <= lam).sum() > c:                          # ties: step down
        smaller = s[s < lam]
        if len(smaller) == 0:
            return 0.0
        lam = smaller[-1]
    return float(lam)


def ece(p: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None, bins: int = 15) -> float:
    """Weighted ECE with equal-mass bins (bins hold equal total weight)."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    w = np.ones(len(p)) if weights is None else np.asarray(weights, float)
    o = np.argsort(p, kind="mergesort")
    p, y, w = p[o], y[o], w[o]
    cw = np.cumsum(w)
    tot = cw[-1]
    edges = np.searchsorted(cw, tot * np.arange(1, bins) / bins, side="left")
    e = 0.0
    for idx in np.split(np.arange(len(p)), edges + 1):
        if len(idx) == 0:
            continue
        ww = w[idx]
        sw = ww.sum()
        e += sw / tot * abs((ww * p[idx]).sum() / sw - (ww * y[idx]).sum() / sw)
    return float(e)


def brier(p: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None) -> float:
    p, y = np.asarray(p, float), np.asarray(y, float)
    w = np.ones(len(p)) if weights is None else np.asarray(weights, float)
    return float((w * (p - y) ** 2).sum() / w.sum())


def calibration_json(weights_path: str | Path, temperatures: dict[str, float], platt_params: dict[str, dict],
                     lambda_allow: float, alpha: float, n_fraud: int, prevalence: float,
                     extra: dict[str, Any] | None = None) -> dict[str, Any]:
    body = {"model_revision": sha12(weights_path),
            "temperatures": {k: float(v) for k, v in temperatures.items()},
            "platt": {k: {"a": float(v["a"]), "b": float(v["b"])} for k, v in platt_params.items()},
            "conformal": {"lambda_allow": float(lambda_allow), "alpha": float(alpha), "n_fraud": int(n_fraud)},
            "prevalence": float(prevalence)}
    h = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:8]
    doc = {"version": f"cal-{time.strftime('%Y%m%d')}-{h}", **body}
    if extra:
        doc["fit"] = extra
    return doc
