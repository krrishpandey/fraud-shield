"""'Is the injected data too easy?' helpers (DESIGN section 10): artifact classifier."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from fraudshield.data.olist import CATEGORY_VOCAB

ARTIFACT_COLUMNS = ("weight_kg", "length_cm", "width_cm", "height_cm", "declared_value", "carrier_cost",
                    "distance_km", "n_items", "category_code", "express", "tod_seconds", "second", "cost_cents",
                    "freight_residual", "scan_lag_h", "scan_missing")


def artifact_frame(df: pd.DataFrame, freight_coef: np.ndarray | None = None, pred: np.ndarray | None = None,
                   scale: np.ndarray | None = None) -> pd.DataFrame:
    """Columns that have a real counterpart (raw values and non-semantic formatting details).
    freight_residual = cost - pred, where pred is the expected real freight for the lane (bin median)
    or, if only freight_coef is given, the linear lane model."""
    t = pd.to_datetime(df.booked_at)
    lag = (pd.to_datetime(df.first_scan_at) - t).dt.total_seconds() / 3600
    if pred is None:
        pred = freight_coef[0] + freight_coef[1] * df.weight_kg + freight_coef[2] * df.distance_km / 1000
    pred = np.asarray(pred, dtype=float)
    scale = np.ones(len(df)) if scale is None else np.maximum(np.asarray(scale, dtype=float), 1e-6)
    return pd.DataFrame({
        "weight_kg": df.weight_kg.astype(float), "length_cm": df.length_cm.astype(float),
        "width_cm": df.width_cm.astype(float), "height_cm": df.height_cm.astype(float),
        "declared_value": df.declared_value.astype(float), "carrier_cost": df.carrier_cost.astype(float),
        "distance_km": df.distance_km.astype(float), "n_items": df.n_items.astype(float),
        "category_code": df.category.map({c: i for i, c in enumerate(CATEGORY_VOCAB)}).fillna(-1).astype(float),
        "express": (df.service == "express").astype(float),
        "tod_seconds": (t - t.dt.floor("D")).dt.total_seconds(), "second": t.dt.second.astype(float),
        "cost_cents": np.round(df.carrier_cost.astype(float) * 100) % 100,
        "freight_residual": (df.carrier_cost.astype(float).to_numpy() - pred) / scale, "scan_lag_h": lag.to_numpy(),
        "scan_missing": lag.isna().astype(float),
    }).reset_index(drop=True)


def artifact_auc(a: pd.DataFrame, b: pd.DataFrame, seed: int = 0, folds: int = 5,
                 groups_a=None, groups_b=None) -> float:
    """Out-of-fold ROC AUC of a LightGBM separating frame a (label 1) from frame b (label 0).
    With groups (campaign / account ids), folds never split a group, so the classifier cannot win by
    memorizing one campaign's repeated values."""
    X = pd.concat([a, b], ignore_index=True).to_numpy(dtype=float)
    y = np.r_[np.ones(len(a)), np.zeros(len(b))]
    oof = np.zeros(len(y))
    if groups_a is not None:
        groups = np.r_[np.asarray(groups_a).astype(str), np.asarray(groups_b).astype(str)]
        splitter = StratifiedGroupKFold(folds, shuffle=True, random_state=seed).split(X, y, groups)
    else:
        splitter = StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y)
    params = dict(objective="binary", learning_rate=0.05, num_leaves=15, min_child_samples=20, verbose=-1,
                  seed=seed, num_threads=4)
    for tr, te in splitter:
        m = lgb.train(params, lgb.Dataset(X[tr], label=y[tr]), num_boost_round=200)
        oof[te] = m.predict(X[te])
    return float(roc_auc_score(y, oof))
