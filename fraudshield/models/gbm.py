"""Baselines: B0 rules only, B1 LightGBM on per-booking features, B2 LightGBM on all features
(account change + graph). Each in tier R (real columns only) and tier F (adds the synthetic layer).
Calibration: Platt scaling on the calibration window (keeps the ranking, unlike isotonic ties).
Saved as artifacts/models/<system>_<tier>.lgb + <system>_<tier>.json (no pickles).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from fraudshield.features.spec import numeric_features

SYSTEMS = ("B1", "B2")
GBM_VERSION = "gbm-v1"
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_child_samples=20,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=7,
              num_threads=4)
NUM_ROUNDS = 400


def system_features(system: str, tier: str) -> list[str]:
    if system == "B1":
        return numeric_features(tier, groups={"booking"})
    if system == "B2":
        return numeric_features(tier)
    raise ValueError(system)


# B0: the weekly post-delivery rules (phase3_eval_demo section 4.5) applied at booking time
RULES_R = (("new_origins_l10", 3, 2.0), ("payoff_z", 3.0, 1.0), ("consignee_accts_30d", 3, 2.0),
           ("distinct_origins_7d", 3, 1.0))
RULES_F = (("new_senders_l10", 3, 2.0), ("payer_senders_30d", 3, 1.0))


def rules_score(df: pd.DataFrame, tier: str) -> np.ndarray:
    s = np.zeros(len(df))
    for col, thr, w in RULES_R + (RULES_F if tier == "F" else ()):
        s += w * (df[col].fillna(-np.inf).to_numpy() >= thr)
    new_hv = (df["tenure_days"].to_numpy() < 30) & (df["high_value"].to_numpy() >= 1) & (df["n_prior"].to_numpy() >= 4)
    s += 1.0 * new_hv
    if tier == "F":
        s += 1.0 * (df["login_device_age_days"].to_numpy() < 1)
    return s


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class GBMModel:
    def __init__(self, system: str, tier: str):
        self.system, self.tier = system, tier
        self.features = system_features(system, tier)
        self.booster: lgb.Booster | None = None
        self.platt = (1.0, 0.0)
        self.metadata: dict = {}

    def _X(self, df) -> np.ndarray:
        return df[self.features].to_numpy(dtype=float)

    def fit(self, train: pd.DataFrame, y_train, cal: pd.DataFrame, y_cal) -> "GBMModel":
        ds = lgb.Dataset(self._X(train), label=np.asarray(y_train).astype(int), feature_name=self.features,
                         free_raw_data=True)
        self.booster = lgb.train(PARAMS, ds, num_boost_round=NUM_ROUNDS)
        z = _logit(self.predict_raw(cal)).reshape(-1, 1)
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(z, np.asarray(y_cal).astype(int))
        self.platt = (float(lr.coef_[0, 0]), float(lr.intercept_[0]))
        self.metadata = {"version": GBM_VERSION, "system": self.system, "tier": self.tier, "features": self.features,
                         "platt": {"a": self.platt[0], "b": self.platt[1]}, "params": PARAMS, "num_rounds": NUM_ROUNDS,
                         "n_train": int(len(train)), "n_train_pos": int(np.asarray(y_train).sum()),
                         "n_cal": int(len(cal)), "n_cal_pos": int(np.asarray(y_cal).sum())}
        return self

    def predict_raw(self, df) -> np.ndarray:
        return self.booster.predict(self._X(df))

    def predict(self, df) -> np.ndarray:
        a, b = self.platt
        return 1.0 / (1.0 + np.exp(-(a * _logit(self.predict_raw(df)) + b)))

    def score_values(self, values: dict) -> float:
        """Calibrated P(fraud) for one FeatureVector.values dict (NaN-safe)."""
        row = pd.DataFrame([{f: (np.nan if values.get(f) is None else values.get(f)) for f in self.features}])
        return float(self.predict(row)[0])

    def save(self, directory: str | Path, extra: dict | None = None) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        name = f"{self.system}_{self.tier}"
        self.booster.save_model(str(d / f"{name}.lgb"))
        meta = {**self.metadata, **(extra or {})}
        (d / f"{name}.json").write_text(json.dumps(meta, indent=2, default=str))

    @classmethod
    def load(cls, directory: str | Path, system: str, tier: str) -> "GBMModel":
        d = Path(directory)
        m = cls(system, tier)
        m.booster = lgb.Booster(model_file=str(d / f"{system}_{tier}.lgb"))
        m.metadata = json.loads((d / f"{system}_{tier}.json").read_text())
        m.features = m.metadata["features"]
        m.platt = (m.metadata["platt"]["a"], m.metadata["platt"]["b"])
        return m
