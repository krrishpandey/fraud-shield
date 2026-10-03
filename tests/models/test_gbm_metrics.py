import numpy as np
import pandas as pd
import pytest

from fraudshield.features.spec import FEATURES, numeric_features
from fraudshield.models.gbm import SYSTEMS, GBMModel, rules_score, system_features
from fraudshield.data.metrics import flag_top_per_day, pr_auc, precision_recall_at_top_per_day


def toy(n=4000, seed=0, prev=0.03):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.normal(size=n) for c in numeric_features("F")})
    y = rng.random(n) < prev
    df.loc[y, "new_origins_l10"] += 4
    df.loc[y, "payoff_z"] += 3
    df["is_fraud"] = y
    df["booked_at"] = pd.Timestamp("2018-06-01") + pd.to_timedelta(rng.uniform(0, 30, n), unit="D")
    return df


def test_precision_at_top_per_day():
    df = pd.DataFrame({"booked_at": pd.to_datetime(["2018-01-01"] * 4 + ["2018-01-02"] * 4),
                       "score": [0.9, 0.1, 0.2, 0.3, 0.1, 0.8, 0.2, 0.3],
                       "y": [1, 0, 0, 0, 0, 0, 1, 0]})
    flags = flag_top_per_day(df.booked_at, df.score.to_numpy(), frac=0.25)
    assert flags.tolist() == [True, False, False, False, False, True, False, False]
    p, r = precision_recall_at_top_per_day(df.booked_at, df.score.to_numpy(), df.y.to_numpy(), frac=0.25)
    assert p == pytest.approx(0.5) and r == pytest.approx(0.5)


def test_pr_auc_perfect_and_random():
    y = np.array([0, 0, 1, 1])
    assert pr_auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)


def test_system_features_respect_tiers():
    for system in SYSTEMS:
        r = system_features(system, "R")
        assert all(FEATURES[f].tier == "R" for f in r)
        assert set(r) <= set(system_features(system, "F"))
    assert set(system_features("B1", "F")) < set(system_features("B2", "F"))


def test_rules_score_ranks_obvious_fraud():
    df = toy()
    s = rules_score(df, "R")
    assert pr_auc(df.is_fraud, s) > 0.5


def test_gbm_fit_calibrate_save_load(tmp_path):
    tr, cal, te = toy(seed=1), toy(seed=2), toy(seed=3)
    m = GBMModel("B2", "R").fit(tr, tr.is_fraud, cal, cal.is_fraud)
    p = m.predict(te)
    assert ((p >= 0) & (p <= 1)).all()
    assert pr_auc(te.is_fraud, p) > 0.8
    # calibrated mean near prevalence
    assert abs(p.mean() - te.is_fraud.mean()) < 0.02
    m.save(tmp_path)
    m2 = GBMModel.load(tmp_path, "B2", "R")
    np.testing.assert_allclose(m2.predict(te), p)
    assert m2.metadata["features"] == m.features
    # single-booking scoring from a feature dict
    row = te.iloc[0].to_dict()
    assert m2.score_values(row) == pytest.approx(p[0])
