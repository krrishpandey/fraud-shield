import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from fraudshield.data.metrics import flag_top_per_day
from fraudshield.models.laya_train import metrics as Mx


def _data(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.random(n) < 0.05
    s = rng.normal(0, 1, n) + 1.5 * y
    w = np.where(y, 1.0, rng.integers(1, 5, n)).astype(float)
    return y, s, w


def test_weighted_pr_auc_equals_duplicated_rows():
    y, s, w = _data()
    wi = w.astype(int)
    ref = average_precision_score(np.repeat(y, wi), np.repeat(s, wi))
    assert Mx.pr_auc_w(y, s, w) == pytest.approx(ref, rel=1e-9)


def test_weighted_roc_auc():
    y, s, w = _data()
    assert Mx.roc_auc_w(y, s, w) == pytest.approx(roc_auc_score(y, s, sample_weight=w))


def test_weighted_top_per_day_matches_unweighted_when_w1():
    rng = np.random.default_rng(0)
    ts = pd.Timestamp("2018-01-01") + pd.to_timedelta(rng.integers(0, 5 * 24 * 60, 2000), unit="min")
    s = rng.random(2000)
    f1 = Mx.flag_top_per_day_w(ts, s, np.ones(2000), 0.01)
    f0 = flag_top_per_day(ts, s, 0.01)
    assert (f1 == f0).all()


def test_weighted_top_per_day_counts_weight():
    ts = [pd.Timestamp("2018-01-01 10:00")] * 4
    s = np.array([0.9, 0.8, 0.7, 0.1])
    w = np.array([1.0, 1.0, 1.0, 197.0])          # day weight 200 -> 2 flagged weight units
    f = Mx.flag_top_per_day_w(ts, s, w, 0.01)
    assert f.tolist() == [True, True, False, False]


def test_threshold_at_friction_and_recall():
    s_legit = np.arange(100) / 100.0
    thr = Mx.threshold_at_friction(s_legit, np.ones(100), 0.05)
    assert ((s_legit >= thr).mean()) == pytest.approx(0.05)
    y = np.r_[np.zeros(100, bool), np.ones(4, bool)]
    s = np.r_[s_legit, [0.99, 0.97, 0.5, 0.1]]
    assert Mx.recall_at_fpr(y, s, np.ones(104), 0.05) == pytest.approx(0.5)


def test_cluster_bootstrap_ci_contains_point_and_is_ordered():
    y, s, w = _data(2000)
    clusters = np.where(y, np.arange(2000) % 20, -1)
    lo, hi = Mx.bootstrap_ci(lambda idx: Mx.pr_auc_w(y[idx], s[idx], w[idx]), y, clusters, B=100, seed=0)
    pt = Mx.pr_auc_w(y, s, w)
    assert lo < pt < hi
