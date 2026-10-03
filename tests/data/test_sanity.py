import numpy as np
import pandas as pd

from fraudshield.data.sanity import artifact_auc, artifact_frame


def _rows(n, rng, shift=0.0):
    t = pd.Timestamp("2018-01-01") + pd.to_timedelta(rng.uniform(0, 100, n), unit="D")
    return pd.DataFrame({
        "weight_kg": rng.lognormal(0, 1, n) + shift, "length_cm": rng.uniform(10, 50, n), "width_cm": 20.0,
        "height_cm": 10.0, "declared_value": rng.lognormal(4, 1, n), "carrier_cost": rng.lognormal(3, 0.5, n).round(2),
        "distance_km": rng.uniform(0, 3000, n), "n_items": 1, "category": "home", "service": "standard",
        "booked_at": t, "first_scan_at": t + pd.to_timedelta(rng.uniform(10, 80, n), unit="h"),
    })


def test_artifact_auc_detects_shift_only_when_present():
    rng = np.random.default_rng(0)
    a, b = _rows(1500, rng), _rows(1500, rng)
    coef = np.array([9.0, 2.9, 11.5])
    assert artifact_auc(artifact_frame(a, coef), artifact_frame(b, coef)) < 0.6
    c = _rows(1500, rng, shift=3.0)
    assert artifact_auc(artifact_frame(a, coef), artifact_frame(c, coef)) > 0.8


def test_grouped_cv_does_not_reward_memorizing_a_group():
    rng = np.random.default_rng(1)
    a = _rows(1500, rng)
    b = _rows(1500, rng)
    # side a: 30 groups, each repeating one exact weight (like one seller's catalogue)
    ga = np.repeat(np.arange(30), 50)
    b_w = b.weight_kg.to_numpy().copy()
    a["weight_kg"] = rng.choice(b_w, 30)[ga]
    pred = np.full(len(a), 20.0)
    fa, fb = artifact_frame(a, pred=pred), artifact_frame(b, pred=np.full(len(b), 20.0))
    plain = artifact_auc(fa[["weight_kg"]], fb[["weight_kg"]])
    grouped = artifact_auc(fa[["weight_kg"]], fb[["weight_kg"]], groups_a=ga, groups_b=np.arange(len(b)) % 300 + 1000)
    assert grouped < plain and grouped < 0.65


def test_residual_scaled_by_lane_spread():
    rng = np.random.default_rng(2)
    a = _rows(10, rng)
    f = artifact_frame(a, pred=np.full(10, 10.0), scale=np.full(10, 2.0))
    np.testing.assert_allclose(f.freight_residual, (a.carrier_cost - 10.0) / 2.0)
