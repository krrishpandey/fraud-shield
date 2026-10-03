"""Account mix features (T2) and rule flags (T3 drop pattern, T6 under-declared parcel)."""
import math

import numpy as np
import pandas as pd

from fraudshield.features import featurize
from fraudshield.features.mix import account_mix_frame, rule_flags
from fraudshield.features.store import FeatureStore
from tests.features.helpers import bk, history_frame


def _store(rows):
    st = FeatureStore()
    for r in rows:
        st.append(r)
    return st


def test_account_mix_uses_only_earlier_bookings():
    rows = history_frame(6)  # category home: not high value; RJ/MG from SP
    rows[2] = bk("h002", "acc1", rows[2].booked_at, category="computers")  # a high-value category
    cur = bk("cur", "acc1", "2018-03-01T00:00:00", category="computers")
    v = featurize(cur, _store(rows)).values
    assert v["acct_hv_share"] == 1 / 6
    assert 0 < v["acct_mean_dist"] < 2000 and 0 <= v["acct_far_share"] <= 1


def test_account_mix_is_missing_for_a_first_booking():
    v = featurize(bk("cur", "accX", "2018-03-01T00:00:00"), _store([])).values
    assert all(math.isnan(v[k]) for k in ("acct_hv_share", "acct_far_share", "acct_mean_dist"))


def test_frame_version_equals_the_live_version_including_same_second_bookings():
    rows = history_frame(12)
    rows += [bk("t1", "acc1", "2018-02-01T00:00:00", category="computers", dest_uf="AM", dest_zip3="690"),
             bk("t2", "acc1", "2018-02-01T00:00:00", dest_uf="RS", dest_zip3="900")]  # same second as t1
    st = _store(rows)
    live = {r.booking_id: featurize(r, st).values for r in rows}
    frame = pd.DataFrame([{**featurize(r, st).values, "booking_id": r.booking_id, "account_id": r.account_id,
                           "booked_at": pd.Timestamp(r.booked_at)} for r in rows])
    off = account_mix_frame(frame).set_index("booking_id")
    for bid, v in live.items():
        for k in ("acct_hv_share", "acct_far_share", "acct_mean_dist"):
            a, b = v[k], off.loc[bid, k]
            assert (math.isnan(a) and math.isnan(b)) or abs(a - b) < 1e-9, (bid, k, a, b)
    assert off.loc["t1", "acct_hv_share"] == off.loc["t2", "acct_hv_share"]  # t2 does not see t1


def test_rule_flags():
    drop = {"consignee_first_seen_days": 10.0, "consignee_bookings_30d": 4, "consignee_other_accts_30d": 2,
            "dims_z": 0.0, "weight_z": 0.0, "n_prior": 50}
    assert rule_flags(drop) == {"drop_pattern": 1, "under_declared": 0}
    old_receiver = {**drop, "consignee_first_seen_days": 200.0}
    assert rule_flags(old_receiver)["drop_pattern"] == 0
    small = {**drop, "consignee_bookings_30d": 0, "dims_z": -3.5, "weight_z": -1.2}
    assert rule_flags(small) == {"drop_pattern": 0, "under_declared": 1}
    new_acct = {**small, "n_prior": 2}  # too little history to know the account's usual parcel
    assert rule_flags(new_acct)["under_declared"] == 0
    assert rule_flags({**small, "dims_z": float("nan")})["under_declared"] == 0


def test_live_featurize_returns_the_flags():
    v = featurize(bk("cur", "acc1", "2018-03-01T00:00:00"), _store(history_frame(8))).values
    assert v["drop_pattern"] in (0, 1) and v["under_declared"] in (0, 1)
    assert np.isfinite(v["acct_mean_dist"])


def test_under_score_live_and_frame_agree():
    from fraudshield.features.mix import under_score, under_score_frame
    rows = [{"dims_z": -2.0, "weight_z": -1.0, "n_prior": 10}, {"dims_z": 1.0, "weight_z": 0.5, "n_prior": 10},
            {"dims_z": -4.0, "weight_z": -2.0, "n_prior": 3}, {"dims_z": float("nan"), "weight_z": -1.0, "n_prior": 20}]
    assert under_score(rows[0]) == 3.0 and under_score(rows[1]) == -1.5
    assert math.isnan(under_score(rows[2])) and math.isnan(under_score(rows[3]))  # too little history / unknown
    fr = under_score_frame(pd.DataFrame(rows))
    assert fr[0] == 3.0 and fr[1] == -1.5 and math.isnan(fr[2]) and math.isnan(fr[3])
