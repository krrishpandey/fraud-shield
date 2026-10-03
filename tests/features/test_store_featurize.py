import dataclasses
import math

import numpy as np
import pandas as pd
import pytest

from fraudshield.features import FEATURES, FeatureStore, featurize, featurize_frame
from fraudshield.features.store import booking_to_row
from tests.features.helpers import bk, history_frame


def store_with(rows):
    s = FeatureStore()
    for r in rows:
        s.append(r)
    return s


def test_store_history_is_strictly_before_and_sorted():
    rows = history_frame(10)
    s = store_with(reversed(rows))
    h = s.history("acc1", rows[5].booked_at)
    assert list(h.booking_id) == [r.booking_id for r in rows[:5]]
    assert s.history("nobody", rows[5].booked_at).empty


def test_basic_account_features():
    rows = history_frame(30)
    s = store_with(rows)
    t = pd.Timestamp(rows[-1].booked_at) + pd.Timedelta(hours=1)
    b = bk("x1", "acc1", t.isoformat(), origin_uf="PR", origin_zip3="806", dest_uf="AM", dest_zip3="690",
           sender_id="snd_9", carrier_cost=150.0, weight_kg=12.0)
    v = featurize(b, s).values
    assert v["n_prior"] == 30
    assert v["tenure_days"] == pytest.approx((t - pd.Timestamp(rows[0].booked_at)).total_seconds() / 86400)
    assert v["bookings_24h"] == sum(pd.Timestamp(r.booked_at) >= t - pd.Timedelta(hours=24) for r in rows)
    assert v["origin_seen"] == 0 and v["is_home_origin"] == 0 and v["home_origin_zip3"] == "013"
    assert v["sender_differs"] == 1 and v["payer_sender_pair_age_days"] == 0
    assert v["lane_seen"] == 0 and v["dest_region_seen"] == 0
    assert v["cost_vs_median"] > 8 and v["payoff_z"] > 3 and v["weight_z"] > 3
    assert v["distinct_origins_7d"] == 2 and v["new_origins_l10"] == 0


def test_thin_history_gives_nan_profile():
    rows = history_frame(3)
    s = store_with(rows)
    t = pd.Timestamp(rows[-1].booked_at) + pd.Timedelta(hours=1)
    v = featurize(bk("x", "acc1", t.isoformat()), s).values
    assert math.isnan(v["weight_z"]) and math.isnan(v["cost_vs_median"])


def test_consignee_graph_counts_other_accounts():
    rows = history_frame(5)
    t0 = pd.Timestamp(rows[-1].booked_at)
    drops = [bk(f"d{i}", f"other{i}", (t0 + pd.Timedelta(hours=i + 1)).isoformat(), consignee_id="dropX")
             for i in range(4)]
    s = store_with(rows + drops)
    t = t0 + pd.Timedelta(days=1)
    v = featurize(bk("x", "acc1", t.isoformat(), consignee_id="dropX"), s).values
    assert v["consignee_other_accts_30d"] == 4 and v["consignee_accts_30d"] == 5
    assert v["consignee_prior"] == 4
    assert v["consignee_first_seen_days"] == pytest.approx((t - t0 - pd.Timedelta(hours=1)).total_seconds() / 86400)


def test_no_leakage_from_rows_at_or_after_booking_time():
    rows = history_frame(20)
    t = pd.Timestamp(rows[-1].booked_at) + pd.Timedelta(hours=2)
    b = bk("x", "acc1", t.isoformat(), consignee_id="cZ")
    base = featurize(b, store_with(rows)).values
    future = [bk("f1", "acc1", t.isoformat(), consignee_id="cZ", origin_zip3="999"),
              bk("f2", "acc1", (t + pd.Timedelta(hours=1)).isoformat(), consignee_id="cZ"),
              bk("f3", "accB", t.isoformat(), consignee_id="cZ")]
    after = featurize(b, store_with(rows + future)).values
    assert _same(base, after)


def test_meta_and_labels_are_never_read():
    rows = history_frame(20)
    t = pd.Timestamp(rows[-1].booked_at) + pd.Timedelta(hours=2)
    b = bk("x", "acc1", t.isoformat())
    poisoned = [dataclasses.replace(r, meta={"is_injected": True, "typology": "T1", "scenario_id": "T1_1"})
                for r in rows]
    a = featurize(b, store_with(rows)).values
    c = featurize(dataclasses.replace(b, meta={"typology": "T1", "is_injected": True}), store_with(poisoned)).values
    assert _same(a, c)
    assert "meta" not in booking_to_row(b)
    # label columns in a frame are dropped on load
    df = pd.DataFrame([booking_to_row(r) for r in rows]).assign(is_injected=1, typology="T1", label_misuse=1)
    s = FeatureStore.from_frame(df)
    assert not ({"is_injected", "typology", "label_misuse"} & set(s.history("acc1", t.isoformat()).columns))


def test_every_feature_has_a_tier():
    assert FEATURES and all(spec.tier in ("R", "F") for spec in FEATURES.values())


def test_tier_r_features_ignore_synthetic_inputs():
    rows = history_frame(25)
    rng = np.random.default_rng(1)
    t = pd.Timestamp(rows[-1].booked_at) + pd.Timedelta(hours=2)
    b = bk("x", "acc1", t.isoformat())

    def scramble(r):
        return dataclasses.replace(r, channel=str(rng.choice(["web", "api", "counter"])),
                                   login_device_age_days=float(rng.uniform(0, 50)),
                                   payment_method=str(rng.choice(["card", "ach"])),
                                   owner_contact_age_days=None, sender_id=f"snd{rng.integers(5)}")
    s2 = store_with([scramble(r) for r in rows])
    s2.register_change("acc1", "new_login", at=t.isoformat(), announced_at=(t - pd.Timedelta(days=2)).isoformat())
    s2.confirm_fraud("acc1", confirmed_at=(t - pd.Timedelta(days=1)).isoformat(), entities=["cons_h001"])
    a = featurize(b, store_with(rows)).values
    c = featurize(scramble(b), s2).values
    r_names = [n for n, sp in FEATURES.items() if sp.tier == "R"]
    f_names = [n for n, sp in FEATURES.items() if sp.tier == "F"]
    assert _same({k: a[k] for k in r_names}, {k: c[k] for k in r_names})
    assert not _same({k: a[k] for k in f_names}, {k: c[k] for k in f_names})


def test_events_drive_change_and_confirmed_fraud_features():
    rows = history_frame(10)
    t = pd.Timestamp(rows[-1].booked_at) + pd.Timedelta(hours=2)
    s = store_with(rows)
    s.register_change("acc1", "new_warehouse", at=t.isoformat(), announced_at=(t - pd.Timedelta(days=3)).isoformat())
    s.confirm_fraud("accX", confirmed_at=(t - pd.Timedelta(days=1)).isoformat(), entities=["dropQ"])
    s.confirm_fraud("acc1", confirmed_at=(t + pd.Timedelta(days=1)).isoformat(), entities=[])  # future: invisible
    v = featurize(bk("x", "acc1", t.isoformat(), consignee_id="dropQ"), s).values
    assert v["verified_change_30d"] == 1 and v["links_confirmed_fraud"] == 1 and v["prior_confirmed_fraud"] == 0


def test_stream_featurizer_equals_reference(toy_bookings):
    from fraudshield.data.billing_layer import build_billing_layer
    zp = toy_bookings[["origin_zip5", "origin_uf"]].drop_duplicates().rename(
        columns={"origin_zip5": "zip5", "origin_uf": "uf"}).assign(lat=0.0, lng=0.0)
    df = build_billing_layer(toy_bookings, zp, seed=2).bookings
    # shared consignees, same-timestamp ties, foreign senders
    df.loc[df.index[::7], "consignee_id"] = "shared_c"
    df.loc[df.index[5], "booked_at"] = df.loc[df.index[4], "booked_at"]
    df.loc[df.index[10:40:3], "sender_id"] = "snd_x"
    changes = pd.DataFrame([{"account_id": df.account_id.iloc[0], "kind": "new_login",
                             "at": df.booked_at.iloc[200], "announced_at": df.booked_at.iloc[190], "announced": True}])
    confirmed = [{"account_id": df.account_id.iloc[3], "confirmed_at": df.booked_at.iloc[300], "entities": ["shared_c"]}]
    fast = featurize_frame(df, changes=changes, confirmed=confirmed).set_index("booking_id")
    store = FeatureStore.from_frame(df, changes=changes, confirmed=confirmed)
    sample = df.sample(250, random_state=0)
    for _, r in sample.iterrows():
        ref = featurize(store.booking_from_row(r), store).values
        got = fast.loc[r.booking_id].to_dict()
        assert _same(ref, got, keys=FEATURES.keys()), r.booking_id


def _same(a, b, keys=None, tol=1e-6):
    keys = list(keys) if keys is not None else sorted(set(a) | set(b))
    for k in keys:
        x, y = a.get(k), b.get(k)
        if isinstance(x, str) or isinstance(y, str):
            if x != y:
                print("diff", k, x, y)
                return False
            continue
        x, y = float(x), float(y)
        if math.isnan(x) and math.isnan(y):
            continue
        if not math.isclose(x, y, rel_tol=tol, abs_tol=tol):
            print("diff", k, x, y)
            return False
    return True
