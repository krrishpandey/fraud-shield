import json

import numpy as np
import pandas as pd
import pytest

from fraudshield.data.billing_layer import build_billing_layer
from fraudshield.data.inject import (
    FRAUD_TYPOLOGIES,
    HELD_OUT,
    LABEL_COLUMNS,
    InjectionContext,
    inject,
)
from fraudshield.data.split import assign_split
from tests.conftest import make_bookings


@pytest.fixture(scope="module")
def world():
    b = make_bookings(n_accounts=500, per_account=(20, 90), seed=5, start="2017-01-01", days=610)
    pool = b[["origin_zip5", "origin_uf"]].drop_duplicates().rename(columns={"origin_zip5": "zip5", "origin_uf": "uf"})
    dz = b[["dest_zip5", "dest_uf"]].drop_duplicates().rename(columns={"dest_zip5": "zip5", "dest_uf": "uf"})
    pool = pd.concat([pool, dz]).drop_duplicates("zip5").reset_index(drop=True)
    rng = np.random.default_rng(0)
    pool["lat"] = rng.uniform(-30, -5, len(pool))
    pool["lng"] = rng.uniform(-55, -38, len(pool))
    pool["n_customers"] = 1.0
    pool["n_sellers"] = 1.0
    real = build_billing_layer(b, pool, seed=1).bookings
    real["split"] = assign_split(real.booked_at).to_numpy()
    ctx = InjectionContext.build(real, pool)
    res = inject(ctx, seed=3)
    return real, pool, ctx, res


def test_injected_rows_have_all_label_columns(world):
    real, pool, ctx, res = world
    inj = res.rows
    assert len(inj) > 0
    for c in LABEL_COLUMNS:
        assert c in inj.columns, c
    assert inj.is_injected.all()
    assert set(inj.camouflage_level) <= {0, 1, 2}
    assert (inj.seed == 3).all()
    for p in inj.params_json.head(20):
        json.loads(p)
    assert set(inj.typology) <= set(FRAUD_TYPOLOGIES) | {"HN_new_channel", "HN_3pl"}
    assert set(FRAUD_TYPOLOGIES) <= set(inj.typology)
    assert (inj.is_fraud == inj.typology.isin(FRAUD_TYPOLOGIES)).all()
    assert inj.booking_id.is_unique and not set(inj.booking_id) & set(real.booking_id)


def test_deterministic(world):
    real, pool, ctx, res = world
    again = inject(ctx, seed=3).rows
    pd.testing.assert_frame_equal(res.rows.reset_index(drop=True), again.reset_index(drop=True))
    other = inject(ctx, seed=4).rows
    assert not other.booking_id.isin(res.rows.booking_id).all()


def test_campaigns_stay_inside_one_split_and_held_out_only_in_test(world):
    inj = world[3].rows
    sp = inj.groupby("campaign_id").split.nunique()
    assert (sp == 1).all()
    assert set(inj.split) <= {"train", "cal", "test"}
    assert set(inj[inj.typology.isin(HELD_OUT)].split) == {"test"}
    assert (assign_split(inj.booked_at).to_numpy() == inj.split.to_numpy()).all()


def test_victims_disjoint_across_splits(world):
    real, pool, ctx, res = world
    inj = res.rows
    realacc = set(real.account_id)
    v = inj[inj.account_id.isin(realacc) & inj.is_fraud]
    by = v.groupby("split").account_id.agg(set)
    names = list(by.index)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            assert not (by[names[i]] & by[names[j]])


def test_values_snapped_to_real_grids(world):
    real, pool, ctx, res = world
    inj = res.rows
    zips = set(pool.zip5)
    assert inj.origin_zip5.isin(zips).all() and inj.dest_zip5.isin(zips).all()
    assert (inj.origin_zip3 == inj.origin_zip5.str[:3]).all()
    parcels = set(map(tuple, real[["weight_kg", "length_cm", "width_cm", "height_cm"]].round(3).to_numpy()))
    normal = inj[inj.typology != "T6"]
    tup = list(map(tuple, normal[["weight_kg", "length_cm", "width_cm", "height_cm"]].round(3).to_numpy()))
    assert all(t in parcels for t in tup)
    t6 = inj[inj.typology == "T6"]
    assert (t6.weight_kg < t6.true_weight_kg).all()
    assert (inj.carrier_cost >= 0).all()
    assert set(inj.service) <= {"standard", "express"}


def test_typology_semantics(world):
    inj = world[3].rows
    t3 = inj[inj.typology == "T3"]
    assert t3.label_drop_consignee.all()
    assert (t3.groupby("consignee_id").account_id.nunique() >= 2).mean() > 0.5
    assert not inj[inj.typology != "T3"].label_drop_consignee.any()
    t1 = inj[inj.typology == "T1"]
    assert (t1.sender_id != t1.account_id).all()
    assert inj[inj.typology == "T4"].label_payoff_max.all()
    t2 = inj[inj.typology == "T2"]
    assert (t2.groupby("account_id").booked_at.min() > pd.Timestamp("2017-01-01")).all()
    hn = inj[inj.typology.str.startswith("HN")]
    assert (~hn.is_fraud).all() and (hn.risk_level == 0).all()
    assert inj[inj.is_fraud].risk_level.between(1, 4).all()


def test_prevalence_near_one_percent(world):
    real, pool, ctx, res = world
    inj = res.rows
    for split in ("train", "test"):
        n_real = (real.split == split).sum()
        rate = (inj[inj.split == split].is_fraud).sum() / n_real
        assert 0.004 < rate < 0.025, (split, rate)


def test_confirmed_events_are_delayed(world):
    inj, conf = world[3].rows, world[3].confirmed
    assert conf
    last = inj.groupby("campaign_id").booked_at.max()
    for c in conf[:20]:
        assert c["confirmed_at"] >= last[c["campaign_id"]] + pd.Timedelta(days=29)


def test_legit_3pl_times_follow_real_time_of_day(world):
    real, pool, ctx, res = world
    hn = res.rows[res.rows.typology == "HN_3pl"]
    tod = (hn.booked_at - hn.booked_at.dt.floor("D")).dt.total_seconds()
    assert (tod // 60).isin(set(np.floor(ctx.tod_seconds / 60))).all()


def test_freight_and_timestamps_on_real_grids(world):
    real, pool, ctx, res = world
    inj = res.rows[res.rows.typology != "T6"]
    assert inj.carrier_cost.round(2).isin(set(real.carrier_cost.round(2))).all()
    assert (inj.booked_at.dt.floor("s") == inj.booked_at).all()
    fs = inj.first_scan_at.dropna()
    assert (fs.dt.floor("s") == fs).all()


def test_seconds_of_minute_come_from_real_distribution(world):
    real, pool, ctx, res = world
    saved = ctx.sec_of_min
    try:
        ctx.sec_of_min = np.array([17.0])
        rows = inject(ctx, seed=9).rows
    finally:
        ctx.sec_of_min = saved
    assert (rows.booked_at.dt.second == 17).all()
