import numpy as np
import pandas as pd

from fraudshield.data.billing_layer import SYNTHETIC_COLUMNS, build_billing_layer
from tests.conftest import make_bookings


def _zip_pool(b):
    return b[["origin_zip5", "origin_uf"]].drop_duplicates().rename(
        columns={"origin_zip5": "zip5", "origin_uf": "uf"}).assign(lat=-23.0, lng=-46.0)


def test_deterministic_for_a_seed(toy_bookings):
    a = build_billing_layer(toy_bookings, _zip_pool(toy_bookings), seed=7)
    b = build_billing_layer(toy_bookings, _zip_pool(toy_bookings), seed=7)
    c = build_billing_layer(toy_bookings, _zip_pool(toy_bookings), seed=8)
    pd.testing.assert_frame_equal(a.bookings, b.bookings)
    assert not a.bookings.device_id.equals(c.bookings.device_id)


def test_synthetic_columns_present_and_valid(toy_bookings):
    out = build_billing_layer(toy_bookings, _zip_pool(toy_bookings), seed=1).bookings
    for col in SYNTHETIC_COLUMNS:
        assert col in out.columns
    assert set(out.channel) <= {"web", "api", "counter"}
    assert set(out.payment_method) <= {"account_billing", "card", "ach"}
    assert (out.login_device_age_days >= 0).all()
    oc = out.owner_contact_age_days.dropna()
    assert (oc >= 0).all() and len(oc) < len(out)  # some accounts have no verified contact
    assert len(out) == len(toy_bookings)


def test_account_created_before_first_booking(toy_bookings):
    res = build_billing_layer(toy_bookings, _zip_pool(toy_bookings), seed=1)
    first = toy_bookings.groupby("account_id").booked_at.min()
    acc = res.accounts.set_index("account_id")
    assert (acc.account_created_at.reindex(first.index) <= first).all()
    n_logins = acc.n_logins
    assert n_logins.between(1, 3).all()


def test_legit_change_rates_and_semantics():
    big = make_bookings(n_accounts=2000, per_account=(25, 40), seed=3)
    res = build_billing_layer(big, _zip_pool(big), seed=11)
    ch = res.changes
    n_est = big.account_id.nunique()
    rate = ch.groupby("kind").account_id.nunique() / n_est
    assert 0.01 <= rate["new_warehouse"] <= 0.03
    assert 0.02 <= rate["new_login"] <= 0.045
    assert 0.035 <= rate["new_instrument"] <= 0.07
    assert 0.003 <= rate["bundled_expansion"] <= 0.02
    assert ch.announced.any() and not ch.announced.all()
    out = res.bookings
    # a new warehouse only changes rows at/after the change date, inside the same state
    wh = ch[ch.kind.isin(["new_warehouse", "bundled_expansion"])]
    for _, c in wh.head(10).iterrows():
        rows = out[out.account_id == c.account_id]
        before = rows[rows.booked_at < c["at"]]
        after = rows[rows.booked_at >= c["at"]]
        assert before.origin_synthetic.sum() == 0
        assert after.origin_synthetic.sum() >= 1
        assert (after.origin_uf == before.origin_uf.iloc[0] if len(before) else True).all()
    # real bookings never carry fraud labels from this layer
    assert "is_injected" not in out.columns
