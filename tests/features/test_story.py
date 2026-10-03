"""Account story: the as-of timeline behind 'who is this account paying for?' on the decision page.

Counts use the same definitions as the features (new_*_l10, base_*_l10), so the page, the
explanation and the model input agree: the window is the 10 bookings BEFORE this one, "new" means
the value first appears there, and "usual" is the rate over the 90 days before that window.
"""
import pandas as pd

from fraudshield.features.store import FeatureStore
from fraudshield.features.story import account_story
from tests.features.helpers import bk, history_frame


def _store(rows):
    st = FeatureStore()
    for r in rows:
        st.append(r)
    return st


def _story(rows, current, n=40):
    st = _store(rows)
    return account_story(st.history(current.account_id, current.booked_at), current, n=n)


def test_own_goods_account_has_no_new_senders_but_new_receivers_are_normal():
    rows = history_frame(30)  # sender = account, a fresh consignee every booking, 25 days
    cur = bk("cur", "acc1", "2018-02-01T10:00:00")
    s = _story(rows, cur)
    assert s["n_prior"] == 30
    assert s["last10"] == {"size": 10, "new_senders": 0, "new_receivers": 10}
    # the account's very first booking is its first sender, as in the features
    assert s["usual"] == {"new_senders_per10": 0.5, "new_receivers_per10": 10.0, "based_on": 20}
    last = s["bookings"][-1]
    assert last["booking_id"] == "cur" and last["current"] is True and last["own_goods"] is True


def test_takeover_shows_new_senders_in_last_ten():
    rows = history_frame(30)
    t0 = pd.Timestamp("2018-02-20")
    rows += [bk(f"f{i}", "acc1", (t0 + pd.Timedelta(hours=i)).isoformat(), sender_id=f"stranger{i}")
             for i in range(8)]
    cur = bk("cur", "acc1", "2018-03-01T10:00:00", sender_id="stranger_cur")
    s = _story(rows, cur)
    assert s["last10"]["new_senders"] == 8  # the 10 bookings before this one: 2 own, 8 strangers
    assert s["bookings"][-1]["new_sender"] is True and s["bookings"][-1]["own_goods"] is False


def test_repeat_stranger_sender_is_new_only_the_first_time():
    rows = history_frame(5) + [bk("s1", "acc1", "2018-02-01T00:00:00", sender_id="snd_x")]
    cur = bk("cur", "acc1", "2018-02-02T00:00:00", sender_id="snd_x")
    s = _story(rows, cur)
    flags = {b["booking_id"]: b["new_sender"] for b in s["bookings"]}
    assert flags["s1"] is True and flags["cur"] is False


def test_only_bookings_before_the_current_one_and_last_n_returned():
    rows = history_frame(60)
    cur = bk("cur", "acc1", "2018-01-20T00:00:00")  # in the middle of the history
    s = _story(rows, cur, n=12)
    assert len(s["bookings"]) == 12
    assert all(b["booked_at"] < "2018-01-20" for b in s["bookings"][:-1])
    assert s["n_prior"] < 60


def test_usual_is_none_without_earlier_history_and_ignores_old_bookings():
    s = _story(history_frame(4), bk("cur", "acc1", "2018-01-10T00:00:00"))
    assert s["last10"]["size"] == 4
    assert s["usual"] is None
    old = _story(history_frame(30), bk("cur", "acc1", "2018-09-01T00:00:00"))  # history ended 6 months ago
    assert old["usual"] is None


def test_matches_the_features_on_the_same_history():
    from fraudshield.features.featurize import featurize
    rows = history_frame(30)
    t0 = pd.Timestamp("2018-01-26")
    rows += [bk(f"f{i}", "acc1", (t0 + pd.Timedelta(hours=i)).isoformat(), sender_id=f"stranger{i}")
             for i in range(6)]
    cur = bk("cur", "acc1", "2018-02-01T10:00:00", sender_id="stranger_cur")
    st = _store(rows)
    s = account_story(st.history("acc1", cur.booked_at), cur)
    v = featurize(cur, st).values
    assert s["last10"]["new_senders"] == v["new_senders_l10"]
    assert s["last10"]["new_receivers"] == v["new_consignees_l10"]
    assert s["usual"]["new_senders_per10"] == round(v["base_senders_l10"], 2)
    assert s["usual"]["new_receivers_per10"] == round(v["base_consignees_l10"], 2)


def test_rows_carry_route_and_cost_and_never_meta():
    cur = bk("cur", "acc1", "2018-03-01T10:00:00", origin_uf="PR", origin_zip3="806",
             dest_uf="AM", dest_zip3="690", carrier_cost=72.78)
    b = _story(history_frame(3), cur)["bookings"][-1]
    assert b["origin"] == "PR 806" and b["dest"] == "AM 690" and b["carrier_cost"] == 72.78
    assert "meta" not in b
