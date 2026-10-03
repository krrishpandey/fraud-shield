"""Wiring of the data agent's featurizer/serializer/GBM into the decision service."""
import json
from dataclasses import replace
from pathlib import Path

import pytest

from fraudshield.contracts import Booking, FeatureVector

ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "data/processed/demo_bookings.json"
pytestmark = pytest.mark.skipif(not DEMO.exists(), reason="demo data not built")


@pytest.fixture(scope="module")
def rc():
    from fraudshield.api import real_components
    real_components.reset()
    return real_components


def _demo(scenario: str) -> Booking:
    item = next(x for x in json.loads(DEMO.read_text()) if x["scenario"] == scenario)
    return Booking(**item["booking"])


def test_featurizer_returns_history_features(rc):
    fv = rc.featurizer(_demo("legit-tenured"))
    assert isinstance(fv, FeatureVector)
    assert fv.values["n_prior"] >= 20  # tenured seller has real history in the demo store


def test_featurizer_appends_so_next_booking_sees_it(rc):
    b = _demo("legit-tenured")
    before = rc.featurizer(replace(b, booking_id="probe-1", booked_at="2018-09-01T10:00:00")).values["n_prior"]
    after = rc.featurizer(replace(b, booking_id="probe-2", booked_at="2018-09-01T11:00:00")).values["n_prior"]
    assert after == before + 1


def test_featurizer_does_not_double_append_known_booking(rc):
    b = replace(_demo("legit-tenured"), booking_id="probe-dup", booked_at="2018-09-02T10:00:00")
    rc.featurizer(b)
    n1 = rc.store_size()
    rc.featurizer(b)
    assert rc.store_size() == n1


def test_serializer_matches_training_format_without_gbm_line(rc):
    b = _demo("takeover")
    text = rc.serializer(b, rc.featurizer(b))
    assert text.startswith("BOOKING ")
    assert "GBM risk" not in text  # training states (states.jsonl) have no GBM line


def test_gbm_ranks_takeover_above_legit(rc):
    legit = rc.gbm(rc.featurizer(_demo("legit-tenured")))
    takeover = rc.gbm(rc.featurizer(_demo("takeover")))
    assert 0.0 <= legit <= 1.0 and 0.0 <= takeover <= 1.0
    assert takeover > legit


def test_gbm_version_reported(rc):
    assert rc.gbm_version().startswith("gbm")


def test_warm_loads_store_and_gbm(rc):
    rc.reset()
    rc.warm()
    assert rc._store is not None and rc._gbm_model is not None


def test_account_story_for_takeover_shows_new_senders(rc):
    b = _demo("takeover")
    rc.featurizer(b)
    s = rc.account_story(b)
    assert s["account_id"] == b.account_id and s["bookings"][-1]["booking_id"] == b.booking_id
    assert s["last10"]["new_senders"] >= 5
    legit = _demo("legit-tenured")
    assert rc.account_story(legit)["last10"]["new_senders"] == 0


def test_stream_source_replays_the_seed0_test_window_in_time_order(rc):
    items = rc.stream_source(0)
    assert len(items) == 22925
    ts = [it.booking.booked_at for it in items]
    assert ts == sorted(ts) and ts[0] >= "2018-05-15" and ts[-1] < "2018-09-01"
    assert 0.010 < sum(it.is_fraud for it in items) / len(items) < 0.014
    assert all(it.offline_score is not None for it in items)
    # ground truth rides beside the booking, never inside it
    assert all("is_fraud" not in (it.booking.meta or {}) for it in items[:500])
    with pytest.raises(ValueError, match="seed 0"):
        rc.stream_source(1)


def test_live_scores_match_offline_scores_booking_for_booking(rc):
    """The stream must not change accuracy: live featurizer + model reproduce the offline score per booking."""
    items = rc.stream_source(0)
    sample = items[:120] + [it for it in items if it.is_fraud][:60]
    for it in sample:
        live = rc.gbm(rc.featurizer(it.booking))
        assert live == pytest.approx(it.offline_gbm, abs=1e-9), it.booking.booking_id
