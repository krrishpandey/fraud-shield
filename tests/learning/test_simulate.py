import json

from fraudshield.learning.label_store import LabelStore
from fraudshield.learning.simulate import simulate_feedback
from tests.learning.helpers import FEATS, make_pipeline, make_pool, make_reference


def _setup(tmp_path):
    store = LabelStore(tmp_path / "labels.jsonl")
    pipe = make_pipeline(tmp_path, label_store=store)
    pool = make_pool(make_reference()).query("split == 'test'")
    return store, pipe, pool


def test_simulated_labels_are_flagged_and_match_truth(tmp_path):
    store, pipe, pool = _setup(tmp_path)
    out = simulate_feedback(pipe, pool, n=60, seed=1, error_rate=0.0)
    assert out["added"] == 60 and out["fraud"] + out["legit"] == 60 and out["simulated"] is True
    assert out["flipped"] == 0 and "SIMULATED" in out["note"]
    labs = store.all()
    assert len(labs) == 60 and all(x["source"] == "simulated_analyst" and x["simulated"] for x in labs)
    truth = dict(zip(pool.booking_id, pool.is_fraud))
    assert all((x["label"] == "fraud") == bool(truth[x["booking_id"]]) for x in labs)
    # snapshot is the as-of feature row, scored through the pipeline
    row = pool.set_index("booking_id").loc[labs[0]["booking_id"]]
    assert labs[0]["features"]["weight_kg"] == float(row["weight_kg"])
    assert pipe.get_by_booking(labs[0]["booking_id"])["analyst"]["simulated"] is True
    ev = [json.loads(x) for x in (tmp_path / "audit.jsonl").read_text().splitlines()]
    assert sum(e["event_type"] == "analyst_feedback" and e["payload"]["simulated"] for e in ev) == 60


def test_error_rate_flips_and_no_relabel_on_second_call(tmp_path):
    store, pipe, pool = _setup(tmp_path)
    out = simulate_feedback(pipe, pool, n=40, seed=2, error_rate=1.0)
    assert out["flipped"] == 40
    truth = dict(zip(pool.booking_id, pool.is_fraud))
    assert all((x["label"] == "fraud") != bool(truth[x["booking_id"]]) for x in store.all())
    simulate_feedback(pipe, pool, n=40, seed=2, error_rate=0.0)
    assert len({x["booking_id"] for x in store.all()}) == 80


def test_enriched_mix_has_injected_rows(tmp_path):
    store, pipe, pool = _setup(tmp_path)
    out = simulate_feedback(pipe, pool, n=100, seed=3, error_rate=0.0, injected_share=0.5)
    assert out["fraud"] >= 30 and out["sampling"]["injected_share"] == 0.5
