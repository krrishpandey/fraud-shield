import json

import pandas as pd
import pytest

from fraudshield.contracts import QUESTIONS, TRAINED_QUESTIONS
from fraudshield.models.laya_train import data as D


def test_gold_yes_no_maps_1_to_a_and_0_to_b(make_row):
    g = D.gold_for(make_row(misuse=1, fs=0, pm=1, rl=3, typology="T1"))
    assert g["misuse"] == {"label": "a", "probabilities": {"a": 1.0, "b": 0.0}}
    assert g["foreign_senders"] == {"label": "b", "probabilities": {"a": 0.0, "b": 1.0}}
    assert g["payoff_max"]["label"] == "a"


def test_gold_risk_level_is_one_hot_score(make_row):
    g = D.gold_for(make_row(misuse=1, rl=3, typology="T1"))["risk_level"]
    assert g["label"] == 3 and g["score"] == 3.0
    assert g["probabilities"] == {"0": 0.0, "1": 0.0, "2": 0.0, "3": 1.0, "4": 0.0}


def test_gold_action_is_argmin_cost_with_cost_vector(make_row):
    cost = {"allow": 227.6, "allow_scan_gated": 137.1, "owner_confirm": 25.8, "review": 66.9, "hold": 23.4,
            "block": 0.0}
    g = D.gold_for(make_row(misuse=1, typology="T1", cost=cost))["action"]
    assert g["label"] == "block"
    assert g["cost"] == cost
    assert g["probabilities"]["block"] == 1.0 and sum(g["probabilities"].values()) == 1.0


def test_gold_never_contains_zero_shot_question(make_row):
    g = D.gold_for(make_row(misuse=1, dc=1, typology="T3"))
    assert "drop_consignee" not in g
    assert set(g) == set(TRAINED_QUESTIONS)


def test_gold_can_drop_risk_level(make_row):
    g = D.gold_for(make_row(), qids=("misuse", "foreign_senders", "payoff_max", "action"))
    assert "risk_level" not in g


def test_notebook_record_has_json_string_fields(make_row):
    rec = D.to_notebook_record(make_row(i=3, misuse=1, typology="T2"))
    assert set(["id", "workflow", "state", "questions", "gold"]) <= set(rec)
    assert json.loads(rec["state"]) == "BOOKING 2018-01-01 Mon 10:00 | channel web | row 3"
    q = json.loads(rec["questions"])
    assert list(q) == list(TRAINED_QUESTIONS)
    assert q["misuse"] == QUESTIONS["misuse"]
    assert json.loads(rec["gold"])["misuse"]["label"] == "a"
    # labels never leak into the state; extra fields kept for audit
    assert rec["typology"] == "T2" and rec["split"] == "train"


def _frame(make_row):
    rows = []
    i = 0
    for t, n in [("T1", 40), ("T2", 20), ("T3", 10), ("T5", 10)]:
        for _ in range(n):
            rows.append(make_row(i=i, typology=t, misuse=1)); i += 1
    for _ in range(60):
        rows.append(make_row(i=i, hn=["new_state"])); i += 1
    for _ in range(30):
        rows.append(make_row(i=i, typology="HN_3pl")); i += 1
    for _ in range(300):
        rows.append(make_row(i=i)); i += 1
    for _ in range(50):
        rows.append(make_row(i=i, split="cal", typology="T1", misuse=1)); i += 1
    return pd.DataFrame(rows)


def test_sample_excludes_held_out_typologies_and_non_train(make_row):
    df = _frame(make_row)
    s = D.sample_training_bookings(df, n_hard_neg=50, n_legit=200, fraud_repeat=1, seed=0)
    assert not s.typology.isin(["T3", "T5"]).any()
    assert (s.split == "train").all()


def test_sample_counts_and_fraud_repeat(make_row):
    df = _frame(make_row)
    s = D.sample_training_bookings(df, n_hard_neg=50, n_legit=200, fraud_repeat=3, seed=0)
    fraud = s[s.labels.map(lambda l: l["misuse"]) == 1]
    assert len(fraud) == 60 * 3                       # all eligible train fraud (T1+T2), repeated 3x
    assert fraud.id.nunique() == 60
    legit = s[s.labels.map(lambda l: l["misuse"]) == 0]
    is_hn = legit.hard_negative.map(len).gt(0) | legit.typology.str.startswith("HN_")
    assert is_hn.sum() == 50 and (~is_hn).sum() == 200
    assert legit.id.is_unique


def test_sample_is_deterministic_and_shuffled(make_row):
    df = _frame(make_row)
    a = D.sample_training_bookings(df, 50, 200, 2, seed=1)
    b = D.sample_training_bookings(df, 50, 200, 2, seed=1)
    assert a.id.tolist() == b.id.tolist()
    assert a.labels.map(lambda l: l["misuse"]).iloc[:60].mean() < 1.0   # not sorted by class


def test_sample_caps_at_available(make_row):
    df = _frame(make_row)
    s = D.sample_training_bookings(df, n_hard_neg=10_000, n_legit=10_000, fraud_repeat=1, seed=0)
    assert len(s) == 60 + 90 + 300


def test_build_items_targets_and_no_drop(make_row, tok):
    recs = [D.to_notebook_record(make_row(i=1, misuse=1, rl=2, typology="T1"))]
    items, stats = D.build_items(tok, recs, max_len=512, head_max_len=192)
    assert stats["dropped"] == 0 and stats["truncated"] == 0
    assert len(items) == len(TRAINED_QUESTIONS)
    by = {it["qid"]: it for it in items}
    assert by["misuse"]["target"] == [1.0, 0.0] and by["misuse"]["qtype"] == 0
    assert by["risk_level"]["target"] == [0, 0, 1.0, 0, 0] and by["risk_level"]["qtype"] == 1
    assert len(by["action"]["markers"]) == 6 and len(by["action"]["cost"]) == 6
    assert by["action"]["cost"][0] == 0.0           # allow, in criteria order
    assert all(len(it["ids"]) <= 512 for it in items)
    assert "cost" not in by["misuse"]


def test_build_items_counts_truncation(make_row, tok):
    r = make_row(i=1)
    r["state"] = "word " * 600
    items, stats = D.build_items(tok, [D.to_notebook_record(r)], max_len=512, head_max_len=192)
    assert stats["truncated"] == len(TRAINED_QUESTIONS)
    assert stats["max_state_tokens"] >= 600


def _eval_frame(make_row):
    rows, i = [], 0
    for _ in range(400):
        rows.append(make_row(i=i, split="test")); i += 1
    for _ in range(40):
        rows.append(make_row(i=i, split="test", hn=["new_state"])); i += 1
    for seed in (0, 1):
        for _ in range(10):
            rows.append(make_row(i=i, split="test", typology="T3", misuse=1, seed=seed)); i += 1
        for _ in range(5):
            rows.append(make_row(i=i, split="test", typology="HN_3pl", seed=seed)); i += 1
    for _ in range(30):
        rows.append(make_row(i=i, split="cal")); i += 1
    return pd.DataFrame(rows)


def test_eval_sample_weights_restore_population(make_row):
    df = _eval_frame(make_row)
    s = D.eval_sample(df, "test", n_legit=100, seed=0)
    real = s[~s.is_injected]
    assert (s.split == "test").all()
    assert len(real[real.hard_negative.map(len) == 0]) == 100
    assert real[real.hard_negative.map(len) == 0].weight.sum() == pytest.approx(400)
    assert (real[real.hard_negative.map(len) > 0].weight == 1).all() and (real.hard_negative.map(len) > 0).sum() == 40
    assert (s[s.is_injected].weight == 1).all()
    assert len(s[s.is_injected]) == 30                # both seeds' injected rows
    assert s.id.is_unique


def test_eval_sample_seed_view(make_row):
    df = _eval_frame(make_row)
    s = D.eval_sample(df, "test", n_legit=100, seed=0)
    v1 = D.seed_view(s, 1)
    assert (v1[v1.is_injected].seed == 1).all() and len(v1[v1.is_injected]) == 15
    assert len(v1[~v1.is_injected]) == 140
