import json

from fraudshield.contracts import SERVED_QUESTIONS
from fraudshield.models.laya_client import LayaClient
from fraudshield.models.laya_train.cache import answers_to_probs, build_cache


def _ans(p):
    return {"answers": {q: {"type": "choice", "choice": "a", "probabilities": {"a": p, "b": 1 - p}}
                        for q in SERVED_QUESTIONS}}


def test_answers_to_probs_keeps_only_probabilities():
    out = answers_to_probs(_ans(0.7), SERVED_QUESTIONS)
    assert out["misuse"] == {"a": 0.7, "b": 0.30000000000000004}
    assert set(out) == set(SERVED_QUESTIONS)


def test_cache_round_trips_through_laya_client(tmp_path):
    doc = build_cache([("STATE ONE", _ans(0.9)), ("STATE TWO", _ans(0.1))], revision="abc123def456",
                      meta={"model": "fraudshield-laya"})
    p = tmp_path / "laya_cache.json"
    p.write_text(json.dumps(doc))
    c = LayaClient.cached(p)
    r = c.predict("STATE ONE")
    assert r.cached and r.raw["misuse"]["a"] == 0.9
    assert c.revision == "abc123def456"
    assert c.predict("STATE TWO").raw["drop_consignee"]["a"] == 0.1
    assert doc["_meta"]["entries"] == 2
