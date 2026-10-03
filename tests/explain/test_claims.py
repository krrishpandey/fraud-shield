import copy
import json

import httpx
import numpy as np
import pytest

from fraudshield.explain.attribution import FEATURE_CODE, agreement, attribution_view, model_of, model_reasons
from fraudshield.explain.claims import (CLAIMS_SCHEMA, claims_view, compose_text, explain_claims, template_claims,
                                        validate_claims)
from fraudshield.explain.groq_client import GroqClient
from fraudshield.explain.llm import check_text
from tests.explain.test_explain import RECORD


def _claims(*items, action="hold"):
    return {"action": action, "claims": [dict(text=t, reason_code=c, cited_fields=f, numbers=n) for t, c, f, n in items]}


GOOD = _claims(
    ("Carrier cost is 11.6x the account's median booking.", "COST_FAR_ABOVE_ACCOUNT_NORM", ["cost_vs_median"], [11.6]),
    ("8 of the last 10 bookings used new senders.", "NEW_SENDERS_UNDER_PAYER", ["new_senders_l10"], [8, 10]),
)


# ---------- template claims ----------
def test_template_claims_one_per_reason_and_all_pass():
    t = template_claims(RECORD)
    assert t["action"] == "hold"
    assert [c["reason_code"] for c in t["claims"]] == RECORD["reasons"][:3]
    assert t["claims"][0]["cited_fields"] == ["cost_vs_median"] and 11.6 in t["claims"][0]["numbers"]
    chk = validate_claims(RECORD, t)
    assert chk["ok"], chk["problems"]
    assert all(c["ok"] for c in chk["claims"])


def test_good_llm_claims_pass_and_composed_text_passes_the_prose_validator():
    chk = validate_claims(RECORD, GOOD)
    assert chk["ok"], chk
    text = compose_text(RECORD, chk["claims"])
    assert text.startswith("Recommended action: hold the booking until verified (hold).")
    assert check_text(RECORD, text)["ok"]


# ---------- per-claim checks ----------
def test_fabricated_number_fails_only_that_claim():
    bad = copy.deepcopy(GOOD)
    bad["claims"][1]["text"] = "37 of the last 10 bookings used new senders."
    chk = validate_claims(RECORD, bad)
    assert not chk["ok"]
    assert chk["claims"][0]["ok"] and not chk["claims"][1]["ok"]
    assert "number not in record: 37" in chk["claims"][1]["problems"]


def test_fabricated_number_only_in_numbers_list_is_caught():
    bad = copy.deepcopy(GOOD)
    bad["claims"][0]["numbers"] = [11.6, 4321]
    chk = validate_claims(RECORD, bad)
    assert not chk["claims"][0]["ok"] and any("4321" in p for p in chk["claims"][0]["problems"])


def test_true_number_moved_to_the_wrong_claim_is_caught():
    # 11.6 is in the record, but it is the cost ratio, not part of the senders reason
    bad = copy.deepcopy(GOOD)
    bad["claims"][1]["text"] = "11.6 of the last 10 bookings used new senders."
    chk = validate_claims(RECORD, bad)
    assert not chk["claims"][1]["ok"]
    assert any("not from this claim's cited fields" in p for p in chk["claims"][1]["problems"])


def test_reason_not_in_decision_is_rejected():
    bad = copy.deepcopy(GOOD)
    bad["claims"].append(dict(text="The booking links to confirmed fraud.", reason_code="LINK_TO_CONFIRMED_FRAUD",
                              cited_fields=[], numbers=[]))
    chk = validate_claims(RECORD, bad)
    assert not chk["ok"] and not chk["claims"][2]["ok"]
    assert "not one of this decision's reasons" in chk["claims"][2]["problems"][0]


def test_mislabelled_reason_code_is_rejected():
    bad = copy.deepcopy(GOOD)
    bad["claims"][0]["reason_code"] = "NEW_LOGIN_DEVICE"  # a real reason, but the sentence is about cost
    chk = validate_claims(RECORD, bad)
    assert not chk["claims"][0]["ok"] and any("does not talk about" in p for p in chk["claims"][0]["problems"])


def test_unknown_field_wrong_action_and_ids_are_rejected():
    bad = copy.deepcopy(GOOD)
    bad["claims"][0]["cited_fields"] = ["secret_score"]
    bad["claims"][1]["text"] = "8 of the last 10 bookings used new senders like snd_4411."
    bad["action"] = "block"
    chk = validate_claims(RECORD, bad)
    assert not chk["ok"] and not chk["action_ok"]
    assert "field not in record: secret_score" in chk["claims"][0]["problems"]
    assert any("snd_4411" in p for p in chk["claims"][1]["problems"])


def test_needs_two_of_top_three_reasons():
    one = _claims(GOOD["claims"][0]["text"] and ("Carrier cost is 11.6x the account's median booking.",
                                                 "COST_FAR_ABOVE_ACCOUNT_NORM", ["cost_vs_median"], [11.6]))
    chk = validate_claims(RECORD, one)
    assert chk["claims"][0]["ok"] and not chk["ok"] and any("covers 1 of top 3" in p for p in chk["problems"])


def test_garbage_shapes_do_not_crash():
    for s in ({}, {"action": "hold", "claims": [None, 3, {"text": 5}]}, {"action": "hold", "claims": "x"}):
        assert validate_claims(RECORD, s)["ok"] is False


# ---------- structured outputs on Groq, with fallbacks ----------
def _groq(handler):
    return GroqClient(api_key="k", http=httpx.Client(transport=httpx.MockTransport(handler)))


def _reply(content):
    return httpx.Response(200, json={"choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                                     "usage": {"total_tokens": 100}})


def test_strict_json_schema_is_requested_and_a_passing_answer_is_shown():
    seen = []

    def handler(req):
        seen.append(json.loads(req.content))
        return _reply(json.dumps(GOOD))

    exp, log, payload = explain_claims(RECORD, client=_groq(handler), model_id="openai/gpt-oss-120b")
    rf = seen[0]["response_format"]
    assert rf["type"] == "json_schema" and rf["json_schema"]["strict"] is True
    assert rf["json_schema"]["schema"] == CLAIMS_SCHEMA
    assert exp.source == "llm" and payload["mode"] == "json_schema" and payload["claims_source"] == "llm"
    assert log["mode"] == "json_schema" and log["usage"] == [{"total_tokens": 100}]
    assert "11.6x" in exp.text and len(seen) == 1


def test_schema_rejected_falls_back_to_json_object():
    modes = []

    def handler(req):
        rf = json.loads(req.content)["response_format"]["type"]
        modes.append(rf)
        if rf == "json_schema":
            return httpx.Response(400, json={"error": {"message": "response_format json_schema not supported"}})
        return _reply(json.dumps(GOOD))

    exp, log, payload = explain_claims(RECORD, client=_groq(handler), model_id="m")
    assert modes == ["json_schema", "json_object"]
    assert exp.source == "llm" and payload["mode"] == "json_object" and log["mode_errors"][0]["mode"] == "json_schema"


def test_both_json_modes_refused_or_ignored_falls_back_to_prose():
    bodies = []

    def handler(req):
        b = json.loads(req.content)
        bodies.append(b)
        if "response_format" in b:
            return _reply("Sure! Here is a paragraph, not JSON.") if b["response_format"]["type"] == "json_object" \
                else httpx.Response(400, json={"error": {"message": "no"}})
        return _reply("Hold this booking: carrier cost is 11.6x the median and new senders appeared under this payer.")

    exp, log, payload = explain_claims(RECORD, client=_groq(handler), model_id="m")
    assert [("response_format" in b) for b in bodies] == [True, True, False]
    assert payload["mode"] == "prose" and log["mode"] == "prose" and exp.source == "llm"
    assert payload["claims_source"] == "template"  # prose has no claims: the console shows the template's


def test_failing_claim_shows_template_but_keeps_the_rejected_claims():
    bad = copy.deepcopy(GOOD)
    bad["claims"][1]["text"] = "37 of the last 10 bookings used new senders."
    calls = []

    def handler(req):
        calls.append(json.loads(req.content))
        return _reply(json.dumps(bad))

    exp, log, payload = explain_claims(RECORD, client=_groq(handler), model_id="m", retry_invalid=1)
    assert len(calls) == 2 and "37" in calls[1]["messages"][-1]["content"]  # retried, told what failed
    assert exp.source == "template" and payload["mode"] == "json_schema"
    assert payload["claims_source"] == "template" and all(c["ok"] for c in payload["claims"])
    assert [c["ok"] for c in payload["rejected_claims"]] == [True, False]


def test_provider_error_gives_template_without_more_calls():
    calls = []

    def handler(req):
        calls.append(1)
        return httpx.Response(500, json={"error": "down"})

    exp, log, payload = explain_claims(RECORD, client=_groq(handler), model_id="m")
    assert len(calls) == 1 and exp.source == "template" and log["error"].startswith("HTTPStatusError")


def test_client_without_structured_outputs_uses_prose_path():
    class Anthropicish:
        class messages:  # noqa: N801
            @staticmethod
            def create(**kw):
                assert "response_format" not in kw
                raise RuntimeError("offline")

    exp, log, payload = explain_claims(RECORD, client=Anthropicish(), model_id="m")
    assert payload["mode"] == "prose" and exp.source == "template"
    exp, log, payload = explain_claims(RECORD, client=None)
    assert payload["mode"] == "template" and payload["ok"]


# ---------- attribution agreement ----------
def test_agreement_definitions():
    a = agreement(["A", "B", "A", "D"], ["B", "C", "A"], reasons=["A", "B", "D"])
    assert a["cited"] == ["A", "B", "D"] and a["hits"] == 2 and a["of"] == 3
    assert a["hit_at_3"] == pytest.approx(2 / 3, abs=1e-4) and a["top1_match"] is False and a["reachable"] == 2
    assert agreement(["B"], ["B"])["top1_match"] is True
    assert agreement(["A"], [])["hits"] is None


@pytest.fixture(scope="module")
def tiny_model():
    lgb = pytest.importorskip("lightgbm")
    rng = np.random.default_rng(0)
    feats = ["cost_vs_median", "new_senders_l10", "dist_km"]
    X = rng.normal(size=(600, 3))
    y = (X[:, 1] + 0.3 * X[:, 0] > 0.5).astype(int)
    booster = lgb.train({"objective": "binary", "verbose": -1, "num_leaves": 7, "seed": 1},
                        lgb.Dataset(X, y, feature_name=feats), num_boost_round=30)

    class M:
        pass
    m = M()
    m.booster, m.features, m.metadata = booster, feats, {"active_version": "tiny"}
    return m


def test_model_reasons_sum_contributions_per_code(tiny_model):
    mr = model_reasons(tiny_model, {"cost_vs_median": 2.0, "new_senders_l10": 3.0, "dist_km": 0.0})
    codes = [x["code"] for x in mr["top_codes"]]
    assert codes[0] == "NEW_SENDERS_UNDER_PAYER" and set(codes) <= set(FEATURE_CODE.values())
    assert mr["top_features"][0]["name"] == "new_senders_l10" and 0 < mr["mapped_share"] <= 1


def test_model_of_finds_the_model_in_a_learning_scorer_closure(tiny_model):
    def gbm_scorer(model):
        def score(fv):
            return model.booster.predict([[0, 0, 0]])[0]
        return score
    assert model_of(gbm_scorer(tiny_model)) is tiny_model
    assert model_of(lambda fv: 0.5) is None


def test_claims_view_template_and_agreement(tiny_model):
    rec = {**RECORD, "feature_values": {"cost_vs_median": 11.6, "new_senders_l10": 8, "dist_km": 1.0},
           "explanation": None}

    def gbm_scorer(model):
        return lambda fv: model
    v = claims_view(rec, gbm_scorer(tiny_model))
    assert v["claims_source"] == "template" and v["ok"] and len(v["claims"]) == 3
    att = v["attribution"]
    assert att["available"] and att["agreement"]["cited"] == RECORD["reasons"][:3]
    assert att["agreement"]["hits"] == len(set(RECORD["reasons"][:3]) & set(att["agreement"]["model"]))
    assert claims_view({**rec, "feature_values": None}, None)["attribution"]["available"] is False
