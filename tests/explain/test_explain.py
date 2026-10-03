from types import SimpleNamespace

import pytest

from fraudshield.explain.llm import build_prompt, explain
from fraudshield.explain.template import build_facts, render_template
from fraudshield.explain.validate import validate

RECORD = {
    "decision_id": "dec_000123", "booking_id": "demo-takeover-01", "action": "hold", "degraded": False,
    "probabilities": {"misuse": 0.91, "foreign_senders": 0.94, "payoff_max": 0.88, "drop_consignee": 0.07},
    "expected_costs": {"allow": 205.1, "hold": 23.0},
    "reasons": ["COST_FAR_ABOVE_ACCOUNT_NORM", "NEW_SENDERS_UNDER_PAYER", "NEW_LOGIN_DEVICE"],
    "top_features": [
        {"name": "cost_vs_median", "value": 11.6, "baseline": 1.0, "code": "COST_FAR_ABOVE_ACCOUNT_NORM"},
        {"name": "new_senders_l10", "value": 8, "baseline": 0.0, "code": "NEW_SENDERS_UNDER_PAYER"},
        {"name": "login_device_age_days", "value": 0.0, "baseline": 30.0, "code": "NEW_LOGIN_DEVICE"},
    ],
    "booking": {"carrier_cost": 212.6, "origin_zip3": "806", "dest_zip3": "690", "account_id": "acc_7c3e",
                "meta": {"scenario": "T1"}},
}


def test_template_is_deterministic_and_names_action_reasons_and_checks():
    t1 = render_template(RECORD)
    assert t1 == render_template(RECORD)
    assert "hold" in t1.lower()
    assert "11.6x" in t1 and "0.91" in t1
    assert "check" in t1.lower()
    assert "—" not in t1  # no em dash


def test_template_passes_its_own_validator():
    facts = build_facts(RECORD)
    ok, problems = validate(render_template(RECORD), facts, template=render_template(RECORD))
    assert ok, problems


def test_facts_never_include_meta():
    facts = build_facts(RECORD)
    assert "T1" not in str(facts) and "scenario" not in str(facts)


def test_validator_rejects_invented_number():
    facts = build_facts(RECORD)
    txt = "Hold this booking: cost is 11.6x the median, new senders appeared, a new login device. Loss could be R$999."
    ok, problems = validate(txt, facts)
    assert not ok and any("999" in p for p in problems)


def test_validator_rejects_unknown_id_or_zip():
    facts = build_facts(RECORD)
    txt = "Hold: cost 11.6x the median and new senders under this payer; consignee cns_9999 is suspicious."
    ok, problems = validate(txt, facts)
    assert not ok and any("cns_9999" in p for p in problems)


def test_validator_requires_action_and_two_of_top_three_reasons():
    facts = build_facts(RECORD)
    ok, problems = validate("Cost is 11.6x the median.", facts)
    assert not ok
    assert any("action" in p for p in problems) and any("reason" in p for p in problems)


def test_validator_bans_hype():
    facts = build_facts(RECORD)
    txt = "Hold. This is confirmed fraud: cost 11.6x the median and new senders under the payer."
    ok, problems = validate(txt, facts)
    assert not ok and any("banned" in p for p in problems)


def test_validator_accepts_percent_form_of_probability():
    facts = build_facts(RECORD)
    txt = "Hold this booking (91% fraud estimate): carrier cost is 11.6x the account median and senders are new."
    ok, problems = validate(txt, facts)
    assert ok, problems


def test_prompt_contains_only_structured_record():
    system, user, h = build_prompt(build_facts(RECORD), render_template(RECORD))
    assert "acc_7c3e" not in user  # account id not needed in facts
    assert "T1" not in user
    assert len(h) == 64


def _fake_client(text):
    calls = []

    def create(**kw):
        calls.append(kw)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)], stop_reason="end_turn")
    return SimpleNamespace(messages=SimpleNamespace(create=create)), calls


def test_llm_valid_output_is_used():
    good = "Hold this booking. Carrier cost is 11.6x the account median and 8 of the last 10 bookings used new senders. Check the senders with the account owner."
    client, calls = _fake_client(good)
    exp, log = explain(RECORD, client=client, model_id="m1")
    assert exp.source == "llm" and exp.valid and exp.text == good and exp.model_id == "m1"
    assert calls[0]["temperature"] == 0 and calls[0]["model"] == "m1"
    assert log["validator"]["ok"] is True and log["prompt_hash"] == exp.prompt_hash


def test_llm_invalid_output_falls_back_to_template():
    client, _ = _fake_client("Block now, definitely a fraudster, loss R$5000.")
    exp, log = explain(RECORD, client=client, model_id="m1")
    assert exp.source == "template" and exp.text == render_template(RECORD)
    assert log["validator"]["ok"] is False and log["llm_output"].startswith("Block now")


def test_llm_error_or_missing_key_falls_back(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    exp, log = explain(RECORD, client=None, model_id="m1")
    assert exp.source == "template" and "no ANTHROPIC_API_KEY" in log["error"]

    def boom(**kw):
        raise RuntimeError("network")
    exp, log = explain(RECORD, client=SimpleNamespace(messages=SimpleNamespace(create=boom)), model_id="m1")
    assert exp.source == "template" and "network" in log["error"]


GOOD = ("Hold this booking. Carrier cost is 11.6x the account median and 8 of the last 10 bookings used new "
        "senders. Check the senders with the account owner.")


def test_check_text_marks_every_number_and_agrees_with_explain():
    from fraudshield.explain.llm import check_text
    r = check_text(RECORD, GOOD)
    assert r["ok"] is True and r["problems"] == []
    assert [n["text"] for n in r["numbers"]] == ["11.6", "8", "10"]
    assert all(n["ok"] for n in r["numbers"])
    n0 = r["numbers"][0]
    assert GOOD[n0["start"]:n0["end"]] == "11.6"


def test_check_text_flags_the_invented_number_at_its_position():
    from fraudshield.explain.llm import check_text
    bad = GOOD.replace("8 of the last", "47 of the last")  # far from every record value (rounding is allowed)
    r = check_text(RECORD, bad)
    assert r["ok"] is False and any("47" in p for p in r["problems"])
    flagged = [n for n in r["numbers"] if not n["ok"]]
    assert [n["text"] for n in flagged] == ["47"] and bad[flagged[0]["start"]:flagged[0]["end"]] == "47"


def test_check_text_applies_the_same_length_rule_as_explain():
    from fraudshield.explain.llm import check_text
    r = check_text(RECORD, GOOD + " More words." * 70)
    assert r["ok"] is False and any("too long" in p for p in r["problems"])
