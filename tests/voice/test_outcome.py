from fraudshield.voice.damping import active_groups, damp_values, groups_for_reason
from fraudshield.voice.outcome import classify_transcript, detail_matches, find_field, parse_outcome


def test_find_field_flat_and_nested_bolna_shapes():
    assert find_field({"owner_confirmed": "yes"}, "owner_confirmed") == "yes"
    nested = {"Tracd Verify": {"Owner Confirmed": {"objective": "no", "confidence": 0.9}}}
    assert find_field(nested, "owner_confirmed") == "no"
    assert find_field({"x": {"change_reason": {"subjective": "new warehouse"}}}, "change_reason") == "new warehouse"
    assert find_field({}, "owner_confirmed") is None


def test_transcript_fallback():
    assert classify_transcript("assistant: Was this you?\nuser: No, that wasn't me.") == "no"
    assert classify_transcript("assistant: Was this you?\nuser: Yes, we booked it.") == "yes"
    assert classify_transcript("assistant: Was this you?\nuser: I'm not sure, let me check.") == "unsure"
    assert classify_transcript("") == "unsure"


def test_branches():
    ex = {"extracted_data": {"owner_confirmed": "yes", "detail_answer": "SIPCOT, Hosur",
                             "change_reason": "new warehouse"}}
    out = parse_outcome(ex, ["Hosur"])
    assert out.branch == "confirm" and out.detail_match is True and out.extraction_source == "bolna_extraction"
    assert parse_outcome({**ex, "extracted_data": {**ex["extracted_data"], "detail_answer": "Guindy"}},
                         ["Hosur"]).branch == "detail_mismatch"
    assert parse_outcome({"extracted_data": {"owner_confirmed": "yes"}}, ["Hosur"]).branch == "detail_mismatch"
    assert parse_outcome({"extracted_data": {"owner_confirmed": "no"}}, ["Hosur"]).branch == "deny"
    assert parse_outcome({"extracted_data": {"owner_confirmed": "unsure"}}, []).branch == "unsure"
    fb = parse_outcome({"transcript": "user: no, not me"}, [])
    assert fb.branch == "deny" and fb.extraction_source == "transcript_fallback"


def test_detail_match():
    assert detail_matches("Plot 14, SIPCOT Phase 2, Hosur", ["Hosur"]) is True
    assert detail_matches("Ambattur", ["Hosur"]) is False
    assert detail_matches(None, ["Hosur"]) is None


def test_change_reason_groups_and_damping():
    assert groups_for_reason("We opened a new warehouse and a new staff login") == ["site", "login"]
    assert groups_for_reason("we moved to a new godown") == ["site"]
    assert groups_for_reason(None) == []
    vals = {"sender_differs": 1, "origin_seen": 0, "login_device_age_days": 0.0, "tenure_days": 400.0,
            "payer_sender_pair_age_days": 0.0}
    out, changed = damp_values(vals, ["site"])
    assert out["sender_differs"] == 0 and out["origin_seen"] == 1 and out["payer_sender_pair_age_days"] == 400.0
    assert out["login_device_age_days"] == 0.0 and "login_device_age_days" not in changed


def test_confirmed_change_is_scoped_and_expires():
    from datetime import datetime
    from types import SimpleNamespace
    ch = [{"groups": ["site"], "scope": {"sender_id": "wh1", "origin_zip3": "880"}, "until": "2018-07-01T10:00:00"}]
    same = SimpleNamespace(sender_id="wh1", origin_zip3="880")
    other = SimpleNamespace(sender_id="someone_else", origin_zip3="880")
    assert active_groups(ch, same, datetime(2018, 6, 10))[0] == ["site"]
    assert active_groups(ch, other, datetime(2018, 6, 10))[0] == []       # out of scope: no free pass
    assert active_groups(ch, same, datetime(2018, 7, 2))[0] == []         # 30 days are over


def test_laya_explanation_falls_back_to_labelled_demo_text(tmp_path, monkeypatch):
    from fraudshield.voice.laya_explain import explain
    monkeypatch.delenv("LAYA_API_KEY", raising=False)
    rec = {"decision_id": "d1", "state_text": "BOOKING not in any cache", "probabilities": {"misuse": 0.97},
           "feature_values": {"sender_differs": 1, "new_senders_l10": 8, "cost_vs_median": 4.0,
                              "consignee_other_accts_30d": 0}}
    out = explain(rec, {"laya": {"cache_path": str(tmp_path / "none.json")}})
    assert out["source"] == "demo" and out["note"].startswith("Demo text: Laya is not connected")
    yes = {i["id"] for i in out["items"] if i["yes"]}
    assert yes == {"misuse", "foreign_senders", "payoff_max"}
    assert "someone other than the owner" in out["summary"].lower()
    assert all(i["probability"] is None for i in out["items"])   # demo text never shows a Laya probability


def test_greeting_yes_is_not_the_answer():
    # the real Bolna call of 31 Oct 2026: "hello yes" to the greeting, then "no" to "Was this you?"
    t = ("assistant: Hello, this is Tracd calling for Sri Murugan Textiles about your courier account.\n"
         "user:  hello yes\n"
         "assistant:  A booking was just made on your account: 18.0 kg express. Was this you?\n"
         "user:  no it was not me\n"
         "assistant: Thank you. We have stopped this booking.")
    assert classify_transcript(t) == "no"


def test_real_bolna_extraction_shape():
    # shape returned by the real call of 31 Oct 2026 (objective null, text under subjective)
    ex = {"Tracd Verify": {
        "owner_confirmed": {"objective": "no", "subjective": "", "confidence": 0.9},
        "detail_answer": {"objective": None, "subjective": "Plot 14, SIPCOT, Hosur", "confidence": 0.8},
        "change_reason": {"objective": None, "subjective": "", "confidence": 0.0}}}
    assert find_field(ex, "owner_confirmed") == "no"
    assert find_field(ex, "detail_answer") == "Plot 14, SIPCOT, Hosur"
    assert find_field(ex, "change_reason") is None


def test_pickup_answer_comes_from_the_owners_words():
    # second real call (31 Oct 2026): owner said yes but gave pickup "delhi id"; Bolna's extraction summarised it with
    # the booking's own route ("Chennai to Delhi"), which wrongly matched the expected pickup "Chennai".
    t = ("assistant: Hello, this is Tracd calling for Sri Murugan Textiles.\nuser:  yes pet\n"
         "assistant:  A booking was just made: 18.0 kg express, Chennai to Delhi. Was this you?\nuser:  yes it was me only\n"
         "assistant:  Thanks. Which pickup address is this going from?\nuser:  delhi id\n"
         "assistant:  Has anything changed, like a new warehouse or a new staff login?\nuser:  no no\n"
         "assistant: Thank you, I have noted that.")
    ex = {"transcript": t, "extracted_data": {"Tracd Verify": {
        "owner_confirmed": {"objective": "yes"},
        "detail_answer": {"subjective": "The person made the booking of 18 kg from Chennai to Delhi, pickup 'delhi id'."}}}}
    out = parse_outcome(ex, ["Chennai", "Ambattur Industrial Estate, Chennai"])
    assert out.owner_confirmed == "yes" and out.detail_answer == "delhi id"
    assert out.detail_match is False and out.branch == "detail_mismatch"
    assert out.change_reason is None
