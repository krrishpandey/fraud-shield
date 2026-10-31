import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fraudshield.api.pipeline import Pipeline
from fraudshield.audit.log import AuditLog, verify_file
from fraudshield.contracts import FeatureVector
from fraudshield.policy.costs import CostConfig
from fraudshield.voice.owners import OwnerDirectory
from fraudshield.voice.router import build_verify_router
from fraudshield.voice.service import InlineRunner, VerifyConfig, VerifyService
from tests.api import fakes
from tests.policy.fixtures import make_booking

ACCT = None  # filled from the fixture booking


def featurize(b):
    return FeatureVector(b.booking_id, b.booked_at, {"new_senders_l10": 8, "sender_differs": 1, "origin_seen": 0,
                                                     "tenure_days": 412.0, "login_device_age_days": 0.0,
                                                     "cost_vs_median": 1.2})


def gbm(fv):
    """Risky while the booking looks new to the account; low once a confirmed change explains it."""
    return 0.97 if fv.values.get("sender_differs") else 0.004


def setup(tmp_path, contact_days=400.0, **cfg):
    pipe = Pipeline(featurize=featurize, serialize=fakes.fake_serialize, gbm=gbm, laya=fakes.DownLaya(),
                    audit=AuditLog(tmp_path / "audit.jsonl"))
    b = make_booking()
    owners = OwnerDirectory({b.account_id: {"company": "Test Co", "phone_env": "PH", "language": "en",
                                            "phone_changed_days_ago": contact_days,
                                            "pickup_addresses": ["Hosur"]}}, env={"PH": "+919800000000"})
    runner = InlineRunner()
    svc = VerifyService(lambda did: (pipe.get(did), pipe), VerifyConfig(mode="simulated", sim_speed=0, **cfg), owners,
                        runner=runner, costs=CostConfig.default())
    r = pipe.score(b)
    assert r["action"] == "hold", r["action"]
    return pipe, svc, runner, r["decision_id"]


def events(tmp_path, kind=None):
    rows = [json.loads(x) for x in (tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    return [x for x in rows if kind is None or x["event_type"] == kind]


YES = {"owner_confirmed": "yes", "detail_answer": "SIPCOT Hosur", "change_reason": "new warehouse", "language": "en"}


def test_yes_with_stated_change_releases_with_first_scan_check(tmp_path):
    pipe, svc, runner, did = setup(tmp_path)
    svc.start(did, sim_answer=YES)
    runner.run_all()
    v = svc.get(did)
    assert v["status"] == "released" and v["final_action"] == "allow_scan_gated"
    assert v["rescore"]["p_before"] == pytest.approx(0.97, abs=1e-3)
    assert v["rescore"]["p_after"] < 0.01
    assert pipe.get(did)["action"] == "hold"            # the audited decision is never edited
    assert pipe.get(did)["owner_verification"]["final_action"] == "allow_scan_gated"
    out = events(tmp_path, "verify_outcome")[-1]["payload"]
    assert out["decision_id"] == did and out["decision_audit_hash"] == pipe.get(did)["audit_hash"]
    assert pipe.confirmed_changes[pipe.get(did)["account_id"]][0]["groups"] == ["site"]
    assert verify_file(tmp_path / "audit.jsonl")["ok"]


def test_no_blocks_locks_and_opens_case(tmp_path):
    pipe, svc, runner, did = setup(tmp_path)
    svc.start(did, sim_answer={"owner_confirmed": "no", "language": "en"})
    runner.run_all()
    v = svc.get(did)
    assert v["final_action"] == "block" and v["account_locked"]
    assert v["case"]["reason"] == "owner_denied" and "that was not me" in v["transcript"]
    assert pipe.get(did)["analyst"]["label"] == "fraud"
    assert events(tmp_path, "account_locked")
    # the locked account's next booking is a hard signal
    nxt = pipe.score(make_booking(booking_id="bk_next"))
    assert nxt["action"] in ("hold", "block") and "ACCOUNT_LOCKED" in nxt["reasons"]


def test_hard_signal_is_never_released_by_a_yes(tmp_path):
    pipe, svc, runner, did = setup(tmp_path)
    pipe.get(did)["hard_signal"] = True                  # e.g. a known reshipping-drop address
    svc.start(did, sim_answer=YES)
    runner.run_all()
    v = svc.get(did)
    assert v["final_action"] is None and v["status"] == "analyst_review"
    assert v["case"]["reason"] == "hard_signal" and v["rescore"]["action"] == "hold"


def test_coached_yes_without_matching_detail_stays_held(tmp_path):
    pipe, svc, runner, did = setup(tmp_path)
    svc.start(did, sim_answer={**YES, "detail_answer": "Ambattur, Chennai"})
    runner.run_all()
    v = svc.get(did)
    assert v["final_action"] is None and v["case"]["reason"] == "detail_mismatch"


def test_unsure_goes_to_analyst(tmp_path):
    pipe, svc, runner, did = setup(tmp_path)
    svc.start(did, sim_answer={"owner_confirmed": "unsure"})
    runner.run_all()
    assert svc.get(did)["case"]["reason"] == "unsure" and svc.get(did)["final_action"] is None


def test_30_day_contact_rule_skips_the_call(tmp_path):
    pipe, svc, runner, did = setup(tmp_path, contact_days=6)
    v = svc.start(did, sim_answer=YES)
    runner.run_all()
    assert v["attempts"] == [] and v["status"] == "analyst_review"
    assert v["case"]["reason"] == "contact_changed_recently"
    assert events(tmp_path, "verify_started")[-1]["payload"]["call"] is False


def test_no_answer_retries_once_then_sms_then_analyst_never_released(tmp_path):
    pipe, svc, runner, did = setup(tmp_path)
    svc.start(did, sim_answer={"owner_confirmed": "no_answer"})
    runner.run_all()
    v = svc.get(did)
    assert len(v["attempts"]) == 2 and all(a["status"] == "no-answer" for a in v["attempts"])
    assert v["sms"]["stub"] is True and v["case"]["reason"] == "no_answer"
    assert v["final_action"] is None and v["status"] == "analyst_review"


def test_webhook_is_idempotent_and_token_protected(tmp_path):
    pipe, svc, runner, did = setup(tmp_path, webhook_secret="s3cret")
    svc.start(did, sim_answer=YES)
    ex = svc.get(did)["attempts"][0]["execution_id"]
    runner.pending.clear()                               # drive the call by webhook only
    app = FastAPI()
    app.include_router(build_verify_router(svc, pipe, tmp_path / "audit.jsonl", None))
    c = TestClient(app)
    done = {"id": ex, "status": "completed", "transcript": "assistant: Was this you?\nuser: yes",
            "extracted_data": {"owner_confirmed": "yes", "detail_answer": "Hosur", "change_reason": "new warehouse"}}
    assert c.post("/voice/webhook", json=done).status_code == 401
    assert c.post("/voice/webhook?token=s3cret", json={"foo": 1}).status_code == 422
    n0 = len(events(tmp_path))
    r1 = c.post("/voice/webhook?token=s3cret", json=done).json()
    n1 = len(events(tmp_path))
    r2 = c.post("/voice/webhook?token=s3cret", json=done).json()
    assert r1["final_action"] == "allow_scan_gated" and r2["duplicate"] is True
    assert len(events(tmp_path)) == n1 > n0               # the duplicate wrote nothing
    assert len(events(tmp_path, "verify_outcome")) == 1


def test_analyst_resolves_case(tmp_path):
    pipe, svc, runner, did = setup(tmp_path)
    svc.start(did, sim_answer={"owner_confirmed": "unsure"})
    runner.run_all()
    v = svc.resolve_case(did, "legit", "called back, new warehouse")
    assert v["final_action"] == "allow_scan_gated" and v["case"]["status"] == "resolved"
    assert pipe.get(did)["analyst"]["label"] == "legit"
    assert svc.metrics(1)["honest_held"] == 1


def test_audit_chain_detects_a_tampered_call_record(tmp_path):
    pipe, svc, runner, did = setup(tmp_path)
    svc.start(did, sim_answer={"owner_confirmed": "no"})
    runner.run_all()
    p = tmp_path / "audit.jsonl"
    assert verify_file(p)["ok"]
    lines = p.read_text(encoding="utf-8").splitlines()
    i = next(k for k, x in enumerate(lines) if '"verify_outcome"' in x)
    lines[i] = lines[i].replace('"final_action":"block"', '"final_action":"allow"')
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    res = verify_file(p)
    assert res["ok"] is False and res["first_bad_index"] == i


def test_metrics_come_from_the_run(tmp_path):
    pipe, svc, runner, did = setup(tmp_path)
    svc.start(did, sim_answer=YES)
    runner.run_all()
    m = svc.metrics(bookings_scored=1)
    assert m["released_by_call"] == 1 and m["calls_answered"] == 1 and m["call_answer_rate"] == 1.0
    assert m["simulated_verifications"] == 1 and m["median_minutes_to_release_honest"] is not None
    assert m["analyst_minutes_saved_per_1000"] == pytest.approx(6.0 * 1000)


def test_disconnected_waits_for_completed_then_uses_extraction(tmp_path):
    pipe, svc, runner, did = setup(tmp_path)
    svc.start(did, sim_answer=YES)
    ex = svc.get(did)["attempts"][0]["execution_id"]
    runner.pending.clear()
    t = "assistant: Was this you?\nuser: yes it was us"
    svc.handle_event({"id": ex, "status": "in-progress", "transcript": t}, "webhook")
    svc.handle_event({"id": ex, "status": "call-disconnected", "transcript": t}, "webhook")
    assert svc.get(did)["final_action"] is None            # not decided at disconnect
    svc.handle_event({"id": ex, "status": "completed", "transcript": t,
                      "extracted_data": {"owner_confirmed": "yes", "detail_answer": "Hosur", "change_reason": "new warehouse"}})
    v = svc.get(did)
    assert v["outcome"]["extraction_source"] == "simulated" or v["outcome"]["owner_confirmed"] == "yes"
    assert v["final_action"] == "allow_scan_gated"
