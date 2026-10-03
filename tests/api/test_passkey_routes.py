"""Owner passkey "was this you?" routes (fraudshield/api/passkey.py) with a software authenticator, no browser."""
import hashlib
import json
from dataclasses import asdict

import pytest
from fastapi.testclient import TestClient

from fraudshield.api.app import build_app
from fraudshield.api.passkey import BOUND_FIELDS, bound_challenge, bound_fields
from fraudshield.audit.log import canonical_hash
from fraudshield.identity.webauthn import b64url_decode
from tests.api.test_app import components
from tests.identity.soft_authenticator import SoftAuthenticator
from tests.policy.fixtures import make_booking

ORIGIN = "http://localhost:8092"


@pytest.fixture
def cfg(tmp_path):
    return {"audit_path": str(tmp_path / "audit" / "audit.jsonl"), "replay_path": str(tmp_path / "none.jsonl"),
            "demo_bookings_path": str(tmp_path / "demo.json"), "web_dist": str(tmp_path / "nodist"),
            "calibration_path": str(tmp_path / "nocal.json"), "learning": {"enabled": False},
            "passkey": {"extra_origins": [ORIGIN]}}


@pytest.fixture
def app(cfg):
    return build_app(cfg, components=components())


@pytest.fixture
def client(app):
    return TestClient(app)


def add_decision(app, action="owner_confirm", **kw) -> str:
    b = make_booking(**kw)
    app.state.pipeline.add_record({"booking_id": b.booking_id, "account_id": b.account_id, "action": action,
                                   "booking": asdict(b), "source": "test"})
    return app.state.pipeline.get_by_booking(b.booking_id)["decision_id"]


def enroll(client, account="acc_7c3e", auth=None) -> SoftAuthenticator:
    auth = auth or SoftAuthenticator(origin=ORIGIN)
    o = client.post("/passkey/enroll/options", json={"account_id": account}).json()
    pk = o["publicKey"]
    assert pk["rp"]["id"] == "localhost" and pk["attestation"] == "none"
    assert pk["pubKeyCredParams"] == [{"type": "public-key", "alg": -7}]
    r = client.post("/passkey/enroll/verify", json={"nonce_id": o["nonce_id"], "account_id": account,
                                                     "credential": auth.create(b64url_decode(pk["challenge"]))})
    assert r.status_code == 200 and r.json()["enrolled"] is True, r.text
    return auth


def confirm(client, did, auth, **get_kw):
    o = client.post(f"/decisions/{did}/owner_confirm/options").json()
    cred = auth.get(b64url_decode(o["challenge"]), **get_kw)
    r = client.post(f"/decisions/{did}/owner_confirm/verify", json={"nonce_id": o["nonce_id"], "credential": cred})
    assert r.status_code == 200, r.text
    return o, cred, r.json()


def audit_events(cfg, kind):
    lines = open(cfg["audit_path"], encoding="utf-8").read().splitlines()
    return [json.loads(x) for x in lines if json.loads(x)["event_type"] == kind]


def test_challenge_is_bound_to_the_booking_fields():
    b = asdict(make_booking())
    f = bound_fields(b)
    assert tuple(sorted(f)) == tuple(sorted(BOUND_FIELDS))
    n = b"\x01" * 32
    base = bound_challenge(n, f)
    assert base == hashlib.sha256(n + json.dumps(f, sort_keys=True, separators=(",", ":")).encode()).digest()
    for k in BOUND_FIELDS:
        other = dict(f, **{k: (f[k] + 100 if isinstance(f[k], float) else f[k] + "x")})
        assert bound_challenge(n, other) != base, k
    assert bound_challenge(b"\x02" * 32, f) != base


def test_enroll_status_and_audit(client, cfg):
    assert client.get("/passkey/accounts/acc_7c3e").json()["enrolled"] is False
    auth = enroll(client)
    st = client.get("/api/passkey/accounts/acc_7c3e").json()
    assert st["enrolled"] is True
    assert st["credential_id_hash"] == hashlib.sha256(auth.cred_id).hexdigest()
    ev = audit_events(cfg, "passkey_enrolled")
    assert len(ev) == 1 and ev[0]["payload"]["account_id"] == "acc_7c3e"
    assert ev[0]["payload"]["simulated_owner_device"] is True
    assert client.get("/audit/verify").json()["ok"] is True


def test_enroll_rejects_wrong_origin_and_reused_nonce(client):
    o = client.post("/passkey/enroll/options", json={"account_id": "acc_x"}).json()
    ch = b64url_decode(o["publicKey"]["challenge"])
    bad = SoftAuthenticator(origin="http://127.0.0.1:8092").create(ch)
    r = client.post("/passkey/enroll/verify", json={"nonce_id": o["nonce_id"], "account_id": "acc_x",
                                                     "credential": bad})
    assert r.json()["enrolled"] is False and r.json()["code"] == "origin"
    good = SoftAuthenticator(origin=ORIGIN).create(ch)
    r = client.post("/passkey/enroll/verify", json={"nonce_id": o["nonce_id"], "account_id": "acc_x",
                                                     "credential": good})
    assert r.json()["enrolled"] is False and r.json()["code"] == "nonce"


def test_valid_confirm_releases_owner_confirm_to_allow(app, client, cfg):
    did = add_decision(app)
    auth = enroll(client)
    o, _, res = confirm(client, did, auth)
    assert o["bound_fields"]["carrier_cost"] == 212.6 and o["release_action"] == "allow"
    assert o["allow_credentials"] == [{"type": "public-key", "id": o["allow_credentials"][0]["id"]}]
    assert res["verified"] is True and res["released_action"] == "allow"
    assert res["bound_fields_hash"] == canonical_hash(o["bound_fields"])
    assert res["verify_ms"] >= 0
    d = client.get(f"/decisions/{did}").json()
    assert d["action"] == "owner_confirm"  # the original decision stays as it was audited
    assert d["owner_confirmation"]["verified"] is True and d["owner_confirmation"]["released_action"] == "allow"
    ev = audit_events(cfg, "owner_confirmed")
    assert len(ev) == 1
    p = ev[0]["payload"]
    assert p["decision_id"] == did and p["credential_id_hash"] == hashlib.sha256(auth.cred_id).hexdigest()
    assert p["bound_fields_hash"] == res["bound_fields_hash"] and p["released_action"] == "allow"
    assert client.get("/audit/verify").json()["ok"] is True
    # already released: no second ceremony
    assert client.post(f"/decisions/{did}/owner_confirm/options").status_code == 409


def test_hold_is_released_only_to_a_scan_gated_label(app, client):
    did = add_decision(app, action="hold")
    auth = enroll(client)
    _, _, res = confirm(client, did, auth)
    assert res["verified"] is True and res["released_action"] == "allow_scan_gated"


def test_allow_and_block_cannot_use_owner_confirm(app, client):
    enroll(client)
    for act in ("allow", "block"):
        did = add_decision(app, action=act, booking_id=f"bk_{act}")
        assert client.post(f"/decisions/{did}/owner_confirm/options").status_code == 409


def test_no_passkey_enrolled_is_409(app, client):
    did = add_decision(app)
    r = client.post(f"/decisions/{did}/owner_confirm/options")
    assert r.status_code == 409 and "enroll" in r.json()["detail"]


def test_wrong_challenge_fails_and_is_audited(app, client, cfg):
    did = add_decision(app)
    auth = enroll(client)
    o = client.post(f"/decisions/{did}/owner_confirm/options").json()
    cred = auth.get(b"\x00" * 32)
    res = client.post(f"/decisions/{did}/owner_confirm/verify", json={"nonce_id": o["nonce_id"], "credential": cred}).json()
    assert res["verified"] is False and res["code"] == "challenge"
    assert client.get(f"/decisions/{did}").json()["owner_confirmation"]["verified"] is False
    ev = audit_events(cfg, "owner_confirm_failed")
    assert len(ev) == 1 and ev[0]["payload"]["code"] == "challenge" and ev[0]["payload"]["tamper_test"] is False


def test_tampered_booking_field_fails(app, client, cfg):
    did = add_decision(app)
    auth = enroll(client)
    o, cred, res = confirm(client, did, auth)
    assert res["verified"] is True
    t = client.post(f"/decisions/{did}/owner_confirm/tamper_test",
                    json={"nonce_id": o["nonce_id"], "credential": cred, "field": "carrier_cost", "delta": 100}).json()
    assert t["verified"] is False and t["code"] == "challenge"
    assert t["tampered_fields"]["carrier_cost"] == pytest.approx(312.6)
    assert t["tampered_bound_fields_hash"] != t["original_bound_fields_hash"] == res["bound_fields_hash"]
    ev = audit_events(cfg, "owner_confirm_failed")
    assert ev[-1]["payload"]["tamper_test"] is True and ev[-1]["payload"]["code"] == "challenge"
    # the tamper test is a dry run: the release stands, nothing else changes
    assert client.get(f"/decisions/{did}").json()["owner_confirmation"]["verified"] is True


def test_tamper_test_with_unchanged_booking_verifies_so_the_failure_is_the_binding(app, client):
    did = add_decision(app)
    auth = enroll(client)
    o, cred, _ = confirm(client, did, auth)
    t = client.post(f"/decisions/{did}/owner_confirm/tamper_test",
                    json={"nonce_id": o["nonce_id"], "credential": cred, "field": "carrier_cost", "delta": 0}).json()
    assert t["verified"] is True


def test_tamper_on_a_string_field(app, client):
    did = add_decision(app)
    auth = enroll(client)
    o, cred, _ = confirm(client, did, auth)
    t = client.post(f"/decisions/{did}/owner_confirm/tamper_test",
                    json={"nonce_id": o["nonce_id"], "credential": cred, "field": "dest_zip3", "value": "011"}).json()
    assert t["verified"] is False and t["tampered_fields"]["dest_zip3"] == "011"
    r = client.post(f"/decisions/{did}/owner_confirm/tamper_test",
                    json={"nonce_id": o["nonce_id"], "credential": cred, "field": "weight_kg", "delta": 1})
    assert r.status_code == 422


def test_replayed_nonce_fails(app, client):
    did = add_decision(app, action="hold")
    did2 = add_decision(app, action="hold", booking_id="bk_second")
    auth = enroll(client)
    o, cred, res = confirm(client, did, auth)
    assert res["verified"] is True
    again = client.post(f"/decisions/{did}/owner_confirm/verify", json={"nonce_id": o["nonce_id"], "credential": cred})
    assert again.json()["verified"] is False and again.json()["code"] == "nonce"
    # the nonce belongs to its decision: it cannot be spent on another booking
    o2 = client.post(f"/decisions/{did2}/owner_confirm/options").json()
    cross = client.post(f"/decisions/{did2}/owner_confirm/verify", json={"nonce_id": o["nonce_id"], "credential": cred})
    assert cross.json()["verified"] is False and cross.json()["code"] == "nonce"
    assert o2["nonce_id"] != o["nonce_id"]


def test_failed_attempt_spends_the_nonce(app, client):
    did = add_decision(app)
    auth = enroll(client)
    o = client.post(f"/decisions/{did}/owner_confirm/options").json()
    bad = auth.get(b64url_decode(o["challenge"]), origin="http://evil.example")
    assert client.post(f"/decisions/{did}/owner_confirm/verify",
                       json={"nonce_id": o["nonce_id"], "credential": bad}).json()["code"] == "origin"
    good = auth.get(b64url_decode(o["challenge"]))
    assert client.post(f"/decisions/{did}/owner_confirm/verify",
                       json={"nonce_id": o["nonce_id"], "credential": good}).json()["code"] == "nonce"


def test_expired_nonce_fails(app, client, monkeypatch):
    import fraudshield.api.passkey as pkmod
    did = add_decision(app)
    auth = enroll(client)
    o = client.post(f"/decisions/{did}/owner_confirm/options").json()
    cred = auth.get(b64url_decode(o["challenge"]))
    real = pkmod.time.monotonic
    monkeypatch.setattr(pkmod.time, "monotonic", lambda: real() + 10_000)
    res = client.post(f"/decisions/{did}/owner_confirm/verify", json={"nonce_id": o["nonce_id"], "credential": cred})
    assert res.json()["verified"] is False and res.json()["code"] == "nonce"


def test_wrong_origin_and_rpid_fail(app, client):
    did = add_decision(app)
    auth = enroll(client)
    for kw, code in (({"origin": "http://127.0.0.1:8092"}, "origin"), ({"rp_id": "127.0.0.1"}, "rp_id")):
        _, _, res = confirm(client, did, auth, **kw)
        assert res["verified"] is False and res["code"] == code


def test_other_accounts_passkey_is_refused(app, client):
    did = add_decision(app)
    enroll(client)  # the real owner's passkey
    attacker = enroll(client, account="acc_attacker")
    _, _, res = confirm(client, did, attacker)
    assert res["verified"] is False and res["code"] == "credential"


def test_credentials_survive_a_restart_and_live_outside_git(cfg, tmp_path):
    a1 = build_app(cfg, components=components())
    auth = enroll(TestClient(a1))
    store = tmp_path / "passkeys" / "credentials.json"
    assert store.exists()
    a2 = build_app(cfg, components=components())
    c2 = TestClient(a2)
    did = add_decision(a2)
    _, _, res = confirm(c2, did, auth)
    assert res["verified"] is True
    assert json.loads(store.read_text())["accounts"]["acc_7c3e"]["sign_count"] == 1


def test_default_origin_is_localhost_on_the_request_port(cfg):
    c = TestClient(build_app({**cfg, "passkey": {}}, components=components()), base_url="http://localhost:8123")
    auth = enroll(c, auth=SoftAuthenticator(origin="http://localhost:8123"))
    assert auth is not None
