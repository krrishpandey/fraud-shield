"""Booking-bound passkey "was this you?" (docs/PASSKEY.md): owner_confirm as a real WebAuthn ceremony.

The account owner enrolls a passkey once (demo: this laptop plays the owner's phone). When a booking is stopped,
the server issues a single-use nonce and the WebAuthn challenge
    challenge = sha256(nonce || canonical JSON of booking_id, account_id, carrier_cost, declared_value, dest_zip3,
                       consignee_id)
so the owner's signature only verifies for these exact booking details (the same idea as W3C Secure Payment
Confirmation). On verify the server recomputes the challenge from the booking it holds; a changed amount or address
gives a different challenge and the signature fails. Every attempt is audited.

Release on success: owner_confirm -> allow (what the action was designed for). Demo extension: hold and review can
also be confirmed, but only to allow_scan_gated (the model saw more risk there, so the depot still weighs it).
block never (it needs a link to confirmed fraud). The decision record keeps its audited action; the release is a new
`owner_confirmation` field and a new audit event.

RP ID is "localhost" (WebAuthn forbids IP addresses as RP IDs), so the console must be opened at
http://localhost:<port>, not http://127.0.0.1:<port>.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from fraudshield.audit.log import canonical_hash, canonical_json
from fraudshield.identity.webauthn import (StoredCredential, WebAuthnError, b64url, b64url_decode, verify_assertion,
                                           verify_registration)

BOUND_FIELDS = ("booking_id", "account_id", "carrier_cost", "declared_value", "dest_zip3", "consignee_id")
NUMERIC_FIELDS = ("carrier_cost", "declared_value")
RELEASE = {"owner_confirm": "allow", "hold": "allow_scan_gated", "review": "allow_scan_gated"}
DEFAULTS: dict[str, Any] = {
    "rp_id": "localhost",
    "rp_name": "tracd",
    "extra_origins": ["http://localhost:5173"],   # vite dev server; the app's own origin is added per request
    "nonce_ttl_s": 120,
    "timeout_ms": 60000,
    "require_uv": True,
    "store_dir": None,     # default: artifacts/passkeys next to artifacts/audit (runtime state, git-ignored)
}


def bound_fields(booking: dict[str, Any]) -> dict[str, Any]:
    """The booking details the owner's signature is bound to, in a fixed canonical form."""
    return {k: (round(float(booking[k]), 2) if k in NUMERIC_FIELDS else str(booking[k])) for k in BOUND_FIELDS}


def bound_challenge(nonce: bytes, fields: dict[str, Any]) -> bytes:
    return hashlib.sha256(nonce + canonical_json(fields).encode("utf-8")).digest()


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


class CredentialStore:
    """One owner passkey per account (demo), in a JSON file. Enrolling again replaces it (audited as replaced)."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self._data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"accounts": {}}

    def get(self, account_id: str) -> tuple[StoredCredential, dict] | None:
        e = self._data["accounts"].get(account_id)
        if e is None:
            return None
        return StoredCredential(b64url_decode(e["credential_id"]), b64url_decode(e["cose_key"]), int(e["sign_count"])), e

    def put(self, account_id: str, cred: StoredCredential) -> tuple[dict, bool]:
        with self._lock:
            replaced = account_id in self._data["accounts"]
            e = {"credential_id": b64url(cred.credential_id), "cose_key": b64url(cred.cose_key),
                 "sign_count": cred.sign_count, "credential_id_hash": _sha(cred.credential_id),
                 "enrolled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "simulated_owner_device": True}
            self._data["accounts"][account_id] = e
            self._save()
            return e, replaced

    def set_count(self, account_id: str, n: int) -> None:
        with self._lock:
            self._data["accounts"][account_id]["sign_count"] = n
            self._save()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=1), encoding="utf-8")
        os.replace(tmp, self.path)


class NonceBook:
    """Single-use server nonces with a short expiry. Spent on the first verify attempt, pass or fail."""

    def __init__(self, ttl_s: float):
        self.ttl_s = ttl_s
        self._lock = threading.Lock()
        self._by_id: dict[str, dict] = {}

    def issue(self, kind: str, scope: str) -> tuple[str, bytes]:
        with self._lock:
            now = time.monotonic()
            for k in [k for k, v in self._by_id.items() if now - v["at"] > max(3600.0, self.ttl_s)]:
                del self._by_id[k]
            nid, nonce = secrets.token_urlsafe(12), secrets.token_bytes(32)
            self._by_id[nid] = {"kind": kind, "scope": scope, "nonce": nonce, "at": now, "used": False}
            return nid, nonce

    def take(self, nid: str, kind: str, scope: str) -> bytes:
        with self._lock:
            e = self._by_id.get(nid)
            if e is None or e["kind"] != kind or e["scope"] != scope:
                raise WebAuthnError("nonce", "unknown challenge for this request")
            if e["used"]:
                raise WebAuthnError("nonce", "this challenge was already used (replay)")
            e["used"] = True
            if time.monotonic() - e["at"] > self.ttl_s:
                raise WebAuthnError("nonce", f"the challenge expired (older than {self.ttl_s:.0f} s)")
            return e["nonce"]

    def peek(self, nid: str, kind: str, scope: str) -> bytes:
        """The nonce without spending it (tamper test only: a dry run that can never release anything)."""
        e = self._by_id.get(nid)
        if e is None or e["kind"] != kind or e["scope"] != scope:
            raise WebAuthnError("nonce", "unknown challenge for this decision")
        return e["nonce"]


class EnrollOptionsIn(BaseModel):
    account_id: str


class EnrollVerifyIn(BaseModel):
    nonce_id: str
    account_id: str
    credential: dict[str, Any]


class ConfirmVerifyIn(BaseModel):
    nonce_id: str
    credential: dict[str, Any]


class TamperIn(BaseModel):
    nonce_id: str
    credential: dict[str, Any]
    field: str = "carrier_cost"
    delta: float = 100.0          # numeric fields: added to the value
    value: str | None = None      # text fields: replaces the value


def build_passkey_router(cfg: dict[str, Any] | None, resolve: Callable[[str], tuple[dict, Any]], audit: Any,
                         default_store_dir: Path) -> APIRouter:
    """resolve(decision_id) -> (decision record, AuditLog of that decision), raising HTTPException 404;
    audit: the main AuditLog (enrollments)."""
    c = {**DEFAULTS, **(cfg or {})}
    rp_id = str(c["rp_id"])
    store = CredentialStore(Path(c["store_dir"] or default_store_dir) / "credentials.json")
    nonces = NonceBook(float(c["nonce_ttl_s"]))
    require_uv = bool(c["require_uv"])
    uv_pref = "required" if require_uv else "preferred"
    r = APIRouter()

    def origins(req: Request) -> set[str]:
        port = req.url.port
        return set(c.get("extra_origins") or []) | {f"http://{rp_id}" + (f":{port}" if port else "")}

    @r.get("/passkey/accounts/{account_id}")
    def passkey_status(account_id: str):
        got = store.get(account_id)
        e = got[1] if got else {}
        return {"account_id": account_id, "enrolled": got is not None, "rp_id": rp_id,
                "credential_id_hash": e.get("credential_id_hash"), "enrolled_at": e.get("enrolled_at"),
                "simulated_owner_device": True}

    @r.post("/passkey/enroll/options")
    def enroll_options(body: EnrollOptionsIn):
        nid, nonce = nonces.issue("enroll", body.account_id)
        user_id = hashlib.sha256(body.account_id.encode("utf-8")).digest()[:16]
        return {"nonce_id": nid, "expires_in_s": nonces.ttl_s, "publicKey": {
            "challenge": b64url(nonce), "rp": {"id": rp_id, "name": c["rp_name"]},
            "user": {"id": b64url(user_id), "name": body.account_id, "displayName": f"Owner of {body.account_id}"},
            "pubKeyCredParams": [{"type": "public-key", "alg": -7}], "timeout": int(c["timeout_ms"]),
            "attestation": "none",
            "authenticatorSelection": {"residentKey": "preferred", "userVerification": uv_pref}}}

    @r.post("/passkey/enroll/verify")
    def enroll_verify(body: EnrollVerifyIn, req: Request):
        try:
            nonce = nonces.take(body.nonce_id, "enroll", body.account_id)
            cred = verify_registration(body.credential, expected_challenge=nonce, rp_id=rp_id, origins=origins(req),
                                       require_uv=require_uv)
        except WebAuthnError as e:
            return {"enrolled": False, "code": e.code, "reason": str(e)}
        entry, replaced = store.put(body.account_id, cred)
        _, h = audit.append("passkey_enrolled", {"account_id": body.account_id, "rp_id": rp_id,
                                                 "credential_id_hash": entry["credential_id_hash"],
                                                 "replaced": replaced, "simulated_owner_device": True})
        return {"enrolled": True, "account_id": body.account_id, "credential_id_hash": entry["credential_id_hash"],
                "replaced": replaced, "audit_hash": h}

    def _stopped(did: str) -> tuple[dict, Any, dict, str]:
        rec, audit = resolve(did)
        act = rec.get("action")
        if act not in RELEASE:
            raise HTTPException(409, f"owner confirmation applies to stopped bookings ({', '.join(RELEASE)}); "
                                     f"this one is {act}")
        b = rec.get("booking")
        if not b:
            raise HTTPException(409, "this decision has no booking details to bind the confirmation to")
        return rec, audit, b, RELEASE[act]

    @r.post("/decisions/{decision_id}/owner_confirm/options")
    def confirm_options(decision_id: str):
        rec, _, b, release = _stopped(decision_id)
        if (rec.get("owner_confirmation") or {}).get("verified"):
            raise HTTPException(409, "the owner already confirmed this booking with their passkey")
        got = store.get(b["account_id"])
        if got is None:
            raise HTTPException(409, f"no owner passkey enrolled for account {b['account_id']}: enroll one first")
        fields = bound_fields(b)
        nid, nonce = nonces.issue("confirm", decision_id)
        return {"nonce_id": nid, "challenge": b64url(bound_challenge(nonce, fields)), "rp_id": rp_id,
                "allow_credentials": [{"type": "public-key", "id": b64url(got[0].credential_id)}],
                "timeout": int(c["timeout_ms"]), "user_verification": uv_pref, "expires_in_s": nonces.ttl_s,
                "bound_fields": fields, "bound_fields_hash": canonical_hash(fields), "release_action": release}

    def _check(rec: dict, fields: dict, nonce: bytes, credential: dict, req: Request) -> tuple[int, float, str]:
        """Shared verification. Returns (new sign count, verify ms, credential id hash)."""
        got = store.get(rec["booking"]["account_id"])
        if got is None:
            raise WebAuthnError("credential", "no passkey enrolled for this account")
        t = time.perf_counter()
        n = verify_assertion(credential, stored=got[0], expected_challenge=bound_challenge(nonce, fields),
                             rp_id=rp_id, origins=origins(req), require_uv=require_uv)
        return n, round((time.perf_counter() - t) * 1000, 3), got[1]["credential_id_hash"]

    def _payload(rec: dict, fields: dict, credential: dict, nid: str) -> dict:
        try:
            cid = _sha(b64url_decode(str(credential.get("rawId") or credential.get("id") or "")))
        except WebAuthnError:
            cid = None
        return {"decision_id": rec["decision_id"], "booking_id": rec["booking_id"],
                "account_id": rec["booking"]["account_id"], "nonce_id": nid, "credential_id_hash": cid,
                "bound_fields_hash": canonical_hash(fields), "rp_id": rp_id, "simulated_owner_device": True}

    @r.post("/decisions/{decision_id}/owner_confirm/verify")
    def confirm_verify(decision_id: str, body: ConfirmVerifyIn, req: Request):
        rec, audit, b, release = _stopped(decision_id)
        fields = bound_fields(b)
        pay = _payload(rec, fields, body.credential, body.nonce_id)
        t0 = time.perf_counter()
        try:
            nonce = nonces.take(body.nonce_id, "confirm", decision_id)
            n, ms, cid = _check(rec, fields, nonce, body.credential, req)
        except WebAuthnError as e:
            _, h = audit.append("owner_confirm_failed", {**pay, "code": e.code, "reason": str(e),
                                                         "tamper_test": False})
            rec["owner_confirmation"] = {"verified": False, "code": e.code, "reason": str(e), "audit_hash": h,
                                         "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            return {"verified": False, "code": e.code, "reason": str(e), "audit_hash": h,
                    "bound_fields_hash": pay["bound_fields_hash"]}
        store.set_count(b["account_id"], n)
        total = round((time.perf_counter() - t0) * 1000, 3)
        _, h = audit.append("owner_confirmed", {**pay, "credential_id_hash": cid, "original_action": rec["action"],
                                                "released_action": release, "sign_count": n, "verify_ms": ms})
        out = {"verified": True, "released_action": release, "original_action": rec["action"],
               "credential_id_hash": cid, "bound_fields_hash": pay["bound_fields_hash"], "sign_count": n,
               "verify_ms": ms, "server_ms": total, "audit_hash": h,
               "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        rec["owner_confirmation"] = out
        return out

    @r.post("/decisions/{decision_id}/owner_confirm/tamper_test")
    def tamper_test(decision_id: str, body: TamperIn, req: Request):
        """Re-checks an assertion the owner already gave against a MODIFIED copy of the booking. A dry run: it never
        releases anything, never moves the counter, and the nonce may already be spent (that is the point)."""
        rec, audit, b, _ = _stopped(decision_id)
        if body.field not in BOUND_FIELDS or body.field in ("booking_id", "account_id"):
            raise HTTPException(422, f"field must be one of {[f for f in BOUND_FIELDS if f not in ('booking_id', 'account_id')]}")
        fields = bound_fields(b)
        tampered = dict(fields)
        if body.field in NUMERIC_FIELDS:
            tampered[body.field] = round(fields[body.field] + float(body.delta), 2)
        else:
            tampered[body.field] = body.value if body.value is not None else fields[body.field] + "0"
        pay = _payload(rec, tampered, body.credential, body.nonce_id)
        try:
            nonce = nonces.peek(body.nonce_id, "confirm", decision_id)
            got = store.get(b["account_id"])
            if got is None:
                raise WebAuthnError("credential", "no passkey enrolled for this account")
            t = time.perf_counter()
            # counter check off: the same assertion is re-used on purpose; only the booking binding is under test
            verify_assertion(body.credential, stored=StoredCredential(got[0].credential_id, got[0].cose_key, 0),
                             expected_challenge=bound_challenge(nonce, tampered), rp_id=rp_id, origins=origins(req),
                             require_uv=require_uv)
            ms = round((time.perf_counter() - t) * 1000, 3)
            ok, code, reason = True, None, "verified: the booking details were not changed"
        except WebAuthnError as e:
            ms, ok, code, reason = None, False, e.code, str(e)
        ev = "owner_confirm_failed" if not ok else "owner_confirmed"
        extra = {"tamper_test": True, "tampered_field": body.field, "dry_run": True}
        if ok:  # only reachable with an unchanged booking (delta 0): audited, but still releases nothing
            _, h = audit.append(ev, {**pay, **extra, "released_action": None})
        else:
            _, h = audit.append(ev, {**pay, **extra, "code": code, "reason": reason})
        return {"verified": ok, "code": code, "reason": reason, "tampered_field": body.field,
                "original_fields": fields, "tampered_fields": tampered,
                "original_bound_fields_hash": canonical_hash(fields), "tampered_bound_fields_hash": canonical_hash(tampered),
                "verify_ms": ms, "audit_hash": h}

    return r
