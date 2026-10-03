"""Minimal WebAuthn (passkey) server verification: registration with attestation "none", and ES256 assertions.

Only what the owner "was this you?" step needs (docs/PASSKEY.md): one algorithm (ES256, COSE -7, P-256), attestation
format "none" (no device attestation is checked: the passkey proves "same device as at enrollment", not "which
model of device"), user verification required by default. Signature check uses `cryptography`.

Registration (W3C WebAuthn L2, section 7.1): clientDataJSON type/challenge/origin, sha256(rpId) == authData rpIdHash,
UP (and UV) flags, attested credential data present, COSE EC2 P-256 key.
Assertion (section 7.2): the same clientDataJSON and authData checks, the ES256 signature over
authenticatorData || sha256(clientDataJSON) with the enrolled key, and the signature counter: if either the stored or
the new counter is non-zero the new one must be larger (a stalled or backwards counter can mean a cloned key).
Every failure raises WebAuthnError with a short machine `code`.
"""
from __future__ import annotations

import base64
import hashlib
import json
import struct
from dataclasses import dataclass
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec

FLAG_UP, FLAG_UV, FLAG_AT = 0x01, 0x04, 0x40
COSE_ES256 = -7


class WebAuthnError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def b64url(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode("ascii")


def b64url_decode(s: str) -> bytes:
    try:
        return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))
    except (ValueError, TypeError) as e:
        raise WebAuthnError("encoding", f"not base64url: {e}")


# ---------------- CBOR (RFC 8949), definite lengths only ----------------
def _cbor_item(d: bytes, i: int) -> tuple[Any, int]:
    if i >= len(d):
        raise WebAuthnError("cbor", "truncated CBOR")
    ib = d[i]
    major, ai = ib >> 5, ib & 0x1F
    i += 1
    if ai < 24:
        n = ai
    elif ai in (24, 25, 26, 27):
        size = 1 << (ai - 24)
        if i + size > len(d):
            raise WebAuthnError("cbor", "truncated CBOR")
        n = int.from_bytes(d[i:i + size], "big")
        i += size
    else:
        raise WebAuthnError("cbor", f"unsupported CBOR additional info {ai}")
    if major == 0:
        return n, i
    if major == 1:
        return -1 - n, i
    if major in (2, 3):
        if i + n > len(d):
            raise WebAuthnError("cbor", "truncated CBOR")
        raw = d[i:i + n]
        try:
            return (raw if major == 2 else raw.decode("utf-8")), i + n
        except UnicodeDecodeError:
            raise WebAuthnError("cbor", "bad UTF-8 in CBOR text")
    if major == 4:
        out = []
        for _ in range(n):
            v, i = _cbor_item(d, i)
            out.append(v)
        return out, i
    if major == 5:
        m: dict = {}
        for _ in range(n):
            k, i = _cbor_item(d, i)
            v, i = _cbor_item(d, i)
            m[k] = v
        return m, i
    if major == 7 and ai < 24:
        simple = {20: False, 21: True, 22: None}
        if n in simple:
            return simple[n], i
    raise WebAuthnError("cbor", f"unsupported CBOR item (major {major})")


def cbor_decode_prefix(d: bytes) -> tuple[Any, int]:
    """First CBOR item in `d` and the number of bytes it used (authData has the COSE key followed by extensions)."""
    return _cbor_item(d, 0)


def cbor_decode(d: bytes) -> Any:
    obj, n = _cbor_item(d, 0)
    if n != len(d):
        raise WebAuthnError("cbor", "trailing bytes after CBOR item")
    return obj


# ---------------- data ----------------
@dataclass(frozen=True)
class StoredCredential:
    credential_id: bytes
    cose_key: bytes       # COSE_Key as enrolled (CBOR)
    sign_count: int = 0

    def public_key(self) -> ec.EllipticCurvePublicKey:
        return cose_to_public_key(cbor_decode(self.cose_key))


def cose_to_public_key(k: Any) -> ec.EllipticCurvePublicKey:
    if not isinstance(k, dict) or k.get(1) != 2 or k.get(3) != COSE_ES256 or k.get(-1) != 1:
        raise WebAuthnError("key", "only EC2 P-256 keys with ES256 (COSE alg -7) are accepted")
    x, y = k.get(-2), k.get(-3)
    if not (isinstance(x, bytes) and isinstance(y, bytes) and len(x) == 32 and len(y) == 32):
        raise WebAuthnError("key", "bad EC2 coordinates")
    try:
        return ec.EllipticCurvePublicNumbers(int.from_bytes(x, "big"), int.from_bytes(y, "big"),
                                             ec.SECP256R1()).public_key()
    except ValueError as e:  # point not on the curve
        raise WebAuthnError("key", f"invalid P-256 point: {e}")


@dataclass(frozen=True)
class AuthData:
    rp_id_hash: bytes
    flags: int
    sign_count: int
    credential_id: bytes | None = None
    cose_key: bytes | None = None


def parse_auth_data(b: bytes) -> AuthData:
    if len(b) < 37:
        raise WebAuthnError("auth_data", "authenticatorData too short")
    flags = b[32]
    count = struct.unpack(">I", b[33:37])[0]
    if not flags & FLAG_AT:
        return AuthData(b[:32], flags, count)
    if len(b) < 55:
        raise WebAuthnError("auth_data", "attested credential data too short")
    n = struct.unpack(">H", b[53:55])[0]
    cid = b[55:55 + n]
    if len(cid) != n:
        raise WebAuthnError("auth_data", "credential id truncated")
    _, used = cbor_decode_prefix(b[55 + n:])
    return AuthData(b[:32], flags, count, cid, b[55 + n:55 + n + used])


# ---------------- shared checks ----------------
def _client_data(cdj: bytes, typ: str, challenge: bytes, origins: set[str]) -> dict:
    try:
        c = json.loads(cdj.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        raise WebAuthnError("client_data", "clientDataJSON is not JSON")
    if c.get("type") != typ:
        raise WebAuthnError("type", f"clientData type is {c.get('type')!r}, expected {typ!r}")
    if b64url_decode(str(c.get("challenge", ""))) != challenge:
        raise WebAuthnError("challenge", "the signed challenge does not match the one the server expects")
    if c.get("origin") not in origins:
        raise WebAuthnError("origin", f"origin {c.get('origin')!r} is not allowed")
    if c.get("crossOrigin"):
        raise WebAuthnError("origin", "cross-origin ceremonies are not allowed")
    return c


def _check_flags(a: AuthData, rp_id: str, require_uv: bool) -> None:
    if a.rp_id_hash != hashlib.sha256(rp_id.encode("utf-8")).digest():
        raise WebAuthnError("rp_id", f"authenticatorData is for a different RP ID than {rp_id!r}")
    if not a.flags & FLAG_UP:
        raise WebAuthnError("user_present", "user presence flag not set")
    if require_uv and not a.flags & FLAG_UV:
        raise WebAuthnError("user_verified", "user verification (PIN, fingerprint, face) did not happen")


def _resp(cred: dict, *names: str) -> list[bytes]:
    r = cred.get("response") or {}
    try:
        return [b64url_decode(r[n]) for n in names]
    except (KeyError, TypeError):
        raise WebAuthnError("format", f"credential response needs {', '.join(names)}")


# ---------------- ceremonies ----------------
def verify_registration(cred: dict, *, expected_challenge: bytes, rp_id: str, origins: set[str],
                        require_uv: bool = True) -> StoredCredential:
    """navigator.credentials.create() result (base64url fields) -> the credential to store."""
    cdj, att_raw = _resp(cred, "clientDataJSON", "attestationObject")
    _client_data(cdj, "webauthn.create", expected_challenge, origins)
    att = cbor_decode(att_raw)
    if not isinstance(att, dict) or not isinstance(att.get("authData"), bytes):
        raise WebAuthnError("attestation", "attestationObject has no authData")
    if att.get("fmt") != "none" or att.get("attStmt") not in ({}, None):
        raise WebAuthnError("attestation", f"attestation format {att.get('fmt')!r}: only \"none\" is accepted")
    a = parse_auth_data(att["authData"])
    _check_flags(a, rp_id, require_uv)
    if a.credential_id is None or a.cose_key is None:
        raise WebAuthnError("auth_data", "no attested credential data")
    cose_to_public_key(cbor_decode(a.cose_key))  # refuse anything but a valid ES256 P-256 key now
    if cred.get("rawId") and b64url_decode(cred["rawId"]) != a.credential_id:
        raise WebAuthnError("credential", "rawId differs from the attested credential id")
    return StoredCredential(a.credential_id, a.cose_key, a.sign_count)


def verify_assertion(cred: dict, *, stored: StoredCredential, expected_challenge: bytes, rp_id: str,
                     origins: set[str], require_uv: bool = True) -> int:
    """navigator.credentials.get() result -> the new signature counter (store it). Raises WebAuthnError."""
    if b64url_decode(str(cred.get("rawId") or cred.get("id") or "")) != stored.credential_id:
        raise WebAuthnError("credential", "assertion is from a credential that is not enrolled for this account")
    cdj, auth_raw, sig = _resp(cred, "clientDataJSON", "authenticatorData", "signature")
    _client_data(cdj, "webauthn.get", expected_challenge, origins)
    a = parse_auth_data(auth_raw)
    _check_flags(a, rp_id, require_uv)
    try:
        stored.public_key().verify(sig, auth_raw + hashlib.sha256(cdj).digest(), ec.ECDSA(hashes.SHA256()))
    except InvalidSignature:
        raise WebAuthnError("signature", "the signature does not verify with the enrolled passkey")
    if (a.sign_count or stored.sign_count) and a.sign_count <= stored.sign_count:
        raise WebAuthnError("sign_count", f"signature counter {a.sign_count} is not above the stored "
                                          f"{stored.sign_count} (possible cloned passkey)")
    return a.sign_count
