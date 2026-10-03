"""A software WebAuthn authenticator for tests and the latency benchmark (no browser, no hardware).

It builds what a real platform authenticator returns: clientDataJSON, authenticatorData, a "none" attestation
object with a COSE EC2 P-256 key, and ES256 assertion signatures over authenticatorData || sha256(clientDataJSON).
Knobs let a test break one thing at a time (origin, rpId, flags, counter).
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
from typing import Any

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec

from fraudshield.identity.webauthn import b64url


def cbor_encode(x: Any) -> bytes:
    """Minimal CBOR encoder (what an authenticator emits for attestation "none")."""
    def head(major: int, n: int) -> bytes:
        if n < 24:
            return bytes([major << 5 | n])
        for ai, fmt in ((24, ">B"), (25, ">H"), (26, ">I"), (27, ">Q")):
            if n < 1 << (8 * struct.calcsize(fmt)):
                return bytes([major << 5 | ai]) + struct.pack(fmt, n)
        raise ValueError(n)

    if isinstance(x, bool):
        return b"\xf5" if x else b"\xf4"
    if x is None:
        return b"\xf6"
    if isinstance(x, int):
        return head(0, x) if x >= 0 else head(1, -1 - x)
    if isinstance(x, bytes):
        return head(2, len(x)) + x
    if isinstance(x, str):
        b = x.encode("utf-8")
        return head(3, len(b)) + b
    if isinstance(x, list):
        return head(4, len(x)) + b"".join(cbor_encode(v) for v in x)
    if isinstance(x, dict):
        return head(5, len(x)) + b"".join(cbor_encode(k) + cbor_encode(v) for k, v in x.items())
    raise TypeError(type(x))


class SoftAuthenticator:
    def __init__(self, rp_id: str = "localhost", origin: str = "http://localhost:8092", uv: bool = True,
                 up: bool = True, counter: int = 0):
        self.rp_id, self.origin, self.uv, self.up = rp_id, origin, uv, up
        self.key = ec.generate_private_key(ec.SECP256R1())
        self.cred_id = os.urandom(32)
        self.counter = counter

    def cose_key(self) -> bytes:
        n = self.key.public_key().public_numbers()
        return cbor_encode({1: 2, 3: -7, -1: 1, -2: n.x.to_bytes(32, "big"), -3: n.y.to_bytes(32, "big")})

    def _flags(self, at: bool) -> int:
        return (0x01 if self.up else 0) | (0x04 if self.uv else 0) | (0x40 if at else 0)

    def _client_data(self, typ: str, challenge: bytes, origin: str | None) -> bytes:
        return json.dumps({"type": typ, "challenge": b64url(challenge), "origin": origin or self.origin,
                           "crossOrigin": False}).encode("utf-8")

    def create(self, challenge: bytes, *, origin: str | None = None, rp_id: str | None = None,
               typ: str = "webauthn.create", fmt: str = "none") -> dict:
        """navigator.credentials.create() result as the console sends it (base64url fields)."""
        cdj = self._client_data(typ, challenge, origin)
        rp_hash = hashlib.sha256((rp_id or self.rp_id).encode()).digest()
        att = (b"\x00" * 16) + struct.pack(">H", len(self.cred_id)) + self.cred_id + self.cose_key()
        auth = rp_hash + bytes([self._flags(True)]) + struct.pack(">I", self.counter) + att
        att_obj = cbor_encode({"fmt": fmt, "attStmt": {}, "authData": auth})
        return {"id": b64url(self.cred_id), "rawId": b64url(self.cred_id), "type": "public-key",
                "response": {"clientDataJSON": b64url(cdj), "attestationObject": b64url(att_obj)}}

    def get(self, challenge: bytes, *, origin: str | None = None, rp_id: str | None = None,
            typ: str = "webauthn.get", bump: int = 1, key: ec.EllipticCurvePrivateKey | None = None) -> dict:
        """navigator.credentials.get() result (base64url fields). bump: how much the signature counter moves."""
        self.counter += bump
        cdj = self._client_data(typ, challenge, origin)
        rp_hash = hashlib.sha256((rp_id or self.rp_id).encode()).digest()
        auth = rp_hash + bytes([self._flags(False)]) + struct.pack(">I", self.counter)
        sig = (key or self.key).sign(auth + hashlib.sha256(cdj).digest(), ec.ECDSA(hashes.SHA256()))
        return {"id": b64url(self.cred_id), "rawId": b64url(self.cred_id), "type": "public-key",
                "response": {"clientDataJSON": b64url(cdj), "authenticatorData": b64url(auth),
                             "signature": b64url(sig), "userHandle": None}}


def b64url_decode(s: str) -> bytes:
    import base64
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))
