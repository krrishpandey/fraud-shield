import os

import pytest
from cryptography.hazmat.primitives.asymmetric import ec

from fraudshield.identity.webauthn import (StoredCredential, WebAuthnError, b64url, b64url_decode, cbor_decode,
                                           verify_assertion, verify_registration)
from tests.identity.soft_authenticator import SoftAuthenticator, cbor_encode

RP = "localhost"
ORIGIN = "http://localhost:8092"


def enroll(auth: SoftAuthenticator) -> StoredCredential:
    ch = os.urandom(32)
    return verify_registration(auth.create(ch), expected_challenge=ch, rp_id=RP, origins={ORIGIN})


def test_cbor_round_trip_of_what_authenticators_emit():
    obj = {"fmt": "none", "attStmt": {}, "authData": b"\x01\x02", 1: 2, 3: -7, -1: 1, "n": [0, 23, 24, 255, 256,
                                                                                           65536, 2**32, -25]}
    assert cbor_decode(cbor_encode(obj)) == obj
    assert cbor_decode(b"\xf5") is True and cbor_decode(b"\xf6") is None


def test_cbor_rejects_trailing_bytes_and_truncation():
    with pytest.raises(WebAuthnError):
        cbor_decode(cbor_encode({"a": 1}) + b"\x00")
    with pytest.raises(WebAuthnError):
        cbor_decode(cbor_encode(b"abcdef")[:-2])
    with pytest.raises(WebAuthnError):
        cbor_decode(b"\x5f")  # indefinite length: not used by authenticators, refused


def test_b64url_round_trip():
    for n in range(0, 40):
        b = os.urandom(n)
        assert b64url_decode(b64url(b)) == b and "=" not in b64url(b)


def test_valid_enroll_returns_the_p256_key_and_counter():
    a = SoftAuthenticator(origin=ORIGIN)
    cred = enroll(a)
    assert cred.credential_id == a.cred_id and cred.sign_count == 0
    n = a.key.public_key().public_numbers()
    assert cred.public_key().public_numbers() == n


def test_enroll_rejects_wrong_challenge_origin_rpid_type_and_flags():
    ch = os.urandom(32)
    cases = [
        (SoftAuthenticator(origin=ORIGIN).create(os.urandom(32)), "challenge"),
        (SoftAuthenticator(origin="http://evil.example:8092").create(ch), "origin"),
        (SoftAuthenticator(origin=ORIGIN).create(ch, rp_id="evil.example"), "rp_id"),
        (SoftAuthenticator(origin=ORIGIN).create(ch, typ="webauthn.get"), "type"),
        (SoftAuthenticator(origin=ORIGIN, uv=False).create(ch), "user_verified"),
        (SoftAuthenticator(origin=ORIGIN, up=False).create(ch), "user_present"),
        (SoftAuthenticator(origin=ORIGIN).create(ch, fmt="packed"), "attestation"),
    ]
    for cred, code in cases:
        with pytest.raises(WebAuthnError) as e:
            verify_registration(cred, expected_challenge=ch, rp_id=RP, origins={ORIGIN})
        assert e.value.code == code, (code, e.value)


def test_valid_assertion_and_counter_moves_forward():
    a = SoftAuthenticator(origin=ORIGIN)
    cred = enroll(a)
    ch = os.urandom(32)
    assert verify_assertion(a.get(ch), stored=cred, expected_challenge=ch, rp_id=RP, origins={ORIGIN}) == 1


def test_assertion_rejects_wrong_challenge():
    a = SoftAuthenticator(origin=ORIGIN)
    cred = enroll(a)
    with pytest.raises(WebAuthnError) as e:
        verify_assertion(a.get(os.urandom(32)), stored=cred, expected_challenge=os.urandom(32), rp_id=RP,
                         origins={ORIGIN})
    assert e.value.code == "challenge"


def test_assertion_rejects_wrong_origin_and_rpid():
    a = SoftAuthenticator(origin=ORIGIN)
    cred = enroll(a)
    ch = os.urandom(32)
    with pytest.raises(WebAuthnError) as e:
        verify_assertion(a.get(ch, origin="http://127.0.0.1:8092"), stored=cred, expected_challenge=ch, rp_id=RP,
                         origins={ORIGIN})
    assert e.value.code == "origin"
    with pytest.raises(WebAuthnError) as e:
        verify_assertion(a.get(ch, rp_id="evil.example"), stored=cred, expected_challenge=ch, rp_id=RP,
                         origins={ORIGIN})
    assert e.value.code == "rp_id"


def test_assertion_rejects_bad_signature_other_key_and_wrong_credential():
    a = SoftAuthenticator(origin=ORIGIN)
    cred = enroll(a)
    ch = os.urandom(32)
    with pytest.raises(WebAuthnError) as e:
        verify_assertion(a.get(ch, key=ec.generate_private_key(ec.SECP256R1())), stored=cred, expected_challenge=ch,
                         rp_id=RP, origins={ORIGIN})
    assert e.value.code == "signature"
    other = SoftAuthenticator(origin=ORIGIN)
    with pytest.raises(WebAuthnError) as e:
        verify_assertion(other.get(ch), stored=cred, expected_challenge=ch, rp_id=RP, origins={ORIGIN})
    assert e.value.code == "credential"


def test_assertion_rejects_flipped_authenticator_data_byte():
    a = SoftAuthenticator(origin=ORIGIN)
    cred = enroll(a)
    ch = os.urandom(32)
    res = a.get(ch)
    auth = bytearray(b64url_decode(res["response"]["authenticatorData"]))
    auth[-1] ^= 0x01  # counter byte: still a valid structure, but no longer what was signed
    res["response"]["authenticatorData"] = b64url(bytes(auth))
    with pytest.raises(WebAuthnError) as e:
        verify_assertion(res, stored=cred, expected_challenge=ch, rp_id=RP, origins={ORIGIN})
    assert e.value.code == "signature"


def test_assertion_needs_user_verification():
    a = SoftAuthenticator(origin=ORIGIN)
    cred = enroll(a)
    a.uv = False
    ch = os.urandom(32)
    with pytest.raises(WebAuthnError) as e:
        verify_assertion(a.get(ch), stored=cred, expected_challenge=ch, rp_id=RP, origins={ORIGIN})
    assert e.value.code == "user_verified"


def test_sign_count_that_goes_backwards_or_stalls_is_rejected():
    a = SoftAuthenticator(origin=ORIGIN)
    cred = enroll(a)
    ch = os.urandom(32)
    cred = StoredCredential(cred.credential_id, cred.cose_key, sign_count=5)
    with pytest.raises(WebAuthnError) as e:
        verify_assertion(a.get(ch, bump=3), stored=cred, expected_challenge=ch, rp_id=RP, origins={ORIGIN})
    assert e.value.code == "sign_count"
    a.counter = 4
    with pytest.raises(WebAuthnError):
        verify_assertion(a.get(ch, bump=1), stored=cred, expected_challenge=ch, rp_id=RP, origins={ORIGIN})


def test_authenticators_without_a_counter_stay_at_zero():
    """Synced passkeys (iCloud, Google, Windows Hello) often report 0 every time: allowed while both are 0."""
    a = SoftAuthenticator(origin=ORIGIN)
    cred = enroll(a)
    ch = os.urandom(32)
    assert verify_assertion(a.get(ch, bump=0), stored=cred, expected_challenge=ch, rp_id=RP, origins={ORIGIN}) == 0
