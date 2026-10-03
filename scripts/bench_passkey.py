"""Owner passkey verification latency (CPU, no browser): cold first check and warm checks.

Builds real ES256 assertions with a software authenticator (tests/identity/soft_authenticator.py: synthetic keys and
bookings, no user data) and times fraudshield.identity.webauthn.verify_assertion, the check the
/decisions/{id}/owner_confirm/verify endpoint runs. Also times the whole endpoint in-process (FastAPI TestClient,
fake decision components, audit write included).

Run:  uv run --no-sync python scripts/bench_passkey.py [--n 1000]
Writes artifacts/results_passkey.json.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

COLD_SNIPPET = """
import os, time
from tests.identity.soft_authenticator import SoftAuthenticator
from fraudshield.identity.webauthn import verify_registration, verify_assertion
a = SoftAuthenticator(origin="http://localhost:8092")
ch = os.urandom(32)
cred = verify_registration(a.create(ch), expected_challenge=ch, rp_id="localhost", origins={"http://localhost:8092"})
ch = os.urandom(32); res = a.get(ch)
t = time.perf_counter()
verify_assertion(res, stored=cred, expected_challenge=ch, rp_id="localhost", origins={"http://localhost:8092"})
print((time.perf_counter() - t) * 1000)
"""


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def summary(xs: list[float]) -> dict:
    return {"n": len(xs), "p50_ms": round(pct(xs, 0.5), 4), "p95_ms": round(pct(xs, 0.95), 4),
            "p99_ms": round(pct(xs, 0.99), 4), "mean_ms": round(statistics.mean(xs), 4),
            "sd_ms": round(statistics.stdev(xs), 4) if len(xs) > 1 else 0.0, "max_ms": round(max(xs), 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--cold-runs", type=int, default=5)
    args = ap.parse_args()

    from dataclasses import asdict

    from fastapi.testclient import TestClient

    from fraudshield.api.app import build_app
    from fraudshield.identity.webauthn import b64url_decode, verify_assertion, verify_registration
    from tests.api.test_app import components
    from tests.identity.soft_authenticator import SoftAuthenticator
    from tests.policy.fixtures import make_booking

    origin = "http://localhost:8092"
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    cold = [float(subprocess.run([sys.executable, "-c", COLD_SNIPPET], capture_output=True, text=True, env=env,
                                 cwd=ROOT, check=True).stdout.strip()) for _ in range(args.cold_runs)]

    a = SoftAuthenticator(origin=origin)
    ch = os.urandom(32)
    cred = verify_registration(a.create(ch), expected_challenge=ch, rp_id="localhost", origins={origin})
    warm = []
    for _ in range(args.n):
        ch = os.urandom(32)
        res = a.get(ch)
        t = time.perf_counter()
        n = verify_assertion(res, stored=cred, expected_challenge=ch, rp_id="localhost", origins={origin})
        warm.append((time.perf_counter() - t) * 1000)
        cred = type(cred)(cred.credential_id, cred.cose_key, n)

    # whole endpoint in-process: options + verify (nonce, binding, signature, counter store, audit write)
    tmp = Path(tempfile.mkdtemp())
    cfg = {"audit_path": str(tmp / "audit" / "audit.jsonl"), "replay_path": str(tmp / "none.jsonl"),
           "demo_bookings_path": str(tmp / "demo.json"), "web_dist": str(tmp / "nodist"),
           "calibration_path": str(tmp / "nocal.json"), "learning": {"enabled": False},
           "passkey": {"extra_origins": [origin]}}
    app = build_app(cfg, components=components())
    c = TestClient(app)
    o = c.post("/passkey/enroll/options", json={"account_id": "acc_7c3e"}).json()
    c.post("/passkey/enroll/verify", json={"nonce_id": o["nonce_id"], "account_id": "acc_7c3e",
                                           "credential": a.create(b64url_decode(o["publicKey"]["challenge"]))})
    endpoint, server = [], []
    m = min(args.n, 300)
    for i in range(m):
        b = make_booking(booking_id=f"bk_bench_{i}")
        app.state.pipeline.add_record({"booking_id": b.booking_id, "account_id": b.account_id, "action": "hold",
                                       "booking": asdict(b), "source": "bench"})
        did = app.state.pipeline.get_by_booking(b.booking_id)["decision_id"]
        o = c.post(f"/decisions/{did}/owner_confirm/options").json()
        res = a.get(b64url_decode(o["challenge"]))
        t = time.perf_counter()
        r = c.post(f"/decisions/{did}/owner_confirm/verify", json={"nonce_id": o["nonce_id"], "credential": res}).json()
        endpoint.append((time.perf_counter() - t) * 1000)
        assert r["verified"], r
        server.append(r["server_ms"])

    out = {"what": "owner passkey ES256 assertion verification latency",
           "data": "SYNTHETIC: software authenticator keys and a fixture booking; no browser, no user data",
           "machine": {"platform": platform.platform(), "python": platform.python_version(),
                       "processor": platform.processor()},
           "cold_first_verify_in_fresh_process": {"runs": cold, **summary(cold)},
           "warm_verify_assertion": summary(warm),
           "endpoint_verify_in_process": {**summary(endpoint), "note": "TestClient round trip, includes audit write"},
           "endpoint_server_ms_field": summary(server)}
    p = ROOT / "artifacts" / "results_passkey.json"
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
