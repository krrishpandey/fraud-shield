# Owner passkey: verification latency

Feature: `docs/PASSKEY.md`. Numbers from `scripts/bench_passkey.py` (`artifacts/results_passkey.json`) and the e2e spec
`e2e/tests/16-passkey.spec.ts`. **All data synthetic**: software or virtual authenticators, fixture bookings, no users.
No train/test split applies (no model involved). Machine: Windows 11 laptop, AMD Family 25 Model 68 CPU, Python 3.12.13,
cryptography 50.0.2, CPU only.

| What is timed | n | p50 | p95 | p99 | mean ± sd |
|---|---|---|---|---|---|
| `verify_assertion` (ES256 check + clientData/authData checks), warm process | 1000 | 0.176 ms | 0.290 ms | 0.681 ms | 0.203 ± 0.205 ms |
| first `verify_assertion` in a fresh Python process (after the authenticator signed) | 5 | 0.299 ms | 0.345 ms | | 0.311 ± 0.024 ms |
| verify endpoint, server side (`server_ms`: nonce, binding, signature, counter file write) | 300 | 2.61 ms | 3.33 ms | 4.48 ms | 2.71 ± 0.40 ms |
| verify endpoint round trip in-process (TestClient, includes the audit write) | 300 | 8.92 ms | 15.16 ms | 19.84 ms | 9.95 ± 2.82 ms |

End to end in the browser (e2e spec 16, Chromium with a CDP virtual authenticator, backend on 127.0.0.1, page at
`http://localhost:8092`), one confirmation per run, 3 runs:

| Run | first signature check in the fresh server process (`verify_ms`) | verify endpoint (`server_ms`) | click "Confirm as owner" to "released" shown |
|---|---|---|---|
| 1 (spec 16 alone) | 22.8 ms | 25.4 ms | 111 ms |
| 2 (full suite) | 37.8 ms | 40.2 ms | 130 ms |
| 3 (full suite) | 16.7 ms | 18.4 ms | 88 ms |

The first check in the live server is 17 to 38 ms (n=3), much slower than a warm check (0.18 ms p50): one-time
initialisation on the first ECDSA verify in that process (the benchmark's fresh-process run had already signed, so it
does not show this). Later checks in the same server are warm. A real Windows Hello prompt adds the person's PIN or face
time on top; that was not measured (needs a person at the PC).

Correctness checks (unit tests, `tests/identity/test_webauthn.py`, `tests/api/test_passkey_routes.py`, all pass):
valid enroll, valid confirm, wrong challenge, tampered booking field (carrier cost, destination ZIP3), replayed nonce,
nonce from another booking, expired nonce, failed attempt spends the nonce, wrong origin, wrong RP ID, another
account's passkey, bad signature, flipped authenticatorData byte, missing user verification, stalled or backwards
signature counter, non-"none" attestation, restart keeps the enrolled key.
