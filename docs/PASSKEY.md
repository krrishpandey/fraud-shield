# Owner passkey: a booking-bound "was this you?"

`owner_confirm` used to be a simulated one-tap message to the owner's contact on file (`fraudshield/sim/outcomes.py`,
which models a fraudster who controls that channel answering "yes, it was me" with probability `p_spoof`). This feature
makes the owner step a real WebAuthn (passkey) ceremony whose signature is **bound to the booking details**.

Sources: FIDO Alliance Passkey Index, Oct 2025
(https://fidoalliance.org/fido-alliance-launches-passkey-index-revealing-significant-passkey-uptake-and-business-benefits/);
W3C Secure Payment Confirmation (https://www.w3.org/TR/secure-payment-confirmation/), which binds the signature to the
transaction details. We do the same binding inside a plain WebAuthn challenge.

## How it works

1. **Enroll** (once per account; demo: one owner per account, enrolling again replaces it and the audit says so).
   `POST /passkey/enroll/options` -> creation options (RP ID `localhost`, ES256 only, attestation `none`, user
   verification required). The browser creates the passkey; `POST /passkey/enroll/verify` checks it
   (`fraudshield/identity/webauthn.py`): clientDataJSON type, challenge and origin, rpIdHash, UP and UV flags, attestation
   format `none`, a valid COSE EC2 P-256 key. The public key is stored in `artifacts/passkeys/credentials.json`
   (runtime state, git-ignored). Audit event `passkey_enrolled`.
2. **Confirm** a stopped booking. `POST /decisions/{id}/owner_confirm/options` issues a 32-byte single-use server nonce
   (expires after 120 s) and the challenge

   `challenge = sha256(nonce || canonical JSON of {booking_id, account_id, carrier_cost, declared_value, dest_zip3, consignee_id})`

   (canonical JSON = sorted keys, no spaces, money rounded to 2 decimals). The owner's passkey signs
   `authenticatorData || sha256(clientDataJSON)`, and clientDataJSON contains the challenge.
   `POST /decisions/{id}/owner_confirm/verify` spends the nonce (pass or fail), **recomputes the challenge from the
   booking the server holds**, and checks the ES256 signature with the enrolled key, the origin, rpIdHash, UP/UV flags
   and the signature counter (if either counter is non-zero, the new one must be larger: a stalled counter can mean a
   cloned key; synced passkeys that always report 0 are accepted).
3. **Audit.** Success writes `owner_confirmed` (decision, booking, account, sha256 of the credential id, sha256 of the
   bound fields, released action, counter, verify time). Every failure writes `owner_confirm_failed` with the reason
   code (`challenge`, `origin`, `rp_id`, `signature`, `nonce`, `sign_count`, `credential`, `user_verified`, ...).
4. **Tamper test** (the stage moment). `POST /decisions/{id}/owner_confirm/tamper_test` re-checks the SAME assertion
   the owner just gave against a modified copy of the booking (default: carrier cost + R$100). The server recomputes
   the challenge from the changed booking, the signature no longer matches, and the check fails with code
   `challenge`. It is a dry run: it never releases anything, never moves the counter, and may use an already spent
   nonce (otherwise it would fail for the boring "replay" reason instead of the binding). With delta 0 it verifies,
   which shows the failure really comes from the changed field.

## What a confirmation releases

| Decision | After a verified owner passkey |
|---|---|
| `owner_confirm` | `allow` (what owner_confirm was designed for: the owner says yes, the label is issued) |
| `hold`, `review` | `allow_scan_gated` (demo extension: the model saw more risk, so the depot still weighs the parcel) |
| `allow`, `allow_scan_gated`, `block` | not applicable (409). A block needs a link to confirmed fraud, an owner cannot override it |

None of the demo bookings lands on `owner_confirm` under the shipped policy (the takeover demo is `hold`), which is why
hold and review are allowed too; the e2e spec uses the takeover demo. The decision record keeps its audited `action`;
the release is a new `owner_confirmation` field on `GET /decisions/{id}` and a new audit event.

## Why this beats a one-tap message

An account takeover means the attacker controls the login, and often the contact channel (SIM swap, changed e-mail),
which is the `p_spoof` path in the cost model. A passkey lives on the owner's device and is bound to the RP ID, so a
phished login or a hijacked inbox does not produce a valid signature, and a signature for one booking cannot approve a
different amount or address. We did **not** change the cost model's `p_spoof` or measure a new rate: there is no
measured number for it here.

Honest limits:
- The OS passkey prompt does not show the amount (Secure Payment Confirmation does; browsers support it only for
  payments). The console shows the bound fields next to the button; the binding guarantees the signature is only valid
  for exactly those fields.
- Enrollment in the demo happens on the same laptop, right before confirming. In production it happens at onboarding,
  behind the existing login, and changing the passkey is itself a risky account event.
- **The owner device is simulated by this laptop.** Labelled in the UI, the audit payloads
  (`simulated_owner_device: true`) and the credential store.
- Attestation `none`: we verify that the same passkey signs, not which device model holds it.

## RP ID, origin and the desktop window

WebAuthn forbids IP addresses as RP IDs, so the RP ID is `localhost` and the expected origin is
`http://localhost:<port of the request>` (plus `passkey.extra_origins`, default `http://localhost:5173` for the vite dev
server). At `http://127.0.0.1:<port>` the panel shows a hint to open `http://localhost:<port>` and disables the buttons.

`fraudshield/desktop.py` now opens the window at `http://localhost:<port>` and prints that URL in the console. The
server still listens on loopback only: on 127.0.0.1 and, when the PC has IPv6, on ::1 (`loopback_sockets`). Windows
resolves `localhost` to ::1 first; with only 127.0.0.1 bound, a refused ::1 connect cost about 1 s per new connection
in Python before the fallback (`tests/test_desktop.py`). The e2e backend still binds 127.0.0.1 only, and spec 16 opens
`http://localhost:8092` in Chromium, so the browser's ::1 -> 127.0.0.1 fallback is verified there.

**Windows Hello inside the pywebview / WebView2 window is unverified** (it needs a person at the PC). Fallback: open
the printed `http://localhost:<port>` in Edge, where Windows Hello passkeys work.

## Demo steps (about 45 s)

1. Start the app (`FraudShield.bat`). If the passkey prompt does not appear in the window, open the URL printed in
   the console (`http://localhost:<port>`) in Edge.
2. Score the takeover demo. The decision is Hold. Scroll to "Was this you? The owner's passkey".
3. Click "Enroll owner passkey (demo: this laptop plays the owner's phone)", then Windows Hello (PIN or face).
4. Click "Confirm as owner", Windows Hello again. Result: "Owner confirmed ... Booking released: Allow, check at first
   scan", with the verify time and audit hash.
5. Click "Tamper test: same signature, carrier cost + R$100". Result: "Rejected ... the signature no longer matches".
6. Optional: Audit page, verify the chain.

If Windows Hello is not set up on the demo laptop:
- Fastest: Settings > Accounts > Sign-in options > PIN (Windows Hello) > Set up (about a minute). Then retry.
- Or in the Edge prompt choose a phone (QR code, needs Bluetooth) or a USB security key.
- Last resort, labelled simulated: in Edge DevTools, More tools > WebAuthn > "Enable virtual authenticator
  environment" > add a ctap2 / internal authenticator with user verification. This is the same virtual authenticator
  the e2e spec uses. Say on stage that it is a software authenticator.
- If the prompt says it was cancelled or timed out, click the button again: each attempt gets a fresh challenge.

## Measured

See `artifacts/results_passkey.md` (`scripts/bench_passkey.py`, synthetic keys and bookings).

## Files

`fraudshield/identity/webauthn.py` (CBOR decoder, authData, COSE key, registration and assertion checks),
`fraudshield/api/passkey.py` (routes, nonce book, credential store, binding), `web/src/components/OwnerPasskey.tsx`,
`web/src/lib/webauthn.ts`, tests in `tests/identity/`, `tests/api/test_passkey_routes.py`,
`e2e/tests/16-passkey.spec.ts`. Dependency: `cryptography>=50` (ES256 verify).
