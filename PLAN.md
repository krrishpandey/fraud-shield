# Tracd Verify: plan

Brief: `reference/Tracd_Verify_Bolna_Buildathon.pdf` (outside the repo, in `F:\PROJECTS\fraud-shield\reference\`; the task text said `docs/`).
Deadline 31 Oct 2026, 11:59 PM IST. Order of work follows the brief's cut order: never cut the owner call.

## What already exists (keep, plug in)

| Piece | Where | Notes |
|---|---|---|
| Scorer | `fraudshield/api/pipeline.py`, `models/gbm.py` | LightGBM (B2_F) + isotonic calibration; Laya v2 lost its gate, so LightGBM decides (`config/app.yaml laya.decide: false`). |
| Cost policy, 6 actions | `policy/costs.py`, `policy/decide.py`, `config/costs.yaml` | allow / allow_scan_gated / owner_confirm / review / hold / block. 30-day owner-contact guard already exists (`owner_trusted`). Block needs a hard signal. |
| Hard signals | `pipeline.HARD_SIGNALS` | `LINK_TO_CONFIRMED_FRAUD`, `LAYA_MISUSE_AND_MODE_HIGH`. |
| Audit log | `audit/log.py`, `python -m fraudshield.audit verify` | Append-only JSONL, SHA-256 chain, verify + tamper detection already tested. |
| Owner "was this you?" | `api/passkey.py` | Passkey release pattern: a release is a NEW record field + NEW audit event; the decision is never edited. Voice copies this pattern. |
| Analyst queue, explanations with number checker | `policy/queue.py`, `explain/*` | LLM explanation + claim checker + template fallback already exist (brief SHOULD item: done). |
| Data | Olist + billing layer + injected fraud, time split | `data/*`, `scripts/build_dataset.py`. Reused as is. |
| UI | `web/` React 19 + Vite + Tailwind, Tracd console (redesign ported to main) | Dense dark console, mono for IDs. Extend, not restyle. |

## Contradictions / assumptions (none block the work)

1. **Brazilian data vs Indian demo story.** Olist lanes are Brazilian states and costs are BRL. The two demo scenarios run through the real model on real Olist accounts and are shown with *demo display labels* (Chennai -> Delhi, Hosur), clearly marked as labels; the real lane and BRL cost stay visible.
2. **Re-score after "yes".** The model gives both demo bookings p ~ 0.997. A Bayes factor alone (P(yes|fraud) = `p_spoof` 0.10 vs P(yes|legit) 0.95, both ASSUMPTIONS) leaves p ~ 0.97 = still held. So "yes" re-scores in two steps, as the brief describes: (a) the owner's stated change neutralises the features it explains (new warehouse -> sender/origin novelty; new staff login -> device age) and LightGBM re-scores; (b) the yes factor applies; (c) the cost policy re-runs with a floor of `allow_scan_gated`. Hard signals keep the booking held for an analyst whatever the answer. A "yes" with no change reason gets only the Bayes factor.
3. **Bolna API** (read 31 Oct 2026 from bolna.ai/docs): `POST /v2/agent`, `PUT /v2/agent/{id}`, `POST /call` (`agent_id`, `recipient_phone_number`, `user_data`, `agent_data.language`), `GET /executions/{id}`, `POST /dispositions/bulk` for the extraction fields. Webhook = POST of the execution object, several per call (one per status). **No signature header is documented**, so the webhook URL carries a secret query token (`?token=`) and Bolna's three source IPs are listed in the README. `extracted_data` shape is documented inconsistently (flat vs `{category: {name: {objective...}}}`): the parser accepts both, and falls back to a keyword classifier on the transcript (marked `extraction_source: transcript_fallback`).
4. **No-answer retry:** own scheduler (retry once after `VERIFY_RETRY_DELAY_S`, then SMS stub, then analyst queue), not a Bolna Workflow: the Workflow API needs a published graph and callback plumbing that cannot be tested without credits today. Bolna's `retry_config` on `/call` is the documented alternative; noted in the README.
5. **Trigger:** calls start on `hold` (brief) and on `owner_confirm` (the existing action that literally means "ask the owner").

## Files

| File | Change | Priority |
|---|---|---|
| `fraudshield/voice/__init__.py` | new package | MUST |
| `fraudshield/voice/agent_config.json` | agent definition as code: prompts en/hi/ta, voices, multilingual config, dispositions | MUST |
| `fraudshield/voice/setup_agent.py` | create/update the agent + dispositions via the Bolna API | MUST |
| `fraudshield/voice/bolna.py` | thin HTTP client (place call, get execution) | MUST |
| `fraudshield/voice/outcome.py` | extraction parsing, outcome mapping, re-score maths, change-reason damping map | MUST |
| `fraudshield/voice/service.py` | verification state machine: start, webhook (idempotent), retry, SMS stub, simulated calls, analyst cases, metrics | MUST |
| `fraudshield/voice/router.py` | FastAPI routes (`/verify/*`, `/voice/webhook`) | MUST |
| `fraudshield/voice/owners.py` + `config/owners.yaml` | owner directory: phone (from env), language, phone-changed date | MUST |
| `fraudshield/api/pipeline.py` | `rescore()` with damped features; active-change damping in `_score` | MUST |
| `fraudshield/api/app.py` | mount the voice router; auto-start verification after `/score` | MUST |
| `fraudshield/audit/log.py` | new event types (`verify_*`) | MUST |
| `data/processed/verify_scenarios.json` | the two seeded demo scenarios | MUST |
| `tests/voice/*` | policy, mapping, idempotency, hard signal, 30-day rule, no-answer, chain verify incl. tamper | MUST |
| `web/src/views/VerifyView.tsx`, `web/src/verify.css`, client/types, nav, DecisionView panel | feed, timeline, call panel, analyst cases, metrics, Run demo | MUST |
| `scripts/expose_webhook.ps1` / `.sh` | cloudflared tunnel | MUST |
| `scripts/bench_verify_latency.py` | p50/p95 scoring latency | MUST |
| `README.md`, `.env.example`, `DEMO.md` | docs | MUST |
| Laya fine-tuning | already done earlier (lost its gate, `artifacts/results_laya_v2.md`): report as is | IF TIME: none |
| Depot voice brief | | IF TIME: cut |

## Order

1. Backend voice package + pipeline rescore + audit events.
2. Tests.
3. Demo scenarios + end-to-end run in simulated mode.
4. UI (Verify page, DecisionView panel).
5. Latency bench, lint/typecheck/build, screenshots.
6. Docs, commits.
