# FraudShield end-to-end tests

Playwright and TypeScript tests that drive the real console (built `web/dist`, served by the FastAPI backend) through the demo storyline in `../DESIGN.md` section 11.

## Run

```
cd e2e
npm install
npm run e2e            # whole suite, about 1 minute
npm run e2e:headed     # same, with a visible browser
npm run e2e:llm        # optional @llm test: keeps GROQ_API_KEY / ANTHROPIC_API_KEY from .env
npm run report         # open the last HTML report (e2e/.tmp-report)
```

Prerequisites: the Python env at `C:/Users/05nik/.venvs/fraudshield` (override with `UV_PROJECT_ENVIRONMENT`), and a current `web/dist` (`cd web && npm run build` if the console changed). Browser: the Playwright Chromium. If it cannot be downloaded, set `E2E_CHANNEL=msedge` to use the installed Edge.

## What the run does

- `playwright.config.ts` starts the backend through `start-server.mjs` on port 8090 (`E2E_PORT` to change it) and stops it at the end. It never reuses a running server.
- `start-server.mjs` deletes `e2e/.tmp/` first, so every run gets a fresh audit log, label store, model registry and model versions. Backend output goes to `e2e/.tmp/server.log`.
- Config: `e2e/app.e2e.yaml`, a copy of `config/app.yaml` with:
  - `laya.mode: cached`. The model is never loaded.
  - `policy.explore_eps: 0`, so actions are deterministic.
  - `audit_path`, `learning.registry_path` and `learning.versions_dir` under `e2e/.tmp/`. The run reads the v1 model from `artifacts/models` and never writes there.
- No GPU use: `CUDA_VISIBLE_DEVICES` is empty, and `e2e/pyshim/torch` is first on `PYTHONPATH`, so `import torch` fails. The `/health` GPU probe imports torch.
- LLM keys are set to empty strings, so explanations come from the template and the tests stay deterministic.
- Specs share one backend and run in file order with one worker.

## Tests

| File | Covers |
|---|---|
| 01-health | Status bar shows cached Laya, CPU only, and degraded or online, matching `/health`. |
| 02-legit | The tenured booking is allowed. Probability, latency and the degraded flag are shown, and the allow cost row is chosen. |
| 03-hard-negative | The first parcel to a new state gets allow or allow_scan_gated. |
| 04-takeover | The takeover booking gets hold, review, block or owner_confirm. Exactly one cost row is chosen, reason codes are shown, and the template explanation shows its source and validator chips. |
| 05-ask | A new question shows either a probability or the API error. In cached mode the API returns 503 and the panel shows the error. No page errors. |
| 06-analyst | Confirming fraud shows a 64-hex audit hash. The queue row shows "Fraud". A newly scored booking raises the bookings KPI by 1. |
| 07-audit | The console verify is ok. A byte flipped in a copy of the log makes `python -m fraudshield.audit verify` exit 1 with `first_bad_index 0`, and the live log is still ok. |
| 08-learning | Simulate shows a result or an error. Retrain with at least 20 new labels gives a result table and gate checks, and the deployed banner matches `gate.passed`. Rollback is offered on non-active versions, and when the candidate was deployed the test rolls back. |
| 09-idempotency | Scoring the same booking twice returns the same decision id and audit hash through the API. In the console, scoring twice adds no booking. |
| 10-llm (`@llm`) | Skipped by default. With a key present, the explanation source is `llm` and the validator passes. |

## Known app bugs pinned by expected-to-fail tests

Each of these is marked `test.fail()`. When the app is fixed, Playwright reports the test as "unexpectedly passed". Then remove the marker, and the workaround where there is one.

1. `POST /learning/simulate_feedback` returns 500 with `KeyError: 'base_consignees_l10'`. `learning/simulate.py` `feature_vector_from_row` drops NaN features, but `features/serialize.py` reads `v['base_consignees_l10']` directly. About 28% of test-window rows have that NaN. Workaround: the retrain test gets its 20 or more new labels from analyst confirmations made through the API.
2. The Learning view crashes on a blank page with "Cannot read properties of null (reading 'replace')". `LearningView.tsx` calls `v.created_at.replace(...)` and `v.metrics.pr_auc`. For the pipeline-trained v1, `created_at` is always null, and `metrics` is null until the first retrain. Workaround: in the browser only, `patchNullVersionFields` replaces those nulls before the view sees them. The API response is not changed.
