# Build rules (every build agent reads this first)

## Context
FraudShield scores parcel-carrier bookings for fraud at booking time. Full approved design: `../DESIGN.md` (project root, one level above this repo). Contracts: `fraudshield/contracts.py` and `docs/API.md`. Do not change contracts without telling the orchestrator in your summary (propose the change; additive optional fields are OK).

## Environment (Windows, Git Bash available)
- Python env lives OUTSIDE OneDrive. Always prefix: `export UV_PROJECT_ENVIRONMENT="C:/Users/05nik/.venvs/fraudshield"` then `uv run pytest ...` / `uv run python ...`. Already synced with `--extra ml` (torch 2.11 cu128, laya 0.3.23, transformers). Add deps with `uv add <pkg>` (or `uv add --optional ml <pkg>`), same env var.
- GPU: RTX 4050 Laptop, 6 GB VRAM. Measured: stock `laya.load("convaiinnovations/laya")` loads in ~2 min first time (cached after), 4 two-option questions = 101 ms p50, 2.5 GB VRAM peak. Full fine-tune of 421M params does NOT fit 6 GB; partial fine-tune (freeze lower layers) does.
- Node 24 / npm 11 for web and e2e.
- Ports: API 8080, laya-serve 8001 (optional), web 5173.
- Real Olist CSVs (gzipped) are in `data/raw/`. Never commit data.

## Skills (mandatory)
- Backend and ML code: Superpowers test-driven development. Read `../hackathon_claude_skills/Superpowers/test-driven-development/SKILL.md` and follow it: write a failing test, run it and see it fail for the right reason, write minimal code, see it pass, refactor. No production code without a failing test first. Also read `.../Superpowers/verification-before-completion/SKILL.md` before claiming done.
- Frontend: `../hackathon_claude_skills/frontend-design/SKILL.md`.
- E2E: Playwright (webapp-testing skill is a one-line description: generate Playwright E2E scripts).
- GSD: you are an isolated loop. Own only your directories. Return a compact summary (what was built, test counts with the actual pytest output tail, interfaces exposed, open issues), not code dumps.

## Ownership
- Data/ML agent: `fraudshield/data/`, `fraudshield/features/`, `fraudshield/models/gbm.py`, `fraudshield/models/calibrate_fit.py`, `notebooks/`, `scripts/` (data/training), `tests/data/`, `tests/features/`, `tests/models/`, `DATA_CARD.md`, `data/processed/`, `artifacts/`.
- Decision-service agent: `fraudshield/models/laya_client.py`, `fraudshield/models/calibration.py` (apply), `fraudshield/policy/`, `fraudshield/explain/`, `fraudshield/audit/`, `fraudshield/api/`, `fraudshield/sim/`, `tests/policy/`, `tests/explain/`, `tests/audit/`, `tests/api/`, `config/`.
- Frontend agent: `web/`.
- E2E agent: `e2e/`.
- Orchestrator: `contracts.py`, `docs/`, README, integration.

## Delivery: desktop app, not a website (user decision, 2026-10-02)
- The user wants a desktop app anyone can run on their PC. Approach: `pywebview` native window (Edge WebView2 on Windows, built into Windows 10/11) that shows the React console from `web/dist` (built once, shipped as static files), with the FastAPI backend running in the same Python process on 127.0.0.1. No browser, no Node needed to run it.
- Entry point: `python -m fraudshield.desktop` (orchestrator owns `fraudshield/desktop.py`); one-click launchers `FraudShield.bat` (Windows) using uv to create the env on first run.
- Must run on PCs without a GPU: Laya mode auto-selects `local` on CUDA, otherwise `cached` (precomputed answers for demo bookings, shown with a "cached" badge) with CPU as an opt-in. The `ml` extra (torch, laya) is optional; the app starts without it.
- Frontend: use relative API paths when served by the desktop app (same origin), keep VITE_API_URL for dev. No features that need a browser (no new tabs/window.open).

## Honesty and style
- Never hard-code results or fake metrics. Anything simulated (owner replies, scan outcomes, delayed labels) is labelled simulated in code, data and UI.
- No em dashes in user-facing text. Plain language.
- `Booking.meta` is never a feature. Leakage tests must enforce this.

## Env gotcha
After any `uv add`, run `uv sync --extra ml` (same UV_PROJECT_ENVIRONMENT) so torch and laya stay installed.
