# FraudShield: project history

NextGenAI hackathon, Theme 3 (FraudShield): score parcel-carrier bookings for fraud at the moment of booking, before the package enters the network.
This file records what we did, in what order, what we decided and why, and what the numbers were. Status as of 2026-10-03 (early morning IST).

---

## 0. START HERE (teammates)

**Where we are:** the app works end to end (desktop window, scoring, explanations, audit log, dashboard, continuous learning). What is NOT done yet: Laya fine-tuning (moves to Kaggle), the final results table, the README, and demo rehearsal. See section 7 for the task list.

**1. Run the app (10 minutes, Windows)**
1. Unzip, open the `fraudshield` folder, double-click `FraudShield.bat`. First run installs `uv` and Python packages (needs internet).
2. A FraudShield window opens. Status bar: "cached" or "local" (Laya), and whether it is "degraded".
   - Today it runs **degraded** (LightGBM backup score) because the fine-tuned Laya model and its answer cache do not exist yet. That is expected until the Kaggle run is imported.
3. Click through: **Score** (pick a demo booking) -> **Decision** (probabilities, cost table, explanation, ask a new question, confirm fraud) -> **Queue** -> **Dashboard** -> **Learning** (simulate feedback, retrain) -> **Audit** (verify).
4. Explanations use Groq (key in `.env`, never commit or post it). Without a key they fall back to a fixed template.

**2. Laptop GPU rule:** never train or bulk-score on a laptop GPU (one teammate's screen was damaged). Live demo use (one Laya call per booking) is fine. Training runs on Kaggle.

**3. Read in this order (30 minutes):** this file -> `../DESIGN.md` sections 5, 6, 11 (pitch, architecture, demo) -> `docs/API.md` -> `DATA_CARD.md` (what is real vs injected) -> `docs/LEARNING_GATE.md` (continuous learning runs).

**4. Developer commands** (Git Bash; the Python env lives outside OneDrive):
```
export UV_PROJECT_ENVIRONMENT="$USERPROFILE/.venvs/fraudshield"
uv sync --extra ml                      # first time (use without --extra ml on PCs without NVIDIA GPU)
uv run pytest -q -m "not gpu"           # Python tests (~2 min)
uv run python -m fraudshield.desktop    # run the desktop app
uv run uvicorn fraudshield.api.app:create_app --factory --port 8080   # API only
cd web && npm install && npm run dev    # console in dev mode (http://localhost:5173); npm run dev:mock = no backend
cd e2e && npm install && npm run e2e    # Playwright end-to-end tests (~40 s, no GPU)
python scripts/make_team_zip.py         # rebuild the team zip (includes .env)
```

**5. Next tasks, suggested split (details in section 7):**
| Who | Task |
|---|---|
| Person with the Kaggle account | Run Laya fine-tuning on Kaggle (`kaggle/KAGGLE_STEPS.md`; upload `kaggle/dist/fraudshield_laya_bundle.zip` (5 MB) as a PRIVATE Kaggle dataset), download the output zip, run `python kaggle/import_kaggle_output.py <zip>` |
| ML person | After import: check `artifacts/results_laya.md` (fine-tuned Laya vs LightGBM on held-out T3/T5), fill the results table in the README |
| Frontend/demo person | Rehearse the 5-minute demo (`../DESIGN.md` section 11); prepare a backup screen recording |
| Writer | README + slides from this file; state limitations honestly (section 6) |

---

## 1. Timeline

| When (2026-10-02/03) | Phase | What happened |
|---|---|---|
| Afternoon | 1. Research | Four parallel research agents: fraud domain, Jev/Laya verification, datasets, RL literature. |
| Afternoon | 2. Ideation | 16 ideas brainstormed, 8 killed, top 3 developed, one recommended. |
| Afternoon | 3. Design | Four parallel design agents: Laya fine-tuning, architecture, RL, evaluation and demo. |
| Evening | 4. Review | Independent judge agent (saw only the design) and a fact-check agent. Design revised. |
| Evening | Approval | Team approved the design. Jev dropped (we use Laya only). Team GPU: RTX 4050 Laptop, 6 GB. |
| Evening | 5. Build | Data/features, decision service and frontend agents in parallel; then continuous learning; then Laya fine-tuning. |
| Evening | Change | Team asked for a desktop app instead of a website: switched to a native window (pywebview). |
| Night | Change | Organizer brief re-checked; added continuous learning. Groq API key supplied for GenAI explanations. |
| Night | Laya fine-tune | Started on the RTX 4050 (9.5 items/s, ~5.5 of 6 GB in use with the desktop). Stopped at step 600 at the team's request: heavy GPU load had damaged a laptop screen before. |
| Night | Change | Fine-tuning moved to Kaggle free GPUs (2x T4): bundle + notebook + import script prepared locally on CPU. Live demo may use the laptop GPU lightly (one Laya call per booking). |

Two pauses happened when the AI session hit its usage limit; agents were resumed afterwards with their work intact.

---

## 2. The idea

**Hook:** "Stolen shipping accounts aren't used to ship a parcel. They're used to open a shipping shop. We catch the moment an account starts paying for parcels from places and people it never shipped for."

Why: DOJ cases (Wilson/Harrod: a company's UPS login used first for the fraudsters' own shipments, then to sell labels to third parties, >$900K) and Hao et al. (CCS 2015, "Drops for Stuff") show stolen carrier accounts are monetized as label-resale businesses. Fraudulent labels sell at a flat price regardless of weight or distance (LabelsBank, about $2 per label), so buyers favor expensive parcels. USPS OIG (Sep 10 2026, 25-072-R26): about $3.1B lost to counterfeit postage; 97% of counterfeit labels were processed for delivery; about 2M paid packages mistakenly intercepted. So: decide at booking, and avoid stopping legitimate parcels.

What we score: each booking against what the paying account normally pays for (senders, origins, logins, devices, package profile, cost per parcel), plus the fraudster's payoff. A fine-tuned Laya model answers named yes/no questions with calibrated probabilities. A cost rule picks one of six actions: allow, allow with a check at first scan, ask the owner to confirm, analyst review, hold, block.

Ideas we killed and why (full list in `../DESIGN.md` section 3): LLM agent reviewing every booking (latency, cost, Rule 1), XGBoost + LLM chatbot (most common submission), Isolation Forest + dashboard, blockchain audit log (a hash chain does the job), canary account numbers (good security control but the AI is removable).

---

## 3. Key decisions and why

| Decision | Why |
|---|---|
| Laya (open weights) fine-tuned; Jev not used | Jev cannot be fine-tuned on customer data (TypeSafe docs) and has ~236-276 ms independent latency; Laya is Apache-2.0 and runs on our GPU. Team chose Laya only. |
| Cost rule decides the action, not RL | With known costs and calibrated probabilities the cheapest-expected-cost action is optimal (Elkan 2001). RL used only for logged exploration and off-policy evaluation, plus an attacker stress test. |
| Yes/no questions as two-option Choice with neutral keys | Laya issue #156: Noul can follow its labels instead of the input. |
| Olist as base data, fraud injected and labelled | No public parcel-fraud dataset exists. Olist is real (CC BY-NC-SA 4.0) with repeat shippers, origin/destination, weight, freight, handoff time. Every injected row is flagged with scenario, seed and parameters. |
| Consignee novelty dropped as a signal | Real Olist data: established sellers send 94.9% of bookings to never-before-seen postcodes. The signal moved to the sender side. |
| Two result tiers (real columns only, and all columns) | Judge review: our strongest signals live in synthetic columns, so we report both. |
| Two fraud types held out of training (T3 mule drops, T5 test-then-burst) | Tests whether the model finds patterns it never saw. T5 is an unsourced hypothesis and is labelled so. |
| Desktop app (pywebview) instead of a website | Team request: easy to run on any PC. One process, native window, no browser, no Node needed at run time. |
| Groq for GenAI explanations (`openai/gpt-oss-120b`) | The team's key is a Groq key. Explanations are validated: any number not in the decision record is rejected and the template is shown instead. |
| Learning gate written down before reruns | Prevents tuning the safety check until it passes. Every run, deployed or not, is recorded in `docs/LEARNING_GATE.md`. |

---

## 4. Results so far (all from real runs)

### Data
- Olist: 100,010 bookings, 3,095 sellers. Splits (real bookings): warm-up 11,547, train 41,042, calibration 17,920, test 22,435.
- Injection: 10 seeds, 15,529 injected rows (8,563 fraud), 300 test fraud campaigns, test prevalence 1.17 to 1.41%.
- Laya input states: median 327 tokens, max 334 (limit 380).

### "Is the injected fraud too easy?" checks
| Check | Result |
|---|---|
| Rules only, hardest camouflage level, PR-AUC (target < 0.8) | 0.042 (real columns), 0.050 (all) : pass |
| Shuffled labels PR-AUC (should be about prevalence 0.012) | 0.0108 : pass |
| Injected rows vs real rows, non-semantic columns (target AUC <= 0.6) | 0.535 : pass |
| Injected legit vs real legit, all comparable columns | 0.617 : narrow fail (started at 0.82 before generator fixes) |

### LightGBM baselines (test window, 10 seeds)
| System | PR-AUC | Precision at top 1%/day | FPR legit |
|---|---|---|---|
| Rules only (real columns) | 0.17 | 0.27 | 0.8% |
| LightGBM, real columns only | 0.73 ± 0.04 | 0.55 | 0.5% |
| LightGBM, all columns | 0.77 ± 0.04 | 0.57 | 0.5% |
- Held-out T3 (mule drops): PR-AUC 0.31, the gap Laya's zero-shot "drop" question is meant to close.
- Known weakness: in the all-columns tier, label resale (T1) and third-party billing (T7) are near-trivial, because real Olist sellers never ship for others. The real-columns tier is the fair headline.

### Laya on the RTX 4050
- Stock Laya, 4 questions per booking: 101 ms p50, 2.5 GB VRAM. Zero-shot answers poor (misuse 0.43 on an obvious takeover), so fine-tuning is required.
- Fine-tuning locally: top 8 of 28 layers + head (125M trainable), 9.2-9.5 items/s, 3.45 GB allocated. Stopped after 600 steps to protect the laptop; now runs on Kaggle. Training set: 5,958 bookings (19.4% fraud; the 386 real-window frauds repeated 3x, so the DESIGN target of 1,200 distinct frauds was not reachable), 29,790 items per epoch, max sequence 438 of 512 tokens, nothing truncated.

### End-to-end pipeline smoke test (LightGBM only, Laya answers not cached yet)
| Demo booking | Truth | Action |
|---|---|---|
| Tenured seller, normal booking | legit | allow |
| Real hard negative (first shipment to a new state) | legit | allow |
| Label-resale takeover | T1 | hold |
| Reshipping drop (held-out pattern) | T3 | allow (missed by LightGBM) |
| Declared-weight manipulation | T6 | allow (only detectable at first scan) |
Warm latency about 90 ms; audit chain verified. GenAI explanation via Groq accepted by the validator.

### Continuous learning
The loop works end to end: analyst labels (and clearly flagged simulated feedback) build a candidate model, which is compared with the active model on data neither trained on, and deployed only if a pre-registered gate passes.
- The retrained model learns the unseen patterns: catch rate on held-out types 0.53 to 0.78 (mule drops 0.28 to 0.69), whole confidence interval above zero.
- The gate rejected every candidate so far, because false positives on legitimate hard cases rose (0.51% to 2.48%). Diagnosis: the new test-then-burst pattern looks like a legitimate account adding a new sales channel (recent login on an old account).
- With stronger regularization (run R3) the candidate beats the current model on every point estimate (PR-AUC 0.817, T3 catch rate 0.28 to 0.58) but still fails two gate checks (hard-negative FPR +0.95 points vs 0.5 allowed; cost CI too wide). We kept the gate unchanged.
- Demo story: the system learns new fraud from analyst confirmations, and the gate stops a model that would hurt legitimate customers.

### Tests
- Python: 344 passing (excluding the GPU-only test, which passes separately). All backend and ML code was written test-first (Superpowers).
- Playwright end-to-end: 14 tests, all passing, about 20-40 s, no GPU (webapp-testing). They found 3 real bugs, now fixed:
  1. "Simulate feedback" crashed (500) on rows with missing features; fixed in `learning/simulate.py`.
  2. The Learning screen went blank for the base model (no date/metrics yet); fixed in `web/src/views/LearningView.tsx`.
  3. Rollback was offered for versions the gate had rejected; now they show "Rejected by gate".
  Also: `/health` no longer imports torch on every call; the console's simulate button now advances the simulated clock 90 days so delayed labels arrive.
- Frontend: lint, type check and build pass.

---

## 5. What was built (repo map)

```
fraudshield/
  FraudShield.bat            one-click Windows launcher
  fraudshield/desktop.py     native window + in-process API
  fraudshield/data/          Olist loader, synthetic billing layer, fraud injection (T1-T7), splits
  fraudshield/features/      feature store, as-of featurizer, Laya input serializer
  fraudshield/models/        LightGBM baselines, Laya client (local/http/cached), calibration
  fraudshield/policy/        cost rule, guards, exploration, off-policy evaluation, review queue
  fraudshield/explain/       template, LLM rewrite (Anthropic or Groq), validator
  fraudshield/audit/         hash-chained audit log + verify command
  fraudshield/learning/      label store, retrain, gate, registry, rollback, simulated feedback
  fraudshield/api/           FastAPI app and wiring
  web/                       React + Tailwind analyst console (desktop layout)
  docs/                      API contract, build rules, learning gate, data outputs, references
  DATA_CARD.md               every column: real, synthetic or injected
```

Skills used: GSD (isolated agent loops with compact summaries), Superpowers (test-first for all backend and ML code), frontend-design (console), webapp-testing (Playwright, pending).

---

## 6. Honest limitations
- The billing/login/device layer is synthetic; Olist has none of it. Results are split into real-column and all-column tiers.
- Owner replies, scan results, analyst labels and delayed disputes in the demo are simulated and labelled as such.
- freight_value in Olist is paid by the buyer; we use it as a stand-in for carrier cost.
- All cost-matrix numbers except freight are assumptions.
- "Test shipment before the big one" (T5) has no carrier-specific source; it is a hypothesis.
- One generator-artifact check narrowly fails (0.617 vs 0.6).
- Continuous learning has not yet produced a model that passes the gate.
- Laptop GPU must not be used for long, heavy jobs (training, bulk inference); those run on Kaggle.

---

## 7. Kaggle package (ready, not yet run)
- `kaggle/fraudshield_laya_kaggle.ipynb`: pilot (picks layers/batch from measured VRAM), train (fp16 on T4, checkpoint + resume each epoch), score stock and fine-tuned Laya, calibrate, evaluate vs LightGBM, build the demo answer cache, zip outputs (~0.8 GB).
- Checked on CPU only (tiny runs, resume, import into a repo copy). Never run on a GPU yet: the first Kaggle run is the real test. Estimated 3-5 hours; the pilot decides and any fallback is written into the results.
- After the run: download `fraudshield_laya_output.zip`, then `uv run python kaggle/import_kaggle_output.py <zip>`. It checks weights, calibration and cache share one model revision, copies them into `artifacts/`, and points `config/app.yaml` at the fine-tuned model.
- Note: on CPU, fine-tuned Laya takes ~4.3 s per booking, so PCs without an NVIDIA GPU use the cached answers.

## 8. Still to do
1. Run Laya fine-tuning on Kaggle (see `kaggle/KAGGLE_STEPS.md`), import its output; calibration; compare stock vs fine-tuned Laya vs LightGBM on held-out T3/T5 and the zero-shot drop question.
2. Record Laya answers for demo bookings (cache) so PCs without a GPU can run the full demo.
3. Replay a test-window sample so the dashboard shows real numbers.
4. Playwright end-to-end tests for the demo flow (webapp-testing skill).
5. README with setup steps and the final results table.
6. Rehearse the 5-minute demo (storyline in `../DESIGN.md` section 11).
7. Share fine-tuned Laya weights with teammates who have a GPU (private Hugging Face repo or drive).
8. Rotate the Groq API key after the hackathon (it was shared in chat).
