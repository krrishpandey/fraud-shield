# Phase 3: FraudShield architecture (2026-10-02)

Chosen idea (from phase2_ideation.md): "Who is this account shipping FOR?" Score each booking against the account's counterparty profile (who it ships for, from where, to whom) and the fraudster's payoff per label. A fine-tuned Laya model reads a compact serialized account story and returns calibrated answers. A cost-sensitive policy with a conformal allow-guard picks one of six actions. A thin bandit layer does logged exploration and off-policy evaluation (OPE) only. An LLM writes analyst explanations from structured outputs only. Every decision lands in a hash-chained audit log.

Labels used below:
- VERIFIED: in phase1 notes or checked on the web today, source given.
- VENDOR: a Convai/TypeSafe figure, not reproduced by us.
- ESTIMATE: our engineering guess, to be measured.
- ASSUMPTION: a business parameter we chose; it goes in config and needs sign-off from the carrier.
- SPECULATIVE: a design idea we have not validated.

---

## 0. Build scope: hackathon vs production

| Component | Hackathon build | Production design |
|---|---|---|
| Event source | Replay of Olist-derived booking stream plus injected typologies (parquet), pushed to POST /score | Booking API pre-commit hook plus Kafka topics for account, payment and scan events |
| Online feature store | In-process Python state (dicts, Counters, ring buffers), snapshotted to parquet | Redis (or Feast online on Redis), keyed by account / payer / consignee |
| Offline store | Parquet partitioned by event date | Same, on object storage, with point-in-time joins |
| Graph | networkx/pandas degree and novelty counts; optional PyG GraphSAGE trained offline | Incremental counters in Redis plus hourly offline GNN embeddings |
| GBM | LightGBM, CPU | Same |
| Laya | laya-serve on one T4 (Kaggle/Colab tunnel or a cloud GPU), fine-tuned checkpoint | 2+ GPU replicas behind a load balancer, pinned HF revision |
| Policy, conformal, exploration | Python module | Same, with policy config in git |
| Explanations | Claude via the Anthropic API, async worker (asyncio queue) | Same, worker pool on a durable queue |
| Audit | Single JSONL file, single writer, hash chain | Per-partition chains, minute-level Merkle root, head hash anchored to WORM storage |
| Console | React + Tailwind, polling the FastAPI backend | Same, with SSO and role-based access |
| First-scan, owner channel, payment | Simulators with documented parameters | Real integrations (section 3) |

---

## 1. Components and data flow

### 1.1 Component diagram

```
                       (sync, pre-commit)                                    (async)
 +-------------+  POST /score   +-----------------------------------------------+
 | Booking API |--------------->|              FraudShield Scoring API          |
 |  (web/API   |<---------------|   FastAPI, one process per core, stateless    |
 |  label buy) | decision JSON  +-----------------------------------------------+
 +-------------+                  |  1 load state       ^ 9 audit.append (WAL)
        |                         v                     |
        |            +----------------------+   +---------------+
        |  booking   | Online feature store |   |  Audit log    |  JSONL, sha256 chain
        |  events    | per-account profile, |   |  (append-only)|-----> daily head hash
        +----------->| counterparty sets,   |   +---------------+       anchored (WORM)
                     | payer/consignee      |           ^
  Account events --->| graph counters,      |           | explanation, analyst,
  (login linked,     | payment status cache |           | outcome events appended
   PIN, contact chg) +----------------------+           |
  Payment events --->        |                          |
  (ACH status,               | 2 featurize(booking, history) -> FeatureVector
   returns R10/R29)          v                          |
                   +-----------------------------+      |
                   | Rules engine (YAML, versioned)  floor action + rule hits
                   | Change detection: novelty rate, diversity jump, CUSUM,
                   |   payoff-per-label z         |      |
                   | Billing graph features (+ GNN embedding lookup)
                   +-----------------------------+      |
                             | 3 LightGBM score p_gbm (isotonic-calibrated)
                             v                          |
                   +-----------------------------+      |
                   | Serializer -> state text    |      |
                   | (~300 tokens, no free text) |      |
                   +-----------------------------+      |
                             | 4 POST /v1/systemone (timeout 180 ms)
                             v                          |
                   +-----------------------------+      |
                   | laya-serve (fine-tuned Laya)|      |
                   | 3 questions, one call       |      |
                   +-----------------------------+      |
                             | 5 calibrate: per-question temperature set vN
                             v                          |
                   +-----------------------------+      |
                   | Policy: expected cost argmin|      |
                   | + rule floor + conformal    |      |
                   |   allow-guard + caps        |      |
                   | + exploration (propensity)  |------+  6 decision record
                   +-----------------------------+
                             | 7 action -> Booking API (and scan / owner systems)
                             | 8 non-allow -> explanation queue
                             v
 +------------------+   +--------------------+   +------------------------------+
 | First-scan system|   | Owner-confirm      |   | Explanation worker           |
 | (scan-gated hold |   | channel (push/SMS/ |   | template -> LLM -> validator |
 |  before linehaul)|   | email, one tap)    |   | -> fallback template         |
 +------------------+   +--------------------+   +------------------------------+
         | scan result          | approve/deny             | explanation
         v                      v                          v
 +-------------------------------------------------------------------------+
 | Analyst console (React+Tailwind): queue by expected loss per minute,    |
 | decision detail, override, labels; dashboard metrics; audit verify      |
 +-------------------------------------------------------------------------+
         | analyst decisions, scan mismatches, owner denials, chargebacks,
         | ACH returns, owner complaints   (delayed labels)
         v
 +-------------------------------------------------------------------------+
 | Label store (label, source, label_time, maturity) -> offline parquet    |
 | -> weekly: GBM retrain, temperature refit, conformal threshold refit    |
 | -> monthly: Laya fine-tune (notebook recipe), held-out typology eval    |
 | -> OPE (IPS/SNIPS/DR via obp) on logged propensities before any policy  |
 |    change -> shadow -> canary                                           |
 +-------------------------------------------------------------------------+
```

### 1.2 Step-by-step flow for one booking

1. Booking API calls POST /score before committing the label. Payload: booking fields (account_id, payer_account_id, login_id, channel, sender, origin zip, consignee, destination zip/country, weight, dims, service, declared value, timestamp).
2. Load state from the online store: account profile, payer and consignee graph counters, latest account events, cached payment status (or a synchronous payment lookup with a 30 ms timeout).
3. featurize(booking, history) builds a FeatureVector: raw booking features, deltas against the account's own history, change-detection signals, graph features, identity and payment signals. The same function runs offline when we replay the event log, so training features and serving features cannot drift apart. That code is shared, not a reimplementation.
4. Rules engine evaluates versioned YAML rules. Output: rule hits and a floor action (the minimum severity, e.g. "at least review").
5. LightGBM scores the FeatureVector. Output: p_gbm, calibrated with isotonic regression on a held-out set. p_gbm is used two ways: as an input band to the Laya state, and as the full decision probability in degrade mode.
6. serialize(features) builds the state text (deterministic template, versioned).
7. laya_client.ask(state, questions) sends one POST /v1/systemone call with 3 questions.
8. Calibration applies our own per-question temperature set (versioned, client-side) to the returned probabilities.
9. policy.decide picks the action: expected-cost argmin, rule floor, conformal allow-guard, segment thresholds, per-account caps, then exploration. It returns the action, its propensity, and the expected cost of every action.
10. audit.append writes the decision record synchronously to the local write-ahead JSONL before the response goes out. No audit record, no decision: if the write fails, the API returns the degrade-mode action and alerts.
11. The response goes to the Booking API. Side effects run async: create the scan-gate flag, the owner-confirm request, or the review case.
12. For every non-allow action (plus a 1% sample of allows for QA), enqueue explain(decision). The worker appends an "explanation" event to the audit log, linked by decision_id.
13. Later events (analyst decisions, owner approve/deny, scan mismatch, chargeback, ACH return, owner complaint) are appended as "feedback" or "outcome" events linked by decision_id, and copied to the label store.

### 1.3 Online feature store contents (per key)

Per account (key `acct:{id}`):
- First-seen timestamp, booking count, tenure segment.
- Counterparty sets with first-seen time: senders (name+address hash), origin zip3/zip5, origin facility, consignees (address hash), destination zip3 and country, login_ids, API keys and integration ids. Exact sets capped at 5,000 members, with a Bloom filter beyond that (production).
- Ring buffer of the last k=20 bookings: counterparty ids, payoff, weight, service, timestamp.
- Distribution sketches: weight and dims quantiles (t-digest in production, plain arrays in the hackathon), service-type mix, hour-of-week histogram, lane counts.
- EWMA volume at 1 day, 7 days and 28 days, plus a platform-wide seasonal multiplier (see 4.4).
- Baseline entropies over 90 days: sender, origin, consignee, destination.
- Billing-correction history: count of bookings whose declared weight/dims were corrected at scan.

Per payer (key `payer:{id}`): distinct senders, origins and accounts billed over 7 and 30 days, payment method, ACH funding status, unpaid exposure, last ACH return code and date.

Per consignee (key `cons:{hash}`): first-seen time, distinct payers and senders over 7 and 30 days, share of its senders that are new accounts (<30 days), count of linked fraud-labelled decisions (label propagation, 2 hops).

Per login (key `login:{id}`): linked-at time, linking method (PIN, invoice data, admin invite), device/IP ASN novelty.

Offline: every raw event goes to parquet (`events/date=YYYY-MM-DD/`). Training sets are built by replaying events in time order through the same featurize code, which makes them point-in-time correct by construction. Labels are joined with their label_time, so a label that was not yet known at time t never enters a training example cut at t.

### 1.4 Change-detection signals (the core of the idea)

Domain note: for e-commerce sellers (Olist), consignees are new on almost every order, so consignee novelty alone is normal. The fraud signature (I2, I5) is novelty on the SENDER side (new senders, origins, logins and integrations billed to this payer) and many-senders-to-one-consignee clustering. Every signal is therefore computed separately for sender-side and consignee-side.

- `novel_sender_rate_k`: share of the last k bookings (k=20, current booking included) whose sender, origin zip3 or origin facility was not in the account's set before the window. Reported next to `baseline_novel_sender_rate` (the account's own long-run rate), so a seller that is genuinely growing has a higher baseline.
- `sender_entropy_jump`: Shannon entropy of the sender distribution over the last 7 days minus the 90-day baseline entropy. Same for origins and package-profile buckets (weight band x service).
- `cusum_novelty`: one-sided CUSUM on the novel-sender indicator, with reference rate = baseline rate + slack. It fires on a sustained shift, not on one odd booking. Reset after analyst-confirmed legitimate expansion.
- `payoff_per_label`: list price of this shipment (weight/dim-weight, zone distance, service level) computed from the carrier rate table. This is what the fraudster saves per label, since resale prices are flat (I7). Features: `payoff_z` against the account's own history (median/MAD), `payoff_ratio` = payoff / account median, `heavy_long_express` flag.
- `new_login_recent`: login linked to the account fewer than 14 days ago, with its linking method. `contact_changed_recent`: owner contact changed fewer than 30 days ago (this also downgrades trust in owner_confirm).
- `dims_mismatch_rate`: share of the last 30 scanned bookings whose measured weight/dims differed from declared beyond tolerance (I8).

### 1.5 Billing graph features

Graph: payer -> sender/origin -> consignee, plus login -> payer edges. Online features are cheap counters, not GNN inference:
- distinct senders and origins per payer over 7 and 30 days, and the ratio of the 7-day rate to the 30-day rate;
- consignee in-degree from distinct payers and senders over 7 and 30 days, consignee first-seen age;
- share of the consignee's payers that are new accounts or recently taken over;
- 2-hop: does this consignee or sender address share edges with any account that has a fraud label (label propagation score, refreshed hourly).

Optional GNN: GraphSAGE trained offline hourly or daily; the embedding is looked up from the store at score time (no online message passing). We validate the GNN component on real labelled graphs (IEEE-CIS entity graph, YelpChi/Amazon via GADBench) before claiming it helps, because our carrier labels are injected. The GNN embedding goes into the GBM only, never into the Laya state.

### 1.6 Serializer and Laya questions

The state text is deterministic and versioned (`serializer_version`). It contains no free text supplied by the booker: no sender names, reference fields or package descriptions. Reason: the Jev docs state that state is not treated as hostile by default (phase1), and Laya is a similar text model, so we close that injection path by construction. Numbers are rendered as words plus values ("3.4x its usual", "first time"). The target length is <= 320 tokens, which leaves room for the questions inside the 512-token English context (VERIFIED: 512 ctx, 192-token option budget).

Example state (illustrative):

```
Account: business, tenure 2.1 years, 1,840 past bookings, segment tenured.
Login: linked 2 days ago via account number + PIN (first new login in 14 months).
Payer: same as account, pays by ACH, last funding pending, no returns.
This booking: origin zip3 never used by this account (account has used 3 origins), sender never seen,
consignee new (normal for this account), destination domestic, zone 8, express, 18.2 kg.
Payoff: list price 3.9x the account's median label; heavy+long+express.
Last 20 bookings: 11 had senders never seen before (account's usual rate: 1 in 20).
Sender diversity: 7-day entropy 2.8 vs 90-day 0.6. CUSUM alarm: yes, since 9 bookings.
Consignee: seen 6 times in 7 days from 5 different payers, 4 of them accounts younger than 30 days.
Rules hit: none. Tabular model risk band: high (top 1%).
```

Questions (one call, 3 questions; keeping the count low matters because vendor latency rises with the question count):

```json
{
  "misuse": {"type": "noul",
    "instructions": "Is this booking being made by someone other than the account's legitimate owner or its authorised staff?"},
  "pattern": {"type": "choice",
    "instructions": "Which pattern best fits this booking and the account's recent activity?",
    "criteria": {
      "legitimate": "the owner's normal business, including normal growth or seasonal peaks",
      "label_resale": "account billed for many new senders and origins it never served",
      "bust_out": "new account shipping high-value parcels quickly before paying",
      "mule_drop": "many unrelated senders shipping to one short-lived consignee",
      "payoff_max": "unusually heavy, long-distance or express shipments for this account",
      "declared_mismatch": "declared weight, size or value inconsistent with history"}},
  "consistency": {"type": "score",
    "instructions": "How consistent is this booking with the account's own history?",
    "criteria": ["very inconsistent", "inconsistent", "somewhat consistent", "consistent", "fully consistent"]}
}
```

- p_fraud = calibrated P(misuse = true). This is the decision probability.
- `pattern` routes the action (e.g. declared_mismatch -> scan gate works well; label_resale -> owner_confirm works well) and seeds the explanation. Typology T5 (test-then-burst) is deliberately left out of the options, because it is an unsourced hypothesis; it is evaluated only as an injected scenario.
- `consistency` feeds monitoring and the explanation, not the decision.

Including the GBM risk band in the state is a design choice. It lets Laya combine the tabular score with the text story, but Laya could simply copy the band. We therefore report four ablations: GBM alone, Laya without the band, Laya with the band, and Laya with the band but with the change-detection lines removed. If Laya does not beat GBM alone on PR-AUC, precision at analyst capacity, ECE and the held-out typology, we say so.

Calibration: we take the probabilities laya-serve returns and apply our own temperature per question (p' = softmax(log p / T_q)). Composing with whatever temperature the server already applied is valid, because temperature scaling composes multiplicatively. T_q is fitted by LBFGS on held-out labelled bookings (<= 400 per question, the same recipe as the notebook) and stored as `temps_vN.json`, hashed into the audit record. Keeping it client-side means a server-side change (VERIFIED: the server clamps temperatures to [0.5, 5.0]) cannot silently move our probabilities.

### 1.7 Delayed-label feedback loop

| Label source | Latency | Strength | Notes |
|---|---|---|---|
| Owner-confirm deny ("not me") | minutes | strong fraud | Only if the contact channel did not change recently |
| Owner-confirm approve | minutes | strong legit | Same caveat |
| Scan-gate mismatch (weight/dims/induction location) | hours to 2 days | medium fraud (T6, resale) | Can also be honest error; analyst confirms |
| Analyst decision | minutes to hours | medium | Analysts are biased toward the model; audited sample |
| ACH unauthorized return R29 (corporate) | within 2 banking days | strong for payment misuse | VERIFIED: R29 window 2 banking days, R10 (consumer) 60 days (Modern Treasury / Ramp ACH return code pages) |
| ACH R10 (consumer payer) | up to 60 days | strong | |
| Owner complaint / invoice dispute | days to weeks | strong | Main label source for allowed bookings |
| Canary billing alias used (product recommendation, phase2 idea 14) | immediate | strong | Only if the carrier deploys aliases |
| No complaint after maturity window | 60-90 days | weak legit | Treat as censored before maturity |

Handling:
- Every training example carries `label`, `label_source`, `label_time`, `matured` (bool). A booking without a label younger than the maturity window (ASSUMPTION: 60 days) is censored, not negative. Option: a delayed-feedback model in the style of Chapelle KDD 2014 for the GBM.
- Selective labels: a blocked booking never produces a chargeback, so its true label is often unknown. The exploration arm (section 4.6) sends a small, logged, randomized share of borderline bookings to allow_scan_gated instead of hold/review, which yields labels at low cost.
- Cadence (ASSUMPTION): GBM retrain weekly; temperature and conformal refit weekly on matured labels; Laya fine-tune monthly or after a new typology. Every new model or policy goes through offline eval, then OPE (IPS, SNIPS, DR via obp) on logged propensities, then shadow mode for 1 week, then a canary on 5% of traffic.
- Feedback-loop guard: the Adyen bandit paper (arXiv 2412.00569) reports oscillation when retraining on the system's own logs. We keep the exploration rate fixed, weight training examples by 1/propensity where labels exist only because of the policy, and keep a never-changed holdout of injected scenarios as a regression set.

---

## 2. Latency budget

Target: synchronous decision p99 < 300 ms measured at the Scoring API, GPU deployment. The booking side's timeout is 350 ms.

| Step | p50 | p99 budget | Basis |
|---|---|---|---|
| HTTP parse, auth, validation | 2 ms | 5 ms | ESTIMATE |
| Online store reads (Redis, pipelined, ~6 keys) | 3 ms | 10 ms | ESTIMATE |
| Payment status (cache hit; sync miss has a 30 ms timeout) | 1 ms | 30 ms | ESTIMATE |
| featurize + rules + change detection + graph counters | 4 ms | 12 ms | ESTIMATE (pure Python) |
| LightGBM single row | <1 ms | 3 ms | ESTIMATE |
| serialize | <1 ms | 2 ms | ESTIMATE |
| Laya call, 3 questions, ~350-token state, T4, fp16 | 60-70 ms | 180 ms (hard timeout) | VENDOR: 39.5 ms for 1 q, 84.5 ms for 5 q on T4; 3 q is our interpolation. Our state is longer than the vendor benchmark states, so MEASURE. The notebook reports ~710 ms per 5-question case on longer 1,024-token states, which is a warning sign. |
| calibration + policy + conformal + exploration | 1 ms | 3 ms | ESTIMATE |
| audit WAL append (buffered, fsync batched every 10 ms) | 1 ms | 12 ms | ESTIMATE |
| network between services | 5 ms | 20 ms | ESTIMATE |
| Total | ~80-90 ms | ~277 ms worst-case sum | Sum of budgets; real p99 is lower because the tails do not all coincide |

Decisions:
- LLM explanation is async and off the critical path. It runs only for non-allow actions plus a 1% QA sample of allows. Expected latency is 1-5 s (ESTIMATE). The console shows the template explanation immediately and swaps in the LLM text when it passes validation. Owner-facing messages always use fixed templates, never LLM text.
- Laya runs one call per booking. Under burst load we micro-batch: the client collects requests for up to 5 ms and sends /v1/systemone/batch (VERIFIED cap 64 states). VENDOR throughput is 103-332 questions/s per T4, i.e. roughly 35-110 bookings/s per GPU at 3 questions. That is the capacity planning number to measure.
- If the 3-question latency on our states is too high, the levers in order are: (a) drop `consistency` (2 questions); (b) cut the state to <= 220 tokens; (c) fine-tune laya-multilingual instead (mmBERT-base 322M, VENDOR 40.1 ms for 5 questions vs 84.5 ms); (d) ONNX export. On (d), the README warns that per-channel INT8 quantization collapsed agreement to 32%. Use per-tensor (now the default) and re-check calibration.

CPU fallback (no GPU at demo time):
- VENDOR: Router(preload=True) reports 193-464 ms per request on CPU (README deployment table). Which checkpoint and question count that row refers to is unclear.
- ESTIMATE for fine-tuned ModernBERT-large (421M) on a laptop CPU, 3 questions, ~350-token state: 0.5-1.5 s per booking. Not measured.
- CPU demo mode: ask only `misuse` (1 question), set LAYA_THREADS to the physical core count, relax the demo SLO to p99 < 1.5 s, and label it on screen as "CPU demo mode, not production latency". Precompute Laya answers for the replay stream offline as a backup, and show the live call for a few hand-picked bookings.

Timeouts and degrade mode:

| Failure | Detection | Behaviour |
|---|---|---|
| Laya timeout (>180 ms), 5xx, or 503 busy (VERIFIED: server sends Retry-After) | client | Degrade for this booking: p_fraud = p_gbm (isotonic-calibrated), stricter degraded thresholds, no exploration. `degraded=true` and the reason are recorded in the audit. |
| Laya error rate > 5% over 1 min | circuit breaker | Breaker open for 30 s, all traffic degraded, alert raised. Half-open probes after that. |
| Online store unavailable | client | Score from booking-only features and rules. Allow is not permitted for accounts younger than 90 days or bookings above the payoff threshold; those get allow_scan_gated or review. |
| Whole Scoring API unreachable (booking-side timeout 350 ms) | Booking API | Booking API applies a local cached fail-safe: deny-list -> hold; new account (<30 days) + international or payoff > threshold -> pending review; everything else -> allow_scan_gated if the scan system is up, else allow. Logged on the booking side and replayed into the audit later. |
| Audit write fails | API | Return the degrade-mode action, page on-call. Never return a decision that is not logged. |

Rule: degraded mode never moves a high-risk booking to plain allow. Degraded thresholds are set so that the degraded false-negative rate on held-out fraud is no worse than full mode; the cost is more scan-gated and review actions. Degraded-mode share is a dashboard metric.

---

## 3. Integration points and action semantics

### 3.1 Booking system (synchronous pre-commit hook)
- The Booking API calls POST /score after validating the booking and before issuing the label or charging.
- The Booking API must support a "pending" label state for owner_confirm, review and hold, because the customer or API client has to be told "processing". API integrations get a pending status plus a webhook when it resolves. This is a real product change for carriers that only support instant labels.
- Idempotency: `booking_id` is the idempotency key. A retry returns the original decision_id and action and does not write a second audit record.

### 3.2 Payment system
- Read: payer authorization (is this login or account allowed to bill this payer, including third-party billing), payment method (invoice/card/ACH), ACH funding status (settled/pending/returned), last return code and date, unpaid exposure versus historical average.
- Why ACH matters: USPS OIG management alert "Enterprise Payment Account Fraud" (Feb 10 2026) on ACH-debit/trust-funded payment accounts; phase1 notes record >$500M past loss, and the OIG page lists ~$1.81B associated monetary value (VERIFIED page title, date and value via search: uspsoig.gov/reports/audit-reports/enterprise-payment-account-fraud). The mechanics are redacted, so our use of the alert is an inference: pending ACH funding plus a sender-side novelty jump raises the floor action to at least owner_confirm.
- Write: none at score time. The payment system subscribes to decisions so it can delay settlement of held labels.
- Return events (R10, R29, R05, etc.) stream in as labels and features.

### 3.3 Account system
- Event stream in: login linked to account (with method: PIN, invoice data, admin invite), PIN reset, invoice-verification attempts and failures, contact detail changes, API key and integration creation, password resets.
- Read at decision time: owner contact channel on file and its last-changed date. owner_confirm goes to the contact on file BEFORE any change in the last 30 days (ASSUMPTION). If none exists, owner_confirm is not offered and review is used instead.
- Write (after a confirmed deny or an analyst fraud verdict): lock label creation for the account, revoke the recently linked login, force re-verification. These are separate account-system actions, triggered by the console, not by the model.

### 3.4 First-scan system
- At decision time, an allow_scan_gated action writes a gate record {tracking_id, expected weight, dims, tolerance, expected origin facility set, expiry}.
- At induction scan: measured weight/dims and the induction facility are compared with the booking. Match -> released to linehaul automatically, event "scan_pass". Mismatch -> held at the facility before linehaul, event "scan_mismatch", review case opened, owner notified by template.
- Not inducted within 7 days (ASSUMPTION) -> label voided and not billed.
- Hackathon: simulated. Tolerances and interception rates are ASSUMPTIONS and are shown as parameters in the demo.

### 3.5 Action semantics

| Action | Label issued? | Package flow | Customer sees | Resolution | Timeout / fallback |
|---|---|---|---|---|---|
| allow | yes, now | normal | normal label | none | n/a |
| allow_scan_gated | yes, now | gate at first scan; held before linehaul on mismatch | normal label | auto at scan; analyst on mismatch | not inducted in 7 days -> void |
| owner_confirm | pending | none yet | "awaiting account owner approval" | owner approves -> allow (sender/origin added to known set); denies -> block + account lock flow | no answer in 30 min (ASSUMPTION) -> review |
| review | pending | none yet | "processing" | analyst: allow / allow_scan_gated / block | SLA 15 min (ASSUMPTION); on miss -> allow_scan_gated if p_fraud < p_hold else hold |
| hold | no | none | "verification required, contact support" | further bookings on the account require review until cleared; analyst or verified owner clears | auto-escalate to senior queue after 72 h |
| block | no | none | "booking refused" plus appeal route | label creation suspended for the account pending verification | analyst must confirm within 24 h or it downgrades to hold |

Block preconditions (all required): p_fraud >= p_block for the segment; at least one hard signal (rule hit, owner deny, deny-listed consignee, or `pattern` != legitimate with answer probability >= 0.8); per-account block cap not exceeded (section 4.5).

---

## 4. False-positive control

### 4.1 Cost matrix (ASSUMPTION, per booking, config `policy_vN.yaml`)

L = loss if fraud is allowed = shipping list price (payoff) + c_ops (claims and handling, ASSUMPTION $8) + c_exposure (ASSUMPTION 0.2 x unpaid exposure share).
V = value at risk if a legitimate customer is disrupted = expected margin over 90 days x churn probability given the action.

| Action | Cost if legit | Cost if fraud |
|---|---|---|
| allow | 0 | L |
| allow_scan_gated | c_gate = $0.40 + 0.002 x V | (1 - r_scan[pattern]) x L + c_gate |
| owner_confirm | c_oc = $1.50 + 0.01 x V | (1 - r_oc) x L + c_oc, where r_oc = 0.9, or 0.3 if contact changed recently |
| review | c_rev = $4 (5 analyst minutes) + 0.01 x V | c_rev + (1 - r_rev) x L, r_rev = 0.8 |
| hold | $10 + 0.05 x V | $2 |
| block | $25 + 0.25 x V | 0 |

r_scan by pattern (SPECULATIVE): declared_mismatch 0.8, label_resale 0.5, payoff_max 0.5, bust_out 0.3, mule_drop 0.3. All of these rates are placeholders until a carrier supplies real numbers. The demo shows a sensitivity table (how actions shift when they move).

Rule: a* = argmin over a in A_allowed of p x C(a, fraud) + (1 - p) x C(a, legit), with p = p_fraud. Because of the per-booking L, a $9 ground parcel and a $120 express parcel with the same p can get different actions. That is the "fraudster economics" part: friction lands where the payoff is.

A_allowed is the action set left after:
1. Rule floor: actions below the floor are removed.
2. Conformal allow-guard: allow is removed unless the guard passes (4.2).
3. Availability: owner_confirm removed if there is no trusted contact; review removed if queue capacity is exhausted (4.5); allow_scan_gated removed if the scan system is down or for international service where the gate is not supported.
4. Block preconditions (3.5).

### 4.2 Conformal allow-guard (class-conditional, Mondrian by segment)

Goal: a finite-sample bound on the fraction of fraud that is plainly allowed, P(allow | fraud, segment) <= alpha (ASSUMPTION alpha = 0.05), valid under exchangeability.
- Calibration set: n_f labelled fraud bookings in the segment (held out from training and from temperature fitting). Nonconformity score s = p_fraud.
- Sort the fraud scores ascending: s_(1) <= ... <= s_(n_f). Let k = floor(alpha x (n_f + 1)). Allow is permitted only if p_fraud < s_(k). If k = 0 (too few fraud examples), allow is never permitted by the guard for that segment, and the segment pools with its parent.
- Guarantee: P(p_fraud(X) < s_(k) | fraud) <= k / (n_f + 1) <= alpha for an exchangeable new fraud booking.
- Honest limits: exchangeability fails under drift and adaptive fraudsters, and our calibration fraud is injected. The guarantee holds for injected typologies seen in calibration only. We report realized allow-rate on the held-out typology separately, and expect it to be worse. Minimum n_f per segment for stable thresholds: ~200 (ASSUMPTION); below that, pool.
- The guard does not block anything. It only moves bookings from allow to the cheapest other allowed action, usually allow_scan_gated, which is why the gated action exists.

### 4.3 Segments and thresholds
Segments: new (<30 days or <20 bookings), growing (30-180 days), tenured (>180 days and >=20 bookings), contract/enterprise (negotiated accounts with a named account manager). Each segment has its own conformal threshold, its own p_block and p_hold floors, and its own cost-matrix V. Thresholds are refit weekly, and changes beyond +-20% need human sign-off (shown as a diff in the console).

### 4.4 Hard negatives
Built into training and evaluation as named scenarios with is_injected=false (real Olist behaviour) or documented perturbations:
- Legit seller ships to a new region: consignee-side novelty only, sender, origin, login and payment unchanged. The serializer states explicitly "consignee new (normal for this account)". Test: the false-positive rate on these bookings, per segment.
- Seasonal surge (Black Friday Nov 2017 in Olist): volume features are divided by a platform-wide seasonal multiplier (this week's platform volume / trailing 8-week median), so an account growing with the market is not flagged.
- New legitimate channel or new warehouse: brings new origins, but via a known login and a settled payment method. Owner_confirm is the intended cheap action. On approve, the new origin goes into the known set and the CUSUM resets.
- Analyst-approved expansion: an analyst can create a time-limited allow-pattern ("account A, origin zip3 X, 30 days"), versioned as a rule, with an audit event.

### 4.5 Caps and capacity
- Per-account cap: at most 1 block per account per 24 h without analyst confirmation; further high-risk bookings become hold. This prevents one model error from shutting down a whole business.
- Global cap: if the block rate over 1 h exceeds 3x its 7-day median, new blocks downgrade to hold and on-call is alerted (likely model or data fault).
- Review capacity: the queue is ordered by expected loss avoided per analyst minute, (p x L x r_rev - c_rev). When the queue is full, review is removed from A_allowed and the policy re-solves. This is supervised scores plus a constraint, in the DeCCaF style, not RL.

### 4.6 Exploration (logged propensities)
- Where: only in the boundary band, defined as bookings where the best action and the second best differ in expected cost by less than delta (ASSUMPTION $1.50), and p_fraud in [0.15, 0.6].
- What: with probability epsilon = 0.05, replace the chosen action with allow_scan_gated (never with plain allow, never when a rule floor is hold or above, never in degraded mode). This is the safe version of Stripe's ~5% let-through: the package is still checked at first scan.
- Logged: `explore=true`, `propensity` of the action actually taken (1 - epsilon for the greedy action, epsilon for the explored one, 1.0 when outside the band), the greedy action, and its expected costs. OPE uses these.

### 4.7 Monitoring false positives on legitimate shippers
Dashboard metrics (daily, by segment):
- Friction rate = share of bookings from accounts with no fraud label in the last 180 days that got any non-allow action other than allow_scan_gated. Alert if > 1% for tenured (ASSUMPTION).
- Owner-confirm approve rate (a high approve rate means wasted friction).
- Analyst "legitimate" rate on reviews and holds.
- Block appeals and their overturn rate.
- Scan-gate pass rate.
- Calibration: ECE and a reliability diagram of p_fraud on matured labels, weekly.
- Degraded-mode share, Laya p99 latency, explanation validation failure rate.

---

## 5. Audit log

### 5.1 Format
Append-only JSONL, one event per line, single writer per chain. Event types: `decision`, `explanation`, `analyst_feedback`, `outcome`, `side_effect` (scan gate created, owner message sent), `config_change` (new model, temps, policy or rules version activated). Records are never edited; corrections are new events that reference the original.

Hash chain:
- `record_hash = sha256(canonical_json(record without record_hash))`, where canonical_json = UTF-8, sorted keys, separators (",", ":"), no NaN, floats rounded to 6 decimals before hashing.
- Each record carries `seq` (monotonic) and `prev_hash` (record_hash of seq - 1). The genesis record has prev_hash = 64 zeros.
- Each day the head hash (seq, record_hash) is written to an external anchor (production: object storage with write-once retention; hackathon: a separate `anchors.jsonl` plus a printed value in the demo). Without an external anchor, a hash chain only shows internal consistency: someone who rewrites the whole file can recompute every hash.

### 5.2 Decision record fields

```json
{
  "event_type": "decision",
  "seq": 18233,
  "prev_hash": "9f2c...e1",
  "decision_id": "dec_01J9X...",
  "timestamp": "2026-10-02T14:03:11.482Z",
  "booking_id": "bk_77812",
  "account_id": "acct_h_4be1...",
  "payer_account_id": "payer_h_91aa...",
  "versions": {
    "laya_model": "convaiinnovations/laya",
    "laya_finetuned_repo": "team/fraudshield-laya",
    "laya_hf_revision": "<40-char commit sha>",
    "laya_package": "0.3.23",
    "temperature_set": "temps_v3",
    "temperature_set_sha256": "...",
    "gbm": "lgbm_2026-10-01",
    "gbm_sha256": "...",
    "rules": "rules_v7",
    "policy": "policy_v4",
    "serializer": "ser_v2",
    "conformal": "conf_v3"
  },
  "segment": "tenured",
  "state_text": "Account: business, tenure 2.1 years, ...",
  "state_sha256": "...",
  "feature_vector_sha256": "...",
  "features_summary": {"novel_sender_rate_k": 0.55, "baseline_novel_sender_rate": 0.05, "payoff_ratio": 3.9, "new_login_recent": true},
  "rule_hits": [],
  "floor_action": "allow",
  "p_gbm": 0.71,
  "laya_raw": {"misuse": {"true": 0.93, "false": 0.07}, "pattern": {"label_resale": 0.81, "...": 0.0}, "consistency": [0.62, 0.25, 0.09, 0.03, 0.01]},
  "laya_latency_ms": 64,
  "calibrated": {"misuse": {"true": 0.84}, "pattern": {"label_resale": 0.72}, "consistency": [0.51, 0.27, 0.13, 0.06, 0.03]},
  "p_fraud": 0.84,
  "loss_if_fraud": 61.4,
  "expected_costs": {"allow": 51.6, "allow_scan_gated": 26.2, "owner_confirm": 6.8, "review": 14.5, "hold": 3.4, "block": 4.6},
  "value_at_risk_legit": 15.0,
  "allowed_actions": ["allow_scan_gated", "owner_confirm", "review", "hold"],
  "block_preconditions_met": false,
  "conformal_allow_passed": false,
  "greedy_action": "hold",
  "action": "hold",
  "propensity": 1.0,
  "explore": false,
  "degraded": false,
  "degraded_reason": null,
  "record_hash": "..."
}
```

Explanation event: `{event_type: "explanation", decision_id, explanation_text, source: "llm"|"template", llm_model_id, prompt_template_version, prompt_sha256, output_sha256, validation: {passed, unmatched_numbers: [], banned_phrases: []}, latency_ms}`. The full prompt is stored too (it contains only the structured record, so no extra PII).

Analyst feedback event: `{event_type: "analyst_feedback", decision_id, analyst_id_hash, verdict: "fraud"|"legitimate"|"unsure", final_action, reason_codes: [], note, allow_pattern_created: null|{...}}`.

Outcome event: `{event_type: "outcome", decision_id, source: "owner_confirm"|"scan"|"chargeback"|"ach_return"|"owner_complaint"|"canary", value, observed_at}`.

### 5.3 PII and retention
- The audit log holds pseudonymous ids (HMAC-SHA256 of account and payer ids, keyed; the key lives in a secrets store), zip3 rather than full addresses, and no names. The state text is built from derived features only, so it holds no names either.
- Raw bookings live in the operational PII store under its own retention policy. Erasing a person there does not break the audit chain, because the chain never held their raw data. Deleting the HMAC mapping makes the pseudonyms unlinkable.
- Retention: ASSUMPTION 7 years for decision records (financial audit), 2 years for full prompts and explanations. Confirm with the carrier's legal team. Analyst ids are hashed; access to the console audit view is role-based.

### 5.4 Verification
Command: `python -m fraudshield.audit verify --path data/audit/decisions.jsonl [--anchor data/audit/anchors.jsonl]`

Algorithm: stream the file; for each line, parse, check seq = previous + 1, check prev_hash = previous record_hash, recompute record_hash and compare; at each anchored seq, compare with the anchor. Output: `{"ok": true, "records": 18233, "head_hash": "...", "anchors_checked": 3}` or `{"ok": false, "first_bad_seq": 1204, "reason": "record_hash mismatch"}`. Exposed as GET /audit/verify. Demo: edit one probability in the file by hand, run verify, show the failure at that seq.

Concurrency: a single writer task owns the file (asyncio queue in the hackathon). Production: one chain per partition (e.g. per region), each minute's partition heads combined into a Merkle root, and the root anchored.

---

## 6. API contracts

All JSON. Times are ISO 8601 UTC. Money is in the carrier's currency, as decimals. Schemas use JSON Schema draft 2020-12 notation, abbreviated.

### 6.1 POST /score
Request:
```json
{
  "type": "object",
  "required": ["booking_id", "account_id", "payer_account_id", "channel", "created_at", "origin", "destination", "parcel", "service"],
  "properties": {
    "booking_id": {"type": "string"},
    "account_id": {"type": "string"},
    "payer_account_id": {"type": "string"},
    "login_id": {"type": ["string", "null"]},
    "integration_id": {"type": ["string", "null"]},
    "channel": {"enum": ["web", "api", "counter", "marketplace"]},
    "created_at": {"type": "string", "format": "date-time"},
    "sender": {"type": "object", "properties": {"sender_id": {"type": "string"}, "address_hash": {"type": "string"}}},
    "origin": {"type": "object", "required": ["postal_code", "country"], "properties": {"postal_code": {"type": "string"}, "country": {"type": "string"}, "facility_id": {"type": ["string", "null"]}}},
    "destination": {"type": "object", "required": ["postal_code", "country"], "properties": {"postal_code": {"type": "string"}, "country": {"type": "string"}, "consignee_hash": {"type": "string"}, "residential": {"type": ["boolean", "null"]}}},
    "parcel": {"type": "object", "required": ["weight_kg"], "properties": {"weight_kg": {"type": "number"}, "length_cm": {"type": "number"}, "width_cm": {"type": "number"}, "height_cm": {"type": "number"}, "declared_value": {"type": ["number", "null"]}}},
    "service": {"enum": ["ground", "express", "overnight", "international_economy", "international_express"]},
    "list_price": {"type": ["number", "null"], "description": "if absent, computed from rate table"},
    "payment_method": {"enum": ["invoice", "card", "ach", "trust", "unknown"]},
    "scenario_id": {"type": ["string", "null"], "description": "demo/eval only; never read by the model; dropped by the API before featurize"}
  }
}
```
Response 200:
```json
{
  "decision_id": "string",
  "booking_id": "string",
  "action": "allow | allow_scan_gated | owner_confirm | review | hold | block",
  "p_fraud": 0.0,
  "pattern": {"label": "string", "probability": 0.0},
  "reason_codes": ["SENDER_NOVELTY_JUMP", "NEW_LOGIN_14D", "PAYOFF_HIGH"],
  "expected_costs": {"allow": 0.0},
  "degraded": false,
  "explanation_status": "pending | not_requested",
  "versions": {"policy": "policy_v4", "laya_hf_revision": "..."},
  "latency_ms": 0
}
```
Errors: 400 invalid booking; 409 booking_id already scored with a different body; 503 never returned to the booking system (degrade instead). Idempotent on booking_id.

`reason_codes` come from a fixed enum, derived deterministically from features and rule hits (top contributing signals by GBM SHAP plus rule hits), not from the LLM.

### 6.2 GET /decisions/{decision_id}
Returns the decision record (5.2), the latest explanation event, the analyst feedback and outcome events, the scan or owner-confirm status, and `audit: {seq, record_hash}`. 404 if unknown. Role-based: analysts see everything; the booking system sees only action and status.

### 6.3 POST /decisions/{decision_id}/analyst
```json
{
  "type": "object",
  "required": ["analyst_id", "verdict", "final_action"],
  "properties": {
    "analyst_id": {"type": "string"},
    "verdict": {"enum": ["fraud", "legitimate", "unsure"]},
    "final_action": {"enum": ["allow", "allow_scan_gated", "hold", "block"]},
    "reason_codes": {"type": "array", "items": {"type": "string"}},
    "note": {"type": "string", "maxLength": 1000},
    "allow_pattern": {"type": ["object", "null"], "properties": {"dimension": {"enum": ["origin_zip3", "sender_id", "login_id", "integration_id"]}, "value": {"type": "string"}, "expires_days": {"type": "integer", "maximum": 90}}}
  }
}
```
Response: `{"ok": true, "audit_seq": 18240, "record_hash": "..."}`. 409 if the decision already has a final analyst verdict (a second verdict needs `supersedes` plus a reason, recorded as a new event).

### 6.4 GET /dashboard/metrics?from=&to=&segment=
```json
{
  "window": {"from": "...", "to": "..."},
  "volume": 0,
  "action_mix": {"allow": 0, "allow_scan_gated": 0, "owner_confirm": 0, "review": 0, "hold": 0, "block": 0},
  "friction_rate_clean_accounts": 0.0,
  "owner_confirm_approve_rate": 0.0,
  "analyst_legit_rate": 0.0,
  "scan_gate_pass_rate": 0.0,
  "degraded_share": 0.0,
  "latency_ms": {"p50": 0, "p95": 0, "p99": 0, "laya_p99": 0},
  "eval": {"pr_auc": 0.0, "precision_at_capacity": 0.0, "recall_by_typology": {"T1": 0.0}, "ece": 0.0, "conformal_realized_allow_fraud_rate": 0.0},
  "loss_avoided_estimate": 0.0,
  "explanation_validation_fail_rate": 0.0,
  "exploration": {"explored": 0, "ope_snips_policy_cost": 0.0}
}
```
In the hackathon, `eval` is computed against injected labels and says so in the UI.

### 6.5 GET /audit/verify
`{"ok": bool, "records": int, "head_hash": str, "anchors_checked": int, "first_bad_seq": int|null, "reason": str|null, "duration_ms": int}`

### 6.6 Internal interfaces (Python)

```python
@dataclass(frozen=True)
class Booking: ...          # mirrors POST /score body, minus scenario_id

@dataclass(frozen=True)
class FeatureVector:
    booking_id: str
    account_segment: Literal["new", "growing", "tenured", "contract"]
    numeric: dict[str, float]     # fixed, versioned key order (FEATURE_SPEC_VERSION)
    flags: dict[str, bool]
    rule_hits: list[str]
    floor_action: Action
    list_price: float
    loss_if_fraud: float
    spec_version: str

def featurize(booking: Booking, history: AccountState) -> FeatureVector: ...
def update_state(history: AccountState, booking: Booking) -> AccountState: ...   # applied after decide, same code offline

def gbm_score(fv: FeatureVector) -> tuple[float, dict[str, float]]: ...   # (p_gbm calibrated, top SHAP contributions)

def serialize(fv: FeatureVector, p_gbm: float) -> str: ...   # deterministic, <= 320 tokens, raises if over budget

class LayaClient:
    def __init__(self, base_url: str, api_key: str | None, timeout_s: float = 0.18, model: str = "english"): ...
    def ask(self, state: str, questions: dict) -> LayaAnswers: ...   # raises LayaUnavailable on timeout/5xx/503
    def ask_batch(self, states: list[str], questions: dict) -> list[LayaAnswers]: ...  # <= 64 states

@dataclass
class LayaAnswers:
    raw: dict[str, dict]          # per question: probabilities as returned
    latency_ms: int
    model: str
    revision: str | None

def calibrate(ans: LayaAnswers, temps: TemperatureSet) -> dict[str, dict[str, float]]: ...

@dataclass
class PolicyResult:
    action: Action
    greedy_action: Action
    propensity: float
    explore: bool
    expected_costs: dict[Action, float]
    allowed_actions: list[Action]
    conformal_allow_passed: bool

def decide(p_fraud: float, pattern: dict[str, float], fv: FeatureVector,
           costs: CostConfig, guard: ConformalGuard, ctx: PolicyContext, rng: random.Random) -> PolicyResult: ...
# ctx: queue capacity, scan system up, owner contact trusted, per-account block count, degraded flag

def explain(record: DecisionRecord) -> Explanation: ...   # async worker; never raises, falls back to template

class AuditLog:
    def append(self, event: dict) -> tuple[int, str]: ...   # returns (seq, record_hash); fsync policy configurable
    def verify(self, anchor_path: str | None = None) -> VerifyResult: ...
```

Contract tests the build agents must ship: same booking and same state give the same state_text and the same decision when explore is off (seeded rng); featurize online equals featurize in offline replay for 1,000 sampled bookings; audit verify passes after 10k appends and fails after a single-byte edit.

---

## 7. Explanation layer

Purpose: analysts read business language; the model never decides. The LLM is not on the decision path and cannot change the action.

Inputs (facts dict, built from the decision record only): action, p_fraud (rounded to 2 decimals), pattern label and probability, reason codes with their feature values, payoff ratio and list price, novelty rates versus baseline, consignee graph counts, rule hits, expected cost of the chosen action versus allow, whether the conformal guard passed, degraded flag. No raw booking text, no names.

Pipeline:
1. Template (always produced, instant): a sentence per reason code from a fixed library, e.g. "11 of the last 20 bookings used senders this account had never used (its usual rate is 1 in 20)."
2. LLM rewrite: Claude via the Anthropic API (model id pinned in config; a small fast model is enough, e.g. a Haiku-class model; confirm the exact id at build time). Prompt: system instructions ("Use only the facts provided. Do not add numbers, places, names or causes not present. Do not state that fraud is confirmed. Say what the analyst should check next from the provided checklist."), the facts dict as JSON, and the template text. Max 120 words. Temperature 0.
3. Validator (deterministic):
   - Every number in the output (regex for integers, decimals, percentages, money, "x" ratios, "N of M") must match a value in the facts dict, allowing for formatting (0.84 = 84%, rounding to the displayed precision).
   - Every postal code, country, pattern name or id-like token must be present in the facts.
   - Banned phrases: "confirmed fraud", "is a fraudster", "criminal", "definitely", "guaranteed", plus any claim about the person's identity.
   - The output must name the chosen action and at least 2 of the top 3 reason codes (keyword match against a synonym list).
   - On any failure, use the template. Log the failed output and the validator result.
4. Log: an explanation event in the audit (5.2) with llm_model_id, prompt_template_version, prompt_sha256, the full prompt, the output, and the validation result.

Failure rate target: < 5% of LLM outputs rejected (ESTIMATE). A higher rate means the prompt or template needs work, and it is visible on the dashboard. Owner-facing and customer-facing text never comes from the LLM.

---

## 8. Suggested repo layout and build split (4 people)

```
fraudshield/
  data/         olist_loader.py, inject.py (T1-T6, seeds, is_injected flags), rate_table.py
  features/     state.py (AccountState), featurize.py, change_detect.py, graph.py, serialize.py
  models/       gbm.py, laya_client.py, calibrate.py, finetune/ (notebook adaptation), gnn/ (optional)
  policy/       costs.py, conformal.py, decide.py, explore.py, ope.py (obp)
  explain/      templates.py, llm.py, validate.py, worker.py
  audit/        log.py, verify.py, __main__.py
  api/          main.py (FastAPI), schemas.py (pydantic, mirrors section 6)
  sim/          scan_sim.py, owner_sim.py, payment_sim.py, replay.py
  eval/         offline_eval.py (PR-AUC, precision@capacity, ECE, held-out typology, ablations)
web/            React + Tailwind console: Queue, DecisionDetail, Dashboard, AuditVerify
configs/        rules_v*.yaml, policy_v*.yaml, temps_v*.json, conf_v*.json
```

Split: (A) data + injection + features + serializer; (B) GBM + Laya fine-tune/serve + calibration + eval; (C) policy + conformal + exploration + OPE + audit + API; (D) console + explanation worker + simulators + demo script.

Critical path risk: Laya fine-tuning time (the notebook says 4-6 min on its data, the README says 4-5 h for ~30k questions). Start fine-tuning on day 1 with a small dataset (~2,000 labelled bookings x 3 questions = 6,000 questions, the same size as the notebook run). Keep GBM-only as a working fallback so the demo never depends on the fine-tune finishing.

## 9. Open items and things we have not verified
- Laya latency on our states (length ~320 tokens, 3 questions) on T4 and on CPU: measure on day 1.
- Whether the laya-serve response exposes the checkpoint revision per request (README: /health reports revision SHAs behind auth). Otherwise record LAYA_REVISION from config at startup.
- Whether a fine-tuned checkpoint can be served by laya-serve under a custom name (LAYA_MODELS seems to take checkpoint names; a custom repo may need the Python API). Fallback: a thin FastAPI wrapper around `laya.load(<our repo>)` exposing the same /v1/systemone body.
- All cost-matrix values, interception rates r_scan/r_oc/r_rev, tolerances and timeouts are ASSUMPTIONS.
- USPS OIG EPA alert mechanics are redacted; the ACH rule is our inference.
- The conformal guarantee holds only under exchangeability with injected calibration fraud; the held-out typology result is the honest number.

Sources used in this phase (beyond phase1/phase2 notes):
- USPS OIG, Management Alert: Enterprise Payment Account Fraud: https://www.uspsoig.gov/reports/audit-reports/enterprise-payment-account-fraud
- ACH R29 two-banking-day window vs R10 60 days: https://www.moderntreasury.com/ach-return-codes/r29 , https://ramp.com/blog/ach-return-codes
- Laya README (local copy laya_readme.md): CPU latency 193-464 ms row, 503 Retry-After, temperature clamp [0.5, 5.0], ONNX quantization warning, batch cap 64, answer_confidence gating.
