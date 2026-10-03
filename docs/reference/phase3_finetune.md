# Phase 3: Laya fine-tuning plan for FraudShield (2026-10-02)

Sources used: phase2_ideation.md, phase1_jev_laya.md, nb_src.txt (official notebook source), laya_readme.md (GitHub README), hf_readme.md (HF card), all in this scratchpad. Token counts were measured with Laya's own tokenizer (downloaded from huggingface.co/convaiinnovations/laya/resolve/main/tokenizer/tokenizer.json; script p3/tok.py, states in p3/states.json).

Tags: [NB] = read in nb_src.txt. [RM] = Laya GitHub README. [HF] = HF card. [VENDOR] = Convai claim with no independent check. [UNVERIFIED] = not confirmed in any source we read; check in code before relying on it. [ASSUMPTION] = our modelling choice or number, to be stated as such in the pitch.

## 0. Facts about Laya that constrain the design

- One forward pass per (state, question) pair. The notebook builds one training item per question: `build_sequence(tok, state, {"t","ins","crit"}, max_len, head_max_len)` [NB]. So 8 questions per booking = 8 sequences at train and at inference (batched).
- English `laya`: 512 tokens, option budget `head_max_len` 192. `head_max_len` is a cap; state room = `max_len - head_len - 1` for the question actually asked [RM].
- Row format the notebook reads: `id`, `workflow`, `state` (JSON string), `questions` (JSON string), `gold` (JSON string). Targets: choice = `gold[qid]["probabilities"][key]`; noul = `[p("false"), p("true")]` (that order); score = `p(str(i))` for i in range(len(criteria)), criteria must be a list. Targets are renormalized; items whose marker count mismatches are silently dropped (`return None`) [NB].
- Eval reads `gold[qid]["label"]`, `gold[qid]["score"]` (score) and `gold[qid]["noul"]` (noul) [NB].
- Noul pitfall: on the English checkpoint a noul can follow its `false:`/`true:` labels instead of the state; a criteria-less noul "answers no whatever the state". Give every noul `criteria` keyed `true`/`false` [RM, issue #156]. Avoid boolean-word labels in choice keys [RM].
- `action.act_probability` carries no usable signal (#185) [RM]. We ignore the act head.
- Temperatures: runtime applies per type, or per (type, option count) bucket, clamped to [0.5, 5.0]; bucket entries override per-type [RM]. No per-question-id temperature in the runtime [UNVERIFIED that none exists; not found in README].
- Score is the weakest primitive (SST-5 0.372) [RM]. Base checkpoints are near chance zero-shot on typed-decisions (0.362 vs majority 0.461) [RM, VENDOR].
- Drift found in the notebook: preprocessing builds items with `cfg["max_len"]`/`cfg["head_max_len"]` read from the downloaded `rl_agent_config.json`, while `train_ddp.py` overrides them to 1024/256 and exports 1024/256. The README also says calibration samples come from training items, but the current code holds out 10% [NB vs RM]. We set both values explicitly in both places.

## 1. Task design (question dict, Laya format)

Eight questions per booking. Two families with different gold sources (this matters for circularity, section 3):

- LATENT questions: gold comes from the injection ground truth, which is not a deterministic function of the state. These carry detection value.
- ANCHOR questions: gold is a documented rule over features that are already in the state. They are reading checks and explanation anchors, not detectors. They are the first thing cut if compute is short.

```json
{
 "misuse": {"type": "noul",
   "instructions": "Is this booking being made by someone misusing this account, rather than by the legitimate account holder for their own business?",
   "criteria": {"true": "misuse: a third party or a fraudulent holder is using the account",
                "false": "legitimate: the holder is shipping for its own business"}},
 "foreign_customers": {"type": "noul",
   "instructions": "Is the account now shipping on behalf of senders, origins or consignees outside its own customer base, beyond its normal growth?",
   "criteria": {"true": "the account is serving parties it never served before, a break from its pattern",
                "false": "the parties fit the account's existing customer base or its normal growth"}},
 "consistent_history": {"type": "noul",
   "instructions": "Is this booking consistent with the account's own history of lanes, weights, sizes, service level and timing?",
   "criteria": {"true": "consistent: similar to what this account usually books",
                "false": "inconsistent: unlike what this account usually books"}},
 "payer_relationship": {"type": "noul",
   "instructions": "Has the billed account paid for shipments from this sender and origin before this booking?",
   "criteria": {"true": "yes, an established payer-sender relationship exists",
                "false": "no, the payer has never billed this sender or origin"}},
 "drop_consignee": {"type": "noul",
   "instructions": "Does the consignee look like a reshipping drop: a recent address receiving parcels from many unrelated senders?",
   "criteria": {"true": "likely a drop address", "false": "an ordinary consignee"}},
 "payoff_max": {"type": "noul",
   "instructions": "Was this booking chosen to maximize the shipping cost charged to the account, relative to the account's norm (heavier, longer, faster)?",
   "criteria": {"true": "cost-maximizing: far above the account's usual cost per parcel",
                "false": "cost is in line with the account's usual parcels"}},
 "risk_level": {"type": "score",
   "instructions": "What is the fraud risk level of this booking, combining whether it is misuse and how much the carrier would lose?",
   "criteria": ["legitimate booking by the account holder",
                "misuse, low loss: cost near the account usual",
                "misuse, moderate loss: up to 3x the account median cost",
                "misuse, high loss: 3x to 8x the account median cost",
                "misuse, severe loss: over 8x median or part of a burst"]},
 "action": {"type": "choice",
   "instructions": "Which action should the carrier take on this booking before it enters the network?",
   "criteria": {"allow": "accept normally",
                "allow_scan_gated": "accept, but hold at first scan if weight, size or drop-off point differ",
                "owner_confirm": "ask the account owner to confirm out of band before accepting",
                "review": "send to a fraud analyst before accepting",
                "hold": "accept the parcel but stop it before linehaul until verified",
                "block": "refuse the booking and lock label creation"}}
}
```

Why each exists:

| qid | family | insight | gold source | job |
|---|---|---|---|---|
| misuse | latent | all | is_fraud from injection | primary calibrated P(fraud) for the decision rule |
| foreign_customers | latent | I2, I5 | generator introduced third-party senders/origins/consignees in an active episode (T1, T3) | separates takeover-resale from organic growth (hard negatives: new channel, 3PL, new region) |
| consistent_history | anchor | profile comparison in the brief | rule: 1 - sigmoid((max(|weight z|, |dims z|, lane_unseen*4, payoff z) - 3)/0.5) | flags mimicry fraud (consistent but misuse) for analysts; reading check |
| payer_relationship | anchor | I1 | deterministic: payer-sender pair age > 0 days | reading check; anchor for explanations |
| drop_consignee | latent | I4 | consignee is a generator-created drop (T3) | mode signal; drives scan-gated/hold rather than owner_confirm |
| payoff_max | latent | I7 | booking drawn from the payoff-skewed sampler (T4 always, T1 with prob 0.6) | mode + severity signal; distinguishes legit heavy outliers |
| risk_level | latent | I7 | 0 if legit; else tier by victim cost relative to account median (cut points 1x/3x/8x, 4 if burst >= 10 in 24h) | analyst triage ordering; P(level >= 1) is a second estimate of P(fraud), used as a head-consistency check |
| action | decision | I6 (false intercepts are costly) | no probability gold; trained with cost reward (section 4) | compared alternative to the rule |

Why "foreign_customers" is not just "novel consignee rate": Olist customers are mostly one-time buyers, so a real B2C seller has about 10 of 10 new consignees in any 10 bookings (see example states). Consignee novelty alone is uninformative for B2C shippers. The resale signature is new SENDERS and ORIGINS under one payer (I5) and the jump relative to the account's own base rate (I2). The question asks for that judgment; the raw rates are in the state.

How they combine (recommended):
1. Each Laya head is calibrated separately per question id (section 5).
2. P(fraud) = calibrated `misuse`. Mode probabilities for action effectiveness come from `foreign_customers`, `drop_consignee`, `payoff_max`, plus deterministic tenure (account < 30 days: owner_confirm is ineffective because the owner may be the fraudster, T2).
3. Action = argmin over a in A(p) of E[cost(a)] = (1 - p) * c_legit(a) + p * sum_m P(m | fraud) * c_fraud(a, m), where A(p) excludes `allow` when p is above the conformal threshold for the booking's tenure group (section 5). Review cost includes a shadow price for analyst capacity tuned on the calibration set.
4. Laya's own `action` Choice is computed and logged, and evaluated as an alternative policy. It is not the decider in v1.

Why the rule decides and not the Choice head: with a known cost matrix and calibrated probabilities the Bayes-optimal action is a threshold/argmin on expected cost (Elkan, IJCAI 2001). The rule lets the business change costs or capacity without retraining, and every action has an arithmetic trace for audit. The Choice head bakes the cost matrix into weights, its output probabilities are not calibrated probabilities of anything (section 4), and its errors are harder to explain. Its possible advantage is using evidence that the scalar P(fraud) and the three mode nouls do not carry. We measure that instead of assuming it.

Illustrative cost matrix [ASSUMPTION, R$, to be stated as assumptions on the slide]. L = carrier cost of the shipment + R$15 dispute/handling.

| action | legit cost | fraud cost (takeover modes) | fraud cost (bust-out, tenure < 30d) |
|---|---|---|---|
| allow | 0 | L | L |
| allow_scan_gated | 0.5 | 0.5 + 0.6L (0.2L if T6 dims/weight mismatch) | same |
| owner_confirm | 3 | 3 + 0.1L | 3 + 1.0L |
| review | 10 | 10 + 0.25L | 10 + 0.25L |
| hold | 12 | 12 + 0.05L | 12 + 0.05L |
| block | 60 | 0 | 0 |

Consequence: allow vs owner_confirm switches at p about 3/(0.9L + 3). For L = R$33 that is about 9%; for L = R$228 (example T1) about 1.4%. High-payoff bookings face friction at lower probability, which is the I7 economics argument made concrete.

## 2. Serialization

Fixed-order, line-based text, `KEY value | value` style, numbers rounded, no free text except a product category from a closed vocabulary. State passed as a plain string (JSON-encoded string in the row) so we control order and length; how Laya renders a dict state is [UNVERIFIED].

Template (one line per group; fields always present, "0"/"none" when empty):
```
BOOKING <date> <dow> <hh:mm> | channel <web|api|counter> | login device seen <d>d | payment <account billing|card|ach>
ACCOUNT tenure <d>d | bookings 90d <n> (<r>/day) [| last 24h <n>] | home origin <UF> <zip3> | prior confirmed fraud <n>
SHIPMENT origin <UF> <zip3> -> dest <UF> <zip3> | <km> km | <kg> kg | <LxWxH> cm | service <standard|express> | category <vocab>
COST carrier cost R$<x> | cost vs acct median x<r> | payoff z <z>
PROFILE lane seen <n>x | weight z <z> | dims z <z> | hour pct <p> | service std <s>% exp <e>%
SENDER <= account|differs from account> | payer-sender pair age <d>d | payer-origin pair age <d>d
CONSIGNEE first seen <d>d | other senders to it 30d <n> | dest region seen by acct <n>x
CHANGE last 10 vs 90d base: new consignees <n> (base <b>) | new senders <n> (base <b>) | new origins <n> (base <b>) | distinct senders 7d <n> (base <b>)
[SEQUENCE <short structured summary of last 72h, optional>]
GRAPH payer->senders 30d <n> | consignee<-accounts 30d <n> | 2-hop links to confirmed fraud <n>
[GBM risk <p>]   (only in variant e)
```

Raw fields: timestamp, channel, payment type, tenure, origin/dest (UF + zip3), distance, weight, dims, service, category, carrier cost.
Pre-computed (as-of, strictly from rows with timestamp < booking time): bookings 90d/24h; lane seen count; weight/dims robust z ((x - median)/(1.4826*MAD), account history, min 5 prior bookings else "n/a"); hour percentile; service mix; cost vs median and payoff z (log cost, robust z); payer-sender and payer-origin first-seen age; consignee first-seen age and distinct senders 30d; change block (counts in last 10 bookings vs 90-day base rate per 10); graph degrees on the billing graph; 2-hop links to entities with a CONFIRMED label whose confirmation time (simulated 30-day delay) is before the booking; GBM out-of-fold, time-respecting score.

Service tier: Olist has no tier. Proxy [ASSUMPTION]: express if (estimated delivery - purchase) is in the account's fastest 20% for that lane distance band; injected express flag otherwise.

Token budget (measured with Laya tokenizer, state only): legit 295, T1 resale 301, T5 test-then-burst 331. Approximate head sizes (instruction + rendered options; exact rendering by `render_options` not run): nouls 43-64, risk_level ~98, action ~97. Worst case 331 + 98 + specials is about 432 < 512. Rule: hard cap state at 380 tokens, assert at preprocessing that no item is truncated.

Example 1, legitimate booking (295 tokens):
```
BOOKING 2018-06-12 Tue 14:05 | channel web | login device seen 388d | payment account billing
ACCOUNT tenure 412d | bookings 90d 138 (1.5/day) | home origin SP 013 | prior confirmed fraud 0
SHIPMENT origin SP 013 -> dest RJ 220 | 362 km | 1.2 kg | 18x12x10 cm | service standard | category housewares
COST carrier cost R$18.40 | cost vs acct median x1.1 | payoff z +0.2
PROFILE lane seen 37x | weight z +0.3 | dims z +0.1 | hour pct 48 | service std 96% exp 4%
SENDER = account | payer-sender pair age 412d | payer-origin pair age 412d
CONSIGNEE first seen 0d | other senders to it 30d 0 | dest region seen by acct 214x
CHANGE last 10 vs 90d base: new consignees 10 (base 9.6) | new senders 0 (base 0.0) | new origins 0 (base 0.0) | distinct senders 7d 1 (base 1)
GRAPH payer->senders 30d 1 | consignee<-accounts 30d 1 | 2-hop links to confirmed fraud 0
GBM risk 0.004
```

Example 2, label-resale account takeover T1 (301 tokens):
```
BOOKING 2018-06-14 Thu 02:41 | channel api | login device seen 0d | payment account billing
ACCOUNT tenure 412d | bookings 90d 138 (1.5/day) | last 24h 19 | home origin SP 013 | prior confirmed fraud 0
SHIPMENT origin PR 806 -> dest AM 690 | 2,874 km | 9.8 kg | 50x40x35 cm | service express | category electronics
COST carrier cost R$212.60 | cost vs acct median x11.6 | payoff z +6.4
PROFILE lane seen 0x | weight z +5.1 | dims z +4.7 | hour pct 1 | service std 96% exp 4%
SENDER differs from account | payer-sender pair age 0d | payer-origin pair age 0d
CONSIGNEE first seen 0d | other senders to it 30d 0 | dest region seen by acct 0x
CHANGE last 10 vs 90d base: new consignees 10 (base 9.6) | new senders 8 (base 0.0) | new origins 7 (base 0.0) | distinct senders 7d 9 (base 1)
GRAPH payer->senders 30d 9 | consignee<-accounts 30d 1 | 2-hop links to confirmed fraud 0
GBM risk 0.71
```

Example 3, test-shipment-then-burst T5 (hypothesis typology, held out; 331 tokens):
```
BOOKING 2018-07-03 Tue 23:12 | channel web | login device seen 2d | payment account billing
ACCOUNT tenure 233d | bookings 90d 41 (0.5/day) | last 24h 6 | home origin MG 357 | prior confirmed fraud 0
SHIPMENT origin MG 357 -> dest SP 041 | 498 km | 14.5 kg | 60x45x40 cm | service express | category furniture
COST carrier cost R$164.10 | cost vs acct median x6.8 | payoff z +4.2
PROFILE lane seen 3x | weight z +3.9 | dims z +3.3 | hour pct 3 | service std 100% exp 0%
SENDER = account | payer-sender pair age 233d | payer-origin pair age 233d
CONSIGNEE first seen 2d | other senders to it 30d 0 | dest region seen by acct 3x
CHANGE last 10 vs 90d base: new consignees 10 (base 9.9) | new senders 0 (base 0.0) | new origins 0 (base 0.0) | distinct senders 7d 1 (base 1)
SEQUENCE 2d ago 1 small parcel 0.3 kg to this consignee, delivered | since then 5 bookings, all express, mean 13.1 kg
GRAPH payer->senders 30d 1 | consignee<-accounts 30d 1 | 2-hop links to confirmed fraud 0
GBM risk 0.38
```
Note T5 keeps sender = account (no resale), so `foreign_customers` should stay low and detection must come from payoff, device age and sequence. This is why it is a good held-out test.

## 3. Training data

Base: Olist orders, booking = (order_id, seller_id) pair, account = seller_id, origin = seller zip, consignee = customer_unique_id + zip, weight/dims from products (summed per booking), carrier cost = freight_value sum. Approximate monthly volumes [from memory, verify on load]: through 2017-05 about 11.7k orders; 2017-06..2018-01 about 41k; 2018-02..04 about 21k; 2018-05..08 about 26k.

Time split (no row of one booking in two sets; features as-of booking time):
- Warm-up 2016-09-04 .. 2017-05-31: profiles only, no training rows.
- Train 2017-06-01 .. 2018-01-31 (contains Black Friday 2017-11-24 surge as real hard negatives).
- Embargo 2018-02-01 .. 2018-02-14 (feature windows and label delay).
- Calibration 2018-02-15 .. 2018-04-30.
- Embargo 2018-05-01 .. 2018-05-14.
- Test 2018-05-15 .. 2018-08-31. (Sep-Oct 2018 is near-empty in Olist; drop.)
- Accounts: compromise different accounts in each period (an account compromised in test was never compromised in train). Profiles still use real history from all earlier periods, which is realistic.

Injection: T1-T6 per phase2. Each injected row: is_injected, scenario_id, typology, params, seed. Fraud rows are appended to the victim account's stream and DO enter later profiles (the production system would have seen them). Generated values are snapped to real Olist value grids (zip prefixes, weights from the product table, freight from a lane-distance regression on real rows) and IDs use the same 32-hex format, to prevent formatting artifacts. Parameters per typology are drawn from ranges anchored to sources (Hao et al. drop lifecycle 5-15 parcels over ~30 days; Harrod escalation; flat resale pricing) and randomized, including a "mimic" intensity knob where the fraudster samples lane/weight from the victim's own profile for 30% of episodes.

Held-out typology: T5 test-then-burst (primary). Reason: it is the unsourced hypothesis, so training on documented typologies and testing on it is the honest test; it also leaves all question golds trainable. Secondary run if compute allows: hold out T3 (drop cluster), with `drop_consignee` evaluated zero-shot from its instructions only.

Gold per question:
- Latent (misuse, foreign_customers, drop_consignee, payoff_max, risk_level): hard 0/1 (or one-hot level) from injection truth. No label smoothing: with a proper scoring rule, hard labels already train toward P(y | state); smoothing would bias calibration. Temperature handles residual over-confidence.
- Anchors (consistent_history, payer_relationship): rule-derived as in section 1, soft via the logistic for consistent_history, hard for payer_relationship.
- action: no probability gold. The row carries a `cost` vector (6 numbers) computed from truth, mode and L. For notebook compatibility `probabilities` holds the zero-uncertainty Bayes action as one-hot, used only for logging; CE weight on action is 0.

Circularity, stated plainly:
- Anchor golds are functions of features in the state. Laya learning them proves it reads the state; it proves nothing about fraud. We report anchor accuracy separately and never count anchors as detection results.
- Latent golds come from our generator. If the generator and the feature code share assumptions, the model learns our assumptions. Mitigations: features written from the domain notes before the injection code is finalized (by a different team member); randomized parameters and the mimic knob; low-signal fraud included (first 1-2 bookings of an episode, before any change features move); held-out typology; injection-detectability test (section 8). Claim scope: "detects our documented injections of these typologies on real shipper histories", not real-world fraud performance.

Class imbalance:
- Injected population: about 4% of active accounts compromised per period, giving a booking-level fraud rate near 1% in calibration and test (base case). Sensitivity at 0.3% and 3% by subsampling fraud rows. Real carrier ATO prevalence is unknown [ASSUMPTION].
- Training sample is enriched: 6,000 bookings = 1,200 fraud (20%) + 1,800 hard negatives + 3,000 random legit. Enrich by sampling, not by loss weights (doing both double-corrects). Prior shift is corrected at calibration (section 5).
- Hard negatives: real first shipment to a new UF; real Black Friday 2017 surge bookings; real heavy/high-freight outliers; injected-legit "new marketplace channel" (an account gains many new consignees, same origin, same payer, same weight profile); injected-legit 3PL accounts with many senders from day one (stable diversity, no jump). Injected-legit rows are labeled legit and flagged is_injected.

## 4. Objective and the RL link

Official recipe [NB, HF]: for each item, sample G = 4 Gaussian perturbations of the logits (sigma 0.4 -> 0.1 across epochs, zero-mean projected over valid options), q = softmax(z + eps). Reward = proper_reward(q, target) mixing log, spherical (w_sph 0.75) and RPS for ordinal (w_rps 1.0). Advantage = reward minus group mean, divided by std. Loss = -adv * log-density of the perturbation (REINFORCE with group-mean baseline, "GRPO-style") + 1.0 * soft cross-entropy to the target.

Why this aims at calibration: a strictly proper scoring rule S has E_{y ~ p}[S(q, y)] maximized only at q = p. With hard labels, the population optimum per state is q = P(y | state); with soft targets t, it is q = t. Log score, spherical score (q_y / ||q||) and RPS (ordinal, on cumulative distributions) are all strictly proper (Gneiting and Raftery 2007).

Honest caveats on the official recipe: (1) the RL term optimizes the score of the noise-smoothed distribution, whose optimum is close to but not exactly t; sigma decay and the CE term (log score, exactly proper) limit the bias. (2) Finite data and a 421M model still over-fit; the vendor's own numbers show raw ECE is poor and temperature refit is what brings it down (0.466 -> 0.081 [VENDOR]). So in our plan calibration rests on held-out temperature/Platt fitting, not on the training objective alone. (3) The "RL" here is a score-function estimator of a known differentiable objective; CE alone may do as well. Ablation: CE-only vs CE + RL on the misuse head (one extra short run if compute permits).

Our extension, kept separate:
- Noul and Score heads: unchanged official objective (proper reward + CE). Nothing cost-weighted touches them. Per-question loss weight: misuse 2.0, others 1.0 [ASSUMPTION].
- Action head only: reward r(q) = -(q . c) / C_SCALE where c is the booking's cost vector and C_SCALE a global constant (95th percentile cost in train), same perturbation and group baseline, CE weight 0. Expected reward over the population is maximized by putting all mass on argmin_a E[c_a | state], the Bayes action. That is what we want from a decision, but it is NOT a proper scoring rule for probabilities: the reward is linear in q, so the head is pushed toward one-hot answers and its "probabilities" do not mean P(action is right). We never threshold or calibrate them as probabilities; we log them and evaluate the head only by realized cost.
- Equivalent and preferred route: apply cost in the decision rule over calibrated P(fraud) and mode probabilities (section 1). Same Bayes target, but cost changes need no retraining.
- Delayed reward (deployment, simulated in the hackathon): outcomes arrive late and selectively. review/owner_confirm/hold resolve in hours to days; allow resolves via non-payment or customer dispute after 30-60 days; block resolves never (selective labels). Use: (a) retrain the latent nouls with proper scoring on resolved labels, importance-weighted by 1/propensity for rows from logged epsilon-exploration near the boundary (Manapat/Stripe style, about 5% let-through; Kilbertus et al. 2020); (b) off-policy evaluation of the rule vs the Choice head with IPS/SNIPS/DR in `obp` (Saito et al. 2021); (c) realized cost per booking becomes the action-head reward in the next training round, with propensity weights. Delayed-feedback correction for unresolved allows: Chapelle KDD 2014 style elapsed-time model. In the hackathon all of this runs on simulated outcomes and must be labelled as such.

## 5. Calibration

Per question id, on the calibration window (2018-02-15 .. 04-30), at the deployment prevalence (1% base).

1. Export the checkpoint with `cfg["temperature"] = [1.0, 1.0, 1.0]` and `temperature_by_options` removed, so the runtime returns raw softmax(z). (1.0 is inside the [0.5, 5.0] clamp.)
2. Apply our own per-qid temperature outside Laya: p_cal = softmax(log p_raw / T_qid). Exact equivalent of scaling logits, because log p = z - logsumexp(z) and softmax is shift-invariant.
3. Fit T_qid by LBFGS on NLL (reuse the notebook's `fit_one_temp`, grouped by qid instead of qtype). For `misuse`, fit 2-parameter Platt (temperature + bias) because the training set is enriched to 20% and temperature alone cannot fix a base-rate shift. Equivalent check: Saerens et al. (2002) prior correction logit += log(pi_dep/(1-pi_dep)) - log(pi_train/(1-pi_train)) then temperature; report both.
4. Calibration set size: legit subsampled to 10,000 bookings with weight w = (true legit count / 10,000), all injected fraud kept with weight 1, giving effective 1% prevalence. Fit and metrics use weights.
5. Metrics before and after, per qid: weighted ECE (15 equal-mass bins), Brier, log loss, reliability diagram with 1,000 account-cluster bootstrap bands; for risk_level also RPS and class-wise ECE. Report on the TEST window, never on the calibration window.
6. Save `calibration.json` = {version, fit window, T per qid, Platt (a, b) for misuse, conformal thresholds, cost matrix version, prevalence assumption}. Hash it into every audit record.

Conformal guarantee on "allow" (split conformal risk control, Angelopoulos et al., "Conformal Risk Control", ICLR 2024):
- Split the calibration window by account into C1 (temperature/Platt fit, 60%) and C2 (conformal, 40%), so the threshold is chosen on data the calibrator did not see.
- Mondrian groups by tenure: g in {<30d, 30-180d, >180d}.
- Guarantee A (miss rate among frauds, class-conditional): within group g, take the n_g fraud bookings in C2 with calibrated scores p_i. Allow if p <= lambda. Loss L_i(lambda) = 1{p_i <= lambda}, monotone in lambda, bounded by 1. Choose lambda_g = max{lambda : (n_g/(n_g+1)) * mean_i L_i(lambda) + 1/(n_g+1) <= alpha}. Then E[P(allowed | fraud, group g)] <= alpha for a new exchangeable fraud booking. alpha = 0.05 needs n_g >= 19; we will have hundreds of injected frauds per group.
- Guarantee B (joint rate): over all C2 bookings in group g with weights for prevalence, L_i(lambda) = 1{y_i = fraud and p_i <= lambda}, target alpha_joint = 0.0005 (fraud-and-allowed per booking). Needs n_g >= 1/alpha - 1 = 1,999 effective bookings. The fraud rate AMONG allowed is joint / P(allow); it is bounded approximately by alpha_joint / P(allow), not exactly. Say "approximately".
- For a high-probability statement instead of in-expectation, use Learn-then-Test (Angelopoulos et al. 2021) with delta = 0.1.
- Decision rule then excludes `allow` when p > lambda_g; scan-gated allow is not covered by the guarantee and is reported separately.
- What the guarantee does NOT cover: exchangeability fails under time drift, adaptive fraudsters, and new typologies. We show this directly: report empirical allowed-fraud rate on the held-out T5 and on the test window; if it exceeds alpha, say so.

## 6. Compute and timeline

Item counts: 6,000 training bookings x 8 questions = 48,000 items per epoch. Calibration inference about 15,000 bookings x 8 = 120k items; test similar.

Throughput, with the conflict stated: notebook text says "~4 to 6 minutes" for 6,000 questions x 4 epochs on 2xT4 (about 70-100 items/s); README says "4-5 hours for 4 epochs over ~30k questions" (about 7-8 items/s at max_len 1024). The two differ by about 10x. We plan with the README figure and scale for our shorter sequences (about 430 tokens max vs up to 1024): assume 11-15 items/s [ASSUMPTION]. RL perturbations act on logits only, so G = 4 adds almost no compute.

- One epoch of 48k items: 53-73 min. Two epochs: about 2-2.5 h. (README-rate worst case at 7 items/s: about 3.8 h.)
- Inference: README claims 103-332 q/s batched on one T4 [VENDOR]; 240k items for cal + test is about 15-40 min.
- Pilot first: 200 optimizer micro-steps, measure items/s, then choose the fallback level.

Kaggle budget (30 GPU h/week, 12 h max session [ASSUMPTION, check Kaggle limits]):
| run | est. GPU h |
|---|---|
| pilot + token/truncation asserts | 0.5 |
| main (d): fine-tuned, no GBM line | 2.5 |
| main (e): fine-tuned, with GBM line | 2.5 |
| held-out T3 run (optional) | 2.5 |
| shuffled-label sanity (2,000 bookings, 1 epoch) | 0.3 |
| zero-shot stock Laya eval | 0.5 |
| cal + test inference, all variants | 2.0 |
| total | about 11 |

Changes vs notebook defaults: EPOCHS 2 (not 4); MICRO_BATCH 16 at 512 if memory allows (T4 16 GB, fp16, gradient checkpointing on), else 8 with GRAD_ACCUM 4; sigma schedule 0.4 -> 0.1 over 2 epochs; LRs unchanged (enc 2.5e-5, head 1e-4).

Colab: free tier is a single T4 (session limits vary [UNVERIFIED]); use it for preprocessing, inference and the shuffled-label run, run training on Kaggle. For single GPU: `--nproc_per_node=1`, GRAD_ACCUM 8.

Fallbacks, in order:
1. Drop anchors (payer_relationship computed in code, consistent_history dropped): 6 questions, -25% compute.
2. max_len 384: drop PROFILE hour pct, SEQUENCE, GRAPH 2-hop; target state <= 250 tokens.
3. 1 epoch.
4. 3,000 bookings (600 fraud).
5. laya-multilingual (322M, 1,024 ctx; about 2x faster per README speed table [VENDOR]); weaker on English, ships with no temperatures, and has a score position bias (#131): drop risk_level on this path.
6. Last resort: freeze the encoder and train only the head (the stuntd approach), cheap but weaker.

## 7. Evaluation

Variants: (a) stock `laya` zero-shot (same states and questions, vendor temperatures, then our temperature fit as a second row); (b) stock Jev zero-shot only if API access arrives (early access/waitlist), same states and questions, 1,000-booking test subsample; (c) LightGBM on the same pre-computed features + graph degree features, isotonic/Platt calibrated on C1; (d) fine-tuned Laya without the GBM line; (e) fine-tuned Laya with the GBM line; plus decision policies (rule on (c), rule on (d)/(e), Laya action head).

Metrics (test window, 1% prevalence, account-cluster bootstrap 95% CIs):
- Detection: PR-AUC; recall at 0.5% and 1% legit friction rate; precision at analyst capacity (top 0.3% of bookings).
- Calibration: ECE, Brier, log loss for misuse; RPS for risk_level.
- Decision: expected cost per 1,000 bookings under the cost matrix with simulated verification outcomes; R$ of fraud allowed; legit friction rate (share of legit bookings not plainly allowed); analyst load.
- Time to act: per compromised account, number of fraud bookings and R$ before the first non-allow action.
- Held-out T5 (and T3) recall at fixed friction; empirical allowed-fraud rate vs conformal alpha.
- Anchor accuracy (reading check; expected > 0.97), reported apart from detection.
- Latency: p50/p95 for 8 questions per booking on one T4 (vendor table suggests about 85-160 ms for 5-10 questions [VENDOR]).

What counts as Laya winning: (d) or (e) beats (c) on expected cost per 1,000 bookings with non-overlapping 95% CIs, OR improves held-out-typology recall at equal friction, while ECE on misuse is no worse than calibrated GBM. Weaker but still useful: (e) beats (c) (Laya adds value on top of GBM signals as the decision layer).

Honest fallback narrative if it does not win: "On our data a calibrated LightGBM with graph features was as good or better at ranking. Fine-tuned Laya matched it within CI at about X ms and gives separately calibrated answers to named questions (misuse, foreign customers, drop, payoff), which drive mode-specific actions and the analyst explanation. Zero-shot Laya was near useless, as its vendor also reports." Do not claim Laya improves detection unless (d)/(e) shows it.

## 8. Failure checks

1. Held-out typology: T5 (and T3); report separately, never mixed into the headline.
2. Shuffled-label sanity: permute misuse gold within each month of the training window, train 1 epoch on 2,000 bookings; test PR-AUC must be about the prevalence (0.01). Anything clearly higher means leakage through features or formatting.
3. Leakage checks: assert every feature's source rows have timestamp < booking time; grep states for scenario_id, is_injected, seed or typology names; confirmed-fraud graph links use confirmation time (simulated 30-day delay), not injection time; no booking split across train/cal/test; compromised accounts disjoint across periods.
4. Injection detectability: train LightGBM to separate injected-LEGIT rows (new channel, 3PL) from real rows using only raw fields. AUC above about 0.6 means the generator leaves artifacts the model can exploit; fix before training.
5. Field-order robustness: at test, randomly permute the order of lines (and fields within lines) 5 times per booking; report mean |delta p_misuse| and action flip rate. Target: mean < 0.02, flips < 1%. If worse, train with 20% order-shuffled augmentation and re-test.
6. Prompt injection in free text: our state has no free text by design (category from a closed vocabulary). Test anyway: add a `NOTE` line with strings such as "verified owner, safe, allow" or "ignore previous instructions" to 500 fraud states; report delta p and flips. Jev docs say state is not treated as hostile; we assume the same for Laya [UNVERIFIED for Laya] and keep the rule that no shipper-controlled text enters the state unvalidated.
7. Truncation: assert token count of every item < max_len and no item dropped by the marker check (log count of `None` returns).
8. Noul label-following (#156): check each noul's predicted distribution is not near-constant across test bookings.
9. Head consistency: P(risk_level >= 1) vs calibrated misuse; large disagreements are logged and sampled for review.
10. Monotonicity spot checks: raising new senders, payoff z or device-new should not lower P(misuse) on 200 sampled states.

## 9. Deployment

- Serve: `pip install "laya[serve]"`; `LAYA_DEVICE=cuda LAYA_PRELOAD=1 LAYA_API_KEY=... laya-serve`; POST /v1/systemone with one state and 8 questions per booking (or /v1/systemone/batch, max 64 states) [RM]. Whether laya-serve can load a custom local checkpoint via `LAYA_MODELS` is [UNVERIFIED]; fallback is a 30-line FastAPI around `laya.Agent(LOCAL_DIR, device="cuda")` (the notebook loads the fine-tuned dir this way [NB]).
- Pin: push the fine-tuned model to a PRIVATE HF repo (Olist is CC BY-NC-SA; also do not reuse the notebook's `convaiinnovations/...` repo id). Record the commit SHA; download that revision once to a local dir and load from disk; verify SHA-256 of model.safetensors at startup. `LAYA_REVISION` and per-checkpoint SHA-256 maps exist per README changelog; exact usage [UNVERIFIED].
- Pin the `laya` package version (0.3.23 at time of writing) and transformers version.
- Audit record per booking (append-only, hash-chained): booking_id, timestamp, model repo + commit SHA + weights SHA-256, laya version, rl_agent_config hash, calibration.json version + hash, cost-matrix version, feature-store snapshot version, exact state text and its SHA-256, raw probabilities per qid, calibrated probabilities per qid, conformal lambda used and tenure group, rule trace (expected cost per action), chosen action, Laya action-head output (logged, not used), latency, prev_hash. Capture raw outputs with an `on_predict_end` hook [RM].
- Set `LAYA_API_KEY` so /health does not expose checkpoint SHAs to unauthenticated callers [RM].

## 10. Key changes to the official notebook

1. Data loader: `load_dataset("json", data_files={"train": ..., "calib": ..., "test": ...})` on our JSONL; keep `id/workflow/state/questions/gold`, add `split`, `booking_ts`, `account_id`, `typology`, `is_injected`, `scenario_id` (never put these in state).
2. Token budgets: set `cfg["max_len"]=512`, `cfg["head_max_len"]=192` explicitly in preprocessing AND in train_ddp.py (fix the 1024/256 drift); assert no truncation; count dropped items.
3. Items carry `qid`, `booking_id`, `weight`, and for action a `cost` vector.
4. Replace the random item-level 10% calibration holdout (seed 20260922) with our time-based, booking-level calibration split loaded from file.
5. Loss: per-qid branches. Noul/score: official proper reward + CE, loss weight per qid. Action: reward = -(q . cost)/C_SCALE, CE weight 0.
6. EPOCHS 2, MICRO_BATCH per pilot, total_updates recomputed.
7. Optional 20% field-order shuffle augmentation (flag).
8. Temperature fit grouped by qid (not qtype); Platt for misuse; write calibration.json; export cfg temperatures [1,1,1] and pop temperature_by_options.
9. Eval cell: replace typed-decisions metrics and the hard-coded Jev/ModernBERT comparison table (numbers from a different benchmark) with our fraud metrics, cost simulation, conformal check, perturbation and injection tests.
10. Push cell: private repo, our org, no auto-generated claims in the model card.
11. Log seeds, item-manifest hash, git hash of data generator.

## 11. Three training records (notebook JSONL schema)

`state`, `questions`, `gold` are JSON-encoded strings, as the notebook calls `json.loads` on each. `questions` is the dict in section 1 (abbreviated here as Q_JSON; the real file repeats it per row). Extra top-level fields are ignored by the notebook loader.

```json
{"id":"bk_2017-09-12_7c3e..a1","workflow":"fraudshield_booking","split":"train","booking_ts":"2017-09-12T14:05:00","account_id":"3504c0cb71d7fa48d967e0e4c94d59d9","typology":"none","is_injected":false,"scenario_id":null,
 "state":"\"BOOKING 2017-09-12 Tue 14:05 | channel web | login device seen 201d | payment account billing\\nACCOUNT tenure 225d | ... \\nGBM risk 0.004\"",
 "questions":"Q_JSON",
 "gold":"{\"misuse\":{\"label\":\"false\",\"noul\":0.0,\"probabilities\":{\"true\":0.0,\"false\":1.0}},\"foreign_customers\":{\"label\":\"false\",\"noul\":0.0,\"probabilities\":{\"true\":0.0,\"false\":1.0}},\"consistent_history\":{\"label\":\"true\",\"noul\":0.98,\"probabilities\":{\"true\":0.98,\"false\":0.02}},\"payer_relationship\":{\"label\":\"true\",\"noul\":1.0,\"probabilities\":{\"true\":1.0,\"false\":0.0}},\"drop_consignee\":{\"label\":\"false\",\"noul\":0.0,\"probabilities\":{\"true\":0.0,\"false\":1.0}},\"payoff_max\":{\"label\":\"false\",\"noul\":0.0,\"probabilities\":{\"true\":0.0,\"false\":1.0}},\"risk_level\":{\"label\":0,\"score\":0.0,\"probabilities\":{\"0\":1.0,\"1\":0.0,\"2\":0.0,\"3\":0.0,\"4\":0.0}},\"action\":{\"label\":\"allow\",\"probabilities\":{\"allow\":1.0,\"allow_scan_gated\":0.0,\"owner_confirm\":0.0,\"review\":0.0,\"hold\":0.0,\"block\":0.0},\"cost\":{\"allow\":0,\"allow_scan_gated\":0.5,\"owner_confirm\":3,\"review\":10,\"hold\":12,\"block\":60}}}"}
```

```json
{"id":"bk_2017-12-04_inj_T1_0042_07","workflow":"fraudshield_booking","split":"train","booking_ts":"2017-12-04T02:41:00","account_id":"<victim seller_id>","typology":"T1","is_injected":true,"scenario_id":"T1_0042",
 "state":"\"BOOKING 2017-12-04 Mon 02:41 | channel api | login device seen 0d | payment account billing\\n... SENDER differs from account | payer-sender pair age 0d ...\\nGBM risk 0.71\"",
 "questions":"Q_JSON",
 "gold":"{\"misuse\":{\"label\":\"true\",\"noul\":1.0,\"probabilities\":{\"true\":1.0,\"false\":0.0}},\"foreign_customers\":{\"label\":\"true\",\"noul\":1.0,\"probabilities\":{\"true\":1.0,\"false\":0.0}},\"consistent_history\":{\"label\":\"false\",\"noul\":0.0,\"probabilities\":{\"true\":0.0,\"false\":1.0}},\"payer_relationship\":{\"label\":\"false\",\"noul\":0.0,\"probabilities\":{\"true\":0.0,\"false\":1.0}},\"drop_consignee\":{\"label\":\"false\",\"noul\":0.0,\"probabilities\":{\"true\":0.0,\"false\":1.0}},\"payoff_max\":{\"label\":\"true\",\"noul\":1.0,\"probabilities\":{\"true\":1.0,\"false\":0.0}},\"risk_level\":{\"label\":4,\"score\":4.0,\"probabilities\":{\"0\":0.0,\"1\":0.0,\"2\":0.0,\"3\":0.0,\"4\":1.0}},\"action\":{\"label\":\"block\",\"probabilities\":{\"allow\":0.0,\"allow_scan_gated\":0.0,\"owner_confirm\":0.0,\"review\":0.0,\"hold\":0.0,\"block\":1.0},\"cost\":{\"allow\":227.6,\"allow_scan_gated\":137.1,\"owner_confirm\":25.8,\"review\":66.9,\"hold\":23.4,\"block\":0}}}"}
```

Hard negative (injected-legit new marketplace channel):
```json
{"id":"bk_2017-11-24_inj_HN_channel_0009_15","workflow":"fraudshield_booking","split":"train","booking_ts":"2017-11-24T10:22:00","account_id":"<seller_id>","typology":"HN_new_channel","is_injected":true,"scenario_id":"HN_0009",
 "state":"\"BOOKING 2017-11-24 Fri 10:22 | channel api | login device seen 0d | payment account billing\\n... SENDER = account | payer-sender pair age 310d ... CHANGE last 10 vs 90d base: new consignees 10 (base 9.7) | new senders 0 (base 0.0) | new origins 0 (base 0.0) ...\"",
 "questions":"Q_JSON",
 "gold":"{\"misuse\":{\"label\":\"false\",\"noul\":0.0,\"probabilities\":{\"true\":0.0,\"false\":1.0}},\"foreign_customers\":{\"label\":\"false\",\"noul\":0.0,\"probabilities\":{\"true\":0.0,\"false\":1.0}},\"consistent_history\":{\"label\":\"true\",\"noul\":0.71,\"probabilities\":{\"true\":0.71,\"false\":0.29}},\"payer_relationship\":{\"label\":\"true\",\"noul\":1.0,\"probabilities\":{\"true\":1.0,\"false\":0.0}},\"drop_consignee\":{\"label\":\"false\",\"noul\":0.0,\"probabilities\":{\"true\":0.0,\"false\":1.0}},\"payoff_max\":{\"label\":\"false\",\"noul\":0.0,\"probabilities\":{\"true\":0.0,\"false\":1.0}},\"risk_level\":{\"label\":0,\"score\":0.0,\"probabilities\":{\"0\":1.0,\"1\":0.0,\"2\":0.0,\"3\":0.0,\"4\":0.0}},\"action\":{\"label\":\"allow\",\"probabilities\":{\"allow\":1.0,\"allow_scan_gated\":0.0,\"owner_confirm\":0.0,\"review\":0.0,\"hold\":0.0,\"block\":0.0},\"cost\":{\"allow\":0,\"allow_scan_gated\":0.5,\"owner_confirm\":3,\"review\":10,\"hold\":12,\"block\":60}}}"}
```
(The state strings are abbreviated with "..." here; real rows carry the full template. New API channel + new device on a long-tenure account is exactly the confusable pattern this negative teaches.)

## Open items to verify in code (not in docs we read)
- How `build_sequence` renders a dict vs string state; exact head token counts from `render_options`.
- Whether `build_sequence` honours a noul `labels` override (the notebook does not pass it; we do not use it).
- Whether laya-serve can serve a local fine-tuned dir; exact `LAYA_REVISION` semantics.
- Real training throughput (pilot), which resolves the 4-6 min vs 4-5 h conflict for our setup.
