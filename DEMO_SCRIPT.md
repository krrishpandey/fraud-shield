# FraudShield: presentation script

This is the script for presenting FraudShield. **Part 1** is what we say, segment by segment, with what is on screen. **Part 2** explains how each feature works under the hood, so whoever is presenting can answer follow-up questions. **Part 3** is Q&A backup.

Target length: about 5 minutes. All numbers come from our own runs (see `PROJECT_HISTORY.md`) or a cited source. Do not round them up.

---

## Before you start (checklist)

- [ ] Open the app with `FraudShield.bat` and confirm the window opens.
- [ ] Sign in **before** you go on stage: the console opens on a sign-in screen (3D sorting line). Type any analyst name and a code of 4 or more characters, then "Open the console". It is a demo gate that labels who is at the desk, not access control (the API has no accounts; the page says so). The sign-in is remembered until you sign out from the account menu at the foot of the sidebar.
- [ ] Look at the status bar. If it says **degraded**, the fine-tuned Laya model is not loaded and the app is using the LightGBM backup score. Say so if asked. Do not hide it.
- [ ] Skip the "ask a new question" segment unless Laya is loaded (it needs Laya to answer).
- [ ] Check that `.env` has the Groq key. Without it, explanations fall back to a fixed template (still fine, just less fluent).
- [ ] Have the backup screen recording ready in case the app fails on stage.
- [ ] For the price demo in segment 4, make sure the booking you use scores between 2.6% and 3.9% risk, or the action will not flip.
- [ ] Passkey beat (5b): Windows Hello must be set up on the demo laptop (Settings > Accounts > Sign-in options > PIN). Try one enroll + confirm before going on stage. If the prompt does not appear in the app window, open the `http://localhost:<port>` URL the app prints, in Edge.
- [ ] Monitor beat (7b): start the live stream at least 30 seconds before you show the Live view, so the monitor card has stops to estimate from (precision shows n/a until the first stop; the drift number needs 500 decisions).
- [ ] Learning beat (6): the banner shows whatever the gate decided on this run. Read it out as it is, green or amber.

---

## Part 1: The script

### 1. The hook (0:00 to 0:35)

**Screen:** title slide, then one slide with the DOJ case and USPS OIG numbers.

**Say:**
> "When criminals steal a business's shipping account, they don't use it to send one parcel. In a real DOJ case, Wilson and Harrod, a company's stolen UPS login was used first for the fraudsters' own shipments, then to sell shipping labels to other people. Over $900,000 in losses. Researchers at ACM CCS 2015 saw the same thing: stolen accounts become label shops.
>
> Today carriers find this after delivery, when the money is gone. But stopping parcels on suspicion has a cost too: the USPS Inspector General found about 2 million paid packages were mistakenly intercepted.
>
> So our goal is simple: decide at the moment of booking, and don't stop the wrong parcels."

---

### 2. A normal booking is allowed (0:35 to 1:00)

**Screen (optional, 5 s):** if you want to show the sign-in screen, sign out first and sign in live: "This is the analyst's console; the sign-in is a demo gate, not security."

**Screen:** Score tab. Pick the demo booking **"Tenured seller, normal booking"**. Click Score. Show the action (**allow**) and the latency number.

**Say:**
> "This is an established seller doing what it always does. FraudShield scores it before the label is issued and allows it in about 90 to 100 milliseconds on this laptop. Most bookings look like this, and most should never notice we exist."

---

### 3. Why we don't flag "new destination" (1:00 to 1:35)

**Screen:** Pick **"Real hard negative (first shipment to a new state)"**. Score it. It is not blocked. Then show the 94.9% slide.

**Say:**
> "Here's a real seller from the Olist dataset shipping to a new state for the first time. The obvious rule would flag that. But we measured it: established sellers send 94.9% of their bookings to postcodes they've never used before. 'New destination' is just normal business.
>
> So we look at the other side: who is this account paying for? A real business ships its own goods from its own places. A stolen account suddenly pays for parcels from senders and places it has never shipped for. That's the shop opening."

---

### 4. A stolen account opening a shop (1:35 to 2:25)

**Screen:** Pick **"Label-resale takeover"**. Score it. Open the Decision view. Point at, in order:
1. The action: **hold**
2. The per-question probabilities
3. The cost table (expected cost of each action)
4. The plain-language explanation

**Say:**
> "Now a takeover. This account is suddenly paying for new senders, at a cost far above its normal parcels, from a login device it has never used. FraudShield holds it: no label until someone checks.
>
> Instead of one mystery score, the model answers named questions. 'Is this account paying for senders outside its customer base?' 'Is this parcel cost-maximizing for the account?' Each one gets a probability.
>
> Then a cost rule picks the action. It compares the expected cost of all six options: allow, allow with a check at first scan, ask the owner to confirm, analyst review, hold, block. It picks the cheapest.
>
> One detail we like: stolen labels sell for a flat price, about $2 on underground markets, no matter how heavy or far the parcel goes. So buyers pick expensive parcels. Our cost rule knows that. For a R$45 parcel, allow stops at about 3.9% risk. For R$120, it stops at about 2.6%.
>
> And the explanation is written by a language model, but checked. If it mentions any number that isn't in the decision record, we reject it and show a fixed template instead."

**Optional, if time:** go back, take a booking with risk around 3%, change only the parcel cost from R$45 to R$120, and show the action change.

---

### 5. Ask a new question (2:25 to 3:10) (only if Laya is loaded)

**Screen:** In the Decision view, type in the "ask a new question" box:
`Does the consignee look like a reshipping drop?`
Show the probability that comes back.

**Say:**
> "An analyst can also ask a question we never trained a dedicated model for, in plain words, and get a probability back with no retraining. A normal gradient-boosted model can't do this: it can only answer the one question it has labels for.
>
> This is our fine-tuned Laya model, an open-weights decision model. It answers questions with probabilities, it doesn't chat."

**Do not say:** that Laya is more accurate than LightGBM, or any Laya accuracy number. Those results are still pending.

**If Laya is not loaded:** skip this segment and say in one line: "The app also lets an analyst type new questions to our Laya model. Its fine-tuning is running now, so we're not showing numbers for it today."

---

### 6. Analyst confirms, the system learns, and the gate says no (3:10 to 3:55)

**Screen:** In the takeover decision, click **confirm fraud**. Then open the **Learning** tab. Show the retrain result: catch rate up, false positives up, status **rejected by gate**.

**Say:**
> "When an analyst confirms fraud, that label feeds back into training. We tested this honestly: we held two fraud types out of training entirely. After retraining on new feedback, the catch rate on one of them, reshipping 'mule drops', went from 0.28 to 0.69. It learned a pattern it had never seen.
>
> But false positives on legitimate shippers went from 0.51% to 2.48%. So the system refused to deploy it. We wrote that safety rule down before running the test, so we couldn't bend it after seeing the results. We'd rather ship nothing than ship a model that hurts real customers."

---

### 7. Dashboard (3:55 to 4:10)

**Screen:** **Dashboard** tab. Point at holds, the trend by fraud type, net loss prevented, and the false-positive rate on legitimate shippers.

**Say:**
> "For the fraud team: what we stopped, what kind of fraud it was, money saved after friction costs, and next to it, how often we bothered a legitimate shipper. The cost assumptions are shown on screen, not hidden."

---

### 8. Tamper-evident audit (4:10 to 4:25)

**Screen:** **Audit** tab. Click verify (passes). Flip one byte in the log. Verify again (fails at that record).

**Say:**
> "Every decision is logged: model versions, the exact input the model saw, probabilities, action, explanation. Each record is chained to the one before it with a hash. Change one byte anywhere, and verification fails at exactly that record."

---

### 9. Results and honest limits (4:25 to 5:00)

**Screen:** results slide (both tiers), then limitations slide.

**Say:**
> "Our app's score reaches PR-AUC 0.79 using only real data columns, 0.82 with all columns, ROC-AUC 0.955. We report them separately because part of our data, the account and login layer, is synthetic. Rules alone score 0.04 to 0.05 on our hardest setting, so the fraud we injected isn't trivially easy to catch. For reshipping drops, a fraud type we never trained on, a rule from published research lifts the share caught from 18% to 53%. And weight fraud, which nobody can see on a booking form, goes from 6% to 40% because suspicious parcels are weighed at the first depot, and one failed scan puts the account's next parcels on the scale.
>
> Our limits: the account and login layer is synthetic, analyst feedback in the demo is simulated and labelled as such, and no retrained model has passed our gate yet.
>
> We built the part that says no, including to us. Thank you."

---

## Part 1b: Round 3 beats (fit them into the 5 minutes)

Each beat below replaces or extends a segment above, so the talk stays about 5 minutes. If time is short, keep 5a, 5b and 6, and say the red-team and monitor numbers in segment 9.

### 4b. (extends segment 4, point 4) Every sentence checked (0:15)

**Screen:** the takeover decision, "Why, in plain words": the claims list, a tick per claim with its reason, and the line "Agrees with the model's top reasons: X of Y". Optional: "Try slipping in a wrong number" shows a cross.

**Say:**
> "The explanation comes back as separate claims, and we check each one against the decision record. Moving a true number into the wrong sentence is the subtle lie: our old whole-text check missed all 33 we planted, the claim check caught all 33. And we show how far the explanation agrees with what the model actually leaned on. Here it's only one of three, and we show that instead of hiding it."

**Before you go on stage:** the Groq free tier allows 200,000 tokens a day for this model, and the measurement run used most of the day's budget on 2026-10-04. If explanations show "template", that's why; the claims and ticks still work.

### 5a. What would change this decision (replaces segment 5 if Laya is not shown; 0:20)

**Screen:** Live tab, stream running. Open a held or blocked row. In the Decision view click **"Show what would change it"**. About 1 in 4 stopped decisions has no answer within the budget (the takeover demo is one: "no change within 50 evaluations"); if so, open another held row. Seen in rehearsal: `blk-06` (`docs/demo_block_bookings.json`) shows "allow if declared weight were 0.85 kg (not 20.00 kg) and declared value were R$30.10".

**Say:**
> "An analyst can ask what would have changed this decision. We search small changes to things the booker controls, the declared weight, value, service, sender, and run each one through the real model and cost rule. 73% of stopped decisions have an answer with two changes or fewer. This is analyst-only: showing it to the booker would teach evasion, so every view is written to the audit log."

### 5b. "Was this you?" with a passkey tied to the booking (0:40)

**Screen:** same Decision view, the "Was this you? The owner's passkey" box. Enroll, Windows Hello. Confirm as owner, Windows Hello: **released**. Then **Tamper test**: **rejected**.

**Say:**
> "When we ask the owner to confirm, it's a real passkey, not a text message a fraudster with the stolen login can answer. The challenge the passkey signs is built from this booking's amount, destination and receiver. Watch: we take the same signature and change only the price by R$100. It no longer verifies. The owner approved this parcel, not any parcel. On this laptop the laptop plays the owner's phone, and the screen says so."

### 6. (extends segment 6) Which model is in use from now on (0:10 extra)

**Screen:** Learning tab after **Retrain**. Read the banner aloud: amber "Still using gbm-B2-F-v1 ... not deployed because: ..." or green "From now on, new bookings are scored by ...". Then open any decision: "LightGBM model that scored this booking".

**Say:**
> "After every retrain the system says in one sentence which model scores new bookings from now on, and if it kept the old one, why, in plain words. Every booking records the exact model version that scored it."

### 7b. A monitor that knows where it is blind (replaces segment 7 if the Dashboard is skipped; 0:20)

**Screen:** Live tab, the **"Label-free monitor"** card.

**Say:**
> "Fraud labels arrive weeks late. This card estimates how precise our stops are and how much fraud we let through, today, from calibrated scores alone. And we measured where it fails: on fraud types the model never learned, it under-counts missed fraud by 1.45 per week. Our drift alarm doesn't see those types either. So we say it on screen instead of trusting a green light."

### 8b. Red team: we attack our own model, live (0:35; replaces segment 8 if short on time)

**Screen:** the **Red team** tab. Target **"Stopped booking blk-06"**, click **Launch attack**. The tries scroll in one per row, each with the decision it got back (Hold, Hold, Scan-gated...). At try 24 the row turns red: **"The attacker got through"**. Point right: the measured table, then the red **"our own gate said no"** box.

Optional second run (8 s): target **"Demo: Label-resale account takeover"**: 50 tries, all Hold, green **"Our model held"**.

**Then harden it, live (0:20):** the right-hand panel **"Harden the model with these attacks"** has counted the attack and kept the evading bookings (3 for blk-06). Click **"Confirm original as fraud (analyst)"**: they become fraud labels. Click **"Retrain with N new evasion labels"** (about 5 s). The banner says which model scores new bookings from now on. In rehearsal (blk-06 and blk-08, 6 labels) it said **"Still using gbm-B2-F-v1 … no proven improvement"**; read out whatever it says. The offline measured run stays underneath for reference.

> "Every attack feeds the same learning loop as the analysts. Once an analyst confirms the original was fraud, the tricks become training data, and the same pre-registered gate decides. Here a handful of new examples isn't enough proof, so the old model stays. In our measured run, 113 of them would have bothered honest shippers, so the gate said no there too."

Any stopped decision also has a **"Red-team this decision"** link that opens this tab with that decision selected.

**Say:**
> "We play the fraudster against our own system. This attacker sees only what a fraudster would see, our decision, never the risk score. It may change two things a fraudster controls, and it gets 50 tries. Watch: hold, hold, scan check... and at try 24 it gets through, by declaring a 20-kilo parcel as 0.85 kilos at a lower value. Across the test window that happens to 18% of the fraud we stop. We publish that number. Then we retrained on those tricks, and our own safety gate rejected the new model, because it would have stopped more honest shippers."

**If asked "how often does it fail?":** the takeover booking held for all 50 tries; on the test window 63% of stopped fraud gets some softer action, but most of those are still stopped (hold instead of block) or still weighed at the depot.

### 9. (extends segment 9) We attacked our own model (one sentence, if 8b was skipped)

**Say:**
> "We also attacked our own model. An attacker who only sees our decision and changes at most two fields gets 18% of the fraud we stop through as a plain allow, and 63% to some softer action, mostly by reusing a sender the account already knows. We retrained on those evasions and our own gate rejected that model too, because it bothered honest shippers."

---

## Part 2: How each feature works

### 2.1 The flow of one booking

```
Booking arrives (before the label is issued)
  1. Account profile        what this account normally does, as of booking time
  2. Rules                  hard signals and minimum actions
  3. Change features        new senders under this payer, cost vs the account's median,
                            new login device, sender variety jump, etc.
  4. LightGBM score         calibrated backup score
  5. Serializer             booking turned into a fixed-order text (~300 tokens)
  6. Laya                   answers 4 named questions in one batched call
  7. Calibration            raw answers turned into honest probabilities
  8. Cost rule              cheapest expected-cost action out of 6
  9. Audit log              hash-chained record written before the answer goes out
 Later: LLM explanation (checked by the validator), analyst review, feedback, retraining
```

If Laya is slow or unavailable, the system falls back to the calibrated LightGBM score with stricter thresholds and marks the decision **degraded**. A high-risk booking never gets a plain allow in degraded mode. If the audit write fails, no decision goes out.

### 2.2 Sender-side scoring ("who is this account paying for?")

- **What it catches:** an account that starts paying for parcels from senders and places it never shipped for. This is the label-resale pattern from the DOJ Wilson/Harrod case and Hao et al. (ACM CCS 2015).
- **Why not destination:** in real Olist data, established sellers send 94.9% of bookings to never-seen postcodes, so destination novelty is noise.
- **Signals used:** new senders under the payer in recent bookings, sender variety in 7 days vs 90 days, origin novelty vs the account's own history, how many senders a payer covers in 30 days.

### 2.3 Named questions (Laya)

Laya is an open-weights decision model that we fine-tune. Instead of one fraud score, it answers four yes/no questions about each booking:

| Question id | Meaning |
|---|---|
| `misuse` | Is this account being misused? |
| `foreign_senders` | Is the account paying for senders outside its customer base? |
| `payoff_max` | Is this parcel cost-maximizing for the account? |
| `drop_consignee` | Does the receiver look like a reshipping drop? |

An analyst can also type a new question. Laya answers it with no retraining, but these new-question answers are **not calibrated** (the API marks them `calibrated: false`).

**Status:** fine-tuning runs on Kaggle. Until it is imported, the app runs in degraded mode on LightGBM. Make no accuracy claims for Laya.

### 2.4 The cost rule and the six actions

The action is chosen by comparing the expected cost of every option, using the calibrated probability and a cost table (friction for a legitimate shipper vs loss if it is fraud). Picking the cheapest expected cost is the standard approach when costs and probabilities are known (Elkan 2001), so we use a rule here, not reinforcement learning.

| Action | What happens |
|---|---|
| `allow` | Normal booking |
| `allow_scan_gated` | Label issued, parcel weighed and checked at first scan before it leaves the depot |
| `owner_confirm` | One-tap message to the account's contact on file (only if that contact is over 30 days old) |
| `review` | Analyst checks it (15-minute target) |
| `hold` | No label until verified |
| `block` | Needs a hard signal plus analyst confirmation within 24 h, otherwise drops to hold |

**Price awareness:** stolen labels sell at a flat price (LabelsBank, about $2 per label), so fraudsters pick expensive parcels. A higher parcel cost raises the loss if it is fraud, which lowers the risk level where friction starts: allow stops at about 3.9% risk for a R$45 parcel and about 2.6% for R$120.

**Note:** all cost numbers except freight are assumptions, and Olist's freight value is paid by the buyer, so we use it as a stand-in for carrier cost.

### 2.5 Protecting legitimate shippers

- `allow_scan_gated` absorbs most doubt at a cost of about R$1 to a legitimate shipper, instead of stopping the parcel.
- We test on real hard cases: 624 test bookings where an established seller ships to a new state for the first time, and 537 new sellers.
- Blocks are capped (one automatic block per account per 24 h) and need analyst confirmation.
- The false-positive rate on legitimate shippers is reported next to every headline metric.

### 2.6 Checked GenAI explanations

- For non-allow decisions, a language model (via Groq) writes a plain-language explanation.
- A validator checks it: every number, postcode and id in the text must appear in the decision record, and the text must name the action and at least 2 of the top 3 reasons.
- If it fails, the fixed template is shown instead. Messages sent to account owners are never written by the LLM.

### 2.7 Continuous learning with a pre-registered gate

- Analyst confirmations become training labels. In the demo, simulated feedback is clearly flagged as simulated.
- Retraining builds a candidate model. It is compared with the active model on data neither was trained on.
- The candidate is deployed only if it passes a gate written down before testing (`docs/LEARNING_GATE.md`): no worse on cost, PR-AUC, calibration and false positives on hard cases, and better on at least one thing.
- **Result so far:** the candidate learned unseen patterns (catch rate on held-out types 0.53 to 0.78; mule drops 0.28 to 0.69), but false positives on legitimate hard cases rose from 0.51% to 2.48%, so the gate rejected it. We did not loosen the gate.
- Every retrain, deploy and rollback is written to the audit log.

### 2.8 Tamper-evident audit log

- Append-only log. Each record holds the hash of the previous record and its own hash.
- A decision record holds: model versions, the exact input text and its hash, raw and calibrated probabilities, expected cost per action, the chosen action, flags (degraded, explored) and latency.
- Verify with `GET /audit/verify` or the Audit tab. Flipping one byte makes verification fail at that record.
- Privacy: hashed ids, 3-digit postcode prefixes only, no names.

### 2.9 Honest testing

- **Data:** Olist (real Brazilian e-commerce, 100,010 bookings, 3,095 sellers). Fraud is injected and every injected row is labelled with its scenario, seed and settings. The account, login and billing layer is synthetic.
- **Held-out fraud types:** reshipping mule drops (T3) and test-then-burst (T5) are never used in training.
- **"Too easy?" checks:** rules alone reach PR-AUC 0.042 (real columns) and 0.050 (all) on the hardest camouflage level. Shuffled labels give 0.0108, about the fraud rate, as expected.
- **Results (artifacts/results_system.md):** LightGBM PR-AUC 0.74 (real columns), 0.78 (all). App score with the drop-address rule: 0.79 (real), 0.82 (all), ROC-AUC 0.955. Share caught, model only to full system: reshipping drops (held out) 18% to 53%, weight fraud 6% to 40% (depot scan + follow-up), all fraud 45% to 53%. Honest bookings stopped 0.4%, honest parcels weighed at the depot 1.8%.
- **Depot scan (T6):** `POST /decisions/{id}/first-scan` takes the measured weight; heavier than declared beyond 0.5 lb or 3% (the UPS / FedEx tolerance) fails the scan, and the account's later parcels get a first-scan check with the reason "A parcel from this account failed a depot weight check". The decision page has a "Depot scan" box; the live stream simulates scans from the dataset's true weights.
- **Depot weighing dial (T6):** the Dashboard has a "Depot weighing" panel. The carrier picks how many parcels to weigh;
  each level shows its measured result, and switching applies to new bookings at once and is written to the audit log
  (`GET`/`POST /first-scan/dial`). The score behind it is how far below the account's usual size and weight a parcel is
  declared (accounts with 5+ earlier bookings).

  | Dial level | Weight fraud caught by the scale | Honest parcels weighed | Weighings per 1,000 bookings |
  |---|---|---|---|
  | Standard (default) | 36% | 1.8% | 19 |
  | Target 3% | 54% | 4.0% | 42 |
  | Target 5% | 73% | 6.2% | 64 |
  | Target 10% | 94% | 11.4% | 116 |

  Mean over 10 seeds on the test window; the thresholds were chosen on separate validation data (the calibration split), so
  the test numbers are honest. "Caught by the scale" counts only parcels the depot weighs; with the model's own flags on
  top, the standard level reaches 40%. Source: `artifacts/first_scan_dial.json` (`scripts/fit_first_scan.py`). Chart:
  `docs/figures/depot_dial.png`.


### 2.10 Desktop app

- One process, a native window (pywebview), no browser needed. Start with `FraudShield.bat`.
- Tabs: Score, Decision, Queue, Dashboard, Learning, Audit.
- About 90 ms warm latency per booking measured on an RTX 4050 laptop GPU.
- Training never runs on the laptop. It runs on Kaggle.

---

### 2.11 What would change this decision (analyst-only counterfactuals)

- `GET /decisions/{id}/counterfactual`, the "Show what would change it" panel on the Decision view. Code: `fraudshield/redteam/search.py`.
- Searches changes to fields the booker controls: declared value, declared weight, parcel size, service, a sender the account already used, booking time. Carrier cost follows the declared weight at the lane freight slope (R$2.727/kg, fitted on the train window); the booker never sets it.
- Every hypothetical booking goes through the real path (features, LightGBM, Platt, rules floor, cost rule) with no stored decision, no feature-store write. Checked: same action and probability as `/score` on 30 real bookings.
- Budget 50 scored bookings, at most 2 changed fields; fewest fields first, then the smallest change.
- Measured (test window, seed 0, 60 of 265 stopped decisions): 73.3% have a softer action within 2 changes (fraud 70.2%, legit 84.6%), 21.7% reach plain allow; median latency 1118 ms.
- Analyst-only: every view writes a `counterfactual_view` audit record first; if that write fails, nothing is shown. There is no role check on the endpoint yet (docs and audit only).
- Sources: Khouna et al., "Optimal Counterfactual Search in Tree Ensembles", arXiv 2605.06561 (May 2026).

### 2.12 Red team: we attack our own model

- `scripts/redteam.py` -> `artifacts/results_redteam.md`. Same search as 2.11, but the attacker sees only the returned action (not the probability). At most 50 queries and 2 changed fields per booking.
- Population: fraud of the trained types that the system stops, test window, seeds 0-7, at most 40 per seed (n = 320), run once. Fraud is injected (synthetic) on real Olist histories.
- Flip to plain allow 18.1 ± 7.6%, to any softer action 63.1 ± 7.3%. By type (allow / softer): T1 18.8% / 74.4%, T2 0.0% / 13.2%, T4 21.9% / 57.5%, T7 26.9% / 84.5%. Never-trained T3/T5 (n = 80): 11.3% / 45.0%. "Softer" includes allow_scan_gated, where the parcel is still weighed at the depot.
- Most evasions reuse a sender the account already used (159 of the 238 changed fields). Under-declaring weight also lowers the carrier cost, so it shrinks the fraudster's payoff.
- Hardening: 113 evasions made from train-window fraud only were added as labels and went through the unchanged retrain and gate. **Rejected**: hard-negative false positives would rise from 0.27% to 1.68% (limit +0.5 points).
- Source: Fok et al., "Foe for Fraud: Transferable Adversarial Attacks in Credit Card Fraud Detection", arXiv 2508.14699 (Aug 2025).

### 2.13 Owner passkey "was this you?"

- Full design: `docs/PASSKEY.md`. Code: `fraudshield/identity/webauthn.py`, `fraudshield/api/passkey.py`.
- The passkey signs `sha256(nonce || booking_id, account_id, carrier_cost, declared_value, dest_zip3, consignee_id)`. The server recomputes it from the booking it holds, so a signature for one booking cannot approve another amount or address. Nonce single use, 120 s expiry.
- A confirmed owner_confirm becomes allow; a confirmed hold or review becomes allow_scan_gated (still weighed at the depot); a block cannot be confirmed by the owner. Success and failure both go to the audit log.
- Measured: signature check 0.176 ms p50 warm (n = 1000), verify endpoint 2.61 ms p50 server side (n = 300). Real Windows Hello prompt time not measured.
- Honest limits: the laptop plays the owner's phone (labelled in UI and audit); enrollment in the demo happens right before confirming, in production it happens at onboarding and changing it is itself a risky event; attestation `none` (we check it's the same passkey, not which device model); the OS prompt doesn't show the amount.
- Sources: FIDO Alliance Passkey Index, Oct 2025; W3C Secure Payment Confirmation (same idea: the signature is bound to the transaction).

### 2.14 Label-free monitor

- `GET /monitor/estimate`, the "Label-free monitor" card on the Live view. Code: `fraudshield/monitor/cbpe.py`. Results: `artifacts/results_monitor.md`.
- From calibrated probabilities p and the actions, without labels: estimated precision of stops = mean p over stopped bookings, estimated missed fraud = sum p over bookings let through, estimated recall. Weekly periods (chosen on the calibration window).
- Calibration window (sanity): precision 0.764 estimated vs 0.745 real, recall 0.810 vs 0.797.
- Test window, all 10 seeds: with held-out T3/T5 present it under-counts missed fraud (52.8 estimated vs 67.4 real; recall 0.803 estimated vs 0.771 real). Per week the paired gap is -1.45 ± 0.43 missed frauds. The 23.5 held-out frauds let through per window add up to only 0.23 expected frauds.
- Also blind: T6 (weight fraud, mean p 0.036), a trained type the booking score can't see. And the score-drift signal (PSI) is the same with or without T3/T5 (difference at most 0.002), so drift alarms don't see new fraud either.
- Sources: Kivimäki et al., arXiv 2505.05295 (ACML 2025); Solozobov, arXiv 2604.15740 (Apr 2026).

### 2.15 Model handover and learning attempt 5

- After every retrain or rollback, `POST /learning/retrain`, `/learning/rollback` and `GET /learning/status` return a `handover` verdict, also written to the audit log. The Learning view shows it as a banner; the Decision view shows the LightGBM version that scored the booking, the Live view the model in use.
- Banners: green "From now on, new bookings are scored by {v} (since HH:MM). Previous model {prev} kept for rollback."; amber "Still using {v} (since HH:MM). Candidate {c} learned the new pattern (+x points recall on new patterns) but was not deployed because: ..." with one plain-language line per failed check; blue after a rollback.
- Two bugs fixed on the way: decisions logged the generic name "gbm" instead of the version, and a running live stream logged the old version name after a deploy.
- Attempt 5 (pre-registered in `docs/LEARNING_GATE.md` before it ran, chosen on training data only, run once): R3 regularisation plus "device age may only interact with channel". **Rejected**, on cost only: cost improvement CI [-9.61, +21.41] BRL per 1k, needs >= -5.98. It passed the other checks: hard-negative FPR 0.32% -> 0.64% (inside +0.5 points), PR-AUC 0.807 -> 0.830, new-pattern recall 0.535 -> 0.577 (CI [+0.016, +0.071]). Caveat: v1 was retrained after the R0-R3 runs, so attempt 5 compares with today's v1, not with the R3 row. No 6th configuration will be tried on this eval set.
- There is no reinforcement learning here: it is gated retraining plus 5% logged exploration. Don't call it RL.

### 2.16 Every sentence checked (claim-by-claim explanations)

- The LLM explainer (Groq `openai/gpt-oss-120b`) is asked for strict JSON (`json_schema`, constrained decoding): `{action, claims: [{text, reason_code, cited_fields, numbers}]}`. If Groq rejects it: `json_object`, then the old prose path. The template produces the same claim structure, so the page works with no LLM. Code: `fraudshield/explain/claims.py`; endpoint `GET /decisions/{id}/claims`.
- `validate_claims()` checks each claim: its reason code must be one of this decision's reasons, every number must be in the record and come from the claim's own cited fields, cited fields must exist, no unknown ids. Any failing claim means the template is shown; rejected claims are kept and shown crossed out.
- Agreement with the model: LightGBM `pred_contrib` (TreeSHAP) summed per reason code (`fraudshield/explain/attribution.py`); hit@3 = share of the model's top reason codes the explanation cites first.
- Measured (`artifacts/results_explanations.md`; seed-0 test window, 60 of 263 stopped decisions, one sample):
  - Planted lies: a true number moved to the wrong claim is rejected 33/33 by the claim check, 0/33 by the old prose validator. Fabricated numbers, invented reasons and invented fields: 57/57 each (by construction).
  - LLM, first try, the 31 decisions where Groq ran (29 not run: daily token limit): rejected 1/31 on both paths; claims grounded 62/62; strict `json_schema` used for all 31.
  - hit@3 against the model's top reasons: template 0.244, prose 0.215, claims 0.204. The ceiling is 0.289, because on average only 0.867 of the model's top 3 codes fired as rules, and reason codes describe only 57.7% of the model's positive push.
- Sources: Groq Structured Outputs docs; arXiv 2512.00163 (LLM self-explanations disagree with SHAP); arXiv 2605.26770 (LLM-written XAI narratives raised confidence without improving accuracy), which is why the checker is fixed code, not an LLM judge.

---

## Part 3: Q&A backup

| Likely question | Our answer |
|---|---|
| "Isn't this just XGBoost plus a chatbot?" | "No. The model answers named questions with probabilities, a cost rule picks one of six actions, and the explanation is checked against the decision record before anyone sees it." |
| "Your account data is synthetic. Why trust it?" | "The pattern comes from a prosecuted DOJ case and peer-reviewed research, the 94.9% is real data, and we report real-column results separately (PR-AUC 0.79 with the app's score, 0.74 for LightGBM alone)." |
| "Does Laya beat LightGBM?" | "We don't know yet. Fine-tuning is running and we won't claim it until the numbers are in." |
| "So continuous learning doesn't work?" | "It learned the unseen pattern, 0.28 to 0.69. It would also have hurt real customers, and a rule we wrote in advance stopped it." |
| "Doesn't the price rule punish big customers?" | "It moves the allow line from about 3.9% to about 2.6% and adds a first-scan check, not a block." |
| "What about a legit 3PL that ships for many senders?" | "We built that in as a legitimate hard case in testing, and a verified pre-announced change flow switches off the novelty signals for 30 days." |
| "Why not reinforcement learning?" | "With known costs and calibrated probabilities, picking the cheapest expected cost is already optimal. RL would add risk without adding value." |
| "What's the latency?" | "About 20 to 30 ms per booking for the decision (p99 under 100 ms), measured on a laptop CPU; the live stream keeps up with 20 bookings per second and peaks near 38. Asking fine-tuned Laya a question takes about 70 ms on the laptop GPU." |
| "Can you catch weight fraud at booking?" | "Mostly not, and we say so: a booking form can't be weighed. Parcels declared far smaller than the account's norm are weighed at the first depot scan, and one failed scan puts the account's next parcels on the scale. That catches 40% of weight fraud while weighing 1.8% of honest parcels. If the carrier wants more, the depot weighing dial on the dashboard goes up to 94% caught for 11.4% of honest parcels weighed; it's their trade-off, and we show the measured cost of each level." |
| "What would a carrier need to change?" | "A 'pending' label state at booking, read access to payment and login events, and a first-scan check record." |
| "Doesn't 'what would change this decision' teach fraudsters how to evade?" | "Only if a booker sees it, so they never do. It's analyst-only, and every view is written to the audit log before it's shown. An attacker can search the same way by trial and error, and our red team measured exactly that: it's why we publish the 18% and 63% numbers instead of hiding them." |
| "So 63% of your caught fraud can evade you?" | "63% can get some softer action with up to two changes and 50 tries, 18% get a plain allow. Most of them do it by claiming a sender the account already uses, which a carrier can check against where the parcel is actually picked up; we haven't tested that check, so we don't claim it. And retraining on those evasions would have hurt honest shippers, so our gate refused it." |
| "Is the passkey real, or a mock-up?" | "It's a real WebAuthn ceremony with a real signature check on our server. The only simulated part is the device: this laptop plays the owner's phone, and the screen and the audit record say so." |
| "Couldn't the fraudster just enroll their own passkey with the stolen login?" | "In the demo we enroll on stage for speed. In production enrollment happens at onboarding, and changing the passkey is itself a risky account event that gets its own checks, like our rule that an owner contact younger than 30 days doesn't count. The binding is what the demo proves: a signature for one parcel can't approve a different price or address." |
| "How do you know your model still works before the fraud labels arrive?" | "We estimate precision and missed fraud from calibrated scores, and we measured where that breaks: it under-counts missed fraud from types the model never learned by 1.45 per week, and the drift alarm doesn't see them. That's why the dashboard says so, and why new types come in through analyst feedback and the gate." |
| "Did the retrained model go live?" | "None of our recorded real-data runs has passed the gate (`docs/LEARNING_GATE.md`), and the Learning tab says which model is in use, in one sentence, after every retrain, including the one we just did on stage. Our latest attempt fixed the false-positive problem and improved every point estimate, but the cost check's confidence interval was still too wide, so v1 keeps scoring new bookings." |
| "Is that 0.28 to 0.69 number from your current model?" | "It's from the run against the earlier version of v1. v1 has been retrained since; on today's v1 the latest attempt raised new-pattern recall from 0.535 to 0.577. The pattern held: learning the new fraud, blocked for safety." |
| "Your explanations agree with the model only about 20% of the time. Aren't they misleading?" | "They're honest about what they are. An explanation may only cite reasons whose rule actually fired for this booking, and those rules describe about 58% of what pushes the model's score. So the ceiling is 0.29 and we're at 0.20 to 0.24. We show the agreement line on every decision instead of letting a fluent paragraph imply more." |
| "Why not have another LLM judge the explanation?" | "Research this year found LLM-written explanations raise confidence without improving accuracy, and make LLM judges worse at spotting bad predictions. Our checker is plain code: every number must come from the record and from the claim's own fields. That's how it catches a true number moved into the wrong sentence, 33 of 33, which the whole-text check missed every time." |
