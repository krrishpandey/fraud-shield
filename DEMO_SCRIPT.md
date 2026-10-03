# FraudShield: presentation script

This is the script for presenting FraudShield. **Part 1** is what we say, segment by segment, with what is on screen. **Part 2** explains how each feature works under the hood, so whoever is presenting can answer follow-up questions. **Part 3** is Q&A backup.

Target length: about 5 minutes. All numbers come from our own runs (see `PROJECT_HISTORY.md`) or a cited source. Do not round them up.

---

## Before you start (checklist)

- [ ] Open the app with `FraudShield.bat` and confirm the window opens.
- [ ] Look at the status bar. If it says **degraded**, the fine-tuned Laya model is not loaded and the app is using the LightGBM backup score. Say so if asked. Do not hide it.
- [ ] Skip the "ask a new question" segment unless Laya is loaded (it needs Laya to answer).
- [ ] Check that `.env` has the Groq key. Without it, explanations fall back to a fixed template (still fine, just less fluent).
- [ ] Have the backup screen recording ready in case the app fails on stage.
- [ ] For the price demo in segment 4, make sure the booking you use scores between 2.6% and 3.9% risk, or the action will not flip.

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

### 2.10 Desktop app

- One process, a native window (pywebview), no browser needed. Start with `FraudShield.bat`.
- Tabs: Score, Decision, Queue, Dashboard, Learning, Audit.
- About 90 ms warm latency per booking measured on an RTX 4050 laptop GPU.
- Training never runs on the laptop. It runs on Kaggle.

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
| "Can you catch weight fraud at booking?" | "Mostly not, and we say so: a booking form can't be weighed. Parcels declared far smaller than the account's norm are weighed at the first depot scan, and one failed scan puts the account's next parcels on the scale. That catches 40% of weight fraud while weighing 1.8% of honest parcels." |
| "What would a carrier need to change?" | "A 'pending' label state at booking, read access to payment and login events, and a first-scan check record." |
