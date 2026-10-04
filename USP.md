# FraudShield: USPs for the presentation round

Judges will hear "real-time AI", "explainable AI" and "reduces fraud losses" from every team. These three USPs are things a typical "XGBoost + LLM explainer" team could not truthfully say. Every number below comes from our own results or a cited source. Do not round them up.

## How we picked them

We drafted 8 candidates and ran each through three tests:

1. **Competitor test:** could an XGBoost + LLM team say this sentence truthfully? If yes, kill it.
2. **Proof test:** does it point to a real number or a live demo moment? If not, kill it.
3. **Judge test:** can we answer the hardest skeptical question without claiming something unproven (Laya accuracy, Laya beating LightGBM, real carrier deployment, real customer data)? If not, kill it.

| # | Candidate | Competitor | Proof | Judge | Result |
|---|---|---|---|---|---|
| 1 | We score who the account pays for, not where it ships | Pass | 94.9% on real Olist data, DOJ case, Hao et al. | Pass | **Keep** |
| 2 | Expensive parcels get friction sooner, because fraud labels are flat-priced | Pass | 3.9% vs 2.6% thresholds, ~$2/label | Pass | **Keep** |
| 3 | Our safety gate blocked our own retrained model | Pass | 0.28 to 0.69 catch, 0.51% to 2.48% FP | Pass | **Keep** |
| 4 | Analysts can ask a new question and get a probability | Pass | Live demo only | Weak | Parked |
| 5 | "Allow, but check at first scan" action | Fail | | | Killed |
| 6 | Two fraud types held out of training | Fail | | | Killed |
| 7 | Hash-chained audit log, number-checked explanations | Fail | | | Killed |
| 8 | Runs on a laptop at ~90-100 ms | Fail | | | Killed |
| 9 | We attack our own model and publish how often we lose (red team) | Pass | 18.1% / 63.1% flip, hardened model rejected by our gate | Pass | **Keep** (USP 5) |
| 10 | Our monitor says where it is blind | Pass | -1.45 ± 0.43 missed frauds per week on unseen types | Pass | **Keep** (merged into USP 5) |
| 11 | The owner approves this parcel, not any parcel (booking-bound passkey) | Pass | Live: tamper test fails on stage | Pass | **Keep** (USP 6) |
| 12 | "What would change this decision" for analysts | Fail | | | Killed (demo only) |
| 13 | One-sentence model handover after every retrain | Fail | | | Killed (supports USP 3) |
| 14 | Every sentence of the explanation checked, claim by claim | Pass | Moved-number lie caught 33/33 vs 0/33 | Weak | Parked (Q&A) |

Rows 9-13 were added for round 3 (2026-10-04) and scored with the same three tests.

- **#12 killed:** any XGBoost team can add a counterfactual library. It stays in the demo (segment 5a), because it is useful and shows the price-aware cost rule.
- **#13 killed:** any team can show which model version is live. It makes USP 3 easier to see on screen.
- **#14 parked:** the claim check is ours and measured, but the obvious judge follow-up, "do the explanations match what the model used?", has a weak answer: hit@3 0.20 to 0.24 against a ceiling of 0.29 (`artifacts/results_explanations.md`). Use it in Q&A, not as a headline.
- **#4 update:** the Laya results are in, and Laya lost at deciding (PR-AUC 0.356 vs 0.777), so #4 stays parked: the hardest judge question now has a clear "no".

### Why the others were cut

- **#8, laptop at ~90-100 ms:** any XGBoost team can truthfully say they score in under 100 ms on a laptop.
- **#7, tamper-evident audit:** the brief asks for auditability, so every team will claim an audit log. The number-check on GenAI explanations is unusual, though. Save it for Q&A.
- **#6, held-out fraud types:** a competitor could do the same, and judges rarely remember methodology. Its numbers support USP 3 instead.
- **#5, check at first scan:** an XGBoost team could add a fifth action label. Good product design, but it doesn't depend on our model. We show it inside the USP 2 demo.
- **#4, ask a new question (parked, not killed):** this is the most XGBoost-proof idea we have. But the hardest judge question is "is it more accurate than your LightGBM?", and we can't answer that until the Laya fine-tune results are in. Promote it to a USP once they land.

## The 3 USPs

### USP 1: We catch the shipping shop, not the parcel

- **Headline:** Stolen accounts open shipping shops. We catch the shop opening.
- **Why it is ours:** Typical teams flag "new destination". We measured that this is noise, so we score which senders an account starts paying for.
- **Proof:** On real Olist data, established sellers send 94.9% of bookings to postcodes they never used before. The pattern comes from the DOJ Wilson/Harrod case (stolen UPS login used to sell labels, >$900K loss) and Hao et al., ACM CCS 2015 ("Drops for Stuff").
- **Demo moment:** Show an account's history where every booking comes from its own sender addresses, with brand-new destinations scoring as normal. Then book a parcel the account pays for on behalf of a stranger sender. The named question "is this account paying for senders outside its customer base?" jumps, and the action changes.
- **Judge challenge:** "Your account data is synthetic. How do you know real stolen accounts look like this?"
- **Our answer:** "The pattern comes from a prosecuted case and peer-reviewed research, the 94.9% is real data, and we report real-column results separately (PR-AUC 0.79 with the app's score, 0.74 for LightGBM alone) so nothing synthetic is hidden."

### USP 2: We price risk the way fraudsters do

- **Headline:** Fraud labels sell at a flat price, so we check expensive parcels sooner.
- **Why it is ours:** Typical teams use one risk threshold for every parcel. Ours moves with the parcel's cost, because label buyers pick expensive parcels.
- **Proof:** LabelsBank sells labels at about $2 regardless of weight or distance. Now in an indictment (S.D. Florida, announced 24 Sep 2026; alleged): LabelsBank.com sold more than 5.1 million counterfeit labels at a flat $2 "regardless of weight, size or destination" to more than 5,000 customers, a $126 million loss to USPS (https://postalemployeenetwork.com/news/2026/09/26/u-s-postal-inspectors-shut-down-website-selling-millions-of-counterfeit-postage-labels/; DOJ release: https://www.justice.gov/usao-sdfl/pr/pakistani-national-charged-defrauding-us-postal-service-over-100-million-selling). Allow stops at about 3.9% risk for a R$45 parcel and about 2.6% for R$120.
- **Demo moment:** Take one booking with risk around 3%. At R$45 it's allowed. Change only the parcel to R$120 and it moves to "allow, but check at first scan". Same customer, same risk, different action.
- **Judge challenge:** "Doesn't that punish your best customers, the ones shipping expensive parcels?"
- **Our answer:** "The shift is about 1.3 points, and it adds friction rather than a block, because USPS OIG found about 2 million paid packages wrongly intercepted."
- **Source for the 2 million:** USPS OIG audit "Efforts to Mitigate Counterfeit Postage" (Sep 2026): about 2 million legitimate packages with valid postage intercepted between Dec 2024 and Feb 2026; the same audit estimates $3.1 billion lost to counterfeit labels (Mar 2024 to Feb 2026), with counterfeit-label parcels delivered 97% of the time. https://www.uspsoig.gov/reports/audit-reports/efforts-mitigate-counterfeit-postage (the OIG page blocks automated fetches; figures cross-checked in https://www.valueaddedresource.net/usps-counterfeit-label-oig-report/).

> Before presenting: check that the demo booking really lands between 2.6% and 3.9% risk, or the action won't flip on stage.

### USP 3: Our system refused to deploy our own model

- **Headline:** Our safety gate blocked our own retrained model from going live.
- **Why it is ours:** Typical teams say "continuous learning" and show a better accuracy number. We can show a pass/fail rule we wrote before testing, and the time it said no.
- **Proof:** Retraining raised the catch rate on held-out mule drops from 0.28 to 0.69. False positives on legitimate shippers rose from 0.51% to 2.48%, so the gate blocked deployment. The block is in the hash-chained audit log.
- **Demo moment:** Open the retrain screen: catch rate up, false positives up, status BLOCKED. Then open the audit log entry for that decision and flip one byte to show the tamper alarm.
- **Judge challenge:** "So your continuous learning doesn't actually work?"
- **Our answer:** "It learned the unseen pattern, 0.28 to 0.69. It also would have hurt real customers, and a rule we wrote in advance caught that before any shipper saw it."

### USP 4: We say what a booking form can't see, and move the check to where it can

- **Headline:** Weight fraud is invisible at booking, so we catch it on the depot scale, and the carrier sets how hard to look.
- **Why it is ours:** Typical teams report one accuracy number and stay quiet about the fraud type their model can't see. Ours admits it: the model alone catches 6% of weight fraud, because a form can't be weighed. So we send suspicious parcels to the first depot scale. One failed weighing puts that account's next parcels on the scale too. A dial on the dashboard shows the measured cost of every level.
- **Proof** (10 seeds, test window, thresholds chosen on separate validation data; `artifacts/first_scan_dial.json`):

  | Dial level | Weight fraud caught by the scale | Honest parcels weighed |
  |---|---|---|
  | Standard (default) | 36% | 1.8% |
  | Target 3% | 54% | 4.0% |
  | Target 5% | 73% | 6.2% |
  | Target 10% | 94% | 11.4% |

  The mismatch tolerance is the one UPS and FedEx publish (0.5 lb or 3%).
- **Demo moment:** Open the Dashboard and click "Target 10%". The level changes live for new bookings and the live stream. Then open the Audit tab and show the change recorded with a hash. Slide: `docs/figures/depot_dial.png`.
- **Judge challenge:** "Weighing 11% of honest parcels is a lot. Is that worth it?"
- **Our answer:** "On freight alone, no, and we don't pretend otherwise. The standard level pays for itself: about R$11 of underpaid freight caught per 1,000 bookings, for 19 weighings at about R$1 each (the R$1 is our assumption). Higher levels are for a carrier that values deterrence, or whose hubs already weigh every parcel in-line, so an extra check costs almost nothing. That's the carrier's call, and the dial shows them the real number for each level."

### USP 5: We publish where we fail

- **Headline:** We attacked our own model, and our monitor tells you where it's blind.
- **Why it is ours:** Typical teams report how often they are right. We measure how often a fraudster beats us, and where our own health check can't see.
- **Proof:**
  - Red team (`artifacts/results_redteam.md`; test window, run once, n = 320 stopped fraud bookings of trained types, seeds 0-7, injected fraud): an attacker who only sees our decision and changes at most 2 fields in 50 tries gets 18.1 ± 7.6% to plain allow and 63.1 ± 7.3% to some softer action. Retraining on evasions made from train-window fraud only was **rejected** by our unchanged gate (hard-negative false positives 0.27% to 1.68%).
  - Monitor (`artifacts/results_monitor.md`; test window, 10 seeds): estimating from calibrated scores alone, it under-counts missed fraud from never-trained types by 1.45 ± 0.43 per week, and the score-drift signal does not change when they are present.
- **Demo moment:** Live tab, the "Label-free monitor" card with its blind-spot line. Then one sentence with the red-team numbers.
- **Judge challenge:** "So 63% of the fraud you catch can get past you?"
- **Our answer:** "63% can get something softer with two changes and 50 tries; 18% get a plain allow. Most do it by claiming a sender the account already uses. We publish it because a fraudster will find it anyway, and our gate stopped us from 'fixing' it in a way that hurts honest shippers."

### USP 6: The owner approves this parcel, not any parcel

- **Headline:** Our "was this you?" is a passkey signature tied to the parcel's price and address.
- **Why it is ours:** Typical teams have no step between allow and block. Ours asks the account owner, with a passkey the login thief doesn't have, and the signature covers this booking's price, destination and receiver.
- **Proof:** Live demo: confirm with Windows Hello and the booking is released; change only the price by R$100 and the same signature fails. Signature check 0.176 ms p50 (n = 1000, `artifacts/results_passkey.md`). Same binding idea as W3C Secure Payment Confirmation; passkey uptake from the FIDO Alliance Passkey Index (Oct 2025).
- **Demo moment:** Decision view, "Confirm as owner", then "Tamper test": rejected.
- **Judge challenge:** "Your laptop is playing the owner. Couldn't a fraudster enroll their own passkey?"
- **Our answer:** "The device is simulated and labelled so. Enrollment belongs at onboarding and changing it is a risky event of its own; what we prove on stage is the binding: one signature, one parcel."

## 20-second spoken version

> "Stolen shipping accounts aren't used to ship one parcel. They're used to open a shipping shop. So we don't flag new destinations, which turned out to be 94.9% of normal bookings. We flag an account paying for senders it never shipped for. Fraudsters buy labels at a flat price, so expensive parcels get checked sooner. And when our own retrained model would have hurt real shippers, our system refused to deploy it."

## Closing line for the last slide

> "We built the part that says no, including to us."

## Do not claim (yet)

- That fine-tuned Laya beats LightGBM. It doesn't: PR-AUC 0.356 vs 0.777 (`artifacts/results_laya_v2.md`), so LightGBM decides
- That the passkey prompt was measured with real users, or any new owner-confirm fraud rate (none was measured)
- That the red-team evasions would work on a real carrier (they are against our model on injected fraud), or that a sender-vs-pickup check would stop them (untested)
- That the LabelsBank defendant is guilty: the case is an indictment (alleged)
- Real carrier deployment or real customer data
- Rounded-up versions of any number above
