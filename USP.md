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
- **Our answer:** "The pattern comes from a prosecuted case and peer-reviewed research, the 94.9% is real data, and we report real-column results separately (LightGBM PR-AUC 0.73) so nothing synthetic is hidden."

### USP 2: We price risk the way fraudsters do

- **Headline:** Fraud labels sell at a flat price, so we check expensive parcels sooner.
- **Why it is ours:** Typical teams use one risk threshold for every parcel. Ours moves with the parcel's cost, because label buyers pick expensive parcels.
- **Proof:** LabelsBank sells labels at about $2 regardless of weight or distance. Allow stops at about 3.9% risk for a R$45 parcel and about 2.6% for R$120.
- **Demo moment:** Take one booking with risk around 3%. At R$45 it's allowed. Change only the parcel to R$120 and it moves to "allow, but check at first scan". Same customer, same risk, different action.
- **Judge challenge:** "Doesn't that punish your best customers, the ones shipping expensive parcels?"
- **Our answer:** "The shift is about 1.3 points, and it adds friction rather than a block, because USPS OIG found about 2 million paid packages wrongly intercepted."

> Before presenting: check that the demo booking really lands between 2.6% and 3.9% risk, or the action won't flip on stage.

### USP 3: Our system refused to deploy our own model

- **Headline:** Our safety gate blocked our own retrained model from going live.
- **Why it is ours:** Typical teams say "continuous learning" and show a better accuracy number. We can show a pass/fail rule we wrote before testing, and the time it said no.
- **Proof:** Retraining raised the catch rate on held-out mule drops from 0.28 to 0.69. False positives on legitimate shippers rose from 0.51% to 2.48%, so the gate blocked deployment. The block is in the hash-chained audit log.
- **Demo moment:** Open the retrain screen: catch rate up, false positives up, status BLOCKED. Then open the audit log entry for that decision and flip one byte to show the tamper alarm.
- **Judge challenge:** "So your continuous learning doesn't actually work?"
- **Our answer:** "It learned the unseen pattern, 0.28 to 0.69. It also would have hurt real customers, and a rule we wrote in advance caught that before any shipper saw it."

## 20-second spoken version

> "Stolen shipping accounts aren't used to ship one parcel. They're used to open a shipping shop. So we don't flag new destinations, which turned out to be 94.9% of normal bookings. We flag an account paying for senders it never shipped for. Fraudsters buy labels at a flat price, so expensive parcels get checked sooner. And when our own retrained model would have hurt real shippers, our system refused to deploy it."

## Closing line for the last slide

> "We built the part that says no, including to us."

## Do not claim (yet)

- That fine-tuned Laya beats LightGBM, or any accuracy figure for Laya (results pending)
- Real carrier deployment or real customer data
- Rounded-up versions of any number above
