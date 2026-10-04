# FraudShield, explained simply

## 1. The problem in one story

A shipping company (like UPS or FedEx) gives business customers an online account to book parcels. The company pays the shipping bill at the end of the month.

A criminal steals the login of one of these businesses. Now the criminal can book parcels, and **the real business gets the bill**.

Criminals don't just ship their own stuff. They **sell shipping labels** to other people, cheaply, using the stolen account. It's like someone stealing your credit card and opening a shop that sells things on your card.

This really happened: in one US court case, a stolen UPS login was used this way and the losses were over $900,000.

**Today**, the carrier usually notices only after the parcels are delivered, when the money is already lost.

**Our goal:** check every booking **at the moment it is made**, before the parcel enters the network, and decide whether it looks like fraud. We also need to avoid annoying honest customers.

---

## 2. Our big idea

Most teams would ask: **"Is this parcel going somewhere new?"**

We checked real data and found that's useless. Honest sellers send about 95% (94.9%) of their parcels to addresses they've never used before. That's just normal business: new customers every day.

So we ask a different question: **"Who is this account paying for?"**

An honest business ships **its own** goods from **its own** places. A stolen account suddenly starts paying for parcels from **strangers**: different senders and different cities it has never shipped from before. That's the sign that someone opened a "shop" on the stolen account.

> Simple version: we don't watch where parcels go. We watch whose parcels the account is paying for.

---

## 3. What happens when someone books a parcel

Think of it as a security guard checking each booking in a fraction of a second:

1. **Look at the account's history.** What does this account normally do? Which senders, which cities, what parcel sizes, what prices?
2. **Spot what changed.** New senders? A much more expensive parcel than usual? A login from a new device?
3. **Get a risk score.** Two models look at it (explained below).
4. **Pick an action.** Allow it, check it, or stop it.
5. **Write it down permanently** in a tamper-proof log.
6. **Explain the decision** in plain English for the analyst.

All of this takes well under a second.

---

## 4. Each feature in simple terms

### Feature 1: Risk scoring with LightGBM (the "experienced checker")

- **What it is:** a standard machine-learning model. It learned from thousands of past bookings, some honest and some fraud, and gives each new booking a fraud score.
- **Analogy:** an experienced bank clerk who has seen many fake cheques and has a gut feeling for new ones.
- **Limit:** it can only spot the kinds of fraud it was trained on. A new trick can slip past it.

### Feature 2: Laya (the "question answerer")

- **What it is:** an AI decision model (open-source; you can download and modify it). Instead of one score, it answers specific yes/no questions about a booking, like:
  - "Is this account paying for senders outside its usual customers?"
  - "Is this parcel unusually expensive for this account?"
  - "Does the receiver look like a 'drop' address used by criminals?"
  - "Is this account being misused?"
- **The special part:** an analyst can **type a brand-new question** in plain English and get an answer, without retraining anything. A normal model like LightGBM can't do that.
- **Status:** to be good at this, Laya must be **fine-tuned** (extra training on our data). That training runs on Kaggle's free GPUs and is **not finished yet**. The plain (stock) version gives poor, unreliable answers, so for now the app relies on LightGBM.

### Feature 3: The cost rule (the "sensible manager")

- **What it is:** a simple calculation that picks the action. For each option it asks: "If I do this, how much money do we expect to lose, counting both fraud losses and annoying an honest customer?" Then it picks the cheapest option.
- **The 6 actions, from softest to strictest:**

| Action | What it means in real life |
|---|---|
| Allow | Ship normally |
| Allow, but check at first scan | Label is issued, but the parcel is weighed and checked at the depot before it travels |
| Ask the owner | Send a one-tap "Was this you?" message to the account owner's trusted contact |
| Review | A human analyst looks at it |
| Hold | No label until someone verifies it |
| Block | Refuse it (only with strong evidence plus a human confirming) |

- **The clever bit, price awareness:** criminals sell stolen labels at a flat price (about $2), no matter how heavy the parcel or how far it goes. So their buyers send **expensive** parcels. Our rule knows this: for an expensive parcel, it adds a check earlier. Allow stops at about 3.9% risk for a R$45 parcel, but at about 2.6% for a R$120 parcel.

### Feature 4: Being gentle with honest customers

- **Why it matters:** the US Postal Service once wrongly stopped about 2 million paid packages. Blocking honest customers costs money and trust.
- **How we handle it:** when we're unsure, we use the soft option ("check at first scan") instead of blocking. Blocks need strong evidence and a human. We also measure how often we bother honest customers and show that number next to every result.

### Feature 5: Plain-English explanations (GenAI)

- **What it is:** for each risky decision, an AI writer (through Groq) writes a short explanation, like: "Held because the account suddenly paid for 9 new senders and the parcel costs far more than usual."
- **The safety check:** AI writers sometimes make up facts. So we check every explanation: if it mentions **any number that isn't in the actual decision**, we throw it away and show a fixed, safe template instead.

### Feature 6: Tamper-proof audit log

- **What it is:** every decision is saved: which model versions were used, what the model saw, the scores, the action and the explanation.
- **How it's tamper-proof:** each record is linked to the one before it with a digital fingerprint (a "hash"). Change even one letter anywhere, and the check fails at exactly that record. It works like numbered, sealed pages in a notebook.
- **Why it matters:** the brief asks for full auditability. Anyone can later prove what the system decided and why.

### Feature 7: Learning from analysts, with a safety gate

- **What it is:** when an analyst marks a booking "fraud" or "honest", that becomes new training data. The system can retrain itself to learn new tricks.
- **The safety gate:** a new model is only switched on if it passes tests we **wrote down before running them** (so we couldn't bend the rules afterwards). It must be no worse at what the old model did well, and not more annoying to honest customers.
- **What actually happened:** the retrained model got much better at catching a fraud type it had never seen (catch rate 0.28 → 0.69). But it also wrongly flagged more honest customers (0.51% → 2.48%). **So the gate refused to switch it on.** That's a feature: the system protects real customers even from its own "improvements".

### Feature 8: Dashboard

- **What it shows:** how many bookings were stopped, which types of fraud, money saved after subtracting the cost of checks, and how often honest customers were bothered.
- **Who it's for:** the carrier's fraud team and managers.

### Feature 9: Desktop app

- **What it is:** a normal Windows app. Double-click `FraudShield.bat` and a window opens. No website or server setup.
- **Tabs:** Score (try a booking), Decision (see details), Queue (bookings waiting for review), Dashboard, Learning (retraining), Audit (check the log).

### Feature 10: "What would change this decision?" (for analysts only)

- **What it is:** for a stopped booking, the app tries small changes the person booking could make (a lower declared value, a lighter declared weight, a different service, a sender the account has used before) and tells the analyst the smallest change that would have made the decision softer. Example: "allowed if the declared weight were 0.85 kg instead of 20 kg".
- **Analogy:** asking a bank clerk "what would you have needed to see to say yes?"
- **Why only analysts:** if the person booking saw it, it would be a recipe for getting fraud through. So it is never shown to them, and every time an analyst looks, it is written in the tamper-proof log.
- **Result:** 73% of stopped bookings have an answer with at most two changes.

### Feature 11: We attack our own model (red team)

- **What it is:** we played the fraudster. Our "attacker" only sees the decision, and may change at most two things on a booking, with up to 50 tries. We counted how often it got stopped fraud through.
- **Result (inserted fraud, test period):** 18% got a plain "allow", 63% got some softer action. Most did it by pretending to ship from a sender the account already uses.
- **What we did with it:** we retrained the model on those tricks. Our own safety gate (Feature 7) refused the new model, because it would have bothered more honest customers.
- **Why it matters:** we report how easily we can be fooled, instead of only how often we are right.

### Feature 12: "Was this you?" with a passkey, tied to the parcel

- **What it is:** when the app asks the account owner to confirm a booking, the owner answers with a **passkey** (the fingerprint, face or PIN unlock on their phone or laptop, like Windows Hello). A text message can be answered by a criminal who stole the account; a passkey lives on the owner's own device.
- **The clever bit:** the passkey signs the details of **this parcel**: its price, destination and receiver. Change the price by R$100 and the same signature no longer fits. The owner approves *this* parcel, not "any parcel".
- **Honest note:** in the demo, the laptop plays the owner's phone, and the screen says so.

### Feature 13: A health check that knows where it is blind

- **What it is:** fraud is only confirmed weeks later. This card on the Live tab estimates **today** how accurate our stops are and how much fraud we are letting through, using only the model's own (calibrated) scores.
- **Analogy:** a weather forecast for our own accuracy, before the "real weather" (the confirmed fraud reports) arrives.
- **Where it fails, measured:** for fraud types the model has never seen, it under-counts the fraud we miss (by about 1.45 per week in our test). The usual "something changed" alarm doesn't notice those types either. We show that warning on screen instead of hiding it.

### Feature 14: Which model is in charge, said in one sentence

- **What it is:** after every retraining, the Learning tab says plainly either **"From now on, new bookings are scored by the new model"** or **"Still using the old model, because..."**, with the reason in everyday words (for example, "honest customers would be wrongly stopped more often"). Every decision shows which model version made it.
- **Latest try:** we made one more, honestly pre-announced attempt to fix the new model. It fixed the "bothers honest customers" problem and was better on every average, but the gate's money check was still too uncertain, so the old model stays in charge.
- **Note:** this is not "reinforcement learning". It is retraining with a safety gate.

### Feature 15: Every sentence of the explanation is checked

- **What it is:** the AI writer (Feature 5) now hands back its explanation as separate **claims**, each one tied to one reason, like "the account paid for 9 new senders". We check every claim on its own: the reason must really apply to this booking, and every number must come from the decision record **and belong to that claim**.
- **Why that matters:** a sneaky mistake is a true number in the wrong sentence. The old check looked at the whole paragraph and missed all 33 we planted; the new check caught all 33.
- **Honest part:** we also measure whether the explanation mentions the same things the model actually relied on most. It's low (about 1 in 5), because our fixed reason rules only describe about 58% of what the model leans on. The page shows that number instead of hiding it.

---

## 5. Where the data comes from (and what is fake)

- **Real part:** Olist, a public dataset of about 100,000 real Brazilian online-shop orders. It has real sellers, cities, parcel weights and shipping prices.
- **Added part:** no public dataset of parcel fraud exists, so we **inserted fraud ourselves**, based on real cases. Every inserted row is labelled, so we always know what's real and what's added.
- **Made-up part:** login and device information is synthetic, because Olist doesn't have any. That's why we report results twice: once using only real data columns, and once using everything.

---

## 6. How we know it works (honest testing)

- **Not too easy:** simple rules alone score very low (PR-AUC 0.04-0.05) on our hardest test, so our inserted fraud isn't trivially easy to spot.
- **LightGBM results:** PR-AUC 0.74 using only real columns, 0.78 using all columns. With the drop-address rule (below), the app's score reaches PR-AUC **0.79** (real columns) and **0.82** (all columns); ROC-AUC 0.955. (PR-AUC runs from 0 to 1. Higher means it catches more fraud while raising fewer false alarms.)
- **Unseen fraud:** we hid two fraud types from training on purpose. For reshipping drops (T3), a rule based on published research (Hao et al., 2015: drops are new addresses that get parcels from several accounts for about 30 days) raised the share caught from 18% to 53%, still without training on them.
- **Weight fraud (T6):** nobody can weigh a booking form, so the model alone catches only 6%. Parcels declared far smaller than the account's usual ones are weighed at the first depot scan; one failed scan means the account's next parcels are weighed too. That catches 40%, while weighing 1.8% of honest parcels. The carrier can turn this up with the **depot weighing dial** on the dashboard: weighing more parcels catches more weight fraud, and each level shows its measured result (weighing 4% of honest parcels catches 54%, 6% catches 73%, 11% catches 94%). Every change of level is written to the audit log.
- **New account fraud (T2):** three new features (what a fresh account ships, how far, how valuable) raised its PR-AUC from 0.38 to 0.46.
- **Live load:** replaying all 22,925 test bookings through the app gives exactly the offline numbers (live PR-AUC 0.807 = offline 0.807).
- **Software tests:** 506 automated code tests and 26 full app click-through tests, all passing (2026-10-04).

---

## 7. Where we are right now

| Part | Status |
|---|---|
| App, scoring, actions, explanations, audit, dashboard, learning | Working |
| LightGBM model | Working, results measured |
| Laya fine-tuning | Done. It lost to LightGBM at deciding (PR-AUC 0.356 vs 0.777), so **LightGBM decides** and Laya only answers analyst questions |
| "Ask a new question" | Works with the fine-tuned Laya; its answers are not calibrated |
| What would change this decision, red team, owner passkey, label-free monitor, model handover banner | Working, tested (added for round 3) |
| Retrained model | Not deployed: no run has passed the safety gate yet |

---

## 8. The whole project in one paragraph

> FraudShield checks every parcel booking the moment it's made. Instead of asking "is this going somewhere new?" (honest sellers do that 95% of the time), it asks "is this account suddenly paying for strangers' parcels?", which is the sign of a stolen account being used as a label shop. A cost rule picks the gentlest action that still makes sense, and checks expensive parcels sooner because criminals prefer them. Every decision is explained in plain English, checked for made-up facts, and saved in a tamper-proof log. The system learns from analysts, but refuses to switch on a new model that would hurt honest customers.
