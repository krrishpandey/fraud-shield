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

---

## 5. Where the data comes from (and what is fake)

- **Real part:** Olist, a public dataset of about 100,000 real Brazilian online-shop orders. It has real sellers, cities, parcel weights and shipping prices.
- **Added part:** no public dataset of parcel fraud exists, so we **inserted fraud ourselves**, based on real cases. Every inserted row is labelled, so we always know what's real and what's added.
- **Made-up part:** login and device information is synthetic, because Olist doesn't have any. That's why we report results twice: once using only real data columns, and once using everything.

---

## 6. How we know it works (honest testing)

- **Not too easy:** simple rules alone score very low (PR-AUC 0.04-0.05) on our hardest test, so our inserted fraud isn't trivially easy to spot.
- **LightGBM results:** PR-AUC 0.73 using only real columns, 0.77 using all columns. (PR-AUC runs from 0 to 1. Higher means it catches more fraud while raising fewer false alarms.)
- **Unseen fraud:** we hid two fraud types from training on purpose. LightGBM does poorly on one of them (PR-AUC 0.31). Closing that gap is the job of the fine-tuned Laya.
- **Software tests:** 344 automated code tests and 14 full app click-through tests, all passing.

---

## 7. Where we are right now

| Part | Status |
|---|---|
| App, scoring, actions, explanations, audit, dashboard, learning | Working |
| LightGBM model | Working, results measured |
| Laya fine-tuning | **Not done yet** (runs on Kaggle) |
| "Ask a new question" | Needs the fine-tuned Laya to work well |
| Demo | Run it in **cached mode** until fine-tuned Laya is ready |

---

## 8. The whole project in one paragraph

> FraudShield checks every parcel booking the moment it's made. Instead of asking "is this going somewhere new?" (honest sellers do that 95% of the time), it asks "is this account suddenly paying for strangers' parcels?", which is the sign of a stolen account being used as a label shop. A cost rule picks the gentlest action that still makes sense, and checks expensive parcels sooner because criminals prefer them. Every decision is explained in plain English, checked for made-up facts, and saved in a tamper-proof log. The system learns from analysts, but refuses to switch on a new model that would hurt honest customers.
