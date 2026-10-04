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

### The round-3 additions at a glance

| Where in the app | What it does, concretely |
|---|---|
| **Decision page → "Show what would change it"** | Lists the smallest edit to the booking that would have changed our decision. Example: a held booking would have been allowed if its declared weight were 0.85 kg instead of 20 kg and its declared value R$30.10 instead of R$121.50. |
| **Red team tab** | A program plays the fraudster: it edits a stopped booking up to 50 times and shows every attempt and our answer. On booking blk-06 it got an "Allow" on attempt 24; on the takeover demo all 50 attempts stayed "Hold". |
| **Red team tab → "Retrain with N new evasion labels"** | Turns the attacker's successful edits into training examples and retrains the model. Our safety gate then decides whether the new model replaces the old one. |
| **Decision page → "Was this you?"** | The account owner approves a stopped booking with Windows Hello (PIN, face or fingerprint). The approval is tied to this booking's price, destination and receiver. |
| **Live tab → "Label-free monitor"** | Estimates how many of today's stopped bookings are really fraud, and how much fraud we let through, weeks before the fraud reports arrive. |
| **Learning tab → banner after retraining** | One sentence saying which model now scores new bookings, and if the old one stayed, the exact reason. |
| **Decision page → ticks next to the explanation** | Each sentence of the AI-written explanation is checked separately against the decision's own numbers. |
| **Sign-in page** | The console opens on a sign-in screen and has a new dark design. |

### Feature 10: "What would change this decision?" (analysts only)

- **Where:** the Decision page of any stopped booking (Hold, Block, Review, Ask the owner, or Allow-with-depot-check), button **"Show what would change it"**.
- **What it does:** it makes up to 50 edited copies of the booking and runs each one through the real model and the real cost rule, without saving anything. Each copy changes at most 2 things that the person booking controls:
  - the declared value;
  - the declared weight (the shipping price is recalculated from the weight, so a lighter parcel also costs less);
  - the parcel size;
  - the service (standard or express);
  - the sender, swapped to one this account has used before;
  - the booking time.
- **What you see** (booking blk-06, which we hold):
  - "Allow if declared weight were 0.85 kg (not 20.00 kg; carrier cost R$34.19) and declared value were R$30.10 (not R$121.50)";
  - "Ask the owner if declared weight were 1.87 kg (not 20.00 kg)".
- **Measured:** we took 60 random stopped bookings from the test period (2018-05-15 to 2018-08-31).
  - 73.3% had at least one edit of 2 fields or fewer that led to a less strict action.
  - 21.7% had one that led to a plain "Allow".
  - A search takes about 1.1 seconds (median).
- **Why only analysts see it:** a booker who saw it would learn exactly how to get fraud through. It is never shown to the booker, and every time an analyst opens it, an entry goes into the tamper-proof audit log.
- **Limit:** about 1 in 4 stopped bookings has no answer within 50 tries. The takeover demo is one of them.

### Feature 11: We attack our own model (red team)

- **Where:** the **Red team** tab. Every stopped booking also has a link, "Red-team this decision".
- **How the attacker works:**
  - It sees only what a real fraudster would see: our answer (Allow, Hold, and so on), never the risk score.
  - It may edit at most 2 of the fields listed in Feature 10.
  - It stops after 50 tries, or as soon as it gets a plain "Allow".
- **What you see:** you pick a stopped booking and click **Launch attack**. Each try appears as a row, with the edit and our answer.
  - On blk-06, tries 1 to 6 change the declared value and all get "Hold".
  - Try 9 (weight 0.85 kg) gets "Allow, check at first scan": the label is issued, but the parcel is weighed at the depot.
  - Try 24 (weight 0.85 kg and value R$30.10) gets a plain "Allow". The row turns red: "The attacker got through".
  - On the takeover demo booking, all 50 tries get "Hold": "Our model held".
- **Measured** (test period, run once, 320 stopped fraud bookings of the 5 fraud types the model was trained on, 8 random variants of the injected fraud):
  - 18.1% got a plain "Allow".
  - 63.1% got any less strict answer. Many of those are still stopped, for example "Hold" instead of "Block", or "Allow, check at first scan".
  - The most common trick, 159 of the 238 edits, was switching the sender to one the account already uses.
- **Turning attacks into training data:**
  1. Each attack keeps its 3 smallest successful edits.
  2. They become training examples labelled "fraud" only once we know the original booking was fraud. An analyst clicks **"Confirm original as fraud"**, or the booking is one of our injected test frauds (then the labels are marked "simulated").
  3. **"Retrain with N new evasion labels"** retrains the model and runs the safety gate. The gate's verdict appears as the same banner as on the Learning tab (Feature 14).
- **What happened when we tried:**
  - In the measured run, we retrained on 113 such edits. The gate refused the new model: honest bookings wrongly stopped in our hard test cases would have gone from 0.27% to 1.68%, and the gate allows at most +0.5 percentage points.
  - In a live test with 2 attacks (6 examples), the gate kept the old model: "no proven improvement".
- **Limit:** the fraud is injected into a real dataset, so these are attacks on our model with our test fraud, not on a real carrier.

### Feature 12: "Was this you?" with a passkey, tied to the booking

- **Where:** the Decision page of a stopped booking, box **"Was this you? The owner's passkey"**.
- **Why:** if a criminal has stolen the account login, they may also control its e-mail or phone number, so a "was this you?" text can be answered by the criminal. A passkey is a key stored on the owner's own device, unlocked with Windows Hello (PIN, face or fingerprint), and a stolen password can't use it.
- **Steps:**
  1. **Enroll owner passkey:** the owner's device creates a key. Done once per account.
  2. **Confirm as owner:** Windows Hello asks for the PIN or face. The device signs a code built from this booking's id, account, shipping price, declared value, destination and receiver.
  3. Our server rebuilds that code from the booking it has stored and checks the signature.
  4. If it matches, the booking is released. A "Hold" becomes "Allow, check at first scan"; an "Ask the owner" becomes "Allow". A "Block" can't be released this way.
- **Tamper test:** this button re-checks the same signature against a copy of the booking with the shipping price raised by R$100 (R$72.78 becomes R$172.78). The result is "Rejected: the signature no longer matches". The owner approved this exact booking, not any booking.
- **Measured:** the signature check takes 0.18 ms (median of 1,000 checks); the whole server step takes 2.6 ms. The time a person takes at the Windows Hello prompt was not measured.
- **Limits:**
  - In the demo, this laptop plays the owner's phone, and the screen says so.
  - In the demo we enroll the passkey just before confirming. In real use it would be enrolled when the account is opened, and changing it would itself need checking.

### Feature 13: The label-free monitor (and where it is blind)

- **Where:** the **Live** tab, card **"Label-free monitor"**. It works while the live booking stream is running.
- **The problem it solves:** we only learn that a booking was fraud days or weeks later, when a chargeback or complaint arrives. Until then we can't count our mistakes directly.
- **How it estimates:** every booking gets a probability from the model, for example 0.30 means "30% chance it is fraud". These probabilities are calibrated (we checked that, across many bookings, about 30 in 100 of the ones scored 0.30 really are fraud). So:
  - adding up the probabilities of the bookings we stopped estimates how many of them are really fraud (the "precision of stops");
  - adding up the probabilities of the bookings we let through estimates how much fraud we missed.
- **What you see** (after about 25 seconds of the stream): "Estimated precision of stops 33.5% (no labels needed)", "Estimated fraud missed 1.7 of 830 let through". Next to it is a grey box with the real answer, which only exists because the demo replays a dataset where we know the truth.
- **Where it fails, measured** (test period, all 10 variants of the injected fraud):
  - For fraud types the model has never seen (T3 and T5), the model gives low probabilities, so the monitor doesn't count them as missed.
  - Over the test period it estimated 52.8 missed frauds when the real number was 67.4: about 1.45 too few per week.
  - The usual "the data has changed" alarm (it compares the spread of today's scores with last month's) gave the same reading with or without these new fraud types.
  - The card shows this warning permanently.

### Feature 14: Which model is in charge, said in one sentence

- **Where:** the **Learning** tab, after you click **Retrain with new labels**. The Decision page also shows "LightGBM model that scored this booking: gbm-B2-F-v1", and the Live tab shows "Model in use".
- **What you see:** a banner with one of three messages:
  - green: "From now on, new bookings are scored by gbm-B2-F-v2 (since 14:02). Previous model gbm-B2-F-v1 kept for rollback."
  - amber: "Still using gbm-B2-F-v1. Candidate gbm-B2-F-v2 was not deployed because:", followed by each failed check in plain words, e.g. "honest hard-case false positives would rise from 0.27% to 0.94%; the limit is +0.5 points".
  - blue, after a rollback: "Rolled back. From now on, new bookings are scored by …".
- **If you leave the tab during a retrain:** the retrain keeps running on the server. When you come back, the progress bar returns (timed from when the run started) and the result appears when it finishes. Clicking Retrain again during a run follows that run instead of starting a second one.
- **Our latest attempt to get a better model accepted** (written down before we ran it, run once):
  - It fixed the earlier problem: honest hard cases wrongly stopped went from 0.32% to 0.64%, inside the +0.5-point limit.
  - It was better on every average: PR-AUC 0.807 to 0.830, and 0.535 to 0.577 of never-seen fraud caught.
  - It was still refused on cost. The 95% range for the saving was −R$9.61 to +R$21.41 per 1,000 bookings, and the gate requires the bottom of that range to be at least −R$5.98. About 180 frauds in the test set is not enough to be that sure.
- **Note:** this is not "reinforcement learning". It is retraining with a pass/fail safety gate written before testing.

### Feature 15: Every sentence of the explanation is checked

- **Where:** the Decision page, section **"Why, in plain words"**.
- **What changed:** the AI writer (Groq's gpt-oss-120b) now returns its explanation as separate claims. Each claim names one reason code and the fields it uses. Example from the takeover booking: "The dimensions are 20 standard units above the account norm", citing reason DIMS_UNUSUAL and field dims_z.
- **What each claim must pass:**
  - its reason code is one of this decision's reasons;
  - every number in it appears in the decision record and in that claim's own fields;
  - every field it cites exists.

  Passing claims get a tick. If any claim fails, the page shows the fixed template instead and crosses out the failed claim.
- **Measured:**
  - We planted 33 explanations where a true number was moved into the wrong sentence. The new check caught 33 of 33; the old whole-paragraph check caught 0.
  - Made-up numbers, reasons or fields were caught 57 of 57 each.
  - These were measured on 31 of 60 sampled decisions; the other 29 were not run because the free Groq quota ran out that day.
- **The honest number on the page:** "Agrees with the model's top reasons: 1 of 3". The model's 3 biggest influences often aren't among our fixed reason rules (those rules describe only about 58% of what drives the score), so the best possible average is 0.29. Our explanations score 0.20 to 0.24.

### Feature 16: Sign-in page and new look

- **What it is:** the console now opens on a sign-in screen with a 3D parcel sorting line, then shows a dark interface with a new sidebar, quick search (Ctrl+K) and status chips.
- **How to sign in:** any analyst name and a code of at least 4 characters.
- **Honest note:** this only records who is at the desk. It is not real security: the API behind the console has no user accounts.
- **What stayed the same:** the action colours (green Allow, teal check-at-depot, blue Ask the owner, purple Review, orange Hold, red Block) and all charts.

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
