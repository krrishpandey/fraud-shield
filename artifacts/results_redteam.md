# Red team: we attack our own model

Real run of scripts/redteam.py on 2026-10-04 04:48, 606 s CPU wall time. Target: the deployed system as config/app.yaml runs it (B2_F LightGBM decides, Platt-calibrated, cost rule and rules floor; Laya not deciding), scored through the real online feature path. Search code: fraudshield/redteam/search.py (shared with GET /decisions/{id}/counterfactual).

Attacker: sees only the returned action; at most 50 queries and 2 changed fields per booking. Fields: declared value, declared weight (carrier cost follows the weight at the lane freight slope, never set freely), parcel size, service, a sender the account already used, booking time.

## Flip rates, test window (2018-05-15..2018-08-31), evaluated once

Population: fraud bookings the system STOPS (owner_confirm, review, hold, block). Injection seeds [0, 1, 2, 3, 4, 5, 6, 7] (as many as fit the time budget); per seed a seeded random subsample of at most 40 trained-type and 10 never-trained bookings. Mean ± sd over seeds; pooled n.

| Type | n attacked | Flip to allow | Flip to any softer action |
|---|---|---|---|
| T1 | 132 | 18.8 ± 12.8% | 74.4 ± 14.5% |
| T2 | 54 | 0.0 ± 0.0% | 13.2 ± 21.7% |
| T4 | 72 | 21.9 ± 15.0% | 57.5 ± 8.8% |
| T6 | 0 | n/a | n/a |
| T7 | 62 | 26.9 ± 21.4% | 84.5 ± 13.4% |
| **all trained** | 320 | 18.1 ± 7.6% | 63.1 ± 7.3% |

Stopped at booking over these seeds (before subsampling): T1 559 of 559, T2 224 of 264, T4 270 of 330, T6 8 of 242, T7 283 of 283. T6 (under-declared weight) is mostly routed to a depot weight check (allow_scan_gated), not stopped, so few or no T6 bookings enter the population.

Never trained (T3, T5; reported only, never used for hardening):

| Type | n attacked | Flip to allow | Flip to any softer action |
|---|---|---|---|
| T3 | 40 | 4.2 ± 7.7% | 17.7 ± 26.9% |
| T5 | 40 | 16.5 ± 11.0% | 70.0 ± 20.1% |
| **all never-trained** | 80 | 11.3 ± 6.4% | 45.0 ± 18.5% |

Fields in the evasions found (trained types): {'sender_id': 159, 'weight_kg': 38, 'dims': 13, 'declared_value': 12, 'service': 9, 'booked_at': 7}.
Median queries to the first softer action, when found: 14.0; to allow: 14.0. Median share of the carrier cost an evading booking keeps: 100.0% (weight changes move the cost, so evading by under-declaring weight also shrinks the payoff).
Offline/online check: 0 of 400 attacked bookings scored differently from the offline feature table (0 expected).

Softer includes allow_scan_gated: the parcel still gets a depot weight check, so a flip to it is a partial evasion at most.

## Analyst counterfactuals (GET /decisions/{id}/counterfactual)

Test window, seed 0: 60 stopped decisions sampled at random from 265 (47 fraud, the rest legit). The endpoint's search (model probability visible, bisection refinement), in process without HTTP.

| Measure | Value |
|---|---|
| Share with a counterfactual of at most 2 changes to a softer action | 73.3% |
| ... on fraud / on legit stopped bookings | 70.2% / 84.6% |
| Share with one that reaches plain allow | 21.7% |
| Latency median / p90 | 1118 ms / 1354 ms |
| Evaluations used, median | 50 of 50 |
| Fields in the counterfactuals | {'sender_id': 28, 'weight_kg': 13, 'declared_value': 6, 'booked_at': 5, 'dims': 5, 'service': 2} |

## Hardening

Evasions generated from TRAIN-window fraud only (seed 0, types T1, T2, T4, T6, T7): 80 stopped train-window fraud bookings attacked, 48 evaded, 113 evasive variants (at most 3 per booking) added as fraud labels (SIMULATED analyst labels: ground truth of synthetic evasions). They went through the existing LearningService retrain and the pre-registered gate (docs/LEARNING_GATE.md), code and thresholds unchanged, in a scratch registry (never artifacts/models/registry.json).

**Gate verdict: REJECTED** (mode noninferiority).

| Check | Passed | Detail |
|---|---|---|
| cost_noninferior | True | cost per 1,000 bookings 1080.27 -> 1040.57 BRL; improvement 95% CI [2.27, 80.01], lower bound must be >= -54.01 |
| pr_auc_noninferior | True | PR-AUC 0.8160 -> 0.8568; must be >= 0.7960 (current - 0.02) |
| ece_noninferior | True | ECE (15 equal-mass bins) 0.0021 -> 0.0005 (change -0.0016, max +0.02) |
| fpr_hard_negative_noninferior | False | FPR on hard negatives 0.27% -> 1.68% (max +0.5 points) |
| superiority | True | needs a lower bound above 0 for either: cost improvement CI [2.27, 80.01] BRL per 1k (yes); new-pattern recall improvement 95% CI [-0.014, 0.001] (no) |
| min_new_labels | True | 113 new labels since the last retrain (need 20) |

Candidate gbm-B2-F-v2 (NOT deployed), test window seed 0, same attack, evaluated once. Each row attacks the test fraud that model stops (seeded subsample), so the two populations differ:

| Model | n stopped fraud attacked | Flip to allow | Flip to softer |
|---|---|---|---|
| v1 (deployed) | 40 | 15.0% | 62.5% |
| candidate | 40 | 5.0% | 12.5% |

Sources: Fok et al., "Foe for Fraud: Transferable Adversarial Attacks in Credit Card Fraud Detection", arXiv 2508.14699 (2025); Khouna et al., "Optimal Counterfactual Search in Tree Ensembles", arXiv 2605.06561 (2026). Fraud rows are synthetic injections on real Olist histories (DATA_CARD.md).
