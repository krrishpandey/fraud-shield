# Retraining deployment gate (pre-registered)

Written 2026-10-02, before the second round of real-data runs. The thresholds below were fixed before those runs and are not tuned on their results. Code: `fraudshield/learning/metrics.py` (`GateConfig`, `gate_checks`). Choose a gate with `learning.gate_mode` in `config/app.yaml`, or per run with `POST /learning/retrain {"gate": "strict" | "noninferiority"}`.

## Gate v2 (default, `noninferiority`)

A candidate GBM is deployed only if ALL of these hold.

**Non-inferiority, on the overall eval set** (the validation window plus the latest 30% of feedback by time, never trained on):

| Check | Rule |
|---|---|
| cost | Lower 95% bound of the cost improvement per 1,000 bookings (current minus candidate) is at least -5% of the current cost per 1,000 bookings. |
| PR-AUC | Candidate PR-AUC is at least current PR-AUC minus 0.02 (point estimates). |
| ECE | Candidate ECE (15 equal-mass bins) is at most current ECE plus 0.02. |
| Hard-negative FPR | Candidate FPR on hard negatives is at most current plus 0.5 percentage points. |
| Labels | At least `min_new_labels` labels have arrived since the last retrain. |

**Superiority, at least one of:**
- the lower 95% bound of the cost improvement on the overall eval set is above 0, or
- the lower 95% bound of the recall improvement on the **new-pattern eval set** is above 0.

The new-pattern eval set holds test-window bookings of the fraud typologies the active model never trained on (T3 and T5 for v1). They are pooled across all available injection seeds, and any campaign that appears in the training feedback is removed.

**Bootstrap:** paired, 95% percentile intervals, B = 200. Resampling is by fraud campaign, and by account where there is no campaign (new-pattern set: by campaign).

**Operating point:** the production cost rule applied to each model's probability, with no exploration. "Stopped" means owner_confirm, review, hold or block.

## Original strict gate (kept as `strict`)

- Lower 95% bound of the cost improvement is at least -2% of the current cost.
- The PR-AUC difference CI is not entirely below 0.
- ECE rises by at most 0.02.
- Hard-negative FPR rises by at most 0.5 points.
- At least `min_new_labels` new labels.

There is no superiority condition.

## Original runs (strict gate, real data)

v1 = `B2_F` (seed 0). The simulation picked random test-window bookings, enriched to 50% injected rows, and gave every one an analyst label with 5% symmetric error. Laya ran in cached mode, so every decision was degraded and nothing was explored.

| Run | Labels | Analyst error | PR-AUC v1 → cand | Cost per 1k BRL v1 → cand (improvement CI) | HN FPR | Result |
|---|---|---|---|---|---|---|
| 1 | 2000 | 5% | 0.685 → 0.671 | 311.9 → 337.3 ([-53.2, -0.4]) | 1.02% → 0.89% | rejected (cost) |
| 2 | 5000 | 5% | 0.611 → 0.592 | 322.8 → 346.9 ([-47.6, -6.0]) | 0.51% → 2.10% | rejected (cost, HN FPR) |
| 3 | 2000 | 0% | 0.837 → 0.837 | 154.8 → 151.4 ([-21.1, +33.0]) | 0.51% → 1.02% | rejected (cost) |

## Why the gate changed

1. **The strict cost rule could not be met.** The overall eval set has about 200 frauds, which fall into far fewer independent campaigns, so the 95% CI of the cost difference is about 50 BRL per 1k wide. That is about 15 to 30% of the current cost. Requiring the lower bound to be at least -2% of current cost (about -3 to -6 BRL) in effect requires a large proven improvement. Run 3 shows this: the candidate was equal or slightly better (-3.4 BRL per 1k as a point estimate) and was still rejected.
2. **The old gate never measured the reason for retraining.** Retraining exists to learn patterns the active model never saw. In the old eval set these appeared only in the latest 30% of feedback, in 2 to 12 rows. The new-pattern eval set pools T3 and T5 test rows across seeds (about 860 rows in 60 campaigns), so recall on new patterns can be measured with a campaign-level CI.
3. **Non-inferiority plus superiority** is the standard design for this question: "is it no worse on what we already do, and better on something?"
4. **The simulation was also unrealistic.** Analysts only see bookings the system stopped. Allowed fraud turns up later as disputes, and allowed legit bookings only count as legit once they mature. The simulation was changed (mode `realistic`). The old one is kept as `uniform_noisy` for the record.

The changes are to the gate design and the simulation. No threshold was chosen by looking at results from runs under the new gate.

## Results under the pre-registered gate (added after the runs, thresholds unchanged)

Scenario "a new pattern emerges". v1 = `B2_F` (seed 0; never trained on T3 or T5). Registry and labels were written to a scratch directory, not `artifacts/`. Laya was not run: a stand-in answered misuse = the v1 GBM probability, so the policy ran in normal mode with 5% logged exploration. Because of that stand-in, the calibration-refresh results are not about Laya.

The new-pattern eval set (b) has 792 rows in 54 campaigns from seeds 1 to 9. The 68 seed-0 rows whose campaigns were in the training feedback were removed.

| Run | Feedback labels | Overall cost per 1k v1 → cand (CI of improvement) | PR-AUC | HN FPR | New-pattern recall v1 → cand (CI of diff) | Gate v2 | Strict |
|---|---|---|---|---|---|---|---|
| realistic, 5% analyst error | 22,925: 363 analyst, 22,562 outcome; 780 explored rows in training | 122.2 → 125.8 ([-20.5, +8.4]) | 0.811 → 0.810 | 0.51% → 2.48% | 0.529 → 0.775 ([+0.177, +0.325]); T3 0.28 → 0.69, T5 0.80 → 0.87 | rejected (cost, HN FPR) | rejected (cost, HN FPR) |
| uniform_noisy, 5% analyst error | 2,000 analyst | 270.8 → 272.2 ([-18.0, +16.4]) | 0.707 → 0.706 | 0.51% → 1.27% | 0.529 → 0.788 ([+0.177, +0.349]) | rejected (cost, HN FPR) | rejected (cost, HN FPR) |

**Reading.** In both runs the loop learned the new patterns: recall on unseen typologies rose by 18 to 35 points, with the whole CI above zero. It also roughly quadrupled false positives on hard negatives (8 to 39 of 1,570) and widened the cost CI past the -5% margin, so the gate correctly held the candidate back. A plausible but untested explanation is that T3 (a reshipping drop) looks like the 3PL hard negatives.

## Training changes after run 4 (realistic run; written before the reruns)

The gate and eval sets are unchanged. Only training changed.

### Diagnosis of run 4
v1 against candidate v3, on the 1,487 hard negatives in the validation window:
- **Stopped bookings rose from 7 to 37.** 25 of the 31 new false positives are `HN_new_channel`: 0 of 63 stopped by v1, 25 of 63 by the candidate. `HN_3pl` was unaffected (0 of 61). `hn_billing_change` went from 3 to 7 and `new_login` from 2 to 6.
- **The cause is `login_device_age_days`.** On the new false positives, its LightGBM `pred_contrib` adds 1.79 log-odds over v1 (median device age 9.4 days, against 200 for all hard negatives). Account age adds +0.63 (payer-origin pair age) and +0.43 (tenure).
- **The new pattern T5 has the same profile as `HN_new_channel`.** T5 has median login device age 5.8 days and tenure 390 days; `HN_new_channel` has 13.3 days and 240 days. They mostly differ on `channel_code` (0 against 1). Learning T5 therefore pulls new-channel hard negatives up with it. My earlier T3/3PL guess was wrong.
- **Weights.**
  - Feedback carries 27.7% of total weight (16,047 rows against 41,823 reference rows).
  - All 780 explored rows have propensity 0.05, so self-normalising gives each weight 1.0, and the clip never binds.
  - New-pattern positives carry 64 rows of weight (T3 36, T5 28).
  - The feedback holds 131 hard negatives, all labelled legit, plus 3,251 hard negatives in the reference window, all kept.

### Fixes, each rerun on the same realistic stream (seed 1, 5% analyst error)
- **(a)** Cap feedback at 30% of total sample weight (`max_feedback_share: 0.3`) and lower the propensity clip from 20 to 5. Expected to be close to a no-op here: the share is already 27.7% and all propensities are equal.
- **(b)** Keep all hard negatives when the train window is subsampled, and give hard-negative feedback its natural weight. This is already true without subsampling, so it is a no-op in these runs. It is kept as a safeguard.
- **(c)** Stronger regularisation (`min_child_samples` 20 → 100, `lambda_l2` 1 → 10) on top of (a) and (b). No features are removed.

**Reruns.** The stream is deterministic given v1's decisions, so it is replayed once and each configuration is retrained on the same labels. A baseline retrain with no fixes checks that run 4 is reproduced.

### Results of the reruns (gate v2 unchanged)

Every run uses the same realistic stream: 22,925 labels, 363 from analysts and 22,562 from outcomes. In every run, v1 has cost 122.2 BRL per 1k, PR-AUC 0.811, hard-negative FPR 0.51% and new-pattern recall 0.529.

| Run | Training change | Feedback weight share | Cost per 1k cand (improvement CI) | PR-AUC cand | HN FPR cand | HN_new_channel stopped (of 63) | New-pattern recall cand (diff CI) | Gate v2 |
|---|---|---|---|---|---|---|---|---|
| 4 / R0 | none (reproduces run 4 exactly) | 27.7% | 125.8 ([-20.5, +8.4]) | 0.810 | 2.48% | 25 | 0.775 ([+0.18, +0.33]) | rejected: cost, HN FPR |
| R1 | (a) cap 30%, clip 5 | 27.7% (cap not binding) | identical to R0 | | | | | rejected |
| R2 | (a)+(b) | 27.7% (no subsampling, so (b) is a no-op) | identical to R0 | | | | | rejected |
| R3 | (a)+(b)+(c) `min_child_samples` 100, `lambda_l2` 10 | 27.7% | 120.6 ([-15.0, +15.7]) | 0.817 | 1.46% | 7 | 0.708 ([+0.12, +0.24]) | rejected: cost CI lower bound -15.0 < -6.1; HN FPR +0.95 points > +0.5 |

**Reading.** Regularisation (c) is the only change that moved anything. It removed most of the HN_new_channel false positives (25 to 7, with `login_device_age_days` dropping from +1.79 to +0.80 log-odds over v1). It also improved every point estimate: cost, PR-AUC and ECE are all better than v1. It kept most of the new-pattern gain: T3 recall 0.28 to 0.58, T5 recall 0.80 to 0.84.

It is still rejected:
- hard-negative stops are 21 against 7, with `new_login` / `hn_new_state` rising by a few bookings each;
- the cost CI (about 30 BRL wide) is still wider than the -5% margin allows.

No configuration passed, so the default retrain configuration is unchanged. All options stay available in `config/app.yaml` (`max_feedback_share`, `ipw_clip`, `lgb_params`).

## Attempt 5 (pre-registered, written 2026-10-04 before the run)

**This is the 5th training configuration evaluated on this eval set** (after R0 = run 4, R1, R2 and R3). The eval set, the stream and gate v2 are unchanged. It will be run once. Whatever the result, it is added below, and no 6th configuration will be tried on this eval set.

### How the change was chosen (training data only)
The choice used only the training data of the real retrain: the seed-0 train window and the earliest 70% of the run-4/R0–R3 feedback (16,047 labels). The calibration window, the latest 30% of feedback and the new-pattern pool (seeds 1–9) were not loaded. Script: `inner_split.py` in the session scratchpad (not committed), output kept beside it.

**Inner split, by time, inside the training data:**
- inner-train: train-window rows before 2017-12-23 (33,458) plus the earliest 70% of the training feedback (11,232);
- inner-eval: train-window rows from 2017-12-23 on (8,365) plus the latest 30% of the training feedback (4,815).

**Inner baseline.** "inner-v1" is the default recipe trained on the inner-train reference rows only. Platt maps are fitted on the inner-eval reference rows, as the real retrain fits them on the calibration window.

**Selection rule, fixed before the inner runs:** among the new configurations, take the one that passes the most gate-v2 checks against inner-v1 on inner-eval. Break ties first by lower hard-negative FPR, then by higher new-pattern recall.

| Config (inner split, training data only) | PR-AUC | Cost per 1k (improvement CI vs inner-v1) | HN FPR | HN_new_channel stopped (of 106) | Inner gate checks passed |
|---|---|---|---|---|---|
| inner-v1 | 0.889 | 93.37 | 0.13% | 1 | |
| R0 (reference) | 0.931 | 63.76 ([-2.74, +74.09]) | 0.13% | 1 | 5/6 |
| R3 (reference) | 0.931 | 73.87 ([-6.08, +63.49]) | 0.26% | 2 | 4/6 |
| N1: R3 + legit rows with login device age < 30 days weighted ×3 | 0.939 | 66.61 ([-15.03, +77.22]) | 0.26% | 1 | 4/6 |
| **N2: R3 + `login_device_age_days` may interact only with `channel_code`** | 0.924 | 67.63 ([-6.75, +69.10]) | 0.00% | 0 | 4/6 |
| N3: R3 + warm start from inner-v1 (+100 rounds) | 0.853 | 128.80 ([-68.93, +13.53]) | 0.92% | 7 | 2/6 |
| N4: N3 + young-device legit ×3 | 0.861 | 135.11 ([-77.83, +6.95]) | 0.52% | 4 | 3/6 |

N1 uses a feature and the label, not the injected typology, because a real deployment would not know which bookings are "hard negatives".

**Selected: N2.** N1, N2 and R3 each pass 4 of 6 checks, and N2 has the lowest hard-negative FPR.

**Limits of this evidence.** It does not show that N2 will pass:
- the inner split did not reproduce the run-4 failure: R0 stops only 1 of 106 `HN_new_channel` bookings there, against 25 of 63 in the real eval set;
- inner-eval holds only 7 new-pattern frauds (all T5), so no configuration showed a measurable new-pattern gain;
- on the inner split, every new configuration fails the cost non-inferiority check (CI too wide) and the superiority check.

N2 is chosen because it is the one change aimed directly at the diagnosed mechanism. T5 and `HN_new_channel` look alike except for `channel_code`. With this constraint, a tree can use device age only together with channel, so "young device" cannot raise the risk of new-channel hard negatives the way it does for T5.

### Exact configuration
- `max_feedback_share: 0.3`, `ipw_clip: 5`, `lgb_params: {min_child_samples: 100, lambda_l2: 10}` (= R3).
- Plus `interaction_limits: {login_device_age_days: [channel_code]}`, which becomes LightGBM `interaction_constraints = [[login_device_age_days, channel_code], [every feature except login_device_age_days]]` (`fraudshield/learning/retrain.py`, `interaction_constraints`).
- Everything else as R0–R3: full train window (no subsampling), B = 200, seed 0, gate v2 (`noninferiority`) with the same thresholds.
- The same realistic stream: the 22,925 labels of the R0–R3 runs (seed 1, 5% analyst error, Laya stand-in, 5% exploration), replayed from their saved label file, not regenerated. The registry is in a scratch directory, never `artifacts/models/registry.json`.
- CPU only.

**If it passes:** it becomes the default retrain configuration only if the demo still works. Otherwise it stays an option in `config/app.yaml`. **If it fails:** it is reported as failed, and the default stays as it is.

### Result of attempt 5 (run once, 2026-10-04; gate v2 unchanged)

**Rejected.** It failed one check, cost non-inferiority, and passed the other five.

**Deviation found during the run.** v1's numbers did not reproduce the R0–R3 baseline (122.2 / 0.811 / 0.51% / 0.529). The reason is that the model and the data changed after the reruns of 2026-10-02:
- `artifacts/models/B2_F.lgb` (v1) was retrained in commit 4220290 ("Better T2/T3/T6 detection");
- the seed feature tables were rebuilt on 2026-10-03 and now have 71 features.

The replayed stream's feature snapshots were taken with the old v1 and lack 13 of the 71 features (`acct_hv_share`, `acct_far_share`, `acct_mean_dist`, `weight_z`, `dims_z`, `value_z`, `payoff_z`, `cost_vs_median`, `hour_pct`, `svc_exp_pct`, `base_consignees_l10`, `base_origins_l10`, `base_senders_l10`). Those features are therefore missing (NaN) in all 16,047 training feedback rows and all 6,878 eval feedback rows, for both models. The labels themselves are identical: 22,925, of which 363 are analyst and 22,562 are outcome labels. Which bookings got analyst labels and which got outcome labels was still decided by the old v1.

So attempt 5 is compared against **today's v1** on today's eval set. The cells are not directly comparable with the R0–R3 rows above.

| Run | Training change | Feedback weight share | Cost per 1k cand (improvement CI) | PR-AUC cand | HN FPR cand | HN_new_channel stopped (of 63) | New-pattern recall cand (diff CI) | Gate v2 |
|---|---|---|---|---|---|---|---|---|
| v1 today (baseline for this row) | none | n/a | 119.69 | 0.807 | 0.32% (5 of 1,570) | 0 | 0.535 (T3 0.29, T5 0.80) | n/a |
| Attempt 5 | R3 + `login_device_age_days` interacts only with `channel_code` | 27.7% | 115.32 ([-9.61, +21.41]) | 0.830 | 0.64% (10 of 1,570) | 0 | 0.577 ([+0.016, +0.071]); T3 0.38, T5 0.79 | rejected: cost CI lower bound -9.61 < -5.98 (5% of 119.69) |

Other measurements:
- ECE 0.0011 → 0.0008.
- PR-AUC difference CI [-0.0033, +0.0504].
- Eval set: 25,089 bookings with 182 frauds in 1,982 clusters.
- New-pattern set: 792 rows in 54 campaigns, 68 rows excluded.
- Hard negatives in the calibration window: 4 stopped by v1, 10 by the candidate (of 1,487).
- The retrain took 11.6 s on CPU. Nothing was subsampled.

**Reading.**
- The hard-negative problem did not appear. `HN_new_channel` stops stayed at 0 of 63, and hard-negative FPR rose by only +0.32 points, inside the +0.5 limit. The best earlier configuration, R3, had 7 of 63 and +0.95 points against the old v1. This cannot be credited to the interaction limit alone. v1 and the features changed too, and R3 was not rerun on today's v1, so that this stays a single run.
- Every point estimate is better than v1's: cost, PR-AUC and ECE.
- Superiority passed on new-pattern recall, with a CI entirely above 0. The gain is much smaller than in R0/R3: about +4 points, mostly on T3. The likely cause is that the 13 missing features in the feedback rows weaken what can be learned from feedback, but this is not tested.
- The candidate still fails, as every earlier configuration did, because the cost CI is too wide (about 31 BRL per 1k against a 6 BRL margin). The wide CI on about 180 eval frauds, not the candidate's point cost, is what blocks deployment.

The default retrain configuration is unchanged. The option stays available as `interaction_limits` in `config/app.yaml`, off by default. No 6th configuration will be tried on this eval set. A fair rerun would need a fresh stream with today's v1 and features, which would be a new eval set with its own pre-registration.
