# Phase 3: RL design (FraudShield, Top-1 idea)

Date 2026-10-02. Inputs: phase2_ideation.md, phase1_rl_lit.md. Runnable sketch with 5 passing unit tests: `scratchpad/rl_sketch/rl_core.py`, `test_rl_core.py`, `thresholds.py`.

Overall position: RL is NOT the per-booking decider. Per-booking action = argmin expected cost on calibrated Laya probabilities (Elkan, IJCAI 2001). RL earns a place in three narrow jobs: (1) logged exploration + off-policy evaluation (OPE) to fix selective labels and gate any policy change, (2) an adaptive attacker that stress-tests the detector and produces hard positives, (3) feeding delayed, propensity-weighted outcomes back into Laya fine-tuning. The offline contextual bandit candidate (A) is kept as a challenger that must pass a DR lower-bound gate; we expect it to tie the Bayes rule unless our assumed cost parameters are wrong for some segment. The review queue (B) is not RL.

Label: PROVEN = from cited literature or our runnable code. ASSUMPTION = number we chose; must be shown in a sensitivity table. EXPECTATION = what we think the experiment will show, not yet run.

---

## A. Action policy

### A1. Cost matrix (BRL, Olist currency)
F = freight value of this booking (real, Olist `freight_value`; injected fraud skews to high F per insight I7). All other numbers are ASSUMPTIONS, stored in one config dict, varied in a sensitivity table.

| param | value | meaning |
|---|---|---|
| c_dispute | 30 | ops + dispute handling when fraud passes |
| margin | 0.30 | carrier contribution margin on freight |
| goodwill | 15 | churn proxy when a legit booking is lost |
| lost = margin*F + goodwill | | cost of losing a legit booking |
| leak = F + c_dispute | | cost of fraud passing (victim disputes the bill, carrier writes off F) |
| scan: c_scan 0.5, q_scan 0.40, c_intercept 8, ab_scan 0.02 | | scan-gated catches 40% of fraud at first scan (dims/weight/induction mismatch) |
| owner_confirm: c_msg 0.2, fatigue 2, p_unreach 0.30, ab_wait 0.30, p_spoof 0.10 | | owner contact must be on file >30 days (takeover often changes contact); spoof = fraudster controls the channel |
| review: 1.00 BRL/analyst-minute, 6 min, ab_review 0.05, analyst_miss 0.10 | | |
| hold: c_hold 3, ab_hold 0.25 | | hold until invoice-challenge (insight I1) |
| block: lost + 5 support | | |

C[a, y] (y=0 legit, y=1 fraud), example F = 45 BRL (computed by `cost_matrix(45)`):

| action | legit | fraud |
|---|---|---|
| allow | 0.00 | 75.00 |
| allow_scan_gated | 1.07 | 48.70 |
| owner_confirm | 4.77 | 6.35 |
| review | 7.42 | 13.50 |
| hold | 10.12 | 3.00 |
| block | 33.50 | 0.00 |

Bayes-rule regions from `thresholds.py` (p = calibrated P(fraud)):
- F=45, owner contact on file: allow <0.039, scan-gated 0.039-0.08, owner_confirm 0.08-0.616, hold 0.616-0.887, block >0.887.
- F=45, no owner contact: allow <0.039, scan-gated 0.039-0.153, review 0.153-0.205, hold 0.205-0.887, block >0.887.
- F=120: allow <0.026, block >0.931; owner_confirm from 0.06. F=15: allow <0.059.
Findings from the cost model itself (PROVEN given the assumptions): high-payoff bookings get friction at lower probabilities (the "make fraud unprofitable" rule is just the cost matrix); owner_confirm dominates review whenever a verified contact exists, so review is mostly the fallback and the label source; with these numbers hold beats block over a wide band. Show these as a sensitivity heat map, because a judge will ask "where do 0.30 and 15 BRL come from".

Note on "probs": Laya returns several calibrated answers (Noul "booking authorized by account owner", Score "counterparty shift", Choice typology). The policy uses one calibrated p_fraud (from the Noul, temperature-refit), plus F and contact-on-file flag. Other Laya outputs enter the bandit context, not the Bayes rule.

### A2. Baseline and candidate
- Baseline pi_B: `bayes_action(p_fraud, F, allowed)`. Deterministic.
- Candidate pi_C: offline contextual bandit. Context x = [p_fraud, other Laya probs, Laya confidence, F (payoff estimate), account tenure, contact-on-file, owner past response rate, segment]. Learner: DR pseudo-rewards (Dudik et al. ICML 2011) Gamma[i,a] = q_hat(x_i,a) + 1{a=A_i}(r_i - q_hat(x_i,A_i))/pi_0(A_i|x_i); fit one LightGBM regressor per action on Gamma (cross-fitted), pi_C(x) = argmax. Alternative: CRM/POEM (Swaminathan & Joachims, JMLR 2015) if time.
- What the candidate can learn that the baseline cannot: the cost matrix entries that are really context-dependent and unknown (owner reachability by tenure, abandonment under hold by segment, scan catch rate by typology). If the assumed matrix is right and p is calibrated, the bandit cannot beat the Bayes rule; it can only add variance.

### A3. Logging policy (what runs in "production" in the simulator)
pi_0(a|x) = (1-eps) * softmax(-EC(a|x)/tau) + eps/|A(x)|, eps=0.05, tau=2 BRL. Propensity of the chosen action is logged with model version. Safety constraints:
- A(x) = allowed set: no owner_confirm without verified contact; `allow` gets zero mass when p_fraud > 0.30 (never explored to full exposure; explore "down" to hold/scan-gated instead, which still reveal labels).
- Daily exposure budget: sum over explored bookings of (EC(explored) - EC(Bayes)) <= B BRL/day; when exhausted, eps -> 0 and log that pscore came from the deterministic policy (rows excluded from OPE of policies that differ there).
- Rationale: Stripe randomizes ~5% of would-be blocks and recommends propensities concentrated near the threshold (Manapat, QCon 2018); randomization is required for consistent estimates (Revelas et al. arXiv 2509.18739; Kilbertus et al. AISTATS 2020).

### A4. What each action reveals (censoring)
| action | label revealed | delay | reward r = -realized cost |
|---|---|---|---|
| allow | dispute / no dispute | days to weeks (simulator: drawn from delay distribution) | -(leak) if disputed else 0; pending until window closes |
| allow_scan_gated | measured weight/dims/induction mismatch (weak signal), then dispute | hours, then delayed | -(c_scan + intercept or leak or abandonment) |
| owner_confirm | owner "yes, me" (strong legit) / "not me" (strong fraud) / no answer (censored) | minutes | answered: immediate; unreachable: falls back to hold |
| review | analyst label (noisy, ~10% miss) | hours | -(review cost + residual leak/abandon) |
| hold | whether shipper passes invoice challenge; abandonment observed | hours-days | partial |
| block | nothing | never | censored; known only via exploration of neighbours |

Reward is filled only when the episode closes (label window W, simulator default 30 simulated days). Pending rows are excluded from OPE or imputed with the delay model of section D; we report both.

### A5. OPE and deployment gate
- Estimators: IPS, SNIPS, DR (q_hat cross-fitted, 5 folds). obp classes `InverseProbabilityWeighting`, `SelfNormalizedInverseProbabilityWeighting`, `DoublyRobust`; each has `estimate_interval(..., alpha, n_bootstrap_samples)` (checked on zr-obp docs). Hand-rolled numpy versions in `rl_core.py` for the paired difference.
- Baseline is also off-policy (it is deterministic, the logger is soft), so evaluate both from the same logs and bootstrap the DIFFERENCE V(pi_C) - V(pi_B) paired (2,000 resamples).
- Report per estimator: value (BRL per 1,000 bookings), 95% CI, ESS = (sum w)^2 / sum w^2, max weight, share of rows where pi_C picks an action with logged propensity 0 (unsupported; must be 0).
- Gate (all must hold): DR 95% lower bound on the difference > 0; ESS >= 500; max weight <= 1/floor; no unsupported actions; per-segment non-inferiority on fraud loss (new accounts, high F, held-out typology) within a tolerance; SNIPS and DR agree in sign. Lower-bound gating follows High-Confidence OPE (Thomas, Theocharous, Ghavamzadeh, AAAI 2015).
- Simulator check: we know the true value of every policy (ground truth for injected fraud and simulated customer response), so we report OPE error = estimate - truth. This is the credibility proof that our OPE works.

### A6. Oscillation risk
Adyen (arXiv 2412.00569) saw policy generations oscillate when retrained on their own logs; this is the performative-prediction effect (Perdomo et al., ICML 2020: deployed predictions change the data they are retrained on). Mitigations:
1. Exploration floor never switches off near the boundary, so the logs never collapse to one action.
2. Train on pooled logs from all generations with each row's own logged propensity, not just the latest generation.
3. Trust region: pi_C may differ from pi_B on at most 10% of bookings per generation.
4. Every generation is re-evaluated on a fresh exploration slice; rollback if the generation-over-generation DR difference is negative.
5. Simulator experiment: run 5 generations with and without mitigations 1-3; plot policy value per generation. EXPECTATION: unmitigated loop shows a drop after it shrinks exploration in the block region.

### A7. Does RL beat the simpler method?
Experiment: simulator with two worlds. World 1: true costs = assumed costs. World 2: heterogeneous truth (e.g. owner reachability 0.9 for tenure > 1 year vs 0.4 for new accounts; abandonment under hold 2x for small sellers). Metric: true total cost per 1,000 bookings, fraud BRL leaked, legit friction rate.
EXPECTATION (honest): World 1, tie within CI (gate rejects, which is the correct outcome and we show it). World 2, a small gain concentrated in the misspecified segments. A hackathon-sized log (tens of thousands of bookings, ~1-3% fraud) may be too small for the DR lower bound to clear zero; if so we say "not deployed; baseline stays", which is still a good demo of the safety gate.

---

## B. Review queue under capacity: NOT RL

Each item i in review/hold has review time t_i (minutes, predicted from case complexity) and value v_i = expected loss prevented = p_i * (leak_i) * (1 - analyst_miss) - (1 - p_i) * (abandon cost from delay). Choose x_i in {0,1} to maximize sum v_i x_i s.t. sum t_i x_i <= analyst minutes this shift. Greedy by v_i / t_i is near-optimal for many small items (fractional knapsack bound); OR-Tools CP-SAT if we want exact. Items not reviewed fall back to their Bayes action without review (hold or owner_confirm), so capacity also feeds back into the action rule: the review row of C uses the shadow price of an analyst minute (lambda from the knapsack) instead of a fixed 1 BRL.

Why not RL: arrivals are roughly independent, decisions do not change future states much at our scale, and the literature's best capacity result is supervised + constrained optimization (DeCCaF, Alves et al. TMLR 2024). Adversarial RL prioritization (Tong et al. AAAI 2020) matters when attackers adapt to the queue; out of scope.
Keep 5% of review capacity randomly assigned among queued items, so analyst labels are not only from top scores (Revelas et al. 2025).
Experiment: on simulated days at 3 capacity levels, compare FIFO, sort-by-p, sort-by-v, greedy v/t. Metric: BRL prevented per analyst hour, precision@capacity. EXPECTATION: v/t beats sort-by-p mainly because it prefers high-F cases. No RL arm.

---

## C. Adaptive fraudster (stress test, the most defensible RL use)

Setting: attacker has taken over one Olist-derived seller account (typology T1/T4) for a takeover window of 30 simulated days, before the owner notices an invoice.
- Policy parameters theta (continuous, 5 dims), each a mutation operator applied to the attacker's desired bookings:
  1. rate_per_day: spread bookings over time.
  2. decoy_frac: share of bookings sent to the account's own historical consignees and lanes (camouflage; earns nothing).
  3. weight_cap_q: cap package weight at this quantile of the account's own history (lighter = lower payoff).
  4. express_match: match the account's historical service mix instead of choosing express.
  5. warmup_days: wait before the first fraudulent booking.
  The attacker cannot change features it does not control (account tenure, payer-origin link history, graph neighbours), following Lunghi et al. arXiv 2307.01390.
- Observation: black box. Only the action each booking got (attacker sees blocked/held/asked; it does not see scores). Matches the limited-knowledge setting of FRAUD-RLA (arXiv 2502.02290).
- Reward per episode: value extracted = sum of F over fraudulent (non-decoy) bookings that pass, until the account is burned (first block, hold failed, owner "not me", or analyst fraud label) or the window ends. Detection is not a separate penalty; burning the account ends the income stream, which is the real cost.
- Algorithm: cross-entropy method (Rubinstein, Methodology and Computing in Applied Probability 1:127-190, 1999): population 48, elite 20%, 8 episodes per theta over random victim accounts, 15 iterations (`attacker_cem` in sketch). Control arm: random search with the same query budget; a hand-written "mimic" heuristic (decoy 0.5, weight at median, service matched).
- Outputs:
  1. Evasion curve: x = attacker iterations (queries), y = evasion rate of elite campaigns and value extracted per stolen account.
  2. Payoff frontier: value per account vs evasion rate across all sampled theta. The economic argument: evasion is only reachable with lighter, slower, camouflaged campaigns, so value per stolen account falls. If value per account drops below the black-market label price times bookings (labels sell ~2-9 USD per Phase 1 notes), resale is no longer worth the takeover.
  3. Adversarial round: passed fraudulent bookings from the top 10% theta become hard positives (tagged `scenario_id=T_adv`, seed logged), Laya re-fine-tuned (or, if GPU time is short, head-only / GBM refit for the demo), temperature refit, attacker re-run from scratch. Report the new curve plus false-positive rate on the hard negatives (first new region, Black Friday 2017, new marketplace channel), because adversarial retraining can over-flag legitimate growth.
- Does RL beat simpler? Comparison is CEM vs random search at equal budget. EXPECTATION: in 5 dims CEM reaches a given evasion with fewer queries, but random search may be close. That is fine: the purpose is the stress test and the hard positives, not an RL win. We do not claim the attacker is a realistic fraudster; it is an upper-bound adversary in our own simulator.

---

## D. Delayed rewards -> Laya fine-tuning labels

Label sources and soft targets (Laya's training loss already includes soft CE on gold probability targets, so soft labels fit directly):
| source | target P(fraud) | when |
|---|---|---|
| owner "not me" | 0.98 | minutes |
| owner "yes, me" (verified old contact) | 0.03 (p_spoof residual) | minutes |
| dispute / chargeback | 1.0 | delayed |
| analyst label | analyst calibrated accuracy (e.g. 0.9 / 0.1), recalibrated against later disputes | hours |
| scan mismatch only | not a label; a feature for the next booking | hours |
| no dispute yet at age t | p S(t) / (p S(t) + 1 - p), S(t) = P(delay > t) | at retrain time |

The last row is the Chapelle (KDD 2014) idea: a not-yet-reported booking is not a negative; jointly model fraud probability and report-delay distribution (fit S(t) on confirmed cases, e.g. exponential or Weibull per typology). Simplest robust option for the hackathon: exclude bookings younger than the label window and use the soft target only as an ablation. Ktena et al. (RecSys 2019) compare loss corrections for delayed labels in continuous training; Dal Pozzolo et al. (TNNLS 2018) train separate models on fast investigator feedback and delayed labels and aggregate them; we copy that split (fast: owner/analyst; slow: disputes).

Selection bias correction: a booking has a label only if its action revealed one. P(labeled | x) = sum over revealing actions of pi_0(a|x) * P(reveal | a). Weight labeled rows by 1/P(labeled|x), clipped at 20, self-normalized. Laya's fine-tune API may not accept sample weights (not verified), so implement by weighted resampling of the training set. Temperature refit on a propensity-weighted held-out set; report weighted ECE.

Experiment (cleanest win available, because the simulator knows ground truth for every booking): train Laya on (a) naive labels (revealed only, unweighted, pending = negative) vs (b) propensity-weighted + delay-handled. Evaluate on ALL bookings with ground truth. Metrics: ECE overall and for p > 0.5 (the block/hold region that naive training never sees labels for), PR-AUC, true cost of the Bayes rule using each model. EXPECTATION: (b) is better calibrated in the high-score region; PR-AUC differences small. This is not RL, but it is the reason the exploration and propensity logging exist.

---

## Implementation sketch (in `scratchpad/rl_sketch/rl_core.py`, tests pass)
```python
ACTIONS = ["allow","allow_scan_gated","owner_confirm","review","hold","block"]
cost_matrix(F: float, prm: dict = DEFAULT_PARAMS) -> np.ndarray  # (6,2) BRL
expected_costs(p_fraud: float, C: np.ndarray, allowed: list|None = None) -> dict[str, float]
bayes_action(p_fraud: float, F: float, prm=DEFAULT_PARAMS, allowed=None) -> str
logging_policy(p_fraud, F, rng, prm=DEFAULT_PARAMS, allowed=None, tau=2.0, eps=0.05,
               no_explore_to_allow_above=0.30) -> tuple[str, float, np.ndarray]  # action, propensity, dist
ips(r, a, pscore, pi_e) -> float          # r = -realized cost, a int index, pi_e (n,6)
snips(r, a, pscore, pi_e) -> float
dr(r, a, pscore, pi_e, q_hat) -> float     # q_hat (n,6) cross-fitted
ess(w) -> float
paired_bootstrap_diff(est, r, a, pscore, pi_c, pi_b, q_hat=None, B=2000, alpha=0.05) -> (lo, hi)
dr_pseudo_rewards(r, a, pscore, q_hat) -> np.ndarray  # (n,6), targets for candidate policy
deployment_gate(lo_dr_diff, ess_val, max_w, floor, min_ess=500) -> bool
attacker_cem(run_campaign, n_iter=15, pop=48, elite_frac=0.2, episodes=8, seed=0) -> (history, theta_mu)
# run_campaign(theta, rng) -> (value_extracted_BRL, n_fraud_bookings, n_passed); wraps simulator + full scoring pipeline
```
Unit tests (5, passing): Bayes action at p=0 is allow and at p=1 is block; expected costs linear in p; logging propensities sum to 1, respect floor, and allow has zero mass above cap; IPS/SNIPS equal the on-policy mean when pi_e = pi_0, DR equals DM with a perfect model, ESS of equal weights = n; CEM runs on a toy env.
Still to write: simulator (customer response, owner reachability, delay draws), q_hat cross-fitting, obp wiring (`OffPolicyEvaluation` with `action_dist` shape (n, 6, 1)), attacker `run_campaign` calling the real scoring service in batch (laya-serve /v1/systemone/batch, max 64).

Build order (time-boxed): cost model + Bayes rule + logging (2h) -> simulator logs + OPE table with truth column (3h) -> attacker CEM + evasion curve (3h) -> adversarial retrain round (GPU-bound) -> candidate bandit + gate (only if time; the gate demo works even with a trivial candidate).

## 30-second demo moment
Screen: the evasion curve and payoff frontier side by side.
Script: "We let an attacker learn against FraudShield. It does learn: after 15 rounds about X% of its bookings get through. But look at the right: to get through it had to ship lighter parcels, slower, mixed with decoys that earn nothing. A stolen account went from R$Y to R$Z of free shipping. Then we fed its successful bookings back as training data, and the curve drops again. We are not claiming we stop every fraud; we make the stolen account not worth stealing." Backup 10 seconds: OPE table showing the bandit candidate did NOT pass the DR lower-bound gate, so the baseline stays, and the OPE estimate vs simulator truth column.
X, Y, Z are placeholders until the experiment runs.

## Sources (verified)
- Elkan, The Foundations of Cost-Sensitive Learning, IJCAI 2001 (Phase 1).
- Dudik, Langford, Li, Doubly Robust Policy Evaluation and Learning, ICML 2011, arXiv 1103.4601 (Phase 1).
- Swaminathan & Joachims, CRM, JMLR 16 (2015) (Phase 1).
- Saito et al., Open Bandit Dataset and Pipeline, NeurIPS 2021 D&B; obp estimator API: https://zr-obp.readthedocs.io/en/latest/_autosummary/obp.ope.estimators.html (checked today).
- Thomas, Theocharous, Ghavamzadeh, High-Confidence Off-Policy Evaluation, AAAI 2015, https://ojs.aaai.org/index.php/AAAI/article/view/9541 (new, verified).
- Perdomo, Zrnic, Mendler-Dunner, Hardt, Performative Prediction, ICML 2020, PMLR v119 pp. 7599-7609, https://proceedings.mlr.press/v119/perdomo20a.html (new, verified).
- Vangara & Egg (Adyen), arXiv 2412.00569 (Phase 1).
- Manapat (Stripe), QCon.ai 2018 talk (Phase 1, industry talk).
- Revelas, Boldea, Werker, arXiv 2509.18739; Kilbertus et al. AISTATS 2020 (Phase 1).
- Alves et al. DeCCaF, TMLR 2024, arXiv 2403.06906; Tong et al. AAAI 2020 (Phase 1).
- Lunghi et al. arXiv 2307.01390; FRAUD-RLA arXiv 2502.02290 (Phase 1).
- Rubinstein, The Cross-Entropy Method for Combinatorial and Continuous Optimization, Methodology and Computing in Applied Probability 1:127-190, 1999, https://link.springer.com/article/10.1023/A:1010091220143 (new, verified).
- Chapelle, KDD 2014; Dal Pozzolo et al. TNNLS 2018 (Phase 1).
- Ktena et al., Addressing delayed feedback for continuous training with neural networks in CTR prediction, RecSys 2019 (new, verified via Semantic Scholar/ResearchGate listing).
