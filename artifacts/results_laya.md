# Laya fine-tune results (FraudShield)

Real runs. Training GPU: Tesla T4, precision fp16. Test window = seed-0 real Olist rows (legit, shared by every seed) plus the injected rows of 10 injection seeds. Legit rows are a 5,000-row random sample plus every real hard negative, weighted back to the full window (22,925 bookings per seed view, prevalence about 1.2%). All three systems score exactly the same rows. T3 and T5 are held out (never in training or calibration). B2 = the data agent's LightGBM + change/graph features, tier F, seed-0 model, Platt-calibrated.

## Training

- Data: 5958 training bookings (5186 unique) from the seed-0 train window: all 386 eligible train fraud bookings repeated 3x (19.4% fraud), 1,800 hard negatives, 3,000 random legit; T3/T5 excluded. 29790 items per epoch (questions: misuse, foreign_senders, payoff_max, risk_level, action). Only 386 train fraud bookings exist in the seed-0 train window, so the DESIGN target of 1,200 distinct fraud bookings was met by repetition, not by new data.
- Token budget: max state 333 tokens over all 64200 states, max sequence 438 of 512; 0 items dropped by the marker check, 0 truncated.
- Pilot (200 micro-steps each): K=12 micro=16: 8.38 items/s, peak VRAM 4.78 GB allocated / 4.98 GB reserved, 174M trainable, est 0.99 h/epoch.
- Run: top 12 of 28 encoder layers + decision head trained (embeddings and lower layers frozen, in bf16), bf16 autocast, gradient checkpointing, micro-batch 16 x accumulation 2 = 32, 2 epochs, LR encoder 2.5e-5 / head 1e-4 cosine to 1e-6, sigma 0.4 -> 0.1, group 4, misuse weight 2.0. Wall time 113.7 min. Run config: planned.
- Objective: notebook recipe (log + spherical proper reward, RPS for risk_level, group-mean REINFORCE on Gaussian logit noise, plus soft CE) on probability heads; action head reward -(q . cost)/C with C = 34.9 BRL and CE weight 0.

## Main table (mean ± sd over 10 injection seeds; top 1% of bookings per day)

| System | PR-AUC | Prec@1%/day | Rec@1%/day | FPR legit (%) | FPR new_state (%) | FPR new_seller (%) | FPR billing_change (%) | FPR injected (%) | FPR multi_account_consignee (%) |
|---|---|---|---|---|---|---|---|---|---|
| B2 GBM+graph (F) | 0.777 ± 0.031 | 0.507 ± 0.050 | 0.452 ± 0.036 | 0.573 ± 0.063 | 0.652 ± 0.214 | 1.793 ± 0.346 | 0.301 ± 0.139 | 4.830 ± 2.103 | 0.135 ± 0.285 |
| B3 Laya stock zero-shot | 0.018 ± 0.003 | 0.023 ± 0.007 | 0.029 ± 0.008 | 1.607 ± 0.026 | 0.766 ± 0.120 | 0.427 ± 0.033 | 1.069 ± 0.058 | 1.258 ± 0.634 | 0.676 ± 0.000 |
| M1 Laya fine-tuned | 0.265 ± 0.032 | 0.187 ± 0.021 | 0.189 ± 0.017 | 1.077 ± 0.048 | 0.553 ± 0.045 | 2.848 ± 0.079 | 0.546 ± 0.035 | 3.641 ± 1.468 | 0.676 ± 0.000 |

PR-AUC per typology (typology vs all legit; held-out marked *):

| System | T1 | T2 | T3* | T4 | T5* | T6 | T7 |
|---|---|---|---|---|---|---|---|
| B2 GBM+graph (F) | 1.000 ± 0.000 | 0.676 ± 0.179 | 0.308 ± 0.197 | 0.724 ± 0.116 | 0.807 ± 0.035 | 0.016 ± 0.024 | 0.999 ± 0.003 |
| B3 Laya stock zero-shot | 0.006 ± 0.002 | 0.002 ± 0.000 | 0.003 ± 0.001 | 0.003 ± 0.001 | 0.007 ± 0.006 | 0.002 ± 0.001 | 0.002 ± 0.001 |
| M1 Laya fine-tuned | 0.196 ± 0.023 | 0.025 ± 0.012 | 0.015 ± 0.012 | 0.060 ± 0.024 | 0.088 ± 0.014 | 0.001 ± 0.000 | 0.108 ± 0.016 |

Recall at top 1%/day per typology:

| System | T1 | T2 | T3* | T4 | T5* | T6 | T7 |
|---|---|---|---|---|---|---|---|
| B2 GBM+graph (F) | 0.694 ± 0.112 | 0.422 ± 0.132 | 0.209 ± 0.123 | 0.530 ± 0.104 | 0.388 ± 0.093 | 0.060 ± 0.070 | 0.672 ± 0.117 |
| B3 Laya stock zero-shot | 0.031 ± 0.026 | 0.022 ± 0.032 | 0.027 ± 0.023 | 0.035 ± 0.035 | 0.044 ± 0.033 | 0.025 ± 0.038 | 0.011 ± 0.014 |
| M1 Laya fine-tuned | 0.283 ± 0.074 | 0.155 ± 0.107 | 0.093 ± 0.103 | 0.185 ± 0.055 | 0.152 ± 0.091 | 0.000 ± 0.000 | 0.390 ± 0.100 |

## Headline (a): held-out T3 and T5 at equal legit friction (pooled seeds 0-9, 95% bootstrap CI by campaign)

| Typology | Legit FPR | n bookings / campaigns | B2 GBM recall | M1 misuse recall | B3 stock recall | M1 - B2 |
|---|---|---|---|---|---|---|
| T3 | 0.5% | 451 / 20 | 0.304 [0.176, 0.429] | 0.222 [0.128, 0.318] | 0.007 [0.000, 0.016] | [-0.123, -0.047] |
| T3 | 1.0% | 451 / 20 | 0.333 [0.205, 0.458] | 0.222 [0.128, 0.318] | 0.011 [0.004, 0.026] | [-0.156, -0.064] |
| T5 | 0.5% | 409 / 40 | 0.807 [0.783, 0.832] | 0.780 [0.752, 0.809] | 0.027 [0.002, 0.067] | [-0.054, -0.007] |
| T5 | 1.0% | 409 / 40 | 0.814 [0.787, 0.841] | 0.780 [0.752, 0.809] | 0.046 [0.010, 0.093] | [-0.061, -0.012] |
| T3 PR-AUC | all | | 0.382 [0.235, 0.514] | 0.069 [0.036, 0.112] | 0.021 [0.017, 0.028] | |
| T5 PR-AUC | all | | 0.836 [0.814, 0.857] | 0.375 [0.337, 0.412] | 0.041 [0.028, 0.073] | |

## Headline (b): zero-shot `drop_consignee` on T3 bookings (451 bookings, 20 campaigns) vs legit

| System | ROC-AUC | PR-AUC | Recall @ 1% legit FPR | Recall @ 5% legit FPR | FPR on multi-account-consignee legit at the 1% threshold |
|---|---|---|---|---|---|
| B3 stock Laya, drop_consignee | 0.488 [0.437, 0.533] | 0.018 | 0.004 [0.000, 0.011] | 0.049 | 0.000 |
| M1 fine-tuned Laya, drop_consignee (never trained) | 0.603 [0.552, 0.661] | 0.067 | 0.173 [0.099, 0.249] | 1.000 | 0.007 |
| B2 GBM general fraud score (reference) | 0.878 [0.816, 0.928] | 0.382 | 0.333 [0.200, 0.454] | 0.503 | n/a |

M1 - B3 ROC-AUC difference, 95% CI: [0.068, 0.165]

## Calibration (test window, seed-0 view, weighted to true prevalence; ECE = 15 equal-mass bins)

| Question | rate | Stock ECE | Stock Brier | M1 raw ECE | M1 raw Brier | M1 calibrated ECE | M1 calibrated Brier | mean p raw -> cal |
|---|---|---|---|---|---|---|---|---|
| misuse | 0.0120 | 0.6008 | 0.3758 | 0.0120 | 0.0208 | 0.0040 | 0.0097 | 0.0240 -> 0.0126 |
| foreign_senders | 0.0064 | 0.5524 | 0.3204 | 0.0008 | 0.0013 | 0.0010 | 0.0012 | 0.0072 -> 0.0074 |
| payoff_max | 0.0053 | 0.4826 | 0.2480 | 0.0088 | 0.0134 | 0.0122 | 0.0124 | 0.0141 -> 0.0175 |

calibration.json `cal-20261003-21e12c1c` (model_revision 77e2caf7e3c7): temperatures misuse 3.025, foreign_senders 1.713, payoff_max 2.475, drop_consignee 1.000; Platt misuse a=0.657 b=-3.017; prevalence 0.0092. Fitted on 3432 calibration bookings (111 fraud, split by account); conformal lambda_allow = 0.0000 from 56 held-out calibration fraud (alpha 0.05).
- Test check of the allow guard: allowed-fraud rate 0.000 over 2954 test fraud (T3* 0.000, T5* 0.000, T6 0.000); share of legit bookings for which allow is ruled out: 1.000. The guarantee covers exchangeable fraud only; campaigns are clustered and T3/T5 are new typologies, so rates above 0.05 there are a real miss of the guarantee, not noise.

## Verdict

Win condition (a) not met; condition (b) met. On held-out T3 the GBM catches more than the fine-tuned Laya (recall 0.333 vs 0.222, difference CI [-0.156, -0.064]); on held-out T5 the GBM catches more than the fine-tuned Laya (recall 0.814 vs 0.780, difference CI [-0.061, -0.012]); and the zero-shot drop_consignee question on the fine-tuned model beats stock Laya on T3 (ROC-AUC 0.603 vs 0.488, difference CI [0.068, 0.165]). Held-out T3 has few campaigns, so its intervals are wide.
