# Laya v2 results: Laya decides, LightGBM testifies

Design and the pre-registered win condition: `docs/LAYA_V2.md` (written before training). Same test rows as v1 (`results_laya.md`): seed-0 real Olist legit (5,000 sampled + all real hard negatives, weighted) plus the injected rows of 10 seeds. T3 and T5 never appear in training or calibration.

Training: 11833 bookings, 3833 distinct fraud from 10 seeds (v1: 386), LightGBM line withheld on 29% of states, out-of-fold LightGBM risk on training states, questions misuse, foreign_senders, payoff_max, action.

## Verdict (pre-registered)

- W1, no loss in ranking (M2 PR-AUC >= B2 - 0.02): **not met** (0.356 vs 0.777)
- W2, adds something (95% CI of M2 - B2 above 0): T3_recall_fpr1 not met, T5_recall_fpr1 not met, cost_saved not met
- **Decision: LightGBM keeps deciding (W1 not met)**

## Ranking (mean ± sd over 10 seeds; top 1% of bookings per day)

| System | PR-AUC | Prec@1%/day | Rec@1%/day | FPR legit (%) | FPR new_state (%) | FPR new_seller (%) | FPR billing_change (%) | FPR injected (%) | FPR multi_account_consignee (%) |
|---|---|---|---|---|---|---|---|---|---|
| B2 LightGBM (current decider) | 0.777 ± 0.031 | 0.507 ± 0.050 | 0.452 ± 0.036 | 0.573 ± 0.063 | 0.652 ± 0.214 | 1.793 ± 0.346 | 0.301 ± 0.139 | 4.830 ± 2.103 | 0.135 ± 0.285 |
| S0 3-input stack (control) | 0.782 ± 0.028 | 0.510 ± 0.050 | 0.454 ± 0.035 | 0.570 ± 0.065 | 0.624 ± 0.269 | 1.832 ± 0.363 | 0.278 ± 0.141 | 4.293 ± 1.658 | 0.270 ± 0.349 |
| M1 Laya v1 | 0.265 ± 0.032 | 0.187 ± 0.021 | 0.189 ± 0.017 | 1.077 ± 0.048 | 0.553 ± 0.045 | 2.848 ± 0.079 | 0.546 ± 0.035 | 3.641 ± 1.468 | 0.676 ± 0.000 |
| M2 Laya v2 decider | 0.356 ± 0.021 | 0.205 ± 0.022 | 0.203 ± 0.015 | 1.031 ± 0.047 | 0.766 ± 0.099 | 2.071 ± 0.037 | 0.200 ± 0.102 | 2.701 ± 0.967 | 0.743 ± 0.214 |
| M2-solo (LightGBM line withheld) | 0.121 ± 0.017 | 0.084 ± 0.016 | 0.100 ± 0.016 | 1.436 ± 0.045 | 0.894 ± 0.096 | 0.728 ± 0.046 | 0.958 ± 0.058 | 1.494 ± 0.889 | 0.676 ± 0.000 |

PR-AUC per typology (held-out marked *):

| System | T1 | T2 | T3* | T4 | T5* | T6 | T7 |
|---|---|---|---|---|---|---|---|
| B2 LightGBM (current decider) | 1.000 ± 0.000 | 0.676 ± 0.179 | 0.308 ± 0.197 | 0.724 ± 0.116 | 0.807 ± 0.035 | 0.016 ± 0.024 | 0.999 ± 0.003 |
| S0 3-input stack (control) | 1.000 ± 0.000 | 0.669 ± 0.181 | 0.314 ± 0.195 | 0.717 ± 0.118 | 0.806 ± 0.034 | 0.020 ± 0.027 | 0.999 ± 0.003 |
| M1 Laya v1 | 0.196 ± 0.023 | 0.025 ± 0.012 | 0.015 ± 0.012 | 0.060 ± 0.024 | 0.088 ± 0.014 | 0.001 ± 0.000 | 0.108 ± 0.016 |
| M2 Laya v2 decider | 0.207 ± 0.023 | 0.097 ± 0.012 | 0.025 ± 0.020 | 0.122 ± 0.019 | 0.091 ± 0.013 | 0.003 ± 0.004 | 0.117 ± 0.009 |
| M2-solo (LightGBM line withheld) | 0.058 ± 0.010 | 0.017 ± 0.005 | 0.009 ± 0.006 | 0.028 ± 0.009 | 0.024 ± 0.005 | 0.001 ± 0.000 | 0.029 ± 0.006 |

## Held-out fraud types at equal legit friction (pooled seeds, 95% campaign-bootstrap CI)

| Type | Legit FPR | n / campaigns | B2 | M2 | M2-solo | S0 | M2 - B2 CI |
|---|---|---|---|---|---|---|---|
| T3 | 0.5% | 451 / 20 | 0.304 | 0.333 | 0.344 | 0.313 | [0.014, 0.050] |
| T3 | 1.0% | 451 / 20 | 0.333 | 0.333 | 0.344 | 0.361 | [-0.024, 0.023] |
| T5 | 0.5% | 409 / 40 | 0.807 | 0.809 | 0.765 | 0.807 | [0.000, 0.007] |
| T5 | 1.0% | 409 / 40 | 0.814 | 0.809 | 0.765 | 0.814 | [-0.012, 0.000] |

## Decisions: realized cost per booking (BRL, cost rule of the live pipeline; mean ± sd over seeds)

| Decider | BRL / booking | fraud leak | legit friction | actions taken (pooled rows) |
|---|---|---|---|---|
| B2 + cost rule | 0.234 ± 0.023 | 0.210 | 0.024 | allow 12065, hold 1803, owner_confirm 579, allow_scan_gated 171 |
| M2 probabilities + cost rule | 0.320 ± 0.021 | 0.244 | 0.076 | allow 11846, owner_confirm 2704, allow_scan_gated 50, hold 18 |
| M2 action head (Laya alone) | 4.662 ± 0.011 | 0.135 | 4.527 | owner_confirm 14618 |
| M2 action head + cost auditor | 0.326 ± 0.021 | 0.243 | 0.083 | allow 11836, owner_confirm 2752, allow_scan_gated 29, hold 1 |

Cost saved by M2 vs B2 (same cost rule): -0.086 BRL per booking, 95% CI [-0.103, -0.070].
The cost auditor overruled Laya's own action on 97.57% of bookings (margin 2.0 BRL).

## Zero-shot `drop_consignee` on T3 (never trained), ROC-AUC

M2 0.644, solo 0.617, B2_general 0.878 (v1 fine-tuned: 0.603, stock: 0.488)

## Calibration

`cal-20261004-0cd1e9e4`: Platt misuse a=0.219 b=-3.457; conformal lambda_allow 0.0000 from 56 calibration fraud.
