# GBM baseline results (test window 2018-05-15..2018-08-31, 10 injection seeds)

Real run of scripts/train_gbm.py. Values are mean ± sd across seeds. Test prevalence 1.29% of bookings (injected fraud on real Olist histories; real rows assumed legit). Operating point for precision, recall and FPR: flag the top 1% of bookings per day. FPR legit = flagged legit / all legit bookings; hard-negative FPR = flagged / bookings in that slice. T3 and T5 are HELD OUT (never in train or calibration). Tier R = real columns only; tier F adds the synthetic billing/login layer.

## Headline

| System | Tier | PR-AUC | Prec@1%/day | Rec@1%/day | Campaign recall | FPR legit (%) |
|---|---|---|---|---|---|---|
| B0 | R | 0.168 ± 0.041 | 0.266 ± 0.034 | 0.214 ± 0.020 | 0.587 ± 0.083 | 0.8 ± 0.0 |
| B0 | F | 0.202 ± 0.046 | 0.278 ± 0.043 | 0.223 ± 0.026 | 0.603 ± 0.069 | 0.8 ± 0.0 |
| B1 | R | 0.086 ± 0.025 | 0.160 ± 0.031 | 0.129 ± 0.026 | 0.623 ± 0.065 | 0.9 ± 0.0 |
| B1 | F | 0.303 ± 0.083 | 0.326 ± 0.061 | 0.263 ± 0.050 | 0.583 ± 0.059 | 0.7 ± 0.1 |
| B2 | R | 0.734 ± 0.042 | 0.548 ± 0.047 | 0.440 ± 0.028 | 0.867 ± 0.063 | 0.5 ± 0.1 |
| B2 | F | 0.773 ± 0.039 | 0.569 ± 0.053 | 0.457 ± 0.030 | 0.870 ± 0.071 | 0.5 ± 0.1 |

## FPR on hard-negative slices (% flagged at top 1%/day)

| System | Tier | new_state | new_seller | billing_change | injected | multi_account_consignee |
|---|---|---|---|---|---|---|
| B0 | R | 1.2 ± 0.5 | 0.7 ± 0.1 | 0.5 ± 0.2 | 21.9 ± 3.9 | 5.5 ± 0.8 |
| B0 | F | 1.0 ± 0.4 | 0.4 ± 0.1 | 0.4 ± 0.3 | 27.1 ± 4.3 | 4.2 ± 1.3 |
| B1 | R | 2.7 ± 0.5 | 1.0 ± 0.3 | 0.6 ± 0.2 | 2.2 ± 1.7 | 1.5 ± 0.5 |
| B1 | F | 1.0 ± 0.4 | 1.3 ± 0.3 | 1.4 ± 0.8 | 14.3 ± 6.7 | 0.9 ± 0.6 |
| B2 | R | 0.3 ± 0.3 | 1.6 ± 0.3 | 1.5 ± 0.5 | 4.1 ± 2.0 | 0.1 ± 0.2 |
| B2 | F | 0.3 ± 0.4 | 1.7 ± 0.3 | 0.3 ± 0.1 | 3.3 ± 1.2 | 0.1 ± 0.3 |

Slice sizes (test, seed 0): new_state 705, new_seller 3,090, billing_change 898, injected 214, multi_account_consignee 148

## PR-AUC per typology (typology vs all legit; held-out marked *)

| System | Tier | T1 | T2 | T3* | T4 | T5* | T6 | T7 |
|---|---|---|---|---|---|---|---|---|
| B0 | R | 0.169 ± 0.051 | 0.006 ± 0.001 | 0.086 ± 0.034 | 0.036 ± 0.036 | 0.105 ± 0.044 | 0.001 ± 0.000 | 0.011 ± 0.010 |
| B0 | F | 0.193 ± 0.053 | 0.005 ± 0.001 | 0.064 ± 0.027 | 0.047 ± 0.043 | 0.136 ± 0.046 | 0.001 ± 0.000 | 0.024 ± 0.018 |
| B1 | R | 0.030 ± 0.020 | 0.047 ± 0.032 | 0.002 ± 0.001 | 0.081 ± 0.050 | 0.022 ± 0.018 | 0.004 ± 0.003 | 0.015 ± 0.011 |
| B1 | F | 0.159 ± 0.122 | 0.346 ± 0.177 | 0.026 ± 0.037 | 0.369 ± 0.165 | 0.149 ± 0.148 | 0.006 ± 0.006 | 0.101 ± 0.133 |
| B2 | R | 0.961 ± 0.022 | 0.377 ± 0.099 | 0.217 ± 0.159 | 0.602 ± 0.097 | 0.755 ± 0.059 | 0.020 ± 0.027 | 0.855 ± 0.117 |
| B2 | F | 0.998 ± 0.005 | 0.635 ± 0.204 | 0.307 ± 0.191 | 0.591 ± 0.149 | 0.801 ± 0.029 | 0.022 ± 0.028 | 0.993 ± 0.011 |

## Recall at top 1%/day per typology

| System | Tier | T1 | T2 | T3* | T4 | T5* | T6 | T7 |
|---|---|---|---|---|---|---|---|---|
| B0 | R | 0.353 ± 0.071 | 0.083 ± 0.055 | 0.270 ± 0.048 | 0.207 ± 0.111 | 0.267 ± 0.065 | 0.003 ± 0.009 | 0.104 ± 0.082 |
| B0 | F | 0.394 ± 0.079 | 0.059 ± 0.038 | 0.195 ± 0.057 | 0.240 ± 0.111 | 0.292 ± 0.074 | 0.003 ± 0.009 | 0.151 ± 0.106 |
| B1 | R | 0.111 ± 0.025 | 0.264 ± 0.076 | 0.004 ± 0.009 | 0.290 ± 0.119 | 0.098 ± 0.049 | 0.047 ± 0.040 | 0.124 ± 0.068 |
| B1 | F | 0.246 ± 0.101 | 0.542 ± 0.159 | 0.078 ± 0.063 | 0.492 ± 0.133 | 0.254 ± 0.145 | 0.063 ± 0.045 | 0.201 ± 0.129 |
| B2 | R | 0.703 ± 0.072 | 0.390 ± 0.159 | 0.173 ± 0.135 | 0.537 ± 0.090 | 0.393 ± 0.077 | 0.059 ± 0.045 | 0.619 ± 0.082 |
| B2 | F | 0.709 ± 0.096 | 0.403 ± 0.134 | 0.228 ± 0.138 | 0.505 ± 0.078 | 0.405 ± 0.084 | 0.068 ± 0.062 | 0.678 ± 0.124 |

## PR-AUC per camouflage level

| System | Tier | cam 0 | cam 1 | cam 2 |
|---|---|---|---|---|
| B0 | R | 0.133 ± 0.051 | 0.069 ± 0.027 | 0.042 ± 0.026 |
| B0 | F | 0.161 ± 0.057 | 0.080 ± 0.030 | 0.050 ± 0.030 |
| B1 | R | 0.061 ± 0.035 | 0.039 ± 0.019 | 0.013 ± 0.010 |
| B1 | F | 0.411 ± 0.132 | 0.065 ± 0.044 | 0.029 ± 0.028 |
| B2 | R | 0.746 ± 0.055 | 0.615 ± 0.090 | 0.566 ± 0.170 |
| B2 | F | 0.803 ± 0.071 | 0.641 ± 0.096 | 0.680 ± 0.177 |

Notes: B0 is a fixed rule sum (ties are common, which lowers its PR-AUC). GBM calibrated with Platt scaling on the calibration window 2018-02-15..2018-04-30. Models saved for seed 0 in artifacts/models/.
