# Label-free monitor: estimated vs realized performance

Real run of scripts/eval_monitor.py, 10 of the 10 injection seeds (runtime 26.9 s on CPU), mean ± sd over seeds. B2 LightGBM tier F, Platt-calibrated on the calibration window; T3 and T5 never trained. Decisions re-created with the service's cost rule in its shipped mode (LightGBM decides). Estimate = CBPE from the decision score (fraudshield/monitor/cbpe.py): precision of stops = mean p over stopped bookings, missed fraud = sum p over bookings let through, recall = caught / (caught + missed). Stopped = owner_confirm, review, hold, block; let through = allow, allow_scan_gated. Realized = the dataset's ground truth (in production only known once labels arrive).

## Period, chosen on the calibration window (2018-02-15..2018-04-30) only

| Period | Precision MAE (cal) | Stopped bookings per period (cal) |
|---|---|---|
| day | 0.145 | 2.6 |
| week | 0.085 | 17.3 |

Chosen: **week** (lower precision MAE). Periods are counted from the window's first day; the last one is partial.

## Estimate vs realized per week

Signed error = estimate minus realized (negative missed-fraud error = the monitor under-counts missed fraud; positive recall error = it thinks recall is better than it is).

| Window, scenario, score | Precision MAE | Precision signed | Missed MAE (count) | Missed signed | Recall MAE | Recall signed | Score PSI vs cal (mean / max) |
|---|---|---|---|---|---|---|---|
| calibration window (sanity, Platt fit here) | 0.085 ± 0.029 | 0.028 ± 0.045 | 2.59 ± 0.72 | -0.17 ± 0.24 | 0.150 ± 0.047 | -0.032 ± 0.073 | 0.015 / 0.044 |
| test, trained types only (T3/T5 rows removed) | 0.145 ± 0.051 | -0.003 ± 0.045 | 2.94 ± 0.70 | 0.54 ± 1.07 | 0.168 ± 0.037 | -0.051 ± 0.072 | 0.140 / 0.588 |
| test, T3/T5 present (real test window) | 0.134 ± 0.034 | -0.039 ± 0.053 | 3.27 ± 0.59 | -0.91 ± 1.32 | 0.148 ± 0.031 | 0.007 ± 0.067 | 0.142 / 0.589 |
| test, trained only, model-only p | 0.146 ± 0.051 | -0.005 ± 0.045 | 2.94 ± 0.70 | 0.54 ± 1.07 | 0.168 ± 0.037 | -0.052 ± 0.072 | n/a |
| test, T3/T5 present, model-only p | 0.159 ± 0.045 | -0.067 ± 0.061 | 3.27 ± 0.59 | -0.91 ± 1.32 | 0.149 ± 0.034 | -0.003 ± 0.066 | n/a |

## Whole window totals (decision score)

| Window, scenario | Stopped | Precision est / real | Caught est / real | Missed est / real | Recall est / real |
|---|---|---|---|---|---|
| calibration | 190.5 | 0.764 / 0.745 | 145.2 / 141.8 | 34.0 / 35.8 | 0.810 / 0.797 |
| test, trained types only | 224.8 | 0.721 / 0.741 | 161.5 / 165.5 | 52.6 / 43.9 | 0.755 / 0.789 |
| test, T3/T5 present | 287.3 | 0.746 / 0.797 | 213.8 / 228.0 | 52.8 / 67.4 | 0.803 / 0.771 |

## Where the missed fraud is, per type (test window, mean over seeds)

Let through = fraud bookings of the type that were allowed or only scan-checked. Sum of p = what the monitor 'sees' of them (it cannot attribute p to a type; this is the diagnostic with labels).

| Type | Bookings | Let through | Sum of p over those | Mean p of the type |
|---|---|---|---|---|
| T1 | 69.7 | 0.1 | 0.02 | 0.983 |
| T2 | 32.1 | 6.4 | 0.58 | 0.594 |
| T3* | 45.1 | 15.5 | 0.18 | 0.465 |
| T4 | 41.6 | 9.3 | 0.37 | 0.516 |
| T5* | 40.9 | 8.0 | 0.05 | 0.780 |
| T6 | 30.9 | 28.1 | 0.32 | 0.036 |
| T7 | 35.1 | 0.0 | 0.00 | 0.980 |

\* held out, never trained.

## Blind spot

Fraud types the model never learned (T3, T5): 23.5 of them were let through per test window on average, while their scores add up to only 0.2 expected frauds. Missed fraud per week, estimate minus actual: -0.91 with them present, +0.54 without; paired gap -1.45 ± 0.43 per week. Whole window: estimated recall 0.803 vs actual 0.771 with them, 0.755 vs 0.789 without (test window, 10 seeds).

Input-drift signal (PSI of the decision score per week against the calibration window): 0.015 ± 0.003 on the calibration window itself, 0.142 ± 0.062 on the test window with T3/T5 present and 0.140 ± 0.062 without them (difference 0.002 at most, in any seed). Whatever moves the score distribution on the test window, the held-out fraud is not what moves it: a few dozen bookings among ~22,900, scored low. A score-distribution proxy cannot flag them (Solozobov, arXiv 2604.15740: proxies catch covariate drift, not concept drift that leaves the features unchanged).

Note: T6 (under-declared parcels) is a trained type the score also does not see (it goes to the depot scan, allow_scan_gated, which counts as let through here). It was in the calibration window too, so Platt scaling folds it into the base rate: the missed-fraud estimate is a sum of small p over ~22,900 mostly honest bookings, right only on average over the fraud mix it was calibrated on, never per type. Without T3/T5 it over-counts missed fraud (52.6 estimated vs 43.9 actual over the window); with them it under-counts (52.8 vs 67.4). Per week the absolute errors are dominated by noise (a few frauds a week), so the signed error and the paired gap are the readable numbers, not the MAE.

