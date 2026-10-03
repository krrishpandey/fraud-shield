# Whole-system results per fraud type

Test window 2018-05-15..2018-08-31, 10 injection seeds, mean ± sd. Real run of scripts/eval_system.py.
Model = B2 LightGBM with the account mix features (fraudshield/features/mix.py). The drop rule and the
first-scan rule use fixed thresholds (drop rule from Hao et al., CCS 2015; first-scan threshold chosen on the
train window with honest scan checks <= 2%). T3 and T5 are never in training. 'Caught' = flagged in the top 1%
of bookings per day by the system score, or sent to a first-scan check, where the depot scale reveals a
T6 parcel's true weight (simulated from the dataset's true weights).

## Tier R (real columns only)

| System | PR-AUC | ROC-AUC | Fraud caught | Honest stopped | Honest scan-checked |
|---|---|---|---|---|---|
| model only | 0.743 ± 0.050 | 0.947 ± 0.016 | 44.6 ± 3.1% | 0.5 ± 0.1% | 0.0 ± 0.0% |
| model + drop rule | 0.790 ± 0.036 | 0.955 ± 0.013 | 46.8 ± 3.1% | 0.4 ± 0.1% | 0.0 ± 0.0% |
| model + drop rule + first-scan check | 0.790 ± 0.036 | 0.955 ± 0.013 | 52.0 ± 3.0% | 0.4 ± 0.1% | 1.7 ± 0.0% |
| model + drop rule + first-scan check + depot follow-up | 0.790 ± 0.036 | 0.955 ± 0.013 | 53.4 ± 3.9% | 0.4 ± 0.1% | 1.8 ± 0.2% |

PR-AUC per type (held-out marked *):

| System | T1 | T2 | T3* | T4 | T5* | T6 | T7 |
|---|---|---|---|---|---|---|---|
| model only | 0.964 | 0.459 | 0.231 | 0.631 | 0.756 | 0.021 | 0.850 |
| model + drop rule | 0.963 | 0.454 | 0.514 | 0.627 | 0.755 | 0.021 | 0.847 |

Share of each type caught (top 1% per day, or first-scan check):

| System | T1 | T2 | T3 | T4 | T5 | T6 | T7 |
|---|---|---|---|---|---|---|---|
| model only | 70% | 43% | 18% | 56% | 39% | 6% | 59% |
| model + drop rule | 69% | 40% | 38% | 54% | 39% | 6% | 58% |
| model + drop rule + first-scan check | 71% | 40% | 53% | 54% | 40% | 27% | 59% |
| model + drop rule + first-scan check + depot follow-up | 71% | 40% | 53% | 54% | 40% | 40% | 59% |

## Tier F (all columns)

| System | PR-AUC | ROC-AUC | Fraud caught | Honest stopped | Honest scan-checked |
|---|---|---|---|---|---|
| model only | 0.777 ± 0.043 | 0.955 ± 0.015 | 46.0 ± 3.1% | 0.4 ± 0.1% | 0.0 ± 0.0% |
| model + drop rule | 0.821 ± 0.028 | 0.961 ± 0.013 | 47.9 ± 2.9% | 0.4 ± 0.1% | 0.0 ± 0.0% |
| model + drop rule + first-scan check | 0.821 ± 0.028 | 0.961 ± 0.013 | 53.0 ± 3.1% | 0.4 ± 0.1% | 1.7 ± 0.0% |
| model + drop rule + first-scan check + depot follow-up | 0.821 ± 0.028 | 0.961 ± 0.013 | 54.4 ± 4.2% | 0.4 ± 0.1% | 1.8 ± 0.2% |

PR-AUC per type (held-out marked *):

| System | T1 | T2 | T3* | T4 | T5* | T6 | T7 |
|---|---|---|---|---|---|---|---|
| model only | 0.997 | 0.703 | 0.305 | 0.615 | 0.796 | 0.024 | 0.997 |
| model + drop rule | 0.997 | 0.693 | 0.599 | 0.609 | 0.796 | 0.024 | 0.997 |

Share of each type caught (top 1% per day, or first-scan check):

| System | T1 | T2 | T3 | T4 | T5 | T6 | T7 |
|---|---|---|---|---|---|---|---|
| model only | 69% | 44% | 23% | 52% | 40% | 6% | 68% |
| model + drop rule | 69% | 43% | 37% | 51% | 40% | 6% | 67% |
| model + drop rule + first-scan check | 71% | 43% | 52% | 51% | 41% | 28% | 69% |
| model + drop rule + first-scan check + depot follow-up | 71% | 43% | 52% | 51% | 41% | 41% | 69% |

