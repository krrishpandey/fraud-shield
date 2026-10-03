# Too-easy checks (real run of scripts/sanity_checks.py)

1. Rules-only B0 PR-AUC at camouflage 2 (target < 0.8): tier R 0.042, tier F 0.050 -> PASS
2. Artifact classifier (target AUC <= 0.6). Folds grouped by campaign (injected) / account (real), so the classifier cannot win by memorizing one campaign's repeated values; plain row-level CV shown too.
   a) injected-legit vs real, all columns with a real counterpart: grouped 0.617, plain 0.751
   b) all injected vs real, non-semantic columns only: grouped 0.535, plain 0.550
   -> grouped FAIL, plain FAIL
   Per-column grouped AUC (a): express 0.533, distance_km 0.519, scan_lag_h 0.51, scan_missing 0.507, freight_residual 0.503, carrier_cost 0.501, n_items 0.5, second 0.499, tod_seconds 0.497, cost_cents 0.495, category_code 0.482, height_cm 0.478, declared_value 0.478, weight_kg 0.468, length_cm 0.467, width_cm 0.465
   Per-column grouped AUC (b): scan_lag_h 0.513, freight_residual 0.511, scan_missing 0.508, second 0.502, cost_cents 0.502
   Placebo (no injection): real campaign-like groups of 25 bookings from one account vs random real rows, grouped AUC 0.531. This is the floor of the test for clustered data.
   For information (not a check, fraud is meant to differ): injected fraud vs real, all columns, grouped AUC 0.813
3. Shuffled-label B2 tier F PR-AUC, mean of 5 shuffles 0.0108 (runs 0.0153, 0.0079, 0.0073, 0.0093, 0.0141) vs prevalence 0.0120 -> PASS
