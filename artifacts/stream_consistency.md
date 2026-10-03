# Live stream consistency check

Measured 2026-10-03 16:12 on Windows 11, AMD64 Family 25 Model 68 Stepping 1, AuthenticAMD, 16 CPU threads. The seed-0 test window was streamed in booked_at
order through the decision service (Laya cached, so each decision uses the LightGBM backup score), and
compared with the offline evaluation of the same model (artifacts/models B2_F, scores in
data/processed/gbm_scores_seed0.parquet) on the same bookings. The decision record stores scores with
4 decimals, so the offline scores are rounded the same way and a per-booking difference up to 0.00005 is
rounding, not a different score.

- Bookings streamed and scored: 22925 of 22925, errors 0
- Fraud bookings among them: 276
- Largest per-booking difference between the live and the offline score: 0.00e+00 (over 22925 bookings)

| Metric | Live stream | Offline, same bookings | Offline, whole window |
|---|---|---|---|
| PR-AUC | 0.767 | 0.767 | 0.767 |
| ROC-AUC | 0.937 | 0.937 | 0.937 |
| Precision, top 1% per day | 0.475 | 0.475 | 0.475 |
| Recall, top 1% per day | 0.413 | 0.413 | 0.413 |

Fraud stopped (owner confirm, review, hold or block): 196; sent to a first-scan check: 21; allowed: 59. Honest bookings stopped: 47 (0.21%).

Result: PASS. The live stream reproduces the offline scores and metrics; accuracy does not drop under continuous scoring.
