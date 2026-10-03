# FraudShield HTTP API contract (v1)

Base URL: `http://localhost:8080` (FastAPI). Laya server (if used) runs separately on port 8001. Web console on 5173 with `VITE_API_URL`.
All responses are JSON. Errors: `{"detail": "..."}` with 4xx/5xx.

## POST /score
Request: a `Booking` (see `fraudshield/contracts.py`).
```json
{"booking_id":"demo-takeover-01","account_id":"acc_7c3e","booked_at":"2018-06-14T02:41:00","channel":"api",
 "login_device_age_days":0,"payment_method":"account_billing","sender_id":"snd_91aa","origin_uf":"PR","origin_zip3":"806",
 "dest_uf":"AM","dest_zip3":"690","consignee_id":"cns_4410","weight_kg":9.8,"length_cm":50,"width_cm":40,"height_cm":35,
 "service":"express","category":"electronics","declared_value":1450.0,"carrier_cost":212.6,"owner_contact_age_days":412,
 "meta":{"scenario":"T1"}}
```
Response 200 (idempotent on booking_id: same id returns the stored decision):
```json
{"decision_id":"dec_000123","booking_id":"demo-takeover-01","action":"hold","greedy_action":"hold",
 "propensity":0.95,"explored":false,"degraded":false,
 "probabilities":{"misuse":0.91,"foreign_senders":0.94,"payoff_max":0.88,"drop_consignee":0.07},
 "raw_probabilities":{"misuse":0.97,"foreign_senders":0.99,"payoff_max":0.93,"drop_consignee":0.12},
 "gbm_score":0.71,
 "expected_costs":{"allow":205.1,"allow_scan_gated":131.0,"owner_confirm":40.2,"review":66.0,"hold":23.0,"block":30.1},
 "reasons":["NEW_SENDERS_UNDER_PAYER","COST_FAR_ABOVE_ACCOUNT_NORM","NEW_LOGIN_DEVICE"],
 "top_features":[{"name":"new_senders_last10","value":8,"baseline":0.0},{"name":"cost_vs_median","value":11.6,"baseline":1.0}],
 "state_text":"BOOKING 2018-06-14 Thu 02:41 | ...",
 "model_versions":{"laya":"...","calibration":"...","gbm":"...","policy":"...","serializer":"..."},
 "latency_ms":{"total":142.3,"features":18.1,"laya":101.0,"decide":0.4,"audit":3.2},
 "explanation_status":"pending",
 "audit_hash":"9f2c..."}
```

## GET /decisions?limit=50&action=hold
List of decision summaries, newest first: `[{decision_id, booking_id, account_id, booked_at, action, misuse, carrier_cost, analyst_label|null, scenario|null}]`.

## GET /decisions/{decision_id}
Full decision (same shape as /score response) plus `booking` (the request), `explanation` (`{text, source, valid, model_id}` or null while pending), `analyst` (`{label, note, at}` or null).

## POST /decisions/{decision_id}/analyst
Request `{"label":"fraud"|"legit","note":"..."}` -> `{"ok":true,"audit_hash":"..."}`. Feeds the label store (and OPE).

## POST /decisions/{decision_id}/ask
Ask a NEW plain-language yes/no question about this booking's stored state (zero-shot Laya).
Request `{"instructions":"Does the consignee look like a reshipping drop?","yes":"likely a drop","no":"ordinary consignee"}`
Response `{"qid":"custom_1","probability_yes":0.82,"raw_probabilities":{"a":0.82,"b":0.18},"latency_ms":40.2,"calibrated":false}`

## GET /dashboard/metrics
```json
{"window":{"from":"2018-05-15","to":"2018-08-31"},
 "totals":{"bookings":19263,"by_action":{"allow":18800,"allow_scan_gated":300,"owner_confirm":60,"review":40,"hold":50,"block":13}},
 "held_shipments":63,
 "revenue_loss_prevented_brl":41250.0, "friction_cost_brl":3100.0, "net_prevented_brl":38150.0,
 "fpr_legit":0.012, "fpr_hard_negative":0.031,
 "trend":[{"date":"2018-06-01","fraud_stopped":3,"held":2,"by_typology":{"T1":2,"T4":1}}],
 "latency":{"p50_ms":120.0,"p99_ms":190.0},
 "assumptions":{"cost_matrix_version":"costs-v1","note":"friction and loss costs are assumptions"}}
```

## GET /audit/verify
`{"ok":true,"records":1234,"head_hash":"...","first_bad_index":null}`

## GET /health
`{"ok":true,"laya_mode":"local|http|cached","gpu":true,"versions":{...}}`

## Additions (v1.1, accepted from frontend agent proposals)
- `GET /demo/bookings` -> `[{"scenario":"takeover","title":"...","description":"...","expected":"hold"|null,"booking":{Booking}}]`. Scenario ids: `legit-tenured`, `hard-negative-new-state`, `takeover`, `reshipping-drop`, `weight-manipulation`. Empty list if no demo file.
- `GET /health` adds `"degraded": bool` (true when Laya is unavailable and the GBM fallback is active).
- `explanation_status` values: `pending` | `ready` | `failed`.
- `reasons[]` codes: the decision service publishes the full list at `GET /reason-codes` -> `{"CODE":"plain-language text"}` (source of truth: `fraudshield/policy/reasons.py`).
- `POST /decisions/{id}/ask`: `yes` and `no` are optional; default to "yes" and "no".
- Dashboard `trend[].by_typology` keys: `T1`..`T7` (plus `unlabelled` for live decisions without ground truth).

## Continuous learning (v1.2)
Analyst confirmations (and, only when explicitly requested, SIMULATED analyst labels drawn from ground truth, flagged `simulated`) become training labels. Retraining builds a candidate, evaluates it against the active model on data neither was trained on, and deploys only if a gate passes. Every run is written to the audit log (`config_change` / `retrain` events).

- `GET /learning/status` ->
```json
{"active_version":"gbm-B2-F-v1","labels_since_last_retrain":42,
 "label_sources":{"analyst":12,"simulated_analyst":30},
 "versions":[{"version":"gbm-B2-F-v1","created_at":"...","parent":null,"n_train":59000,"n_feedback_labels":0,
   "metrics":{"pr_auc":0.61,"ece":0.012,"cost_per_1k_brl":410.0,"fpr_hard_negative":0.03,"recall_new_pattern":0.20},"deployed":true}],
 "calibration_version":"cal-...","laya_export":{"path":"artifacts/feedback/laya_feedback.jsonl","rows":42}}
```
- `POST /learning/simulate_feedback` `{"n":200,"seed":1}` -> `{"added":200,"fraud":31,"legit":169,"simulated":true}` (demo only: labels come from injected ground truth of replayed bookings).
- `POST /learning/retrain` `{"min_new_labels":20}` ->
```json
{"run_id":"rt_0003","n_new_labels":200,"n_fraud":31,"n_legit":169,
 "eval_set":{"n":5000,"description":"validation window + latest 30% of feedback labels by time, never trained on"},
 "current":{"version":"gbm-B2-F-v1","pr_auc":0.61,"ece":0.012,"cost_per_1k_brl":410.0,"fpr_hard_negative":0.03,"recall_new_pattern":0.20},
 "candidate":{"version":"gbm-B2-F-v2","pr_auc":0.66,"ece":0.013,"cost_per_1k_brl":372.0,"fpr_hard_negative":0.03,"recall_new_pattern":0.55},
 "gate":{"passed":true,"checks":[{"name":"cost_not_worse","passed":true,"detail":"..."}]},
 "deployed_version":"gbm-B2-F-v2","audit_hash":"..."}
```
- `POST /learning/rollback` `{"version":"gbm-B2-F-v1"}` -> `{"active_version":"gbm-B2-F-v1","audit_hash":"..."}`

## Depot weighing dial (v1.3)

`GET /first-scan/dial` returns `{current, levels, version}`. Each level has `name`, `threshold` (null for standard),
`t6_caught`, `t6_caught_sd`, `honest_weighed` and `weighs_per_1k_bookings`, fitted and measured by
`scripts/fit_first_scan.py` (`artifacts/first_scan_dial.json`).

`POST /first-scan/dial {"level": "5%"}` switches the level for the app and the live stream, writes a `config_change`
audit record and returns the same body plus `audit_hash`. An unknown level returns 422. A booking whose
`under_score` (`-(dims_z + weight_z)`, accounts with 5+ earlier bookings) reaches the level's threshold gets the reason
`UNDER_DECLARED_PARCEL` and at least `allow_scan_gated`. The starting level is `first_scan.level` in the config
(default `standard`).

## Model handover (v1.4, continuous learning)

Additive fields that say which GBM scores new bookings from now on. Existing fields are unchanged. Code:
`fraudshield/learning/handover.py`.

- `POST /learning/retrain` adds `handover`; `POST /learning/rollback` adds `handover` and `previous_version`:
```json
{"event":"retrain","run_id":"rt_0002","at":"2026-10-04T14:02:11",
 "verdict":"previous_model_kept",
 "active_version":"gbm-B2-F-v1",
 "previous_version":"gbm-B2-F-v1",
 "candidate_version":"gbm-B2-F-v3",
 "active_since":"2026-10-04T13:40:02",
 "rollback_target":null,
 "failed_checks":[{"name":"fpr_hard_negative_noninferior",
   "plain":"honest hard-case false positives would rise from 0.51% to 2.48%; the limit is +0.5 points",
   "detail":"FPR on hard negatives 0.51% -> 2.48% (max +0.5 points)"}],
 "new_pattern":{"current":0.529,"candidate":0.775,"ci95":[0.177,0.325],"typologies":["T3","T5"],"source":"new-pattern eval set"},
 "message":"Still using gbm-B2-F-v1 (since 2026-10-04T13:40:02). Candidate gbm-B2-F-v3 learned the new pattern (...) but was not deployed because: (1) ..."}
```
  - `verdict`: `new_model_in_use` (gate passed, candidate deployed), `previous_model_kept` (gate failed) or
    `rolled_back`.
  - `active_version` scores new bookings from now on. `previous_version` was active before this event (equal to
    `active_version` when the candidate was kept). `candidate_version` is null for a rollback.
  - `active_since`: when the active version became active (the service start time if it never changed).
  - `rollback_target`: the latest other version that passed the gate before, or null.
  - `failed_checks[].plain`: each failed gate check in plain words, built from the gate's numbers.
- `GET /learning/status` adds `model_in_use` (what the decision service actually scores with) and `last_handover`
  (the latest retrain or rollback handover above plus its `audit_hash`; null before the first one). Versions add
  `activated_at`.
```json
"model_in_use":{"version":"gbm-B2-F-v1","since":"2026-10-04T13:40:02","since_reason":"activated",
  "registry_active_version":"gbm-B2-F-v1","matches_registry":true,"rollback_target":null}
```
- Every decision's `model_versions.gbm` (API record and audit log) is the exact registry version of the scorer that
  produced its score (e.g. `gbm-B2-F-v1`, no longer the generic `gbm` when `learning.wire_active` is on). A deploy or
  rollback also swaps the GBM of a running live stream. The retrain and rollback audit payloads include `handover`.
