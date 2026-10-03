# Data and model outputs (from the Data/ML agent)

Rebuild everything (about 8 minutes on the dev laptop):
```
export UV_PROJECT_ENVIRONMENT="C:/Users/05nik/.venvs/fraudshield"
uv run python scripts/build_dataset.py --seeds 10 --workers 5   # bookings, injection, features
uv run python scripts/train_gbm.py --seeds 10                   # models + artifacts/results_gbm.md
uv run python scripts/sanity_checks.py --seeds 10               # artifacts/sanity_checks.md
uv run python scripts/export_outputs.py                         # states.jsonl, demo store, demo bookings
```

## For the API / desktop app (no torch or transformers needed)
```python
from fraudshield.data.demo import load_demo_store, load_demo_bookings
from fraudshield.features import featurize            # featurize(booking, store) -> FeatureVector
from fraudshield.features.serialize import serialize, SERIALIZER_VERSION   # serialize(fv, gbm_risk=None) -> str
from fraudshield.models.gbm import GBMModel
store = load_demo_store()                    # about 0.6 s, 101,567 rows (seed-0 history, demo bookings excluded)
gbm = GBMModel.load("artifacts/models", "B2", "F")  # calibrated P(fraud); gbm.score_values(fv.values)
fv = featurize(booking, store)               # 20-50 ms per booking on the demo store
store.append(booking)                        # after scoring
```
- `data/processed/feature_store_demo.parquet` + `feature_store_demo.events.json`: history rows (Booking fields only, no
  labels) and simulated events (pre-announced verified changes, confirmed-fraud feedback). Features only read rows
  strictly before `booked_at`, so later rows never leak.
- `FeatureStore.register_change(account_id, kind, at, announced_at)` and `FeatureStore.confirm_fraud(account_id,
  confirmed_at, entities)` feed the tier F event features (the analyst "fraud" label can call `confirm_fraud`).
- `fraudshield/features/zip3_centroids.csv` ships inside the package (needed for distance).
- `data/processed/demo_bookings.json`: list of `{"scenario","title","description","expected","booking"}`, scenarios in
  order `legit-tenured`, `hard-negative-new-state`, `takeover`, `reshipping-drop`, `weight-manipulation`.
  `expected` is the ground truth only ("legit", "T1", "T3", "T6"), never a model output. `booking` is a
  `contracts.Booking` dict; labels live in `booking.meta` only (never a feature). The first two are real Olist rows from
  the test window; the other three are injected rows (seed 0). T3 is a held-out typology.

## For the Laya fine-tune / evaluation
- `data/processed/states.jsonl` (64,200 rows, about 100 MB): one JSON object per booking:
  `id, workflow, seed, split, booking_ts, account_id, typology, is_injected, scenario_id, campaign_id, camouflage_level,
  state, labels{misuse, foreign_senders, payoff_max, drop_consignee, risk_level}, action_cost{allow..block},
  hard_negative[list of slices], gbm_b2f`.
  Contents: seed 0 all calibration and test rows, all injected and hard-negative train rows, plus 15,000 random legit
  train rows (so the train part is already enriched; sample from it); seeds 1-9 test-window injected rows only (for
  about 300 test campaigns). `gbm_b2f` = calibrated B2 tier F score (seed-0 model) for cal/test rows, null for train
  (in-sample scores would leak). `state` has no GBM line; pass `serialize(fv, gbm_risk=...)` for the M2 variant.
- `labels`: hard 0/1 from injection truth; `drop_consignee` is 1 only for T3 (zero-shot question, never trained).
  `risk_level`: 0 legit; misuse 1..4 by cost vs the victim's pre-onset median (>1.5x, >3x, >8x) or 4 for T5 bursts.
- `action_cost`: cost per action in BRL (`fraudshield/data/labels.py`, mirrors docs/reference/rl_core.py, version
  costs-v1, all [ASSUMPTION]); T6 scan gate catches 95%; T2 owner_confirm reaches the fraudster; no verified contact ->
  owner_confirm costs at least review.
- Token counts with Laya's tokenizer on 2,000 sampled states: median 327, p99 331, max 334 (cap 380; worst case with
  every number inflated is under 380, tested).
- Serializer deviations from the DESIGN template: ACCOUNT line ends with `verified change 30d yes|no`; CHANGE line adds
  `distinct origins 7d`; SEQUENCE line is always present (`last 72h n bookings | express k | mean x kg`); counts above
  999 render `999+`; missing profile stats render `n/a` (fewer than 5 prior bookings).

## Full tables
- `data/processed/bookings_all.parquet`: real rows once (`seed` = -1, present in every seed) + injected rows of all 10
  seeds (`seed` = k), all Booking columns, labels, split. Seed k = real rows + injected rows with seed k.
- `data/processed/features/seed_K.parquet`: every booking of seed K with all features (`fraudshield/features/spec.py`,
  tier R/F per feature), labels, split and hard-negative slice flags. `features.parquet` = seed 0.
- `data/processed/features/injected_seed_K.parquet`, `events/seed_K.json` (changes + confirmed events).
- `data/processed/gbm_scores_seed0.parquet`: B2 tier F and R calibrated scores for all seed-0 rows (train rows in-sample).
- `data/processed/accounts_synthetic.parquet`: synthetic account records; `build_manifest.json`, `states_manifest.json`.
- `artifacts/models/{B1,B2}_{R,F}.lgb` + `.json` (features, Platt a/b, training counts, seed-0 test metrics).
- `artifacts/results_gbm.md|json`, `artifacts/sanity_checks.md|json`.

Note: `states.jsonl` and the json files in data/processed are not covered by the current .gitignore (only parquet is).
