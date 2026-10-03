# Laya v2: Laya decides, LightGBM testifies

Written 2026-10-03, **before** the v2 model was trained or scored. The win condition below is fixed now and
is not changed after the results arrive (same rule as `docs/LEARNING_GATE.md`).

## Why v1 lost

v1 (`artifacts/results_laya.md`): fine-tuned Laya PR-AUC 0.265 vs LightGBM 0.777, so LightGBM decides and Laya
only answers questions. Three causes, all fixable:

1. **Too little fraud.** 386 distinct fraud bookings (seed-0 train window), repeated 3x.
2. **Laya saw less than LightGBM.** The v1 state omits signals LightGBM uses (account mix, burst ratio, value z,
   entropy jumps, consignee parcels in 30 days, owner contact age) and never shows the rule flags.
3. **Wrong job.** v1 asked Laya to rebuild a tabular ranker from text. That is LightGBM's strength, not Laya's.

## The v2 design

Laya is the decision maker. LightGBM and the published rules are witnesses whose testimony Laya reads.

| Change | What | Why |
|---|---|---|
| ser-v2 state | v1 lines + `MORE` (the GBM-only signals) + `EVIDENCE` (LightGBM risk, drop rule, under-declared rule) | Laya sees everything LightGBM saw, plus LightGBM's opinion |
| Out-of-fold witness | Training states carry 5-fold out-of-fold LightGBM risk (grouped by account, per seed) | An in-sample GBM score is nearly always right; Laya would learn to copy it. OOF PR-AUC per seed 0.75 to 0.85, close to its 0.777 on test, so the training witness is as noisy as the live one |
| Witness dropout | The LightGBM line reads "not available" on 30% of training states | Laya cannot be a pass-through, and it still decides when LightGBM is down or being retrained |
| 10 seeds of fraud | Train windows of all 10 injection seeds: about 3,700 distinct fraud bookings (v1: 386) | More, and more varied, fraud. T3 and T5 stay held out of everything |
| Questions | misuse, foreign_senders, payoff_max, action (risk_level dropped) | risk_level duplicated misuse + cost; dropping it buys token room for the new lines |
| Laya picks the action | The action head (trained on the cost vector) proposes; the cost rule audits | Laya is the decider; the auditor overrules only an action that is clearly costlier under Laya's own probabilities, and every overrule is logged |

Evaluation uses the **same bookings** as v1 (`artifacts/laya/evalset_*.parquet`), re-serialized, so v1, v2 and
LightGBM are compared row for row. Each test booking is scored twice: with the LightGBM line, and with it withheld.

## Systems compared

- **B2**: LightGBM (tier F, served model), the current decider.
- **M1**: v1 fine-tuned Laya (reference).
- **M2**: v2 Laya with the evidence line (the proposed decider).
- **M2-solo**: v2 Laya with the LightGBM line withheld (what Laya learned by itself).
- **S0**: a 3-input logistic stack (LightGBM logit, drop rule, under-declared rule) fitted on the calibration
  window. Control: does Laya add more than a trivial combiner of the same witnesses?

## Win condition (pre-registered)

Laya becomes the live decider (`config/app.yaml` `laya.decide: true`) only if **both** hold:

- **W1, no loss in ranking:** M2 mean PR-AUC over the 10 seeds >= B2 mean PR-AUC - 0.02.
- **W2, adds something:** at least one of the following, with the 95% campaign-bootstrap CI of (M2 - B2)
  entirely above 0:
  - (a) held-out T3 recall at 1% legit FPR
  - (b) held-out T5 recall at 1% legit FPR
  - (c) cost saved per booking (BRL) by the full decision (M2 probabilities + cost rule) vs B2 + the same cost rule

If W1 holds and W2 fails, we report "Laya matches LightGBM and adds named fraud modes", and LightGBM keeps deciding.
If W1 fails, LightGBM keeps deciding. Either way the numbers are published as they come out.

Also reported, without a pass/fail: M2-solo vs M1 (did 10x data + more signals fix Laya on its own?), M2 vs S0,
the action head's cost vs the cost rule, how often the auditor overrules Laya, and zero-shot `drop_consignee` on T3.

## Files

| Step | Command / file |
|---|---|
| Build data (CPU, ~1 min) | `uv run python scripts/laya_v2_build.py --legit 6000` -> `artifacts/laya_v2/` |
| Kaggle bundle | `uv run python kaggle/build_kaggle_bundle.py --v2` -> `kaggle/dist/fraudshield_laya_v2_bundle.zip` |
| Kaggle notebook | `kaggle/fraudshield_laya_v2_kaggle.ipynb` (regenerate: `kaggle/make_notebook_v2.py`) |
| Import + evaluate (CPU) | `uv run python kaggle/import_kaggle_v2_output.py <fraudshield_laya_v2_output.zip>` -> `artifacts/results_laya_v2.md` |
| Demo cache (GPU, seconds) | `FS_LAYA_ART=artifacts/laya_v2 uv run python scripts/laya_cache.py --v2 --ckpt artifacts/laya_v2/fraudshield-laya` |
| App: Laya decides | `fraudshield/api/pipeline.py` (`laya_action`), `fraudshield/policy/decide.py` (`audit_laya_action`), `real_components.serializer_v2` |

## Switching the app to "Laya decides" (only if the win condition is met)

`config/app.yaml`:
```yaml
laya:
  decide: true
  action_head: true            # false = Laya's probabilities + cost rule pick the action (use if the action head costs more)
  model_path: artifacts/laya_v2/fraudshield-laya
  cache_path: artifacts/laya_v2/laya_cache_v2.json
laya_ask:
  model_path: artifacts/laya_v2/fraudshield-laya
calibration_path: artifacts/laya_v2/calibration_v2.json
components:
  serializer: fraudshield.api.real_components:serializer_v2
```
When Laya is unavailable (timeout, no GPU and no cached answer) LightGBM decides, with the stricter degraded guard,
and the console says so.
