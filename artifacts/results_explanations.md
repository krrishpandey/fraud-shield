# Every sentence checked: claim-by-claim explanations

Measured 2026-10-04 05:39 by `scripts/eval_explanations.py`. Split: seed-0 TEST window only (22925 bookings; 22920 scored through the served pipeline from precomputed as-of features, 5 demo bookings outside the demo store skipped). 263 decisions were stopped (owner_confirm, review, hold, block); **n = 60** were drawn at random from them (numpy seed 0, no replacement): owner_confirm 12, review 1, hold 47, block 0. One seed, one sample: no ± over seeds.

Model attributions: LightGBM `pred_contrib` (TreeSHAP) of the served B2_F model, summed per reason code (fraudshield/explain/attribution.py). hit@3 = share of the model's top reason codes (at most 3) that the explanation cites among its first 3 distinct codes; top-1 = the first cited code is the model's top code.

## Planted lies (no LLM; planted into the template's claims)

| Lie | Rejected / planted |
|---|---|
| Fabricated number (in no field of the record) | 57/57 (100.0%) |
| True number moved to the wrong claim (claim checker) | 33/33 (100.0%) |
| Same moved number in the prose template (old prose validator) | 0/33 (0.0%) |
| Claim about a reason this decision does not have | 57/57 (100.0%) |
| Cites a field the record does not have | 57/57 (100.0%) |

The fabricated-number, invented-reason and invented-field rows are rejected by construction (the checker compares against the record): they show the checker is wired end to end, not a property of any LLM. The moved-number row is the one the old record-wide prose check cannot see.

## Prose vs claims (Groq openai/gpt-oss-120b, first try, no retry)

| | Template (control, no LLM) | (a) Prose path | (b) Claims path |
|---|---|---|---|
| Decisions | 60 | 31 | 31 |
| Rejected on first try -> template | 0 | 1/31 (3.2%) | 1/31 (3.2%) |
| Grounded units | claims 122/122 (100.0%) | sentences 136/136 (100.0%) | claims 62/62 (100.0%) |
| Claims grounded, record-wide test only | | | 62/62 (100.0%) |
| hit@3 vs model top reasons, mean (sd) | 0.244 (0.244), n=60 | 0.215 (0.220), n=31 | 0.204 (0.254), n=31 |
| Top-1 match | 10/60 (16.7%) | 4/31 (12.9%) | 4/31 (12.9%) |
| hit@3, template on the same 31 decisions | 0.247 (0.258), n=31 | | |
| Mode used | | | json_schema 31 |
| API errors / not run (rate-limited or skipped) | | 0 / 29 | 0 / 29 |

Ceiling: a validated explanation may only cite the decision's own reasons (codes whose rule fired); on average 0.867 of the model's 3.000 top codes are among them, so no checked explanation can score a mean hit@3 above 0.289. Share of the model's positive push that any reason code describes: mean 0.577 (sd 0.252). Low agreement is therefore mostly a gap between the fixed reason rules and what LightGBM leans on, not the explainer's wording.

LLM rows are paired: (a) and (b) on the same 31 decisions where both ran. The other 29 sampled decisions were **not run**: the Groq free tier's 200,000 tokens/day limit for this model (shared with everything else using the key today) was reached mid-run (16 rate-limited calls logged in the calls file). Nothing was filled in for them.

Why LLM claims failed (counts over all claims):

- none

Why whole explanations were rejected on first try: prose mentions 1 of top 2 reasons, needs 2; claims covers 1 of top 3 reasons, needs 2.

Notes:

- Prose sentences are judged with the record-wide number/id test (the prose validator's own rule); claims additionally must use their own reason's numbers, a real reason code and real fields, so the claims column is the stricter test. The record-wide row puts both on the same test.
- Prose 'cited reasons' are found by keyword (REASON_WORDS) in order of first mention: approximate.
- First try only (retry_invalid=0); the service retries once with the checker's problems.
- Every LLM answer used here is in artifacts/results_explanations_calls.jsonl (one line per call, rate-limited calls included as a log); reruns reuse them (`--cache-only` makes no new calls). New Groq calls in this run: 0.
- Sources: Groq Structured Outputs (strict json_schema, constrained decoding on openai/gpt-oss-120b; https://console.groq.com/docs/structured-outputs). arXiv 2512.00163: LLM self-explanations disagree with SHAP; LightGBM SHAP is more reliable on financial tabular data. arXiv 2605.26770: LLM-written XAI narratives raised confidence without improving accuracy and made LLM judges worse, hence a fixed validator, not an LLM judge.
