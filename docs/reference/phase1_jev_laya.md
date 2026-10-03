# Phase 1: Jev / Laya verification notes (2026-10-02)

Both products EXIST. Raw downloads in this scratchpad: `hf_readme.md` (HF card convaiinnovations/laya), `laya_readme.md` (GitHub README, 100 KB), `nb.ipynb` + `nb_src.txt` (fine-tune notebook source), `jev_llms.txt`, `jev_full.txt` (TypeSafe docs llms-full.txt, 900 KB).

## LAYA (Convai Innovations)

### HF org listing (https://huggingface.co/api/models?author=convaiinnovations)
- convaiinnovations/laya: Apache-2.0, created 2026-09-18
- convaiinnovations/laya-multilingual: Apache-2.0, created 2026-09-19
- convaiinnovations/laya-typed-decisions: Apache-2.0, created 2026-09-18 (fine-tuned on LocalLLaMA/typed-decisions)

### Model card (https://huggingface.co/convaiinnovations/laya/raw/main/README.md)
- English: "ModernBERT-large backbone (395M params) + decision head (2 transformer layers, option-marker scorer, act/escalate head) = 421M total"
- Multilingual: mmBERT-base (307M, 22 layers, hidden 768, 256k vocab) + head = 322M total, 100+ languages
- Context: English 512 tokens (head budget 192 tokens for options); multilingual & typed-decisions 1,024 default (head 256); "up to 8,192 tokens with `max_len=8192`" (multilingual encoder)
- Training: "strictly proper scoring rules (log + spherical, plus ranked probability score for ordinal questions)"; "Updates are REINFORCE with a group-mean baseline (GRPO-style)."
- Calibration: ECE "0.466 -> 0.081" (laya) and "0.314 -> 0.106" (multilingual) after temperature refit. "Shipped over-confident; temperature calibration essential". Multilingual card: "Ships uncalibrated" temp 1.0, mean confidence 0.75-0.83.
- Known issue: if laya.load() hangs due to TensorFlow import, set USE_TF=0.
- Limitations: base checkpoints "near chance on typed-decisions zero-shot"; score is weakest primitive (SST-5 0.372); >20-50 options degrade.

### GitHub https://github.com/NandhaKishorM/laya (NOT under a convaiinnovations GitHub org; api.github.com/users/convaiinnovations returns 404)
- PyPI `laya` latest 0.3.23, Apache-2.0, author Convai Innovations. Docs https://nandhakishorm.github.io/laya/ ; demo https://huggingface.co/spaces/convaiinnovations/laya-demo
- Extras: laya[serve], laya[mcp], laya[langchain], laya[onnx], laya[fast]. Python 3.10+.

Speed table (Tesla T4, "measured"):
| q/call | laya | laya-multilingual |
| 1 | 39.5 ms | 32.8 ms |
| 5 | 84.5 ms | 40.1 ms |
| 10 | 158.6 ms | 72.3 ms |
| 50 | 771 ms | 337 ms |
"Batched throughput reaches 103-332 questions/sec on a single T4."
Notebook eval reports ~710 ms per case (typed-decisions case = 5 questions on longer state, fine-tuned 421M).

typed-decisions (400 cases / 2,000 decisions):
| model | acc | soft acc | Brier | ECE | score MAE |
| laya-typed-decisions | 0.766 | 0.471 | 0.062 | 0.213 | 0.242 |
| laya | 0.362 | 0.332 | 0.316 | 0.175 | 0.694 |
| laya-multilingual | 0.352 | 0.328 | 0.463 | 0.314 | 0.760 |
| Jev 1.13.0 (published) | 0.727 | 0.580 | 0.148 | 0.144 | 0.391 |
Majority-class baseline 0.461, random 0.318. README itself: "The laya-typed-decisions row and its four per-workflow scores are from the fine-tuning run and have no committed result file behind them yet."
Jev comparisons: "Jev figures are third-party published, never measured here (no TypeSafe API access)". The "ECE 0.246 vs 0.081, 3x better" headline compares Jev's ECE on nibzard's S5 uncertainty task to Laya's post-temperature ECE on different data -> apples-to-oranges.
"Where Jev leads": >20 options (Banking77 Jev 0.870 vs Laya 0.425), soft accuracy, raw (pre-temperature) calibration.

### laya-serve (README "Self-Hosting: HTTP Server (Jev-compatible)")
```
pip install "laya[serve]"          # adds fastapi + uvicorn + python-multipart
LAYA_DEVICE=cuda LAYA_PRELOAD=1 laya-serve   # binds 0.0.0.0:8000, preloads all 3 checkpoints
curl -s localhost:8000/v1/systemone -H 'content-type: application/json' -d '{
  "state": {"body": "billed twice, refund please or we cancel"},
  "questions": {"dept": {"type": "choice", "instructions": "which team?",
                "criteria": {"billing": "refunds", "tech": "bugs"}}}
}'
```
- POST /v1/systemone/batch with "states": [...] (cap 64), returns "results" + "total_usage"
- GET /health, GET /models. Env: LAYA_HOST, LAYA_PORT, LAYA_DEVICE, LAYA_PRELOAD, LAYA_MODELS, LAYA_THREADS, LAYA_DEFAULT_MODEL, LAYA_MAX_LOADED, LAYA_API_KEY (Bearer), LAYA_ROOT_PATH.
- body `model` field: english / multilingual / typed-decisions, else auto-route. Extra fields: task, lang, lang_guess, min_confidence, max_len, head_max_len.
- Quote: "Laya's answer payload is already schema-identical to what Jev returns ... an existing Jev client ... just needs its baseUrl repointed".
- Differences: options limited by head_max_len token budget (192/256) not Jev's 255; server caps 100 choice options (413); `confidence` = 1 - normalized entropy, NOT Jev's (n*p_max-1)/(n-1) -> "A threshold carried over from Jev does not transfer"; gate on `answer_confidence`.

### Python API (HF card)
```python
from laya import Router
router = Router(preload=True)
questions = {
  "department": {"type": "choice", "instructions": "Which department handles this?",
                 "criteria": {"billing": "invoices, payments, refunds", "technical": "bugs, outages, system errors"}},
  "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["not urgent", "soon", "critical"]},
  "churn_risk": {"type": "noul", "instructions": "Does user threaten to leave?"}}
result = router.predict(state, questions)
result["answers"]["department"]["choice"]; result["answers"]["churn_risk"]["noul"]
import laya; agent = laya.load("convaiinnovations/laya"); agent.predict(state, questions)
agent.predict_batch(states, questions, batch_size=N, sort_by_length=True)  # 2.15x on length-grouped
min_confidence=... -> low_confidence flag + abstention field
```
Card also mentions `laya.load("convaiinnovations/laya", subfolder="multilingual")` AND separate repo `convaiinnovations/laya-multilingual` (exact HF id of the 322M checkpoint = convaiinnovations/laya-multilingual).

### Fine-tune notebook
https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb
- "Accelerator: Select GPU T4 x2"; torchrun --standalone --nproc_per_node=2 (DDP, NCCL)
- pip install -q -U "laya>=0.1.6" "transformers>=4.48.0" "datasets>=3.0.0" ...
- MODEL_ID = "convaiinnovations/laya"; dataset LocalLLaMA/typed-decisions ("all", train: 1,200 cases / 6,000 decisions); rows have JSON `state`, `questions`, `gold` (gold[qid]["probabilities"]) -> soft targets
- cfg max_len=1024, head_max_len=256, max_tokens_per_batch=4096
- EPOCHS=4, MICRO_BATCH=8/GPU, GRAD_ACCUM=4 (eff. 64), GROUP_SIZE=4, LR_ENCODER=2.5e-5, LR_HEAD=1e-4, SIGMA 0.4->0.1, AdamW-style param groups, CosineAnnealingLR eta_min=1e-6, fp16 autocast + GradScaler, grad clip 1.0, gradient checkpointing; FULL fine-tune (no LoRA)
- Loss: Gaussian-noise-perturbed logits (G=4 samples), reward = proper_reward(q, target, qtype, mask, w_sph=0.75, w_rps=1.0), group-mean/std normalized advantage, loss = loss_rl + 1.0 * soft cross-entropy(target)
- Post-train: LBFGS temperature fit per question type on held-out calib items (<=400), clamp
- Pushes to convaiinnovations/laya-typed-decisions
- RUNTIME CONFLICT: notebook markdown "Runs on both T4 GPUs in parallel (~4 to 6 minutes total)." vs README "Runtime on 2xT4 is roughly 4-5 hours for 4 epochs over ~30k questions." (different data sizes: 6k vs 30k questions). Also README says "The notebook's calibration samples come from its training items" while current notebook code holds them out -> doc/code drift.
- Apple Silicon alternative: notebooks/laya_finetune_typed_decisions_mps.py --micro-batch 1 --grad-accum 32
- Third-party: stuntd (github.com/bladedevoff/stuntd) trains per-decision head on frozen encoder behind Jev API.

### Papers
- No Convai tech report/arXiv found. Third-party arXiv 2609.28940 "Calibrated Decision Models for Autonomous Penetration-Testing Harnesses: JEV and Laya..." (not read in detail).

## JEV (TypeSafe AI)
- Site https://typesafe.ai ; docs https://docs.typesafe.ai (llms.txt, llms-full.txt) ; console https://console.typesafe.ai
- Launch post https://typesafe.ai/blog/introducing-system-one-models-and-jev timestamp "Sep 28, 2026, 7:32 PM UTC"; MarkTechPost article dated 2026-09-19; Terms "Last updated: Sep 19, 2026"; nibzard pilot ran 2026-09-26. => released ~Sep 19 2026 (blog timestamp likely an update). Access: "early access", "bringing developers off the waitlist as quickly as we can". No weights, no param count, no architecture beyond "new architecture, a parallel sampler, and RLCD".
- Founder: Diogo Almeida (InstructGPT/RLHF co-author) per docs.
- Models page: jev-1.13.0 (aliases jev-latest, jev-preview). "$42 / $0.042" per Btok/Mtok input, output free. Rate limits "100K tokens per second / 40 requests per second" (dynamic). "Context length: 64k tokens per request; 32k tokens for `state` plus the longest question". Text only. English primary language.
- Customization quote (docs.typesafe.ai/models): "Jev is not fine-tuned or LoRA-adapted with customer data. It is trained with RLCD to return calibrated decisions, and the same weights serve every account. You shape its answers to your domain through the request rather than through per-account weights:" state / instructions+criteria / decompose into atomic questions + combine in code (composite scoring; AutoResearch cookbook trains downstream CatBoost on Jev probabilities).
- "Jev is not trained on customer requests or responses." ZDR for enterprise.
- RLCD (docs machine-learning-primer): "Reinforcement learning for calibrated decisions trains TypeSafe to return decisions and calibrated probabilities instead of generated text." Contract: no text, decisions + probabilities, higher prob = higher chance correct. No algorithm, objective or paper disclosed by TypeSafe.
- Latency vendor claims: "70ms-500ms"; homepage "0.114 seconds", "193.6x Faster, 244.6x Cheaper".
- Calibration vendor claim: "Calibrated: higher confidence means higher accuracy." No ECE numbers published by TypeSafe.
- Jaggedness page (jev-1.13): double negatives/multi-hop hurt; large irrelevant state hurts; "State is data, and jev-1.13 does not treat it as hostile by default" (adversarial content can move answers) -> relevant to fraud.
- API: POST https://api.typesafe.ai/v1/systemone, Bearer key. Body {state (string|object|array), model, questions{id:{type, instructions, criteria}}}. Choice criteria map option->description|null, max 255. Score criteria ordered array 2..10 levels. Noul criteria optional {"true":..., "false":...}. Response {model, answers{id:{type, choice/score/noul, probabilities, confidence, legend(score)}}, usage{input_tokens, output_tokens}}. Noul answers carry no confidence field.
- Python SDK: pip install typesafe-sdk; from typesafe_sdk import TypeSafeClient, Choice, Noul, Score; client.system_one(state=..., questions={...}); response.nouls[k].noul / .choices[k].choice / .scores[k].score. GitHub typesafe-ai/typesafe-sdk-python.
- Terms of Use (site): govern site only; product agreements separate; no output license / high-risk-use clause found there.

### Independent Jev evidence
- AbdelStark/jev-benchmarks: jev-1.13.0, from France, p50 ~236-256 ms; AG News 0.910, Banking77 0.870 (72 labels), DAIR Emotion 0.480; DAIR "Jev is substantially worse calibrated", zero prob on true label 16%; n=100/condition.
- nibzard/decision-model-benchmark: p50 264-316 ms; Banking77 full test 79.2%, CLINC150 88.6%; ECE 0.246 on S5 uncertainty task (source of Laya's "0.246").
- dev.to "Jev After Eight Days of Independent Tests" (aggregator): median ECE 0.071 on public English tasks, 0.161 vs frontier labels; temperature fit on 50 labels cut calibration error up to 74%; option-name sensitivity 32.5%; poor non-English.
- arXiv 2609.29429 "Just Ask Jev" (third-party authors, 2026-09-24): median AUROC 0.886 zero-shot on alignment-failure detection; search snippet: "Jev's probabilities are calibrated when pooled but not within a benchmark", pooled Noul ECE 0.047, per-benchmark median ECE 0.168 vs null 0.074 (snippet only, not verified on page).
