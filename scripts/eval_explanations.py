"""Every sentence checked: prose explanations vs claim-by-claim (structured) explanations, and how the reasons they
cite agree with the LightGBM model's own attributions. Writes artifacts/results_explanations.md + .json.

Decisions: the seed-0 TEST window (2018-05-15 .. 2018-08-31) scored in booked_at order through the served decision
pipeline (config/app.yaml: LightGBM B2_F decides, Laya answers questions only, cost rule, seed-0 exploration) from
the precomputed as-of features (data/processed/features/seed_0.parquet; artifacts/stream_consistency.md shows the
live featurizer gives the same scores). Stopped = owner_confirm, review, hold or block. N decisions are drawn at
random (numpy seed 0, without replacement) from all stopped decisions.

Per decision:
- template claims (no LLM): per-claim check and attribution agreement (the control);
- (a) prose path (explain/llm.py) and (b) claims path (explain/claims.py, strict json_schema on Groq), each FIRST TRY
  only (retry_invalid=0, to keep Groq free-tier calls modest; the service retries once, so its fallback rate can
  only be lower than the first-try rejection rate measured here);
- planted lies, no LLM: a fabricated number, a true number moved to the wrong claim, a reason the decision does not
  have, a field the record does not have. Each is planted into the template claims and must be rejected.

Groq calls are paced to stay under the free tier's 8000 tokens/minute and cached in
artifacts/results_explanations_calls.jsonl (a rerun reuses them). LLM rows are paired: only decisions where both
paths got an answer. With no GROQ_API_KEY (or --no-llm), or after repeated rate limits, the rest are reported as
not run.

Usage: uv run --no-sync python scripts/eval_explanations.py [--n 60] [--no-llm | --cache-only]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fraudshield.api import real_components as RC  # noqa: E402
from fraudshield.api.app import _path, load_config  # noqa: E402
from fraudshield.api.pipeline import Pipeline  # noqa: E402
from fraudshield.contracts import FeatureVector  # noqa: E402
from fraudshield.explain.attribution import agreement, model_reasons  # noqa: E402
from fraudshield.explain.claims import explain_claims, template_claims, validate_claims  # noqa: E402
from fraudshield.explain.groq_client import GroqClient, load_dotenv  # noqa: E402
from fraudshield.explain.llm import check_text, explain  # noqa: E402
from fraudshield.explain.template import build_facts, render_template  # noqa: E402
from fraudshield.explain.validate import NUM_RE, REASON_WORDS, _number_ok, _record_values, validate  # noqa: E402
from fraudshield.features.mix import rule_flags, under_score  # noqa: E402
from fraudshield.features.spec import FEATURES  # noqa: E402
from fraudshield.features.store import FeatureStore  # noqa: E402
from fraudshield.models.calibration import load_calibration  # noqa: E402
from fraudshield.models.laya_client import LayaClient  # noqa: E402
from fraudshield.policy.costs import load_costs  # noqa: E402
from fraudshield.policy.decide import PolicyConfig  # noqa: E402
from fraudshield.policy.reasons import REASON_PLAIN, REASON_RULES  # noqa: E402

ART = ROOT / "artifacts"
CALLS = ART / "results_explanations_calls.jsonl"
STOP = ("owner_confirm", "review", "hold", "block")
TPM_BUDGET = 6500  # Groq free tier: 8000 tokens/minute for openai/gpt-oss-120b; keep a margin
MAX_429 = 4


class _NullAudit:
    def append(self, event_type, payload):
        return 0, "eval"


# ---------- decisions ----------
def score_test_window() -> tuple[list[dict], dict]:
    cfg = load_config(None)
    pol = cfg.get("policy") or {}
    policy = PolicyConfig(**{k: v for k, v in pol.items() if k in PolicyConfig.__dataclass_fields__})
    laya = LayaClient("cached", None, model="none", cache={}, mode_reason="laya.decide is false")
    pipe = Pipeline(lambda b: None, RC.serializer, RC.gbm, laya, _NullAudit(),
                    calibration=load_calibration(_path(cfg["calibration_path"])),
                    costs=load_costs(_path(cfg["costs_path"])), policy=policy, seed=int(pol.get("seed", 0)))
    df = pd.read_parquet(ROOT / "data/processed/features/seed_0.parquet")
    te = df[df.split == "test"].sort_values(["booked_at", "booking_id"], kind="mergesort")
    frame = RC._get_store().frame.set_index("booking_id", drop=False)
    cols = [c for c in te.columns if c in FEATURES]
    t0 = time.perf_counter()
    recs, skipped = [], 0
    for row in te[["booking_id", "booked_at", *cols]].to_dict("records"):
        bid = row["booking_id"]
        if bid not in frame.index:  # the five demo bookings live outside the demo store
            skipped += 1
            continue
        b = FeatureStore.booking_from_row(frame.loc[bid])
        v = {c: (None if isinstance(row[c], float) and np.isnan(row[c]) else row[c]) for c in cols}
        v.update(rule_flags(v))
        v["under_score"] = under_score(v)
        out = pipe.score(b, FeatureVector(bid, str(row["booked_at"]), v))
        recs.append(pipe.get(out["decision_id"]))
    info = {"test_bookings": int(len(te)), "scored": len(recs), "skipped_demo": skipped,
            "score_seconds": round(time.perf_counter() - t0, 1)}
    return recs, info


# ---------- Groq pacing and cache ----------
class PacedClient:
    """Wraps GroqClient: sleeps so the last minute's tokens stay under TPM_BUDGET; counts calls and tokens."""
    supports_response_format = True

    def __init__(self, inner: GroqClient):
        self.inner = inner
        self.window: deque = deque()
        self.calls = 0
        self.tokens = 0
        self.messages = self

    def create(self, **kw):
        while True:
            now = time.time()
            while self.window and now - self.window[0][0] > 60:
                self.window.popleft()
            if sum(t for _, t in self.window) + 1500 <= TPM_BUDGET:
                break
            time.sleep(max(1.0, 60 - (now - self.window[0][0]) + 0.5))
        r = self.inner.messages.create(**kw)
        used = int((r.usage or {}).get("total_tokens") or 1500)
        self.window.append((time.time(), used))
        self.calls += 1
        self.tokens += used
        return r


def _load_calls() -> dict[str, dict]:
    if not CALLS.exists():
        return {}
    out = {}
    for line in CALLS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            x = json.loads(line)
            if "429" in str(x.get("error")):  # rate-limited: kept in the file as a log, retried on the next run
                continue
            out[f"{x['booking_id']}|{x['path']}"] = x
    return out


def _save_call(x: dict) -> None:
    with CALLS.open("a", encoding="utf-8") as f:
        f.write(json.dumps(x, default=str) + "\n")


# ---------- measures ----------
SENT_RE = re.compile(r"(?<=[.!?])\s+")
GROUND_PROBLEMS = ("number not in record", "id or code not in record", "banned phrase")


def prose_sentences_grounded(rec: dict, text: str) -> tuple[int, int]:
    """Prose has no claims: count sentences whose numbers and ids are all in the record (the prose validator's
    record-wide test, applied per sentence)."""
    facts, template = build_facts(rec), render_template(rec)
    sents = [s for s in SENT_RE.split(text.strip()) if s.strip()]
    ok = 0
    for s in sents:
        _, probs = validate(s, facts, template=template)
        ok += not any(p.startswith(GROUND_PROBLEMS) for p in probs)
    return ok, len(sents)


def prose_cited_codes(rec: dict, text: str) -> list[str]:
    """Reasons a prose text mentions, in order of first mention (keyword match with the validator's REASON_WORDS,
    over the decision's own reasons: approximate, 'sender' fits three codes)."""
    low = text.lower()
    pos = []
    for c in rec.get("reasons") or []:
        hits = [low.find(w) for w in REASON_WORDS.get(c, [c.lower()]) if w in low]
        if hits:
            pos.append((min(hits), c))
    return [c for _, c in sorted(pos)]


def record_level_ok(claim: dict) -> bool:
    return not any(p.startswith(GROUND_PROBLEMS + ("field not in record",)) for p in claim["problems"])


# ---------- planted lies ----------
def plant(rec: dict, rng: np.random.Generator) -> dict[str, Any]:
    base = template_claims(rec)
    facts, template = build_facts(rec), render_template(rec)
    nums, _ = _record_values(facts, template)
    out: dict[str, Any] = {}
    if not base["claims"]:
        return out
    c0 = base["claims"][0]

    def run(claims_obj):
        return not validate_claims(rec, claims_obj)["ok"]

    # 1. fabricated number: a number that appears nowhere in the record
    cand = [n for n in range(2, 100) if not _number_ok(str(n), nums)]
    fake = str(int(rng.choice(cand)))
    m = NUM_RE.search(c0["text"])
    text = (c0["text"][:m.start()] + fake + c0["text"][m.end():]) if m else c0["text"].rstrip(".") + f", {fake} times."
    lie = {**base, "claims": [{**c0, "text": text, "numbers": [float(fake)]}, *base["claims"][1:]]}
    out["fabricated_number"] = run(lie)
    # 2. a true number from another reason moved into the first claim (record-wide check would accept it)
    own: list[float] = []
    for n in NUM_RE.findall(c0["text"]):
        own.append(float(n))
    others = [t.get("value") for t in rec.get("top_features") or [] if t.get("code") != c0["reason_code"]]
    others = [o for o in others if isinstance(o, (int, float)) and not any(abs(o - x) < 1e-9 for x in own)]
    if m and others:
        moved = others[0]
        tok = str(int(moved)) if float(moved).is_integer() else f"{moved:.1f}" if abs(moved) >= 1 else f"{moved:.2f}"
        if not _number_ok(tok, own):
            text2 = c0["text"][:m.start()] + tok + c0["text"][m.end():]
            lie2 = {**base, "claims": [{**c0, "text": text2, "numbers": [float(tok)]}, *base["claims"][1:]]}
            out["moved_number"] = run(lie2)
            # the same lie in the prose template, judged by the prose validator
            i = template.find(c0["text"][:1].lower() + c0["text"][1:-1])
            if i >= 0:
                j = i + m.start()
                prose = template[:j] + tok + template[j + (m.end() - m.start()):]
                out["moved_number_prose_rejected"] = not check_text(rec, prose)["ok"]
    # 3. a reason the decision does not have
    other = next(r.code for r in REASON_RULES if r.code not in (rec.get("reasons") or []))
    lie3 = {**base, "claims": [*base["claims"], {"text": REASON_PLAIN[other] + ".", "reason_code": other,
                                                 "cited_fields": [], "numbers": []}]}
    out["invented_reason"] = run(lie3)
    # 4. a field the record does not have
    lie4 = {**base, "claims": [{**c0, "cited_fields": [*c0["cited_fields"], "fraud_ring_score"]},
                               *base["claims"][1:]]}
    out["invented_field"] = run(lie4)
    return out


# ---------- main ----------
def _mean_sd(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None, 0
    a = np.asarray(xs, float)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0, len(a)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--llm-first", type=int, default=None, help="call the LLM only for the first K sampled decisions")
    ap.add_argument("--cache-only", action="store_true", help="no new Groq calls: report the cached LLM answers")
    a = ap.parse_args()

    recs, info = score_test_window()
    stopped = [r for r in recs if r["action"] in STOP]
    rng = np.random.default_rng(0)
    idx = sorted(rng.choice(len(stopped), size=min(a.n, len(stopped)), replace=False))
    sample = [stopped[i] for i in idx]
    info.update(stopped=len(stopped), sampled=len(sample),
                actions={k: sum(r["action"] == k for r in sample) for k in STOP})
    print(json.dumps(info), flush=True)

    model = RC._get_gbm()
    load_dotenv(ROOT / ".env")
    key = os.environ.get("GROQ_API_KEY")
    llm = None if a.no_llm or a.cache_only or not key else PacedClient(GroqClient(key, timeout=60.0))
    use_llm = llm is not None or a.cache_only
    model_id = (load_config(None).get("explain") or {}).get("groq_model_id", "openai/gpt-oss-120b")
    cache = _load_calls()
    rows = []
    n429 = 0  # consecutive rate-limited calls; after MAX_429 the remaining LLM rows are reported as not run
    lie_rng = np.random.default_rng(1)
    for k, rec in enumerate(sample):
        mr = model_reasons(model, rec["feature_values"])
        mcodes = [x["code"] for x in mr["top_codes"]]
        reasons = list(rec.get("reasons") or [])
        t = validate_claims(rec, template_claims(rec))
        row: dict[str, Any] = {
            "booking_id": rec["booking_id"], "action": rec["action"], "reasons": reasons, "model_top": mcodes,
            "mapped_share": mr["mapped_share"],
            "template": {"ok": t["ok"], "claims": len(t["claims"]), "claims_ok": sum(c["ok"] for c in t["claims"]),
                         "agreement": agreement([c["reason_code"] for c in t["claims"] if c["ok"]], mcodes, reasons)},
            "lies": plant(rec, lie_rng),
        }
        if use_llm and a.llm_first is not None and k >= a.llm_first:
            rows.append(row)
            continue
        if use_llm:
            for path in ("prose", "claims"):
                ck = f"{rec['booking_id']}|{path}"
                if ck not in cache and (llm is None or n429 >= MAX_429):
                    continue
                if ck not in cache:
                    t0 = time.perf_counter()
                    if path == "prose":
                        exp, log = explain(rec, client=llm, model_id=model_id, use_default_client=False,
                                           retry_invalid=0)
                        res = {"source": exp.source, "text": log.get("llm_output"), "error": log.get("error"),
                               "validator": log.get("validator")}
                    else:
                        exp, log, payload = explain_claims(rec, client=llm, model_id=model_id, retry_invalid=0)
                        att = (log.get("attempts") or [{}])[-1]
                        res = {"source": exp.source, "mode": payload["mode"], "error": log.get("error"),
                               "mode_errors": log.get("mode_errors"), "llm_output": log.get("llm_output"),
                               "validator": log.get("validator"), "claims": att.get("claims") or []}
                    cache[ck] = {"booking_id": rec["booking_id"], "path": path, "model_id": model_id,
                                 "seconds": round(time.perf_counter() - t0, 2), **res}
                    _save_call(cache[ck])
                    if "429" in str(res.get("error")):  # the shared key is over its limit: back off before the next
                        n429 += 1
                        if n429 < MAX_429:
                            time.sleep(120)
                    else:
                        n429 = 0
                row[path] = cache[ck]
        rows.append(row)
        print(f"{k + 1}/{len(sample)} {rec['booking_id']} {rec['action']}"
              + (f" prose={row.get('prose', {}).get('source', 'not run')} "
                 f"claims={row.get('claims', {}).get('source', 'not run')}" if use_llm else ""), flush=True)

    summary = summarize(rows, sample, use_llm)
    summary["data"] = info
    summary["llm"] = {"ran": use_llm, "model_id": model_id if use_llm else None, "cache_only": a.cache_only,
                      "calls_this_run": llm.calls if llm else 0, "tokens_this_run": llm.tokens if llm else 0,
                      "rate_limited_calls_logged": sum("429" in line for line in CALLS.read_text(encoding="utf-8")
                                                       .splitlines()) if CALLS.exists() else 0,
                      "why_not": None if use_llm else ("--no-llm" if a.no_llm else "no GROQ_API_KEY")}
    (ART / "results_explanations.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1,
                                                              default=str), encoding="utf-8")
    write_md(summary)
    print(json.dumps(summary, indent=1, default=str))


def _rate(xs):
    xs = [x for x in xs if x is not None]
    return (sum(xs), len(xs))


def summarize(rows: list[dict], sample: list[dict], llm_ran: bool) -> dict:
    recs = {r["booking_id"]: r for r in sample}
    s: dict[str, Any] = {}
    tm = [r["template"] for r in rows]
    s["template"] = {
        "n": len(rows), "rejected": sum(not x["ok"] for x in tm),
        "claims_grounded": (sum(x["claims_ok"] for x in tm), sum(x["claims"] for x in tm)),
        **_agree([x["agreement"] for x in tm]),
    }
    s["reachable_mean"] = _mean_sd([r["template"]["agreement"]["reachable"] for r in rows])[0]
    s["model_top_n_mean"] = _mean_sd([len(r["model_top"]) for r in rows])[0]
    s["hit_at_3_ceiling"] = _mean_sd([r["template"]["agreement"]["reachable"] / len(r["model_top"])
                                      for r in rows if r["model_top"]])[0]
    s["mapped_share"] = _mean_sd([r["mapped_share"] for r in rows])
    lies: dict[str, Any] = {}
    for k in ("fabricated_number", "moved_number", "invented_reason", "invented_field", "moved_number_prose_rejected"):
        lies[k] = _rate([r["lies"].get(k) for r in rows])
    s["lies"] = lies
    if not llm_ran:
        s["prose"] = s["claims"] = None
        return s
    rate_limited = lambda x: bool(x.get("error")) and "429" in str(x.get("error"))  # noqa: E731
    # paired: only decisions where BOTH LLM paths ran (not rate-limited), so (a) and (b) see the same decisions
    pr = [r for r in rows if "prose" in r and "claims" in r and not rate_limited(r["prose"])
          and not rate_limited(r["claims"])]
    s["template_paired"] = _agree([r["template"]["agreement"] for r in pr])
    s["prose"] = {"n": len(pr), "not_run": len(rows) - len(pr),
                  "api_errors": sum(bool(r["prose"]["error"]) and not r["prose"]["text"] for r in pr),
                  "rejected_first_try": sum(r["prose"]["source"] != "llm" for r in pr)}
    g_ok = g_n = 0
    ag = []
    for r in pr:
        if r["prose"]["text"]:
            o, n = prose_sentences_grounded(recs[r["booking_id"]], r["prose"]["text"])
            g_ok, g_n = g_ok + o, g_n + n
            ag.append(agreement(prose_cited_codes(recs[r["booking_id"]], r["prose"]["text"]), r["model_top"],
                                r["reasons"]))
    s["prose"]["sentences_grounded"] = (g_ok, g_n)
    s["prose"].update(_agree(ag))
    cl = pr
    s["claims"] = {"n": len(cl), "not_run": len(rows) - len(cl),
                   "modes": {m: sum(r["claims"]["mode"] == m for r in cl) for m in
                             ("json_schema", "json_object", "prose", "template")},
                   "api_errors": sum(bool(r["claims"]["error"]) for r in cl),
                   "rejected_first_try": sum(r["claims"]["source"] != "llm" for r in cl)}
    allc = [c for r in cl for c in r["claims"]["claims"]]
    s["claims"]["claims_grounded"] = (sum(c["ok"] for c in allc), len(allc))
    s["claims"]["claims_record_level_grounded"] = (sum(record_level_ok(c) for c in allc), len(allc))
    probs: dict[str, int] = {}
    for c in allc:
        for p in c["problems"]:
            key = re.sub(r":.*|\d+(\.\d+)?|[A-Z_]{4,}", "", p).strip()
            probs[key] = probs.get(key, 0) + 1
    s["claims"]["claim_problem_counts"] = dict(sorted(probs.items(), key=lambda kv: -kv[1]))
    s["claims"].update(_agree([agreement([c["reason_code"] for c in r["claims"]["claims"]], r["model_top"],
                                         r["reasons"]) for r in cl]))
    for path in ("prose", "claims"):  # why whole explanations were rejected (first try)
        s[path]["rejections"] = [p for r in pr for p in ((r[path].get("validator") or {}).get("problems") or [])
                                 if r[path]["source"] != "llm"]
    return s


def _agree(ags: list[dict]) -> dict:
    ok = [x for x in ags if x.get("hits") is not None]
    m, sd, n = _mean_sd([x["hit_at_3"] for x in ok])
    return {"agreement_n": n, "hit_at_3_mean": m, "hit_at_3_sd": sd,
            "top1_match": _rate([x["top1_match"] for x in ok]),
            "hits_hist": {h: sum(x["hits"] == h for x in ok) for h in range(4)}}


def _pct(pair) -> str:
    k, n = pair
    return f"{k}/{n} ({100 * k / n:.1f}%)" if n else "n/a"


def _f3(x) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def write_md(s: dict) -> None:
    d = s["data"]
    L = ["# Every sentence checked: claim-by-claim explanations", "",
         f"Measured {time.strftime('%Y-%m-%d %H:%M')} by `scripts/eval_explanations.py`. Split: seed-0 TEST window "
         f"only ({d['test_bookings']} bookings; {d['scored']} scored through the served pipeline from precomputed "
         f"as-of features, {d['skipped_demo']} demo bookings outside the demo store skipped). "
         f"{d['stopped']} decisions were stopped (owner_confirm, review, hold, block); **n = {d['sampled']}** were "
         f"drawn at random from them (numpy seed 0, no replacement): "
         + ", ".join(f"{k} {v}" for k, v in d["actions"].items()) + ". One seed, one sample: no ± over seeds.",
         "", "Model attributions: LightGBM `pred_contrib` (TreeSHAP) of the served B2_F model, summed per reason "
         "code (fraudshield/explain/attribution.py). hit@3 = share of the model's top reason codes (at most 3) "
         "that the explanation cites among its first 3 distinct codes; top-1 = the first cited code is the model's "
         "top code.", ""]
    L += ["## Planted lies (no LLM; planted into the template's claims)", "",
          "| Lie | Rejected / planted |", "|---|---|"]
    names = {"fabricated_number": "Fabricated number (in no field of the record)",
             "moved_number": "True number moved to the wrong claim (claim checker)",
             "moved_number_prose_rejected": "Same moved number in the prose template (old prose validator)",
             "invented_reason": "Claim about a reason this decision does not have",
             "invented_field": "Cites a field the record does not have"}
    for k, nm in names.items():
        L.append(f"| {nm} | {_pct(s['lies'][k])} |")
    L += ["", "The fabricated-number, invented-reason and invented-field rows are rejected by construction (the "
          "checker compares against the record): they show the checker is wired end to end, not a property of any "
          "LLM. The moved-number row is the one the old record-wide prose check cannot see.", ""]
    L += ["## Prose vs claims (Groq " + (s["llm"]["model_id"] or "not run") + ", first try, no retry)", ""]
    t = s["template"]
    if s["prose"] is None:
        L += [f"LLM paths: **not run** ({s['llm']['why_not']}). Template (control) only:", ""]
    L += ["| | Template (control, no LLM) | (a) Prose path | (b) Claims path |", "|---|---|---|---|"]
    p, c = s["prose"], s["claims"]
    nr = "not run"
    L.append(f"| Decisions | {t['n']} | {p['n'] if p else nr} | {c['n'] if c else nr} |")
    L.append(f"| Rejected on first try -> template | {t['rejected']} | "
             f"{(_pct((p['rejected_first_try'], p['n']))) if p else nr} | "
             f"{(_pct((c['rejected_first_try'], c['n']))) if c else nr} |")
    L.append(f"| Grounded units | claims {_pct(t['claims_grounded'])} | "
             f"{('sentences ' + _pct(p['sentences_grounded'])) if p else nr} | "
             f"{('claims ' + _pct(c['claims_grounded'])) if c else nr} |")
    if c:
        L.append(f"| Claims grounded, record-wide test only | | | {_pct(c['claims_record_level_grounded'])} |")
    for nm, key in (("hit@3 vs model top reasons, mean (sd)", "hit"), ("Top-1 match", "top1")):
        cells = []
        for x in (t, p, c):
            if x is None:
                cells.append(nr)
            elif key == "hit":
                cells.append(f"{_f3(x['hit_at_3_mean'])} ({_f3(x['hit_at_3_sd'])}), n={x['agreement_n']}")
            else:
                cells.append(_pct(x["top1_match"]))
        L.append(f"| {nm} | " + " | ".join(cells) + " |")
    if c:
        tp = s["template_paired"]
        L.append(f"| hit@3, template on the same {c['n']} decisions | {_f3(tp['hit_at_3_mean'])} "
                 f"({_f3(tp['hit_at_3_sd'])}), n={tp['agreement_n']} | | |")
        L.append(f"| Mode used | | | " + ", ".join(f"{k} {v}" for k, v in c["modes"].items() if v) + " |")
        L.append(f"| API errors / not run (rate-limited or skipped) | | {p['api_errors']} / {p['not_run']} | "
                 f"{c['api_errors']} / {c['not_run']} |")
    L += ["", f"Ceiling: a validated explanation may only cite the decision's own reasons (codes whose rule fired); "
          f"on average {_f3(s['reachable_mean'])} of the model's {_f3(s['model_top_n_mean'])} top codes are among "
          f"them, so no checked explanation can score a mean hit@3 above {_f3(s['hit_at_3_ceiling'])}. Share of the "
          f"model's positive push that any reason code describes: mean "
          f"{_f3(s['mapped_share'][0])} (sd {_f3(s['mapped_share'][1])}). Low agreement is therefore mostly a gap "
          "between the fixed reason rules and what LightGBM leans on, not the explainer's wording.", ""]
    if c:
        L += [f"LLM rows are paired: (a) and (b) on the same {c['n']} decisions where both ran. The other "
              f"{c['not_run']} sampled decisions were **not run**: the Groq free tier's 200,000 tokens/day limit for "
              f"this model (shared with everything else using the key today) was reached mid-run "
              f"({s['llm']['rate_limited_calls_logged']} rate-limited calls logged in the calls file). Nothing was "
              "filled in for them.", ""]
        L += ["Why LLM claims failed (counts over all claims):", ""]
        L += [f"- {k}: {v}" for k, v in c["claim_problem_counts"].items()] or ["- none"]
        L += ["", "Why whole explanations were rejected on first try: prose "
              + (", ".join(p["rejections"]) or "none") + "; claims " + (", ".join(c["rejections"]) or "none") + ".",
              ""]
    L += ["Notes:", "",
          "- Prose sentences are judged with the record-wide number/id test (the prose validator's own rule); "
          "claims additionally must use their own reason's numbers, a real reason code and real fields, so the "
          "claims column is the stricter test. The record-wide row puts both on the same test.",
          "- Prose 'cited reasons' are found by keyword (REASON_WORDS) in order of first mention: approximate.",
          "- First try only (retry_invalid=0); the service retries once with the checker's problems.",
          "- Every LLM answer used here is in artifacts/results_explanations_calls.jsonl (one line per call, "
          "rate-limited calls included as a log); reruns reuse them (`--cache-only` makes no new calls). "
          f"New Groq calls in this run: {s['llm']['calls_this_run']}.",
          "- Sources: Groq Structured Outputs (strict json_schema, constrained decoding on openai/gpt-oss-120b; "
          "https://console.groq.com/docs/structured-outputs). arXiv 2512.00163: LLM self-explanations disagree with "
          "SHAP; LightGBM SHAP is more reliable on financial tabular data. arXiv 2605.26770: LLM-written XAI "
          "narratives raised confidence without improving accuracy and made LLM judges worse, hence a fixed "
          "validator, not an LLM judge."]
    (ART / "results_explanations.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
