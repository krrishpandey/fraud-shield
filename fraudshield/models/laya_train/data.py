"""states.jsonl -> official Laya notebook JSONL records -> tokenized training items.

Record schema (notebook): id, workflow, state, questions, gold, where state/questions/gold are JSON
strings. Extra fields (split, typology, ...) are kept for audit and ignored by the loader; they never
enter the state. Golds are hard 0/1 from injection truth (no smoothing; DESIGN section 7).
drop_consignee is never trained (zero-shot question).
"""
from __future__ import annotations

import json
from typing import Any, Iterable

import numpy as np
import pandas as pd

from fraudshield.contracts import QUESTIONS, TRAINED_QUESTIONS

HELD_OUT_TYPOLOGIES = ("T3", "T5")
YES_NO = ("misuse", "foreign_senders", "payoff_max")
ACTION_KEYS = tuple(QUESTIONS["action"]["criteria"].keys())
N_RISK = len(QUESTIONS["risk_level"]["criteria"])
EXTRA_FIELDS = ("seed", "split", "booking_ts", "account_id", "typology", "is_injected", "scenario_id",
                "campaign_id", "camouflage_level")


def _yes_no(v: int) -> dict[str, Any]:
    v = int(v)
    return {"label": "a" if v else "b", "probabilities": {"a": float(v), "b": float(1 - v)}}


def gold_for(row: dict | pd.Series, qids: Iterable[str] = TRAINED_QUESTIONS) -> dict[str, Any]:
    labels = row["labels"]
    out: dict[str, Any] = {}
    for q in qids:
        if q in YES_NO:
            out[q] = _yes_no(labels[q])
        elif q == "risk_level":
            lv = int(labels["risk_level"])
            out[q] = {"label": lv, "score": float(lv),
                      "probabilities": {str(i): float(i == lv) for i in range(N_RISK)}}
        elif q == "action":
            cost = {k: float(row["action_cost"][k]) for k in ACTION_KEYS}
            best = min(ACTION_KEYS, key=lambda k: (cost[k], ACTION_KEYS.index(k)))
            out[q] = {"label": best, "cost": cost, "probabilities": {k: float(k == best) for k in ACTION_KEYS}}
        else:
            raise ValueError(f"{q} is not a trained question")
    return out


def _clean(v):
    if isinstance(v, float) and np.isnan(v):
        return None
    if hasattr(v, "item"):
        return v.item()
    return v


def to_notebook_record(row: dict | pd.Series, qids: Iterable[str] = TRAINED_QUESTIONS) -> dict[str, Any]:
    qids = tuple(qids)
    rec = {"id": row["id"], "workflow": row["workflow"],
           "state": json.dumps(row["state"]),
           "questions": json.dumps({q: QUESTIONS[q] for q in qids}),
           "gold": json.dumps(gold_for(row, qids))}
    for k in EXTRA_FIELDS:
        if k in row:
            rec[k] = _clean(row[k])
    return rec


def _is_fraud(df: pd.DataFrame) -> pd.Series:
    return df.labels.map(lambda l: int(l["misuse"]) == 1)


def _is_hard_neg(df: pd.DataFrame) -> pd.Series:
    return df.hard_negative.map(lambda h: len(h) > 0) | df.typology.astype(str).str.startswith("HN_")


def sample_training_bookings(df: pd.DataFrame, n_hard_neg: int, n_legit: int, fraud_repeat: int = 1,
                             seed: int = 0) -> pd.DataFrame:
    """All eligible train fraud (x fraud_repeat), n_hard_neg hard negatives, n_legit random legit.

    Only split == train; held-out typologies T3 and T5 are excluded entirely. Deterministic in seed,
    rows shuffled.
    """
    rng = np.random.default_rng(seed)
    tr = df[(df.split == "train") & ~df.typology.isin(HELD_OUT_TYPOLOGIES)]
    fraud = _is_fraud(tr)
    hn = ~fraud & _is_hard_neg(tr)
    legit = ~fraud & ~hn
    parts = [pd.concat([tr[fraud]] * max(1, int(fraud_repeat)))]
    for mask, n in ((hn, n_hard_neg), (legit, n_legit)):
        pool = tr[mask]
        k = min(int(n), len(pool))
        parts.append(pool.iloc[np.sort(rng.choice(len(pool), size=k, replace=False))] if k else pool.iloc[:0])
    out = pd.concat(parts, ignore_index=True)
    return out.iloc[rng.permutation(len(out))].reset_index(drop=True)


def build_items(tok, records: list[dict], max_len: int = 512, head_max_len: int = 192):
    """Tokenize records into training items (notebook build_training_item, plus qid and cost).

    Returns (items, stats). stats counts items dropped by the marker check and items whose state was
    truncated; callers assert both are zero.
    """
    from laya.common import QTYPES, build_sequence, render_options  # noqa: PLC0415

    items: list[dict] = []
    stats = {"records": len(records), "items": 0, "dropped": 0, "truncated": 0, "max_state_tokens": 0,
             "max_len_seen": 0}
    for rec in records:
        state = json.loads(rec["state"])
        questions = json.loads(rec["questions"])
        gold = json.loads(rec["gold"])
        for qid, q in questions.items():
            if qid not in gold:
                continue
            g = gold[qid]
            t, crit = q["type"], q.get("criteria", {})
            if t == "choice":
                keys = list(crit.keys())
                target = [float(g["probabilities"].get(k, 0.0)) for k in keys]
            elif t == "score":
                target = [float(g["probabilities"].get(str(i), 0.0)) for i in range(len(crit))]
            else:
                raise ValueError(f"unsupported question type {t}")
            s = sum(target)
            target = [v / s for v in target] if s > 0 else [1.0 / len(target)] * len(target)
            qi = {"t": t, "ins": q["instructions"], "crit": crit}
            seq, markers, tstats = build_sequence(tok, state, qi, max_len, head_max_len,
                                                  return_truncation_stats=True)
            stats["max_state_tokens"] = max(stats["max_state_tokens"], tstats["state_tokens"])
            if tstats["truncated"]:
                stats["truncated"] += 1
            if len(markers) != len(render_options(qi)):
                stats["dropped"] += 1
                continue
            it = {"ids": seq, "markers": markers, "qtype": QTYPES[t], "target": target,
                  "label": target.index(max(target)), "qid": qid, "rec_id": rec["id"]}
            if "cost" in g:
                it["cost"] = [float(g["cost"][k]) for k in crit]
            stats["max_len_seen"] = max(stats["max_len_seen"], len(seq))
            items.append(it)
    stats["items"] = len(items)
    return items, stats


def eval_sample(df: pd.DataFrame, split: str, n_legit: int, seed: int = 0) -> pd.DataFrame:
    """Evaluation sample for one split: all injected rows (fraud and injected hard negatives, every
    seed, weight 1), all real hard-negative rows (weight 1), and n_legit random other real rows with
    weight N_other / n_legit, so weighted totals equal the real population."""
    rng = np.random.default_rng(seed)
    d = df[df.split == split]
    inj = d[d.is_injected.astype(bool)].assign(weight=1.0)
    real = d[~d.is_injected.astype(bool)]
    hn = real.hard_negative.map(lambda h: len(h) > 0)
    real_hn = real[hn].assign(weight=1.0)
    other = real[~hn]
    k = min(int(n_legit), len(other))
    pick = other.iloc[np.sort(rng.choice(len(other), size=k, replace=False))]
    pick = pick.assign(weight=len(other) / k if k else 1.0)
    return pd.concat([pick, real_hn, inj], ignore_index=True)


def seed_view(sample: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Real rows (shared across seeds) plus the injected rows of one injection seed."""
    inj = sample.is_injected.astype(bool)
    return sample[~inj | (sample.seed == seed)]
