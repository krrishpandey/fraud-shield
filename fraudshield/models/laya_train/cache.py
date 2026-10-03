"""Write artifacts/laya_cache.json in the format LayaClient.cached reads:
{cache_key(state, served qids): {qid: {option: prob}}, "_revision": <weights sha12>, "_meta": {...}}.
Probabilities are the raw (temperature 1.0) model outputs; calibration is applied at serve time."""
from __future__ import annotations

from typing import Any, Iterable

from fraudshield.contracts import SERVED_QUESTIONS
from fraudshield.models.laya_client import cache_key


def answers_to_probs(resp: dict[str, Any], qids: Iterable[str] = SERVED_QUESTIONS) -> dict[str, dict[str, float]]:
    ans = resp.get("answers", resp)
    return {q: {str(k): float(v) for k, v in ans[q]["probabilities"].items()} for q in qids}


def build_cache(entries: Iterable[tuple[str, dict]], revision: str, meta: dict | None = None,
                qids: tuple[str, ...] = SERVED_QUESTIONS) -> dict[str, Any]:
    doc: dict[str, Any] = {}
    for state, resp in entries:
        doc[cache_key(state, list(qids))] = answers_to_probs(resp, qids)
    doc["_revision"] = revision
    doc["_meta"] = {**(meta or {}), "entries": len(doc) - 1, "questions": list(qids)}
    return doc
