"""LLM rewrite of the template explanation (Anthropic SDK). Never on the decision path.

The prompt contains ONLY the structured facts and the template. temperature 0, ~120 words.
Missing ANTHROPIC_API_KEY, API error or validator failure -> the template is shown.
Returns (Explanation, log) where log is the audit 'explanation' event payload.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any

from fraudshield.contracts import Explanation
from fraudshield.explain.template import build_facts, render_template
from fraudshield.explain.validate import validate

DEFAULT_MODEL_ID = "claude-haiku-4-5-20251001"
PROMPT_VERSION = "explain-v1"
SYSTEM = (
    "You write short explanations of parcel-booking fraud decisions for fraud analysts. "
    "Use only the facts provided. Do not add numbers, places, names, ids or causes that are not in the facts. "
    "Do not state that fraud is confirmed. Name the recommended action and at least two of the listed reasons, "
    "and say what the analyst should check next, using the provided checks. Plain business language, "
    "at most 120 words, no lists, no em dashes."
)


def build_prompt(facts: dict[str, Any], template: str) -> tuple[str, str, str]:
    user = "FACTS (JSON):\n" + json.dumps(facts, sort_keys=True) + "\n\nTEMPLATE EXPLANATION:\n" + template
    h = hashlib.sha256((PROMPT_VERSION + "\n" + SYSTEM + "\n" + user).encode("utf-8")).hexdigest()
    return SYSTEM, user, h


def _default_client():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    import anthropic  # noqa: PLC0415
    return anthropic.Anthropic(timeout=15.0, max_retries=1)


def explain(record: dict[str, Any], client: Any = None, model_id: str = DEFAULT_MODEL_ID,
            use_default_client: bool = True) -> tuple[Explanation, dict[str, Any]]:
    facts = build_facts(record)
    template = render_template(record)
    system, user, h = build_prompt(facts, template)
    log: dict[str, Any] = {"decision_id": record.get("decision_id"), "prompt_version": PROMPT_VERSION,
                           "prompt_hash": h, "model_id": model_id, "template": template,
                           "llm_output": None, "validator": None, "error": None}
    if client is None and use_default_client:
        client = _default_client()
    if client is None:
        log["error"] = "no ANTHROPIC_API_KEY; template used"
        log["source"] = "template"
        return Explanation(template, "template", True, None, h), log
    try:
        resp = client.messages.create(model=model_id, max_tokens=400, temperature=0, system=system,
                                      messages=[{"role": "user", "content": user}])
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()
        if getattr(resp, "stop_reason", "end_turn") == "refusal" or not text:
            raise RuntimeError(f"no usable LLM text (stop_reason={getattr(resp, 'stop_reason', None)})")
    except Exception as e:  # any SDK or network error -> template
        log["error"] = f"{type(e).__name__}: {e}"
        log["source"] = "template"
        return Explanation(template, "template", True, model_id, h), log
    log["llm_output"] = text
    words = len(text.split())
    ok, problems = validate(text, facts, template=template)
    if words > 140:
        ok, problems = False, problems + [f"too long: {words} words"]
    log["validator"] = {"ok": ok, "problems": problems}
    if ok:
        log["source"] = "llm"
        return Explanation(text, "llm", True, model_id, h), log
    log["source"] = "template"
    return Explanation(template, "template", True, model_id, h), log
