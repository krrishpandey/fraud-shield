"""Claim-by-claim explanations: every sentence is checked on its own.

The explainer asks the LLM for JSON {action, claims: [{text, reason_code, cited_fields[], numbers[]}]}. On Groq this
uses Structured Outputs with a strict json_schema (constrained decoding on openai/gpt-oss-120b: every field required,
additionalProperties false, no streaming; https://console.groq.com/docs/structured-outputs). If the API rejects the
schema (HTTP 400) or the answer is not the asked-for JSON, it tries json_object mode, then the prose path
(explain/llm.py). Which mode produced the text is recorded ("mode").

validate_claims() checks each claim: its reason_code is one of the decision's reasons; the sentence talks about that
reason; every number in the text and in numbers[] is in the decision record (the prose validator's matching) and
comes from the claim's own cited fields or its reason's fixed wording (so a true number moved to the wrong claim is
caught); every cited field exists; no unknown id, no banned phrase. The explanation passes only if it names the
decision's action, every claim passes, and it covers at least 2 of the top 3 reasons (as the prose validator).
Any failing claim -> the template is shown, and the rejected claims are kept so the console can show which one
failed. A fixed rule check, not an LLM judge: LLM-written XAI narratives raised confidence without improving accuracy
and made LLM judges worse (arXiv 2605.26770).

The template produces the same structure (one claim per reason), so the console and tests work with no LLM.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

import httpx

from fraudshield.contracts import Explanation
from fraudshield.explain.llm import MAX_WORDS, explain
from fraudshield.explain.template import build_facts, render_template
from fraudshield.explain.validate import BANNED, ID_RE, NUM_RE, REASON_WORDS, _number_ok, _record_values
from fraudshield.policy.reasons import REASON_PLAIN, RULES_BY_CODE

CLAIMS_PROMPT_VERSION = "claims-v1"
MAX_CLAIMS = 5
CLAIMS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["action", "claims"],
    "properties": {
        "action": {"type": "string"},
        "claims": {"type": "array", "items": {
            "type": "object",
            "additionalProperties": False,
            "required": ["text", "reason_code", "cited_fields", "numbers"],
            "properties": {
                "text": {"type": "string"},
                "reason_code": {"type": "string"},
                "cited_fields": {"type": "array", "items": {"type": "string"}},
                "numbers": {"type": "array", "items": {"type": "number"}},
            },
        }},
    },
}
MODES = ("json_schema", "json_object")  # then "prose" (explain/llm.py)
SYSTEM_CLAIMS = (
    "You explain parcel-booking fraud decisions to fraud analysts as a short list of claims, returned as JSON. "
    "Use only the facts provided. Each claim is one plain-language sentence about exactly one of the listed reasons: "
    "set reason_code to that reason's code, list in cited_fields the FIELDS whose values the sentence uses, and list "
    "in numbers every number the sentence states. Do not add numbers, places, names, ids or causes that are not in "
    "the facts. Do not state that fraud is confirmed. Set action to the decision's action code. Cover at least two "
    "of the listed reasons (all of them if fewer; an empty claims list if none are listed), most important first, "
    "at most 25 words per claim, no em dashes."
)
RETRY_NOTE = ("The automatic checker rejected your claims: {problems}. Return corrected JSON: use only numbers from "
              "the cited FIELDS or the reason's own text, only reason codes from the facts, and the action code.")


def _reason_words(code: str) -> list[str]:
    return REASON_WORDS.get(code, [code.replace("_", " ").lower()])


def record_fields(record: dict[str, Any]) -> dict[str, Any]:
    """Field name -> value: what a claim may cite. Facts scalars, the question probabilities, and each reason's
    feature value (name) and usual value (<name>_usual)."""
    f = build_facts(record)
    out: dict[str, Any] = {}
    for k in ("p_fraud", "carrier_cost_brl", "origin_zip3", "dest_zip3", "origin_uf", "dest_uf",
              "expected_cost_action_brl", "expected_cost_allow_brl"):
        if f.get(k) is not None:
            out[k] = f[k]
    for q, v in (f.get("question_probabilities") or {}).items():
        out[f"p_{q}"] = v
    fv = record.get("feature_values") or {}
    for t in record.get("top_features") or []:
        if t.get("name"):
            out[t["name"]] = t.get("value")
            if t.get("baseline") is not None:
                out[f"{t['name']}_usual"] = t.get("baseline")
    for code in record.get("reasons") or []:
        r = RULES_BY_CODE.get(code)
        if r is not None and r.feature not in out and fv.get(r.feature) is not None:
            out[r.feature] = fv[r.feature]
    return out


def _claim_fields_for(record: dict[str, Any], code: str) -> list[str]:
    fields = record_fields(record)
    t = next((x for x in record.get("top_features") or [] if x.get("code") == code), None)
    if t is None:
        r = RULES_BY_CODE.get(code)
        return [r.feature] if r is not None and r.feature in fields else []
    out = [t["name"]]
    r = RULES_BY_CODE.get(code)
    if r is not None and "{b}" in r.text and f"{t['name']}_usual" in fields:
        out.append(f"{t['name']}_usual")
    return out


def template_claims(record: dict[str, Any]) -> dict[str, Any]:
    """The template's reasons as claims: one sentence per reason (top 3), with its code, fields and numbers."""
    f = build_facts(record)
    claims = []
    for r in f["reasons"][:3]:
        text = r["text"][:1].upper() + r["text"][1:] + "."
        claims.append({"text": text, "reason_code": r["code"], "cited_fields": _claim_fields_for(record, r["code"]),
                       "numbers": [float(m.replace(",", ".")) for m in NUM_RE.findall(text)]})
    return {"action": f["action"], "claims": claims}


def _num_list(xs: Any) -> list[Any]:
    return list(xs) if isinstance(xs, (list, tuple)) else []


def _num_token(x: Any) -> str | None:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    return str(int(x)) if float(x).is_integer() else repr(float(x))


def _check_claim(c: dict[str, Any], *, reasons: list[str], fields: dict[str, Any], nums: list[float], blob: str,
                 reason_text: dict[str, str]) -> list[str]:
    problems: list[str] = []
    text = c.get("text") if isinstance(c.get("text"), str) else ""
    code = c.get("reason_code") if isinstance(c.get("reason_code"), str) else ""
    cited = [x for x in _num_list(c.get("cited_fields")) if isinstance(x, str)]
    if not text.strip():
        return ["empty claim"]
    low = text.lower()
    if code not in reasons:
        problems.append(f"reason {code or '(none)'} is not one of this decision's reasons")
    elif not any(w in low for w in _reason_words(code)):
        problems.append(f"the sentence does not talk about {code}")
    for name in cited:
        if name not in fields:
            problems.append(f"field not in record: {name}")
    # numbers this claim may state: its cited fields' values and its reason's fixed wording ("of the last 10")
    own: list[float] = []
    _walk_vals([fields[n] for n in cited if n in fields] + [reason_text.get(code, "")], own)
    stated = [m for m in NUM_RE.findall(text)] + [t for t in (_num_token(x) for x in _num_list(c.get("numbers"))) if t]
    for x in _num_list(c.get("numbers")):
        if _num_token(x) is None:
            problems.append(f"not a number: {x!r}")
    for tok in dict.fromkeys(stated):
        if not _number_ok(tok, nums):
            problems.append(f"number not in record: {tok}")
        elif not _number_ok(tok, own):
            problems.append(f"number {tok} is not from this claim's cited fields")
    for tok in ID_RE.findall(text):
        if re.fullmatch(r"x?\d+(?:\.\d+)?x?|\d+(?:st|nd|rd|th|h|d|kg|cm|km)", tok, re.I):
            continue
        if tok not in blob:
            problems.append(f"id or code not in record: {tok}")
    for b in BANNED:
        if b in low:
            problems.append(f"banned phrase: {b!r}")
    return problems


def _walk_vals(vals: list[Any], out: list[float]) -> None:
    for v in vals:
        if isinstance(v, bool) or v is None:
            continue
        if isinstance(v, (int, float)):
            out.append(float(v))
        elif isinstance(v, str):
            out.extend(float(m.replace(",", ".")) for m in NUM_RE.findall(v))


def validate_claims(record: dict[str, Any], structured: dict[str, Any]) -> dict[str, Any]:
    """Per-claim verdicts and the overall verdict for a {action, claims} explanation (LLM or template)."""
    facts = build_facts(record)
    template = render_template(record)
    nums, strs = _record_values(facts, template)
    blob = " ".join(strs)
    fields = record_fields(record)
    reasons = list(record.get("reasons") or [])
    reason_text = {r["code"]: r["text"] for r in facts["reasons"]}
    problems: list[str] = []
    action = structured.get("action") if isinstance(structured, dict) else None
    action_ok = action in (facts["action"], facts["action_label"])
    if not action_ok:
        problems.append(f"does not name the action {facts['action']} (says {action!r})")
    raw = _num_list(structured.get("claims")) if isinstance(structured, dict) else []
    claims = []
    for c in raw[:MAX_CLAIMS]:
        c = c if isinstance(c, dict) else {}
        p = _check_claim(c, reasons=reasons, fields=fields, nums=nums, blob=blob, reason_text=reason_text)
        claims.append({"text": str(c.get("text") or ""), "reason_code": str(c.get("reason_code") or ""),
                       "reason_label": REASON_PLAIN.get(str(c.get("reason_code") or ""), None),
                       "cited_fields": [str(x) for x in _num_list(c.get("cited_fields"))],
                       "numbers": _num_list(c.get("numbers")), "ok": not p, "problems": p})
    if len(raw) > MAX_CLAIMS:
        problems.append(f"too many claims: {len(raw)} (at most {MAX_CLAIMS})")
    if not claims and reasons:  # with no reasons, no claim can pass: the action sentence alone is right
        problems.append("no claims")
    bad = sum(1 for c in claims if not c["ok"])
    if bad:
        problems.append(f"{bad} of {len(claims)} claims failed the check")
    top = [r["code"] for r in facts["reasons"][:3]]
    need = min(2, len(top))
    hit = len({c["reason_code"] for c in claims if c["ok"]} & set(top))
    if hit < need:
        problems.append(f"covers {hit} of top {len(top)} reasons, needs {need}")
    words = sum(len(c["text"].split()) for c in claims)
    if words > MAX_WORDS:
        problems.append(f"too long: {words} words")
    return {"ok": not problems, "action_ok": action_ok, "problems": problems, "claims": claims}


def compose_text(record: dict[str, Any], claims: list[dict[str, Any]]) -> str:
    """The prose shown for a passing claims explanation: the template's action sentence, then each claim."""
    f = build_facts(record)
    parts = [f"Recommended action: {f['action_label']} ({f['action']})."]
    for c in claims:
        t = c["text"].strip()
        parts.append(t if t.endswith((".", "!", "?")) else t + ".")
    return " ".join(parts)


def build_claims_prompt(record: dict[str, Any]) -> tuple[str, str, str]:
    facts = build_facts(record)
    user = ("FACTS (JSON):\n" + json.dumps(facts, sort_keys=True) + "\n\nFIELDS (name: value, what cited_fields may "
            "name):\n" + json.dumps(record_fields(record), sort_keys=True) +
            '\n\nReturn JSON {"action": "<action code>", "claims": [{"text": "...", "reason_code": "<code>", '
            '"cited_fields": ["<field>"], "numbers": [<number>]}]}.')
    h = hashlib.sha256((CLAIMS_PROMPT_VERSION + "\n" + SYSTEM_CLAIMS + "\n" + user).encode("utf-8")).hexdigest()
    return SYSTEM_CLAIMS, user, h


def response_format(mode: str) -> dict[str, Any]:
    if mode == "json_schema":
        return {"type": "json_schema", "json_schema": {"name": "explanation_claims", "strict": True,
                                                       "schema": CLAIMS_SCHEMA}}
    return {"type": "json_object"}


class SchemaIgnored(ValueError):
    """The answer is not the {action, claims} JSON that was asked for."""


def parse_claims(text: str) -> dict[str, Any]:
    t = text.strip()
    if t.startswith("```"):  # a model ignoring json mode sometimes fences its JSON
        t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        obj = json.loads(t)
    except ValueError as e:
        raise SchemaIgnored(f"not JSON: {e}") from e
    if not isinstance(obj, dict) or not isinstance(obj.get("claims"), list) or "action" not in obj:
        raise SchemaIgnored("JSON without action and claims")
    return obj


def _schema_rejected(e: Exception) -> bool:
    """HTTP 400 from the API: the schema or the JSON mode was refused (or its output failed Groq's own check)."""
    return isinstance(e, httpx.HTTPStatusError) and e.response is not None and e.response.status_code == 400


def template_payload(record: dict[str, Any], mode: str = "template") -> dict[str, Any]:
    chk = validate_claims(record, template_claims(record))
    return {"mode": mode, "claims_source": "template", "ok": chk["ok"], "claims": chk["claims"],
            "rejected_claims": [], "rejected_problems": []}


def explain_claims(record: dict[str, Any], client: Any = None, model_id: str | None = None,
                   retry_invalid: int = 1) -> tuple[Explanation, dict[str, Any], dict[str, Any]]:
    """(Explanation, audit log, claims payload). Clients without structured outputs (Anthropic, none) use the
    prose path; the claims shown are then the template's. Never raises for LLM problems."""
    if client is None or not getattr(client, "supports_response_format", False):
        exp, log = explain(record, client=client, model_id=model_id or "none", use_default_client=False,
                           retry_invalid=retry_invalid)
        mode = "prose" if client is not None else "template"
        log["mode"] = mode
        return exp, log, template_payload(record, mode)
    template = render_template(record)
    system, user, h = build_claims_prompt(record)
    log: dict[str, Any] = {"decision_id": record.get("decision_id"), "prompt_version": CLAIMS_PROMPT_VERSION,
                           "prompt_hash": h, "model_id": model_id, "template": template, "mode": None,
                           "mode_errors": [], "attempts": [], "llm_output": None, "validator": None, "error": None,
                           "usage": []}
    for mode in MODES:
        messages = [{"role": "user", "content": user}]
        last = None
        try:
            for _ in range(retry_invalid + 1):
                resp = client.messages.create(model=model_id, max_tokens=800, temperature=0, system=system,
                                              messages=list(messages), response_format=response_format(mode))
                log["usage"].append(getattr(resp, "usage", None))
                text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()
                parsed = parse_claims(text)
                last = validate_claims(record, parsed)
                log["llm_output"] = text
                log["validator"] = {"ok": last["ok"], "problems": last["problems"]}
                log["attempts"].append({"mode": mode, "llm_output": text, "validator": log["validator"],
                                        "claims": last["claims"]})
                if last["ok"]:
                    log["mode"], log["source"] = mode, "llm"
                    payload = {"mode": mode, "claims_source": "llm", "ok": True, "claims": last["claims"],
                               "rejected_claims": [], "rejected_problems": []}
                    return Explanation(compose_text(record, last["claims"]), "llm", True, model_id, h), log, payload
                fails = last["problems"] + [f"claim {i + 1}: {p}" for i, c in enumerate(last["claims"])
                                            for p in c["problems"]]
                messages += [{"role": "assistant", "content": text},
                             {"role": "user", "content": RETRY_NOTE.format(problems="; ".join(fails))}]
        except (SchemaIgnored, httpx.HTTPStatusError) as e:
            if isinstance(e, httpx.HTTPStatusError) and not _schema_rejected(e):
                return _template_after_error(record, log, e, h, model_id, mode)
            log["mode_errors"].append({"mode": mode, "error": f"{type(e).__name__}: {e}"[:300]})
            continue
        except Exception as e:  # network, timeout, rate limit after retries: the template, not more calls
            return _template_after_error(record, log, e, h, model_id, mode)
        # the model answered in this mode but a claim failed every try: template, keep what was rejected
        log["mode"], log["source"] = mode, "template"
        payload = template_payload(record, mode)
        payload["rejected_claims"] = last["claims"] if last else []
        payload["rejected_problems"] = last["problems"] if last else []
        return Explanation(template, "template", True, model_id, h), log, payload
    # both JSON modes refused or ignored: the prose path
    exp, plog = explain(record, client=client, model_id=model_id, use_default_client=False,
                        retry_invalid=retry_invalid)
    plog["mode"] = "prose"
    plog["mode_errors"] = log["mode_errors"]
    return exp, plog, template_payload(record, "prose")


def _template_after_error(record, log, e, h, model_id, mode):
    log["error"] = f"{type(e).__name__}: {e}"[:300]
    log["mode"], log["source"] = mode, "template"
    return Explanation(render_template(record), "template", True, model_id, h), log, template_payload(record, mode)


def claims_view(record: dict[str, Any], gbm: Any = None) -> dict[str, Any]:
    """GET /decisions/{id}/claims: the shown claims with per-claim verdicts, any rejected model claims, and how the
    cited reasons agree with the model's own attributions. Decisions explained before claims existed (or by an
    injected explainer) get the template's claims, computed now."""
    from fraudshield.explain.attribution import attribution_view  # noqa: PLC0415

    exp = record.get("explanation") or {}
    payload = exp.get("claims") if isinstance(exp.get("claims"), dict) else None
    if payload is None:
        payload = template_payload(record, "prose" if exp.get("source") == "llm" else "template")
    cited = [c["reason_code"] for c in payload["claims"] if c["ok"]]
    return {"decision_id": record.get("decision_id"), "action": record.get("action"),
            "explanation_status": record.get("explanation_status"), "explanation_source": exp.get("source"),
            **payload, "attribution": attribution_view(record, gbm, cited)}
