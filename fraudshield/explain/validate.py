"""Deterministic validator for LLM explanations.

Reject unless: every number and every id-like token in the text appears in the facts (or the
template built from them); the text names the chosen action; it mentions at least 2 of the top 3
reasons (all of them if fewer than 2); and it contains no banned phrase.
"""
from __future__ import annotations

import re
from typing import Any

NUM_RE = re.compile(r"(?<![A-Za-z_\d.])\d+(?:[.,]\d+)?")
ID_RE = re.compile(r"\b(?=[A-Za-z0-9_-]*\d)(?=[A-Za-z0-9_-]*[A-Za-z])[A-Za-z0-9_-]{3,}\b")
BANNED = ("confirmed fraud", "fraudster", "criminal", "definitely", "guaranteed", "certainly",
          "without a doubt", "100% sure", "stolen for sure", "—")

ACTION_WORDS = {
    "allow": [r"\ballow"], "allow_scan_gated": [r"\bscan"], "owner_confirm": [r"\bowner", r"\bconfirm"],
    "review": [r"\breview"], "hold": [r"\bhold"], "block": [r"\bblock"],
}
REASON_WORDS = {
    "COST_FAR_ABOVE_ACCOUNT_NORM": ["cost", "median", "price", "payoff"],
    "NEW_SENDERS_UNDER_PAYER": ["sender"], "NEW_ORIGINS": ["origin"], "NEW_LOGIN_DEVICE": ["device", "login"],
    "SENDER_DIFFERS_FROM_ACCOUNT": ["sender"], "PAYER_MANY_SENDERS": ["sender"],
    "CONSIGNEE_MANY_SENDERS": ["consignee"], "WEIGHT_UNUSUAL": ["weight"], "DIMS_UNUSUAL": ["dimension", "size"],
    "BURST_LAST_24H": ["24 hours", "burst"], "NEW_ACCOUNT": ["new account", "days old"],
    "UNUSUAL_HOUR": ["hour"], "LINK_TO_CONFIRMED_FRAUD": ["link"],
    "DROP_ADDRESS_PATTERN": ["receiver", "consignee", "drop"], "UNDER_DECLARED_PARCEL": ["declared", "weight", "size"],
    "ACCOUNT_FAILED_DEPOT_SCAN": ["depot", "weigh", "scan"],
}


def _walk(obj: Any, nums: list[float], strs: list[str]) -> None:
    if isinstance(obj, bool) or obj is None:
        return
    if isinstance(obj, (int, float)):
        nums.append(float(obj))
    elif isinstance(obj, str):
        strs.append(obj)
        nums.extend(float(m.replace(",", ".")) for m in NUM_RE.findall(obj))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            strs.append(str(k))
            _walk(v, nums, strs)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _walk(v, nums, strs)


def _number_ok(tok: str, nums: list[float]) -> bool:
    x = float(tok.replace(",", "."))
    dec = len(tok.split(".")[1]) if "." in tok else 0
    tol = 0.5 * 10 ** (-dec) + 1e-9
    return any(abs(x - v) <= tol or abs(x - 100 * v) <= tol for v in nums)


def _record_values(facts: dict[str, Any], template: str | None) -> tuple[list[float], list[str]]:
    nums: list[float] = []
    strs: list[str] = []
    _walk(facts, nums, strs)
    if template:
        _walk(template, nums, strs)
    return nums, strs


def number_spans(text: str, facts: dict[str, Any], template: str | None = None) -> list[dict[str, Any]]:
    """Every number in the text, where it is, and whether the decision record contains it."""
    nums, _ = _record_values(facts, template)
    return [{"text": m.group(), "start": m.start(), "end": m.end(), "ok": _number_ok(m.group(), nums)}
            for m in NUM_RE.finditer(text)]


def validate(text: str, facts: dict[str, Any], template: str | None = None) -> tuple[bool, list[str]]:
    problems: list[str] = []
    nums, strs = _record_values(facts, template)
    blob = " ".join(strs)
    low = text.lower()

    for tok in NUM_RE.findall(text):
        if not _number_ok(tok, nums):
            problems.append(f"number not in record: {tok}")
    for tok in ID_RE.findall(text):
        if re.fullmatch(r"x?\d+(?:\.\d+)?x?|\d+(?:st|nd|rd|th|h|d|kg|cm|km)", tok, re.I):
            continue
        if tok not in blob:
            problems.append(f"id or code not in record: {tok}")
    for b in BANNED:
        if b in low:
            problems.append(f"banned phrase: {b!r}")

    action = facts.get("action", "")
    if not any(re.search(p, low) for p in ACTION_WORDS.get(action, [re.escape(action)])):
        problems.append(f"does not name the action {action}")

    top = [r["code"] for r in facts.get("reasons", [])[:3]]
    need = min(2, len(top))
    hit = sum(1 for c in top if any(w in low for w in REASON_WORDS.get(c, [c.lower()])))
    if hit < need:
        problems.append(f"mentions {hit} of top {len(top)} reasons, needs {need}")
    return not problems, problems
