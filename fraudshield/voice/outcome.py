"""From a finished call to an outcome: parse the agent's extraction, check the detail answer, pick the branch.

Bolna documents `extracted_data` two ways (flat keys, or {category: {field: {"objective"|"subjective": value}}}), so
`find_field` accepts both. When the extraction is missing (agent without dispositions, extraction still running),
a keyword classifier reads the owner's lines in the transcript; the outcome then says so
(`extraction_source = "transcript_fallback"`).
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

FIELDS = ("owner_confirmed", "detail_answer", "change_reason", "language")
CONFIRM_VALUES = ("yes", "no", "unsure")

# Bolna execution statuses (GET /executions/{id}); terminal ones end an attempt
ACTIVE_STATUSES = ("scheduled", "queued", "rescheduled", "initiated", "ringing", "in-progress")
# "call-disconnected" = the phone line closed, but Bolna's extraction only arrives with "completed" (seen on a real
# call, 31 Oct 2026). The service waits for "completed" and falls back to the transcript after a timeout.
DISCONNECTED = "call-disconnected"
ANSWERED_TERMINAL = ("completed",)
UNANSWERED_TERMINAL = ("no-answer", "busy", "failed", "canceled", "stopped", "error", "balance-low")
TERMINAL_STATUSES = ANSWERED_TERMINAL + UNANSWERED_TERMINAL

_YES = re.compile(r"\b(yes|yeah|yep|correct|that was me|it was me|i booked|we booked)\b", re.I)
_NO = re.compile(r"\b(no|nope|not me|wasn'?t me|didn'?t|did not|never)\b", re.I)
_UNSURE = re.compile(r"\b(not sure|don'?t know|maybe|can'?t say|i'?ll check|unsure)\b", re.I)


@dataclass(frozen=True)
class CallOutcome:
    owner_confirmed: str          # yes | no | unsure
    detail_answer: str | None
    change_reason: str | None
    language: str | None
    extraction_source: str        # bolna_extraction | transcript_fallback | simulated
    detail_match: bool | None     # None = no detail asked or answered
    branch: str                   # deny | confirm | unsure | detail_mismatch

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _value(v: Any) -> Any:
    """Unwrap Bolna's cells. A real call returned {"objective": null, "subjective": "text", "confidence": ..}: take the
    first NON-EMPTY of objective / subjective / value / answer."""
    if isinstance(v, dict):
        for k in ("objective", "subjective", "value", "answer"):
            got = _value(v.get(k))
            if got not in (None, ""):
                return got
        return None
    return v


def find_field(extracted: Any, name: str) -> Any:
    """Depth-first search for `name` (case/space/underscore-insensitive) in a flat or nested extraction."""
    want = re.sub(r"[\s_]+", "", name).lower()
    if isinstance(extracted, dict):
        for k, v in extracted.items():
            if re.sub(r"[\s_]+", "", str(k)).lower() == want:
                got = _value(v)
                if got not in (None, ""):
                    return got
        for v in extracted.values():
            if isinstance(v, dict):
                got = find_field(v, name)
                if got not in (None, ""):
                    return got
    return None


def normalise_confirm(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return "yes" if v else "no"
    s = str(v).strip().lower()
    if s in CONFIRM_VALUES:
        return s
    if _UNSURE.search(s):
        return "unsure"
    if _NO.search(s):
        return "no"
    if _YES.search(s):
        return "yes"
    return None


def owner_lines(transcript: str | None) -> list[str]:
    """The callee's lines. Bolna transcripts are 'assistant: ...' / 'user: ...' lines."""
    out = []
    for line in (transcript or "").splitlines():
        m = re.match(r"\s*(user|customer|owner|human)\s*:\s*(.*)", line, re.I)
        if m:
            out.append(m.group(2).strip())
    return out


def classify_transcript(transcript: str | None) -> str:
    """The owner's first decisive answer AFTER the agent asked "Was this you?" (a "hello, yes" to the greeting is not
    an answer, seen on a real call). No beats yes within one line ("no, that wasn't me")."""
    lines = (transcript or "").splitlines()
    asked = max((i for i, ln in enumerate(lines) if re.search(r"was this you", ln, re.I)), default=-1)
    for line in owner_lines("\n".join(lines[asked + 1:])):
        if _UNSURE.search(line):
            return "unsure"
        if _NO.search(line):
            return "no"
        if _YES.search(line):
            return "yes"
    return "unsure"


def answer_after(transcript: str | None, question: str) -> str | None:
    """The owner's first line after the agent's last line matching `question` (a regex). Bolna's extraction summarised
    the pickup answer on a real call ("...booking of 18 kg from Chennai to Delhi ... pickup 'delhi id'"), which put
    the booking's own city in the answer; the owner's own words are the reliable source."""
    lines = (transcript or "").splitlines()
    idx = max((i for i, ln in enumerate(lines)
               if re.match(r"\s*(assistant|agent)\s*:", ln, re.I) and re.search(question, ln, re.I)), default=None)
    if idx is None:
        return None
    for ln in lines[idx + 1:]:
        m = re.match(r"\s*(user|customer|owner|human)\s*:\s*(.*)", ln, re.I)
        if m and m.group(2).strip():
            return m.group(2).strip()
        if re.match(r"\s*(assistant|agent)\s*:", ln, re.I):
            return None
    return None


def _tokens(s: str) -> set[str]:
    return {t for t in re.findall(r"\w+", s.lower()) if len(t) >= 3}


def detail_matches(answer: str | None, expected: list[str]) -> bool | None:
    """True if the owner's pickup answer names the booking's pickup place (any expected place shares a word)."""
    if not answer or not expected:
        return None
    a = _tokens(answer)
    return any(a & _tokens(e) for e in expected if e)


def parse_outcome(execution: dict[str, Any], expected_pickup: list[str], simulated: bool = False) -> CallOutcome:
    ex = execution.get("extracted_data") or {}
    confirmed = normalise_confirm(find_field(ex, "owner_confirmed"))
    source = "simulated" if simulated else "bolna_extraction"
    if confirmed is None:
        confirmed = classify_transcript(execution.get("transcript"))
        source = "simulated" if simulated else "transcript_fallback"
    # the owner's own words first; Bolna's (summarising) extraction only when the transcript has no answer
    t = execution.get("transcript")
    detail = answer_after(t, r"pickup address|going from") or find_field(ex, "detail_answer")
    reason = answer_after(t, r"anything changed|new warehouse") or find_field(ex, "change_reason")
    lang = find_field(ex, "language") or (execution.get("context_details") or {}).get("language")
    detail_s = str(detail).strip() if detail not in (None, "") else None
    reason_s = str(reason).strip() if reason not in (None, "") else None
    # "no no", "nothing changed", "no, nothing" = no stated change
    if reason_s and re.fullmatch(r"(?:\s*(?:no|nope|none|nothing|null|n/?a|not really|no change|nothing changed|"
                                 r"nothing has changed)[\s,.!]*)+", reason_s, re.I):
        reason_s = None
    match = detail_matches(detail_s, expected_pickup)
    if confirmed == "no":
        branch = "deny"
    elif confirmed == "yes":
        branch = "confirm" if match is not False and detail_s is not None else "detail_mismatch"
    else:
        branch = "unsure"
    return CallOutcome(confirmed, detail_s, reason_s, str(lang) if lang else None, source, match, branch)
