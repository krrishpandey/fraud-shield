"""Owner-confirmed changes and the warning features they explain.

When the owner says "yes, we opened a new warehouse" on the verification call, the novelty features that the new
warehouse explains are set to the values they would have if the warehouse were already part of the account. LightGBM
then re-scores. The same neutral values apply to the account's later bookings for 30 days, but only within the
confirmed scope (the same sender / origin), so a confirmed change is never a free pass for other novelty.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any

DAMP_DAYS = 30

# change group -> the reason text that names it (English keywords; case-insensitive)
GROUP_PATTERNS: dict[str, re.Pattern[str]] = {
    "site": re.compile(r"warehouse|godown|go-down|branch|new (address|location|site|office|shop|unit|factory)|"
                       r"moved|shift(ed)?|relocat|pickup (point|address)", re.I),
    "login": re.compile(r"staff|employee|new (person|user|login|device|phone|laptop|computer|manager|accountant)|"
                        r"login|logged in", re.I),
}

# feature -> neutral value; a callable gets the feature values (tenure-relative ages)
_SITE = {
    "sender_differs": 0, "sender_seen": 1, "sender_entropy_jump": 0.0, "new_senders_l10": 0,
    "origin_seen": 1, "is_home_origin": 1, "origin_entropy_jump": 0.0, "new_origins_l10": 0,
    "payer_sender_pair_age_days": lambda v: _tenure(v), "payer_origin_pair_age_days": lambda v: _tenure(v),
    "payer_senders_30d": lambda v: max(1, int(v.get("payer_senders_30d", 1) or 1) - 1),
    "payer_origins_30d": lambda v: max(1, int(v.get("payer_origins_30d", 1) or 1) - 1),
    "distinct_senders_7d": lambda v: v.get("base_senders_7d", v.get("distinct_senders_7d")),
    "distinct_origins_7d": lambda v: v.get("base_origins_7d", v.get("distinct_origins_7d")),
}
_LOGIN = {"login_device_age_days": float(DAMP_DAYS + 1)}  # just past the NEW_LOGIN_DEVICE reason threshold
GROUP_FEATURES: dict[str, dict[str, Any]] = {"site": _SITE, "login": _LOGIN}


def _tenure(v: dict[str, Any]) -> float:
    try:
        return float(v.get("tenure_days") or 365.0)
    except (TypeError, ValueError):
        return 365.0


def groups_for_reason(reason: str | None) -> list[str]:
    """Change groups named in the owner's stated reason (empty = no stated change)."""
    text = (reason or "").strip()
    return [g for g, rx in GROUP_PATTERNS.items() if text and rx.search(text)]


def damp_values(values: dict[str, Any], groups: list[str] | tuple[str, ...]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Returns (new values, {feature: [old, new]} for the features that changed)."""
    out, changed = dict(values), {}
    for g in groups:
        for k, nv in GROUP_FEATURES.get(g, {}).items():
            if k not in out:
                continue
            new = nv(values) if callable(nv) else nv
            if new is None or out[k] == new:
                continue
            changed[k] = [out[k], new]
            out[k] = new
    return out, changed


def change_scope(booking: dict[str, Any], groups: list[str]) -> dict[str, Any]:
    """What the confirmed change covers: the sender and origin of this booking for a site change."""
    scope: dict[str, Any] = {}
    if "site" in groups:
        scope.update(sender_id=booking.get("sender_id"), origin_zip3=booking.get("origin_zip3"))
    return scope


def active_groups(changes: list[dict[str, Any]], booking: Any, at: datetime) -> tuple[list[str], list[dict]]:
    """Groups of the account's confirmed changes that still apply to this booking (unexpired, in scope)."""
    groups: list[str] = []
    used: list[dict] = []
    for ch in changes or []:
        if at > _ts(ch["until"]):
            continue
        scope = ch.get("scope") or {}
        if any(str(getattr(booking, k, None)) != str(v) for k, v in scope.items() if v is not None):
            continue
        new = [g for g in ch["groups"] if g not in groups]
        if new:
            groups += new
            used.append(ch)
    return groups, used


def until_from(at: datetime) -> str:
    return (at + timedelta(days=DAMP_DAYS)).isoformat(timespec="seconds")


def _ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=None)
