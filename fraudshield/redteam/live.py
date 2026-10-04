"""Red team you can watch (docs/API.md, "Red team"): the attacker of scripts/redteam.py run live on one decision.

The attacker sees only the action the system returns (never the risk score), changes at most MAX_FIELDS fields the
booker controls, and stops after BUDGET tries. attack_payload lists every try in order so the console can replay
the attack. load_results summarises the measured run (artifacts/results_redteam.json) for the Red team tab.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fraudshield.contracts import ACTIONS, Booking
from fraudshield.redteam.search import MAX_FIELDS, MUTABLE_FIELDS, STOP_ACTIONS, SearchResult, describe


def attack_payload(res: SearchResult) -> dict[str, Any]:
    """Every try in order (changes and the action the attacker saw), and the first evasion found."""
    cur = ACTIONS.index(res.original.action)
    attempts = []
    for n, t in enumerate(res.tried, start=1):
        i = ACTIONS.index(t.outcome.action)
        attempts.append({"n": n, "changes": [describe(c, res.booking)["text"] for c in t.changes],
                         "action": t.outcome.action, "softer": i < cur, "allow": i == 0})
    first_softer = next((a["n"] for a in attempts if a["softer"]), None)
    first_allow = next((a["n"] for a in attempts if a["allow"]), None)
    hit = first_allow or first_softer
    evasion = None
    if hit is not None:
        t = res.tried[hit - 1]
        evasion = {"n": hit, "action": t.outcome.action, "changes": [describe(c, res.booking) for c in t.changes]}
    best = min((a["action"] for a in attempts if a["softer"]), key=ACTIONS.index, default=None)
    best_action = "allow" if first_allow else best  # softest action the attacker reached (None: none softer)
    if cur == 0:
        msg = "This booking is already allowed: nothing to attack."
    elif first_allow:
        msg = f"Evaded: plain allow after {first_allow} tries."
    elif best is not None and best not in STOP_ACTIONS:
        msg = (f"Partly evaded: {best} after {first_softer} tries. The label is issued, but the parcel is still "
               "weighed at the first depot scan.")
    elif best is not None:
        msg = (f"Still stopped: the best the attacker got in {len(attempts)} tries was {best} instead of "
               f"{res.original.action}.")
    else:
        msg = f"The model resisted: {len(attempts)} tries, no softer action."
    return {"booking_id": res.booking.booking_id, "original_action": res.original.action,
            "attacker_sees": "action only", "budget": res.budget, "max_fields": MAX_FIELDS,
            "mutable_fields": list(MUTABLE_FIELDS), "queries": len(attempts), "attempts": attempts,
            "first_softer_at": first_softer, "first_allow_at": first_allow,
            "evaded": "allow" if first_allow else ("softer" if first_softer else None),
            "best_action": best_action, "still_stopped": best_action is None or best_action in STOP_ACTIONS,
            "evasion": evasion, "message": msg, "latency_ms": res.latency_ms}


def _rate(r: dict[str, Any]) -> dict[str, Any]:
    return {k: r.get(k) for k in ("n", "n_seeds", "flip_allow_mean", "flip_allow_sd", "flip_softer_mean",
                                  "flip_softer_sd")}


def load_results(path: Path) -> dict[str, Any] | None:
    """The measured red-team run, trimmed for the console. None if the file is missing or unreadable."""
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    a, h = d.get("attack") or {}, d.get("hardening") or {}
    trained, held = a.get("rates_trained") or {}, a.get("rates_held_out") or {}
    gate = h.get("gate") or {}
    return {
        "run_at": d.get("run_at"), "budget": d.get("budget"), "max_fields": d.get("max_fields"),
        "split": a.get("split"), "seeds": a.get("seeds"), "per_seed": a.get("per_seed"),
        "trained": {"all": _rate(trained.get("all") or {})},
        "by_type": [{"type": k, **_rate(v)} for k, v in trained.items() if k != "all" and (v or {}).get("n")],
        "held_out": {"all": _rate(held.get("all") or {}),
                     "by_type": [{"type": k, **_rate(v)} for k, v in held.items() if k != "all"]},
        "fields_used": a.get("fields_used"), "median_queries_softer": a.get("median_queries_softer"),
        "hardening": {"passed": gate.get("passed"), "mode": gate.get("mode"),
                      "n_attacked": h.get("n_attacked"), "n_evaded": h.get("n_evaded"), "n_labels": h.get("n_labels"),
                      "failed_checks": [c for c in gate.get("checks") or [] if not c.get("passed")],
                      "current": h.get("current"), "candidate": h.get("candidate")},
        "synthetic": "fraud rows are injected (synthetic) on real Olist histories",
        "evidence": "artifacts/results_redteam.md",
    }


def load_targets(path: Path) -> list[dict[str, Any]]:
    """Ready-made bookings to attack on stage (docs/demo_block_bookings.json: bookings the system stopped)."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    out = []
    for x in raw if isinstance(raw, list) else []:
        b = x.get("booking", x) if isinstance(x, dict) else None
        try:
            Booking(**b)
        except TypeError:
            continue
        out.append({"title": f"Stopped booking {b['booking_id']} ({b.get('origin_uf')} to {b.get('dest_uf')}, "
                             f"R${float(b.get('carrier_cost') or 0):.2f})", "booking": b})
    return out
