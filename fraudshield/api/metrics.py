"""Dashboard metrics computed only from stored decisions (live + replay). Nothing invented:
money and FPR are computed only over decisions that have a label (analyst label, or a `label`
field on a replayed record from the evaluation data); unlabelled decisions count only in totals.
Money uses the cost matrix (assumptions) per booking: prevented = leak - C[action, fraud] for
labelled fraud that was not plainly allowed; friction = C[action, legit] for labelled legit.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np

from fraudshield.contracts import ACTIONS
from fraudshield.policy.costs import CostConfig, cost_matrix

FRICTION_ACTIONS = {"owner_confirm", "review", "hold", "block"}
TYPOLOGY_KEYS = tuple(f"T{i}" for i in range(1, 8)) + ("unlabelled",)


def _typology(meta: dict) -> str | None:
    t = str(meta.get("typology") or meta.get("scenario") or "")
    key = t.split("_")[0].upper()
    return key if key in TYPOLOGY_KEYS[:-1] else None


def _label(r: dict[str, Any]) -> tuple[str | None, str | None]:
    a = r.get("analyst") or {}
    if a.get("label"):
        return a["label"], "analyst"
    if r.get("label") in ("fraud", "legit"):
        return r["label"], r.get("label_source", "replay")
    return None, None


def _meta(r):
    return ((r.get("booking") or {}).get("meta") or {})


def _hard_negative(r) -> bool:
    m = _meta(r)
    return bool(m.get("hard_negative")) or str(m.get("scenario", "")).upper().startswith("HN")


def dashboard_metrics(records: list[dict[str, Any]], costs: CostConfig) -> dict[str, Any]:
    by_action = Counter({a: 0 for a in ACTIONS})
    prevented = friction = 0.0
    legit_n = legit_fric = hn_n = hn_fric = 0
    label_sources: Counter = Counter()
    trend: dict[str, dict[str, Any]] = defaultdict(lambda: {"fraud_stopped": 0, "held": 0,
                                                             "by_typology": Counter({k: 0 for k in TYPOLOGY_KEYS})})
    dates, lats = [], []
    for r in records:
        act = r.get("action")
        by_action[act] += 1
        date = str(r.get("booked_at", ""))[:10] or None
        if date:
            dates.append(date)
        lt = (r.get("latency_ms") or {}).get("total")
        if lt is not None:
            lats.append(float(lt))
        if act in ("hold", "block") and date:
            trend[date]["held"] += 1
        label, src = _label(r)
        if label is None:
            if act != "allow" and date:
                trend[date]["by_typology"]["unlabelled"] += 1
            continue
        label_sources[src] += 1
        F = float((r.get("booking") or {}).get("carrier_cost", r.get("carrier_cost", 0.0)) or 0.0)
        C = cost_matrix(F, costs.params)
        if label == "fraud" and act != "allow":
            prevented += C["allow"][1] - C[act][1]
            if date:
                trend[date]["fraud_stopped"] += 1
                typ = _typology(_meta(r))
                if typ:
                    trend[date]["by_typology"][typ] += 1
        elif label == "legit":
            friction += C[act][0]
            legit_n += 1
            legit_fric += act in FRICTION_ACTIONS
            if _hard_negative(r):
                hn_n += 1
                hn_fric += act in FRICTION_ACTIONS
    return {
        "window": {"from": min(dates) if dates else None, "to": max(dates) if dates else None},
        "totals": {"bookings": len(records), "by_action": dict(by_action)},
        "held_shipments": by_action["hold"] + by_action["block"],
        "revenue_loss_prevented_brl": round(prevented, 2),
        "friction_cost_brl": round(friction, 2),
        "net_prevented_brl": round(prevented - friction, 2),
        "fpr_legit": legit_fric / legit_n if legit_n else None,
        "fpr_hard_negative": hn_fric / hn_n if hn_n else None,
        "labels": {"labelled": sum(label_sources.values()), "sources": dict(label_sources)},
        "trend": [{"date": d, "fraud_stopped": v["fraud_stopped"], "held": v["held"],
                   "by_typology": dict(v["by_typology"])} for d, v in sorted(trend.items())],
        "latency": {"p50_ms": float(np.percentile(lats, 50)) if lats else None,
                    "p99_ms": float(np.percentile(lats, 99)) if lats else None},
        "assumptions": {"cost_matrix_version": costs.version,
                        "note": "friction and loss costs are assumptions (config/costs.yaml); money and FPR use "
                                "labelled decisions only"},
    }
