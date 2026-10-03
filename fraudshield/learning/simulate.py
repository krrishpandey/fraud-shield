"""SIMULATED analyst feedback for demos (POST /learning/simulate_feedback).

Picks n test-window bookings that have injected ground truth (real Olist rows are assumed legit,
injected rows carry their scenario's truth), scores them through the pipeline if not already scored
(with the precomputed as-of feature row, so the snapshot is exactly what the model saw), and records
an analyst label drawn from ground truth, flipped with probability `error_rate` (analyst mistakes).
Every label is source "simulated_analyst" and flagged simulated in the label store, audit log and
Laya export. The sample is enriched: `injected_share` of picks come from injected rows (fraud and
hard negatives), like an analyst queue that sees more suspicious bookings than average.
"""
from __future__ import annotations

import math
import random
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from fraudshield.contracts import Booking, FeatureVector
from fraudshield.features.spec import FEATURES

NOTE = ("SIMULATED analyst labels drawn from injected ground truth with a {err:.0%} analyst error rate. "
        "Demo only: they are not real analyst decisions.")
BOOKING_FIELDS = ("booking_id", "account_id", "booked_at", "channel", "login_device_age_days", "payment_method",
                  "sender_id", "origin_uf", "origin_zip3", "dest_uf", "dest_zip3", "consignee_id", "weight_kg",
                  "length_cm", "width_cm", "height_cm", "service", "category", "declared_value", "carrier_cost")


def _val(v):
    if isinstance(v, (np.floating, float)):
        return None if math.isnan(float(v)) else float(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    return v


def booking_from_row(r: dict[str, Any]) -> Booking:
    kw = {k: _val(r[k]) for k in BOOKING_FIELDS}
    kw["booked_at"] = str(kw["booked_at"]).replace(" ", "T")
    for k in ("booking_id", "account_id", "sender_id", "consignee_id", "origin_zip3", "dest_zip3", "category"):
        kw[k] = str(kw[k])
    oc = _val(r.get("owner_contact_age_days"))
    typ = r.get("typology")
    meta = {"simulated_replay": True, "typology": typ if typ not in (None, "none") else "none",
            "campaign_id": _val(r.get("campaign_id")) if isinstance(r.get("campaign_id"), str) else None}
    if typ not in (None, "none"):
        meta["scenario"] = typ
    return Booking(**kw, owner_contact_age_days=oc, meta=meta)


def feature_vector_from_row(r: dict[str, Any]) -> FeatureVector:
    vals = {k: _val(r[k]) for k in FEATURES if k in r}
    # Keep missing features as None (same keys as the online featurizer); the serializer prints them as n/a.
    return FeatureVector(str(r["booking_id"]), str(_val(r["booked_at"])).replace(" ", "T"), vals)


def simulate_feedback(pipeline, pool: pd.DataFrame, n: int = 200, seed: int = 1, error_rate: float = 0.05,
                      injected_share: float = 0.5) -> dict[str, Any]:
    labelled = {x["booking_id"] for x in (pipeline.label_store.all() if pipeline.label_store else [])}
    cand = pool[~pool.booking_id.astype(str).isin(labelled)]
    inj = cand[cand.is_injected.astype(bool)] if "is_injected" in cand else cand.iloc[0:0]
    real = cand.drop(inj.index)
    k_inj = min(len(inj), int(round(n * injected_share)))
    k_real = min(len(real), n - k_inj)
    pick = pd.concat([inj.sample(k_inj, random_state=seed), real.sample(k_real, random_state=seed)])
    pick = pick.sort_values("booked_at")
    rng = random.Random(seed)
    added = fraud = flipped = scored = 0
    for r in pick.to_dict("records"):
        bid = str(r["booking_id"])
        rec = pipeline.get_by_booking(bid)
        if rec is None:
            pipeline.score(booking_from_row(r), fv=feature_vector_from_row(r))
            rec = pipeline.get_by_booking(bid)
            scored += 1
        truth = bool(r["is_fraud"])
        flip = rng.random() < error_rate
        lab = "fraud" if truth != flip else "legit"
        pipeline.analyst_feedback(rec["decision_id"], lab, NOTE.format(err=error_rate), source="simulated_analyst")
        added += 1
        fraud += lab == "fraud"
        flipped += flip
    return {"added": added, "fraud": fraud, "legit": added - fraud, "simulated": True, "flipped": flipped,
            "error_rate": error_rate, "newly_scored": scored, "source": "simulated_analyst",
            "sampling": {"injected_share": injected_share, "n_injected": k_inj, "n_real": k_real,
                         "pool": "test-window bookings with injected ground truth"},
            "note": NOTE.format(err=error_rate)}


# ---------------------------------------------------------------------------------------------
# Realistic mode (default): labels arrive the way they would in production.
GATED_ACTIONS = ("allow_scan_gated", "owner_confirm", "review", "hold", "block")
ANALYST_DELAY_DAYS = 1.0          # ASSUMPTION: analyst resolves a stopped booking within a day
OUTCOME_DELAY_DAYS = (7.0, 60.0)  # ASSUMPTION: dispute/chargeback for allowed fraud arrives 7 to 60 days later
MATURITY_DAYS = 60.0              # ASSUMPTION: allowed booking with no dispute after 60 days counts as legit
REALISTIC_NOTE = ("SIMULATED feedback (realistic mode): analyst labels ({err:.0%} error) only on bookings the system "
                  "stopped or gated; allowed fraud reported as a dispute 7 to 60 days later; allowed legit labelled "
                  "only after 60 days without a dispute. Ground truth comes from injected scenarios. Demo only.")


def _parse(ts) -> datetime:
    return datetime.fromisoformat(str(ts).replace(" ", "T").replace("Z", ""))


def _iso(t: datetime) -> str:
    return t.isoformat(timespec="seconds")


class RealisticFeedbackSim:
    """Streams test-window bookings in time order with a simulated clock `now` (= latest booking time,
    plus any `advance_days`). Labels are scheduled per decision and released once due <= now:
    - stopped or gated (and not explored): analyst label after ANALYST_DELAY_DAYS, wrong with prob error_rate,
      source simulated_analyst;
    - allowed or explored, fraud: accurate dispute label after U(OUTCOME_DELAY_DAYS), source simulated_outcome;
    - allowed or explored, legit: accurate legit label only after MATURITY_DAYS, source simulated_outcome.
    Immature bookings stay unlabelled (never assumed legit)."""

    def __init__(self, pool: pd.DataFrame, seed: int = 1, error_rate: float = 0.05):
        self.pool = pool.sort_values(["booked_at", "booking_id"]).reset_index(drop=True)
        self.rng = random.Random(seed)
        self.error_rate = float(error_rate)
        self.cursor = 0
        self.now: datetime | None = None
        self.pending: list[tuple[datetime, str, str, str, bool]] = []  # due, decision_id, label, source, flipped

    def step(self, pipeline, n: int = 500, advance_days: float = 0.0) -> dict[str, Any]:
        labelled = {x["booking_id"] for x in (pipeline.label_store.all() if pipeline.label_store else [])}
        rows = self.pool.iloc[self.cursor:self.cursor + int(n)].to_dict("records")
        scheduled = 0
        for r in rows:
            bid = str(r["booking_id"])
            rec = pipeline.get_by_booking(bid)
            if rec is None:
                pipeline.score(booking_from_row(r), fv=feature_vector_from_row(r))
                rec = pipeline.get_by_booking(bid)
            booked = _parse(rec.get("booked_at") or r["booked_at"])
            self.now = booked if self.now is None else max(self.now, booked)
            if bid in labelled:
                continue
            truth = bool(r["is_fraud"])
            if rec.get("explored") or rec["action"] == "allow":
                if truth:
                    due = booked + timedelta(days=self.rng.uniform(*OUTCOME_DELAY_DAYS))
                else:
                    due = booked + timedelta(days=MATURITY_DAYS)
                self.pending.append((due, rec["decision_id"], "fraud" if truth else "legit", "simulated_outcome",
                                     False))
            elif rec["action"] in GATED_ACTIONS:
                flip = self.rng.random() < self.error_rate
                self.pending.append((booked + timedelta(days=ANALYST_DELAY_DAYS), rec["decision_id"],
                                     "fraud" if truth != flip else "legit", "simulated_analyst", flip))
            scheduled += 1
        self.cursor += len(rows)
        if advance_days and self.now is not None:
            self.now = self.now + timedelta(days=float(advance_days))
        due_now = sorted((p for p in self.pending if self.now is not None and p[0] <= self.now), key=lambda p: p[0])
        self.pending = [p for p in self.pending if not (self.now is not None and p[0] <= self.now)]
        by_source: dict[str, int] = {}
        fraud = flipped = 0
        note = REALISTIC_NOTE.format(err=self.error_rate)
        for due, did, label, source, flip in due_now:
            pipeline.analyst_feedback(did, label, note, source=source, at=_iso(due))
            by_source[source] = by_source.get(source, 0) + 1
            fraud += label == "fraud"
            flipped += flip
        return {"mode": "realistic", "simulated": True, "scored": len(rows), "scheduled": scheduled,
                "added": len(due_now), "fraud": fraud, "legit": len(due_now) - fraud, "by_source": by_source,
                "flipped": flipped, "pending": len(self.pending), "now": _iso(self.now) if self.now else None,
                "cursor": self.cursor, "pool_size": len(self.pool), "exhausted": self.cursor >= len(self.pool),
                "error_rate": self.error_rate,
                "assumptions": {"analyst_delay_days": ANALYST_DELAY_DAYS, "outcome_delay_days": OUTCOME_DELAY_DAYS,
                                "maturity_days": MATURITY_DAYS},
                "note": note}
