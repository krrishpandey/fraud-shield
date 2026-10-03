"""What would change this decision: a small search over the fields a booker controls, scored through the REAL
feature, LightGBM, calibration and cost-rule path. Shared by the analyst counterfactual endpoint
(GET /decisions/{id}/counterfactual) and the red team (scripts/redteam.py).

Pure: no audit write, no stored decision, no feature-store append. Hypothetical bookings keep the original booking id
(so the logged-exploration draw is the same as in /score) and read history through a HistoryView that hides the
original booking (already in the store once it was scored).

Mutable fields (what a fraudster or booker types into the booking form):
  declared_value  x0.5, x0.75, x1.25, x1.5, x2, or the account's median
  weight_kg       x0.75, x1.25, or the account's median. carrier_cost moves with it: KG_SLOPE_BRL per kg, the slope
                  of the lane freight model cost ~ a + b*kg + c*km/1000 fitted on the train window's real Olist rows
                  (same model as fraudshield/data/inject.py). The attacker never sets the carrier's price directly.
  dims            all three sides x0.8 or x1.25, or the account's median sides (no cost change: the data's freight
                  has no volume term)
  service         standard <-> express (no cost change: in this data service is a delivery-window proxy and the
                  freight model has no service term)
  sender_id       a sender this account already used (its two most frequent senders before the booking)
  booked_at       the account's most common booking hour on the same day, or 6 hours earlier / later
Not mutable: account, login device, payment method, origin, destination, consignee, category, owner contact.

Search: every single-field change first (one batch, about 18 bookings), then pairs of different fields, best-ranked
singles first, in batches of 8, until a plain allow is found or the budget (50) is spent. With feedback="probability"
(analyst view) singles are ranked by the model's probability and a found single numeric change is narrowed by
bisection (fewest fields, then smallest change). With feedback="action" (red team) the attacker sees only the
returned action: ranking by action, ties broken at random (seeded), no refinement.

Sources: Khouna et al., "Optimal Counterfactual Search in Tree Ensembles", arXiv 2605.06561 (2026): exact search is
possible for a tree ensemble alone; here the decision also runs through account-history features recomputed per
change, a rules floor and the cost rule, so we use a small budgeted grid/greedy search and report what it finds.
Fok et al., "Foe for Fraud: Transferable Adversarial Attacks in Credit Card Fraud Detection", arXiv 2508.14699
(2025): fraud models can be evaded by small realistic input changes; scripts/redteam.py tests that on our model.
"""
from __future__ import annotations

import math
import random
import time
from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import timedelta
from typing import Any, Callable

import numpy as np
import pandas as pd

from fraudshield.api.pipeline import DROP_PRIOR, _maybe_float, _parse_ts
from fraudshield.contracts import ACTIONS, SERVED_QUESTIONS, Booking, FeatureVector
from fraudshield.features.featurize import featurize as featurize_reference
from fraudshield.features.store import to_ts
from fraudshield.models.calibration import apply_calibration, p_yes
from fraudshield.models.laya_client import LayaUnavailable
from fraudshield.policy.decide import PolicyContext, audit_laya_action, decide_with_trace
from fraudshield.policy.reasons import booking_signals, reason_codes

BUDGET = 50
MAX_FIELDS = 2
PAIR_BATCH = 8
REFINE_STEPS = 3
# Lane freight slope (BRL per kg): lstsq of carrier_cost on [1, weight_kg, distance_km/1000] over the 41,042
# train-window real (not injected) rows of data/processed/bookings_all.parquet -> [8.677, 2.727, 10.608].
KG_SLOPE_BRL = 2.727
MIN_CARRIER_COST = 1.0
MUTABLE_FIELDS = ("declared_value", "weight_kg", "dims", "service", "sender_id", "booked_at")
STOP_ACTIONS = ("owner_confirm", "review", "hold", "block")
COST_RULE = (f"carrier cost moves with declared weight at R${KG_SLOPE_BRL:.3f} per kg (lane freight model fitted on "
             f"the train window's real rows), floor R${MIN_CARRIER_COST:.2f}; size, service, time, sender and "
             f"declared value do not change it")


# ---------- read-only history ----------
class HistoryView:
    """Read-only view of a FeatureStore for what-if scoring. Hides `exclude_ids` (the booking under test, already in
    the store once scored) and memoises lookups for one search. Never appends."""

    def __init__(self, store, exclude_ids=(), lock=None):
        self._store, self._ex, self._lock = store, set(exclude_ids), lock
        self.changes, self.confirmed = store.changes, store.confirmed
        self._memo: dict[tuple, pd.DataFrame] = {}

    def _get(self, kind: str, key: str, before) -> pd.DataFrame:
        k = (kind, key, to_ts(before))
        if k not in self._memo:
            fn = self._store.history if kind == "a" else self._store.consignee_history
            if self._lock is not None:
                with self._lock:
                    df = fn(key, before)
            else:
                df = fn(key, before)
            self._memo[k] = df[~df.booking_id.isin(self._ex)] if self._ex and len(df) else df
        return self._memo[k]

    def history(self, account_id: str, before) -> pd.DataFrame:
        return self._get("a", account_id, before)

    def consignee_history(self, consignee_id: str, before) -> pd.DataFrame:
        return self._get("c", consignee_id, before)


# ---------- pure scoring ----------
@dataclass(frozen=True)
class Outcome:
    action: str
    probability: float     # probabilities["misuse"] as /score returns it (after the drop-address prior)
    gbm_score: float
    greedy_action: str


def gbm_model(gbm: Callable) -> Any:
    """The GBMModel behind a scorer, for batch scoring (learning scorers carry .model, the base one get_model)."""
    m = getattr(gbm, "model", None)
    if m is None and callable(getattr(gbm, "get_model", None)):
        m = gbm.get_model()
    return m if hasattr(m, "predict") and hasattr(m, "features") else None


def batch_gbm(gbm: Callable, fvs: list[FeatureVector]) -> list[float]:
    """Same numbers as gbm(fv) one by one (GBMModel.score_values: NaN for missing), in one LightGBM call."""
    m = gbm_model(gbm)
    if m is None:
        return [float(gbm(fv)) for fv in fvs]
    rows = pd.DataFrame([{f: (np.nan if fv.values.get(f) is None else fv.values.get(f)) for f in m.features}
                         for fv in fvs])
    return [float(x) for x in m.predict(rows)]


def _blocks_24h(pipe, account_id: str, at, skip_block_at=None) -> int:
    """Pipeline._blocks_24h, minus the block of the booking under test itself (a what-if replaces that booking)."""
    n = pipe._blocks_24h(account_id, at)
    own = pipe._blocks.get(account_id, [])
    if skip_block_at is not None and skip_block_at in own and timedelta(0) <= at - skip_block_at < timedelta(hours=24):
        n -= 1
    return n


def decide_pure(pipe, b: Booking, fv: FeatureVector, gbm_score: float, skip_block_at=None) -> Outcome:
    """Pipeline._score from the GBM score to the action, without the audit write or the decision store.
    Keep in step with fraudshield/api/pipeline.py (tests/redteam checks both give the same action)."""
    values = {**booking_signals(b), **dict(fv.values)}
    ser = pipe.serialize
    state = ser(b, fv, gbm_score) if getattr(ser, "wants_gbm", False) else ser(b, fv)
    raw: dict[str, dict[str, float]] = {}
    act_probs = None
    try:
        res = pipe.laya.predict(state, SERVED_QUESTIONS + ("action",)) if pipe.laya_action else pipe.laya.predict(state)
        raw = dict(res.raw)
        act_probs = raw.pop("action", None)
        degraded = False
    except LayaUnavailable:
        degraded = True
    if degraded:
        probs = {"misuse": gbm_score}
    else:
        cal_probs, _ = apply_calibration(raw, pipe.calibration)
        probs = {q: p_yes(v) for q, v in cal_probs.items()}
    floor, hard, rule_hits = pipe.rules(b, values, probs, degraded)
    us = _maybe_float(values.get("under_score"))
    if (pipe.scan_threshold is not None and us is not None and us == us and us >= pipe.scan_threshold
            and "UNDER_DECLARED_PARCEL" not in rule_hits):
        rule_hits = [*rule_hits, "UNDER_DECLARED_PARCEL"]
        if floor == "allow":
            floor = "allow_scan_gated"
    if b.account_id in pipe.scan_failed_accounts:
        rule_hits = [*rule_hits, "ACCOUNT_FAILED_DEPOT_SCAN"]
        if floor == "allow":
            floor = "allow_scan_gated"
    if "DROP_ADDRESS_PATTERN" in rule_hits and "misuse" in probs:
        probs = {**probs, "misuse": 1.0 - (1.0 - probs["misuse"]) * (1.0 - DROP_PRIOR)}
    at = _parse_ts(b.booked_at)
    ctx = PolicyContext(carrier_cost=float(b.carrier_cost), account_tenure_days=_maybe_float(values.get("tenure_days")),
                        owner_contact_age_days=b.owner_contact_age_days, rule_floor=floor, hard_signal=hard,
                        blocks_last_24h=_blocks_24h(pipe, b.account_id, at, skip_block_at), degraded=degraded)
    codes, _ = reason_codes(values)
    rng = random.Random(f"{pipe.seed}:{b.booking_id}")
    d, trace = decide_with_trace(probs, ctx, pipe.costs, pipe.policy, rng, reasons=codes)
    if not degraded and act_probs:
        d, _ = audit_laya_action(max(act_probs, key=act_probs.get), d, trace, ctx, pipe.policy)
    return Outcome(d.action, float(probs["misuse"]), float(gbm_score), d.greedy_action)


class WhatIf:
    """Scores hypothetical versions of a booking through a pipeline's own components, purely.

    store/lock: the feature store the pipeline's featurizer reads (real components); history is then read through a
    HistoryView. Without a store, pipe.featurize must itself be pure (fallback and test featurizers are)."""

    def __init__(self, pipe, store=None, lock=None):
        self.pipe, self.store, self.lock = pipe, store, lock

    @classmethod
    def for_pipeline(cls, pipe) -> "WhatIf":
        f = pipe.featurize
        get_store = getattr(f, "store", None)
        return cls(pipe, get_store() if callable(get_store) else None, getattr(f, "lock", None))

    def view(self, booking: Booking) -> HistoryView | None:
        return None if self.store is None else HistoryView(self.store, {booking.booking_id}, self.lock)

    def featurize(self, b: Booking, view: HistoryView | None) -> FeatureVector:
        return featurize_reference(b, view) if view is not None else self.pipe.featurize(b)

    def evaluate(self, bookings: list[Booking], view: HistoryView | None, skip_block_at=None) -> list[Outcome]:
        fvs = [self.featurize(b, view) for b in bookings]
        scores = batch_gbm(self.pipe.gbm, fvs)
        return [decide_pure(self.pipe, b, fv, s, skip_block_at) for b, fv, s in zip(bookings, fvs, scores)]


# ---------- candidate changes ----------
@dataclass(frozen=True)
class Change:
    field: str
    before: Any
    after: Any
    size: float                  # |log ratio| for numbers, hours/12 for time, 1 for a swap
    factor: float | None = None  # numeric scale factor (refinable)


def _ratio_ok(r: float) -> bool:
    return r > 0 and abs(math.log(r)) >= math.log(1.05)


def single_changes(b: Booking, H: pd.DataFrame) -> list[Change]:
    out: list[Change] = []
    n = len(H)
    dv = float(b.declared_value)
    if dv > 0:
        for f in (0.5, 0.75, 1.25, 1.5, 2.0):
            out.append(Change("declared_value", dv, round(dv * f, 2), abs(math.log(f)), f))
        if n >= 5:
            r = float(np.median(H.declared_value)) / dv
            if _ratio_ok(r) and not any(abs(r - f) < 0.05 for f in (0.5, 0.75, 1.25, 1.5, 2.0)):
                out.append(Change("declared_value", dv, round(dv * r, 2), abs(math.log(r)), r))
    w = float(b.weight_kg)
    if w > 0:
        for f in (0.75, 1.25):
            out.append(Change("weight_kg", w, round(w * f, 3), abs(math.log(f)), f))
        if n >= 5:
            r = max(float(np.median(H.weight_kg)), 0.01) / w
            if _ratio_ok(r) and not any(abs(r - f) < 0.05 for f in (0.75, 1.25)):
                out.append(Change("weight_kg", w, round(w * r, 3), abs(math.log(r)), r))
    dims = (float(b.length_cm), float(b.width_cm), float(b.height_cm))
    if min(dims) > 0:
        for f in (0.8, 1.25):
            out.append(Change("dims", dims, tuple(round(x * f, 1) for x in dims), abs(math.log(f)), f))
        if n >= 5:
            med = (float(np.median(H.length_cm)), float(np.median(H.width_cm)), float(np.median(H.height_cm)))
            if min(med) > 0:
                g = (med[0] * med[1] * med[2] / (dims[0] * dims[1] * dims[2])) ** (1 / 3)
                if _ratio_ok(g):
                    out.append(Change("dims", dims, med, abs(math.log(g))))
    out.append(Change("service", b.service, "express" if b.service == "standard" else "standard", 1.0))
    if n:
        for s, _ in Counter(H.sender_id.astype(str)).most_common(3):
            if s != b.sender_id and sum(c.field == "sender_id" for c in out) < 2:
                out.append(Change("sender_id", b.sender_id, s, 1.0))
    t = to_ts(b.booked_at)
    times = []
    if n >= 5:
        h = Counter(H.booked_at.dt.hour).most_common(1)[0][0]
        tm = t.replace(hour=int(h))
        if abs((tm - t).total_seconds()) >= 2 * 3600:
            times.append(tm)
    times += [t - timedelta(hours=6), t + timedelta(hours=6)]
    for tm in times:
        out.append(Change("booked_at", b.booked_at, tm.isoformat(), abs((tm - t).total_seconds()) / 3600 / 12))
    return out


def apply_changes(b: Booking, changes: tuple[Change, ...]) -> Booking:
    kw: dict[str, Any] = {}
    for c in changes:
        if c.field == "weight_kg":
            kw["weight_kg"] = float(c.after)
            kw["carrier_cost"] = carrier_cost_for(b, float(c.after))
        elif c.field == "dims":
            kw["length_cm"], kw["width_cm"], kw["height_cm"] = (float(x) for x in c.after)
        else:
            kw[c.field] = c.after
    return replace(b, **kw)


def carrier_cost_for(b: Booking, new_weight: float) -> float:
    return round(max(MIN_CARRIER_COST, float(b.carrier_cost) + KG_SLOPE_BRL * (new_weight - float(b.weight_kg))), 2)


def scaled(c: Change, f: float) -> Change:
    """The same numeric change at another scale factor."""
    if c.field == "dims":
        return Change("dims", c.before, tuple(round(x * f, 1) for x in c.before), abs(math.log(f)), f)
    nd = 3 if c.field == "weight_kg" else 2
    return Change(c.field, c.before, round(float(c.before) * f, nd), abs(math.log(f)), f)


# ---------- search ----------
@dataclass
class Tried:
    changes: tuple[Change, ...]
    outcome: Outcome

    @property
    def size(self) -> float:
        return sum(c.size for c in self.changes)


@dataclass
class SearchResult:
    booking: Booking
    original: Outcome
    tried: list[Tried] = field(default_factory=list)
    best: dict[str, Tried | None] = field(default_factory=dict)  # "softer", "allow"
    budget: int = BUDGET
    latency_ms: float = 0.0

    @property
    def evaluations(self) -> int:
        return len(self.tried)

    def flipped(self, target: str) -> bool:
        return self.best.get(target) is not None


def _hits(t: Tried, target: str, cur: int) -> bool:
    i = ACTIONS.index(t.outcome.action)
    return i == 0 if target == "allow" else i < cur


def _minimal(ts: list[Tried]) -> Tried | None:
    return min(ts, key=lambda t: (len(t.changes), t.size, t.outcome.probability)) if ts else None


def search(whatif: WhatIf, booking: Booking, original: Outcome | None = None, budget: int = BUDGET,
           max_fields: int = MAX_FIELDS, feedback: str = "probability", seed: int = 0) -> SearchResult:
    """Minimal changes that move the action to a softer one and to plain allow, within `budget` evaluations.
    `original` (the booking's own outcome) is not counted against the budget."""
    t0 = time.perf_counter()
    view = whatif.view(booking)
    if original is None:
        original = whatif.evaluate([booking], view)[0]
    res = SearchResult(booking, original, budget=budget)
    cur = ACTIONS.index(original.action)
    skip = _parse_ts(booking.booked_at) if original.action == "block" else None
    if cur == 0:
        res.best = {"softer": None, "allow": None}
        res.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        return res
    H = view.history(booking.account_id, booking.booked_at) if view is not None else pd.DataFrame(
        {c: [] for c in ("declared_value", "weight_kg", "length_cm", "width_cm", "height_cm", "sender_id")}
        | {"booked_at": pd.Series([], dtype="datetime64[ns]")})
    rng = random.Random(seed)

    def run(change_sets: list[tuple[Change, ...]]) -> list[Tried]:
        change_sets = change_sets[: max(0, budget - len(res.tried))]
        if not change_sets:
            return []
        outs = whatif.evaluate([apply_changes(booking, cs) for cs in change_sets], view, skip)
        new = [Tried(cs, o) for cs, o in zip(change_sets, outs)]
        res.tried.extend(new)
        return new

    singles = run([(c,) for c in single_changes(booking, H)])
    if feedback == "probability":
        for target in ("softer", "allow"):
            _refine(run, _minimal([t for t in singles if _hits(t, target, cur)]), target, cur)
    allow_found = any(_hits(t, "allow", cur) for t in res.tried)
    if max_fields >= 2 and not allow_found and len(res.tried) < budget:
        if feedback == "probability":
            ranked = sorted(singles, key=lambda t: (ACTIONS.index(t.outcome.action), t.outcome.probability))
        else:  # the attacker sees the action only
            keyed = [(ACTIONS.index(t.outcome.action), rng.random(), t) for t in singles]
            ranked = [t for *_, t in sorted(keyed, key=lambda x: (x[0], x[1]))]
        pairs = []
        for i, a in enumerate(ranked):
            for j in range(i + 1, len(ranked)):
                bb = ranked[j]
                if a.changes[0].field != bb.changes[0].field:
                    pairs.append((i + j, j, (a.changes[0], bb.changes[0])))
        pairs.sort(key=lambda x: (x[0], x[1]))
        queue = [p[2] for p in pairs]
        while queue and len(res.tried) < budget:
            batch, queue = queue[:PAIR_BATCH], queue[PAIR_BATCH:]
            if any(_hits(t, "allow", cur) for t in run(batch)):
                break
    res.best = {target: _minimal([t for t in res.tried if _hits(t, target, cur)]) for target in ("softer", "allow")}
    res.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    return res


def _refine(run: Callable, hit: Tried | None, target: str, cur: int) -> None:
    """Bisect a single numeric change toward no change (log scale) while it still reaches the target."""
    if hit is None or len(hit.changes) != 1 or hit.changes[0].factor is None:
        return
    c = hit.changes[0]
    lo, hi = 0.0, math.log(c.factor)  # lo: no change (does not reach the target); hi: reaches it
    for _ in range(REFINE_STEPS):
        mid = (lo + hi) / 2
        got = run([(scaled(c, math.exp(mid)),)])
        if not got:
            return
        if _hits(got[0], target, cur):
            hi = mid
        else:
            lo = mid


# ---------- analyst view ----------
def _money(x: float) -> str:
    return f"R${x:,.2f}"


def describe(c: Change, b: Booking) -> dict[str, Any]:
    """One change as data plus a plain-language phrase ("declared value were R$300.00")."""
    d: dict[str, Any] = {"field": c.field, "from": c.before, "to": c.after}
    if c.field == "declared_value":
        d["text"] = f"declared value were {_money(float(c.after))} (not {_money(float(c.before))})"
    elif c.field == "weight_kg":
        cost = carrier_cost_for(b, float(c.after))
        d["carrier_cost_to"] = cost
        d["text"] = (f"declared weight were {float(c.after):.2f} kg (not {float(c.before):.2f} kg; carrier cost "
                     f"{_money(cost)})")
    elif c.field == "dims":
        f = "x".join(f"{x:.0f}" for x in c.after)
        d["from"], d["to"] = list(c.before), list(c.after)
        d["text"] = f"parcel size were {f} cm (not {'x'.join(f'{x:.0f}' for x in c.before)} cm)"
    elif c.field == "service":
        d["text"] = f"service were {c.after} (not {c.before})"
    elif c.field == "sender_id":
        d["text"] = f"the sender were {c.after}, one this account already used (not {c.before})"
    elif c.field == "booked_at":
        d["text"] = f"it were booked at {str(c.after)[:16].replace('T', ' ')} (not {str(c.before)[:16].replace('T', ' ')})"
    return d


def counterfactual_payload(res: SearchResult) -> dict[str, Any]:
    """API body of GET /decisions/{id}/counterfactual (docs/API.md, analyst-only)."""
    items, seen = [], set()
    for target in ("allow", "softer"):
        t = res.best.get(target)
        if t is None or id(t) in seen:
            continue
        seen.add(id(t))
        ch = [describe(c, res.booking) for c in t.changes]
        items.append({"target": target, "n_changes": len(ch), "changes": ch, "action": t.outcome.action,
                      "probability": round(t.outcome.probability, 4),
                      "summary": f"{t.outcome.action} if " + " and ".join(c["text"] for c in ch)})
    found = bool(items)
    closest = None
    if not found and res.tried:  # what came nearest: lowest model probability among the changes tried
        t = min(res.tried, key=lambda x: (ACTIONS.index(x.outcome.action), x.outcome.probability, len(x.changes)))
        ch = [describe(c, res.booking) for c in t.changes]
        closest = {"n_changes": len(ch), "changes": ch, "action": t.outcome.action,
                   "probability": round(t.outcome.probability, 4),
                   "summary": f"{t.outcome.action} at {100 * t.outcome.probability:.1f}% if "
                              + " and ".join(c["text"] for c in ch)}
    if ACTIONS.index(res.original.action) == 0:
        msg = "This booking is already allowed."
    elif found:
        msg = "Smallest changes found within the budget. Analyst-only: never show these to the booker."
    else:
        msg = (f"No change of at most {MAX_FIELDS} booker-controlled fields softens this decision within "
               f"{res.budget} evaluations.")
    return {"booking_id": res.booking.booking_id, "analyst_only": True,
            "current": {"action": res.original.action, "probability": round(res.original.probability, 4)},
            "found": found, "message": msg, "counterfactuals": items, "closest": closest,
            "evaluations": res.evaluations,
            "budget": res.budget, "max_fields": MAX_FIELDS, "latency_ms": res.latency_ms,
            "mutable_fields": list(MUTABLE_FIELDS), "cost_rule": COST_RULE}
