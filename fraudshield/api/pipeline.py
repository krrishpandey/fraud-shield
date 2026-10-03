"""Scoring pipeline with dependency injection.

featurize(booking) -> FeatureVector; gbm(fv) -> calibrated P(fraud); serialize(booking, fv) -> state text;
laya: LayaClient-like (predict/ask/health); calibration: calibration.json dict or None;
audit: AuditLog-like (append). If the audit write fails, the decision is not stored or returned.
"""
from __future__ import annotations

import hashlib
import random
import threading
import time
from dataclasses import asdict
from datetime import datetime, timedelta
from typing import Any, Callable

from fraudshield.contracts import Booking
from fraudshield.explain.llm import DEFAULT_MODEL_ID, explain
from fraudshield.models.calibration import apply_calibration, p_yes
from fraudshield.models.laya_client import LayaUnavailable
from fraudshield.policy.costs import CostConfig
from fraudshield.policy.decide import PolicyConfig, PolicyContext, decide_with_trace
from fraudshield.policy.reasons import booking_signals, reason_codes

MODE_QUESTIONS = ("foreign_senders", "payoff_max", "drop_consignee")


def default_rules(booking: Booking, values: dict[str, Any], probs: dict[str, float], degraded: bool):
    """Rules floor and hard signal. Hard signal: Laya says misuse and names a fraud mode, each >= 0.8,
    or the booking links to confirmed fraud (DESIGN 6, phase3 3.5)."""
    hits = []
    if probs.get("misuse", 0) >= 0.8 and any(probs.get(q, 0) >= 0.8 for q in MODE_QUESTIONS):
        hits.append("LAYA_MISUSE_AND_MODE_HIGH")
    if float(values.get("links_confirmed_fraud", 0) or 0) >= 1:
        hits.append("LINK_TO_CONFIRMED_FRAUD")
    return "allow", bool(hits), hits


def _parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=None)


class Pipeline:
    def __init__(self, featurize: Callable, serialize: Callable, gbm: Callable, laya: Any, audit: Any,
                 calibration: dict | None = None, costs: CostConfig | None = None,
                 policy: PolicyConfig | None = None, rules: Callable = default_rules, seed: int = 0,
                 llm_client: Any = None, llm_model_id: str = DEFAULT_MODEL_ID, llm_for_allow: bool = False,
                 versions: dict[str, str] | None = None, explainer: Callable | None = None,
                 label_store: Any = None):
        self.featurize, self.serialize, self.gbm, self.laya, self.audit = featurize, serialize, gbm, laya, audit
        self.calibration = calibration
        self.costs = costs or CostConfig.default()
        lam = ((calibration or {}).get("conformal") or {}).get("lambda_allow")
        policy = policy or PolicyConfig()
        if lam is not None and policy.lambda_allow is None:
            policy = PolicyConfig(**{**asdict(policy), "lambda_allow": float(lam)})
        self.policy = policy
        self.rules = rules
        self.seed = seed
        self.llm_client = llm_client
        self.llm_model_id = llm_model_id
        self.llm_for_allow = llm_for_allow
        self.versions = versions or {}
        self.explainer = explainer
        self.label_store = label_store  # learning.LabelStore-like (append); None = labels only audited
        self.on_label: Callable | None = None  # optional hook(label_record), e.g. the Laya export
        self.last_degraded: bool | None = None
        self._lock = threading.RLock()
        self._by_id: dict[str, dict] = {}
        self._by_booking: dict[str, str] = {}
        self._order: list[str] = []
        self._blocks: dict[str, list[datetime]] = {}
        self._n = 0
        self._custom_q = 0

    # ---------- store ----------
    def _next_id(self) -> str:
        self._n += 1
        return f"dec_{self._n:06d}"

    def add_record(self, rec: dict[str, Any]) -> None:
        """Insert a pre-scored decision (replay file). Keeps idempotency on booking_id."""
        with self._lock:
            if rec["booking_id"] in self._by_booking:
                return
            rec = dict(rec)
            rec.setdefault("decision_id", self._next_id())
            if rec["decision_id"] in self._by_id:
                rec["decision_id"] = self._next_id()
            rec.setdefault("source", "replay")
            rec.setdefault("explanation", None)
            rec.setdefault("analyst", None)
            rec.setdefault("explanation_status", "ready" if rec.get("explanation") else "none")
            self._by_id[rec["decision_id"]] = rec
            self._by_booking[rec["booking_id"]] = rec["decision_id"]
            self._order.append(rec["decision_id"])

    def get(self, decision_id: str) -> dict | None:
        return self._by_id.get(decision_id)

    def get_by_booking(self, booking_id: str) -> dict | None:
        did = self._by_booking.get(booking_id)
        return self._by_id.get(did) if did else None

    def list(self) -> list[dict]:
        return [self._by_id[i] for i in reversed(self._order)]

    def _blocks_24h(self, account_id: str, at: datetime) -> int:
        return sum(1 for t in self._blocks.get(account_id, []) if timedelta(0) <= at - t < timedelta(hours=24))

    # ---------- scoring ----------
    def score(self, booking: Booking, fv: Any = None) -> dict[str, Any]:
        """fv: optional precomputed as-of FeatureVector (replays); otherwise featurize(booking)."""
        with self._lock:
            existing = self.get_by_booking(booking.booking_id)
            if existing is not None:
                return public_view(existing)
            return public_view(self._score(booking, fv))

    def swap_gbm(self, scorer: Callable, version: str) -> tuple[Callable, str]:
        """Atomically replace the GBM scorer and its version (under the scoring lock). Returns the old pair."""
        with self._lock:
            old = (self.gbm, self.versions.get("gbm", "unknown"))
            self.gbm = scorer
            self.versions = {**self.versions, "gbm": version}
            return old

    def _score(self, b: Booking, fv: Any = None) -> dict[str, Any]:
        lat: dict[str, float] = {}
        t0 = time.perf_counter()

        def lap(name, t):
            lat[name] = round((time.perf_counter() - t) * 1000, 2)

        t = time.perf_counter()
        if fv is None:
            fv = self.featurize(b)
        values = {**booking_signals(b), **dict(fv.values)}
        lap("features", t)

        t = time.perf_counter()
        gbm_score = float(self.gbm(fv))
        lap("gbm", t)

        t = time.perf_counter()
        state = self.serialize(b, fv)
        lap("serialize", t)

        t = time.perf_counter()
        raw: dict[str, dict[str, float]] = {}
        laya_info: dict[str, Any] = {"mode": getattr(self.laya, "mode", None), "cached": False, "error": None}
        try:
            res = self.laya.predict(state)
            raw = res.raw
            laya_info.update(cached=res.cached, revision=res.revision, mode=res.mode)
            degraded = False
        except LayaUnavailable as e:
            degraded = True
            laya_info["error"] = str(e)
        self.last_degraded = degraded
        lap("laya", t)

        t = time.perf_counter()
        if degraded:
            cal_probs, cal_info = {}, {"calibrated": self.calibration is not None,
                                       "version": (self.calibration or {}).get("version", "uncalibrated")}
            probs = {"misuse": gbm_score}
        else:
            cal_probs, cal_info = apply_calibration(raw, self.calibration)
            probs = {q: p_yes(v) for q, v in cal_probs.items()}
        floor, hard, rule_hits = self.rules(b, values, probs, degraded)
        at = _parse_ts(b.booked_at)
        ctx = PolicyContext(
            carrier_cost=float(b.carrier_cost),
            account_tenure_days=_maybe_float(values.get("tenure_days")),
            owner_contact_age_days=b.owner_contact_age_days,
            rule_floor=floor, hard_signal=hard,
            blocks_last_24h=self._blocks_24h(b.account_id, at), degraded=degraded,
        )
        codes, top = reason_codes(values)
        rng = random.Random(f"{self.seed}:{b.booking_id}")
        d, trace = decide_with_trace(probs, ctx, self.costs, self.policy, rng, reasons=codes)
        trace["rule_hits"] = rule_hits
        lap("decide", t)

        versions = {
            "laya": str(laya_info.get("revision") or getattr(self.laya, "revision", None) or "none"),
            "calibration": cal_info["version"], "gbm": self.versions.get("gbm", "unknown"),
            "policy": self.policy.version, "costs": self.costs.version,
            "serializer": self.versions.get("serializer", "unknown"),
            "features": self.versions.get("features", "unknown"),
        }
        rec: dict[str, Any] = {
            "decision_id": None, "booking_id": b.booking_id, "account_id": b.account_id, "booked_at": b.booked_at,
            "action": d.action, "greedy_action": d.greedy_action, "propensity": d.propensity,
            "explored": d.explored, "degraded": d.degraded,
            "probabilities": {k: round(v, 4) for k, v in probs.items()},
            "raw_probabilities": {q: round(p_yes(v), 4) for q, v in raw.items()},
            "gbm_score": round(gbm_score, 4), "expected_costs": d.expected_costs, "reasons": d.reasons,
            "top_features": top, "state_text": state,
            "state_sha256": hashlib.sha256(state.encode("utf-8")).hexdigest(),
            "model_versions": versions, "calibrated": cal_info["calibrated"],
            "laya_mode": laya_info["mode"], "laya_cached": laya_info["cached"], "laya_error": laya_info["error"],
            "policy_trace": trace, "explanation_status": "pending", "source": "live",
            # as-of snapshot at decision time, used by the label store (never recomputed later)
            "feature_values": dict(fv.values),
        }
        with self._lock:
            rec["decision_id"] = self._next_id()
            t = time.perf_counter()
            lat["total"] = round((time.perf_counter() - t0) * 1000, 2)
            rec["latency_ms"] = lat
            try:
                _, h = self.audit.append("decision", {k: v for k, v in rec.items() if k != "explanation_status"})
            except Exception:
                self._n -= 1
                raise
            lap("audit", t)
            lat["total"] = round((time.perf_counter() - t0) * 1000, 2)
            rec["audit_hash"] = h
            rec["booking"] = asdict(b)
            rec["explanation"] = None
            rec["analyst"] = None
            self._by_id[rec["decision_id"]] = rec
            self._by_booking[b.booking_id] = rec["decision_id"]
            self._order.append(rec["decision_id"])
            if d.action == "block":
                self._blocks.setdefault(b.account_id, []).append(at)
        return rec

    # ---------- async explanation ----------
    def explain_decision(self, decision_id: str) -> None:
        rec = self._by_id.get(decision_id)
        if rec is None:
            return
        try:
            if self.explainer is not None:
                exp, log = self.explainer(rec)
            else:
                client = self.llm_client if (rec["action"] != "allow" or self.llm_for_allow) else None
                exp, log = explain(rec, client=client, model_id=self.llm_model_id, use_default_client=False)
        except Exception as e:  # never lose the decision because of the explainer
            try:
                self.audit.append("explanation", {"decision_id": decision_id, "error": f"{type(e).__name__}: {e}",
                                                  "source": None})
            except Exception:
                pass
            with self._lock:
                rec["explanation"] = None
                rec["explanation_status"] = "failed"
            return
        try:
            self.audit.append("explanation", log)
        except Exception:
            pass
        with self._lock:
            rec["explanation"] = {"text": exp.text, "source": exp.source, "valid": exp.valid, "model_id": exp.model_id}
            rec["explanation_status"] = "ready"

    # ---------- analyst / ask ----------
    def analyst_feedback(self, decision_id: str, label: str, note: str = "", source: str = "analyst",
                         at: str | None = None) -> str:
        """at: label time; defaults to now (simulations pass their simulated clock)."""
        rec = self._by_id[decision_id]
        at = at or datetime.now().isoformat(timespec="seconds")
        simulated = source.startswith("simulated")
        _, h = self.audit.append("analyst_feedback", {"decision_id": decision_id, "booking_id": rec["booking_id"],
                                                      "label": label, "note": note, "at": at, "source": source,
                                                      "simulated": simulated})
        with self._lock:
            rec["analyst"] = {"label": label, "note": note, "at": at, "source": source, "simulated": simulated}
        if self.label_store is not None:
            from fraudshield.learning.label_store import label_from_decision  # noqa: PLC0415
            lab = self.label_store.append(label_from_decision(rec, label, source, labelled_at=at, note=note))
            if self.on_label is not None:
                self.on_label(lab)
        return h

    def ask(self, decision_id: str, instructions: str, yes: str, no: str) -> dict[str, Any]:
        rec = self._by_id[decision_id]
        with self._lock:
            self._custom_q += 1
            qid = f"custom_{self._custom_q}"
        q = {"type": "choice", "instructions": instructions, "criteria": {"a": yes, "b": no}}
        res = self.laya.ask(rec["state_text"], q, qid=qid)
        probs = res.raw[qid]
        out = {"qid": qid, "probability_yes": round(p_yes(probs), 4), "raw_probabilities": probs,
               "latency_ms": round(res.latency_ms, 2), "calibrated": False, "cached": res.cached}
        model_version = str(getattr(res, "revision", None) or getattr(self.laya, "revision", None) or "none")
        _, h = self.audit.append("ask", {"decision_id": decision_id, "booking_id": rec["booking_id"], "qid": qid,
                                         "instructions": instructions, "criteria": {"a": yes, "b": no},
                                         "probability_yes": out["probability_yes"], "raw_probabilities": probs,
                                         "calibrated": False, "cached": res.cached, "model_version": model_version,
                                         "state_sha256": rec.get("state_sha256")})
        out["audit_hash"] = h
        return out


def _maybe_float(x):
    try:
        return None if x is None else float(x)
    except (TypeError, ValueError):
        return None


SCORE_FIELDS = ("decision_id", "booking_id", "action", "greedy_action", "propensity", "explored", "degraded",
                "probabilities", "raw_probabilities", "gbm_score", "expected_costs", "reasons", "top_features",
                "state_text", "model_versions", "latency_ms", "explanation_status", "audit_hash",
                "calibrated", "laya_mode", "laya_cached", "source")


def public_view(rec: dict[str, Any]) -> dict[str, Any]:
    return {k: rec.get(k) for k in SCORE_FIELDS}
