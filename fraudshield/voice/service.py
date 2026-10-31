"""Tracd Verify state machine: held booking -> owner call -> outcome -> re-score -> final action. Every step is audited
as a NEW record that carries the decision id and the decision's audit hash; the decision record is never edited.

    hold / owner_confirm
      -> owner contact unchanged >= 30 days?  no  -> analyst queue (no call: a fraudster may have changed the number)
      -> call (Bolna, or a clearly labelled simulated call when there is no API key / no number / the call fails)
      -> call events (webhook, poller or simulator; idempotent on (execution id, status))
      -> answered:   no      -> block, lock account, analyst case with transcript, label "fraud"
                     yes     -> re-score (stated change damps its features, yes factor, policy re-runs with a
                                first-scan floor); hard signal -> analyst; else usually released with a first-scan check
                     unsure / detail mismatch -> stays held, analyst case
      -> unanswered: retry once, then SMS (stub), then analyst case. Never auto-released.
"""
from __future__ import annotations

import os
import statistics
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from fraudshield.voice.bolna import BolnaClient, BolnaError
from fraudshield.voice.damping import change_scope, groups_for_reason, until_from
from fraudshield.voice.outcome import (ACTIVE_STATUSES, ANSWERED_TERMINAL, DISCONNECTED, TERMINAL_STATUSES,
                                       UNANSWERED_TERMINAL,
                                       parse_outcome)
from fraudshield.voice.owners import LANGUAGE_NAMES, Owner, OwnerDirectory
from fraudshield.voice.script import simulated_transcript

TRIGGER_ACTIONS = ("hold", "owner_confirm")
RELEASE_ACTIONS = ("allow", "allow_scan_gated")
OPEN = ("pending_verification", "calling", "ringing", "in_call", "retry_scheduled", "awaiting_simulated_answer")


@dataclass
class VerifyConfig:
    enabled: bool = True
    auto_start: bool = True
    mode: str = "auto"                 # auto: real Bolna call when BOLNA_API_KEY + agent id + number exist; simulated
    agent_id: str | None = None
    from_phone: str | None = None
    webhook_secret: str | None = None
    min_contact_days: float = 30.0
    max_attempts: int = 2              # the first call + one retry
    retry_delay_s: float = 60.0        # demo pace; production: an hour (brief)
    poll_interval_s: float = 3.0       # GET /executions while a real call is active (works without a tunnel too)
    poll_timeout_s: float = 600.0
    extraction_wait_s: float = 45.0    # after "call-disconnected", wait this long for "completed" (with the extraction)
    price_per_min_usd: float | None = 0.08  # Bolna dashboard "Cost per min" for this agent, 31 Oct 2026
    sim_speed: float = 1.0             # 1 = a ~25 s simulated call; 0 = instant (tests)
    p_yes_given_legit: float = 0.95    # ASSUMPTION: an honest owner confirms their own booking
    p_yes_given_fraud: float | None = None  # None = costs.yaml p_spoof (fraudster controls / coaches the owner channel)
    release_floor: str = "allow_scan_gated"
    bypass_guardrails: bool = False
    sms_text: str = ("Tracd: a booking on your account is on hold. We tried to call you to confirm it. Reply or call "
                     "your courier's support line. We never ask for OTPs or passwords.")

    @classmethod
    def from_sources(cls, cfg: dict[str, Any] | None, env: dict[str, str] | None = None) -> "VerifyConfig":
        env = env if env is not None else dict(os.environ)
        c = dict(cfg or {})
        out = cls(**{k: v for k, v in c.items() if k in cls.__dataclass_fields__})
        out.agent_id = env.get("BOLNA_AGENT_ID") or out.agent_id
        out.from_phone = env.get("BOLNA_FROM_NUMBER") or out.from_phone
        out.webhook_secret = env.get("VERIFY_WEBHOOK_SECRET") or out.webhook_secret
        if env.get("VERIFY_MODE"):
            out.mode = env["VERIFY_MODE"]
        if env.get("VERIFY_RETRY_DELAY_S"):
            out.retry_delay_s = float(env["VERIFY_RETRY_DELAY_S"])
        if env.get("VERIFY_SIM_SPEED"):
            out.sim_speed = float(env["VERIFY_SIM_SPEED"])
        if env.get("VERIFY_BYPASS_GUARDRAILS"):
            out.bypass_guardrails = env["VERIFY_BYPASS_GUARDRAILS"].strip().lower() in ("1", "true", "yes")
        return out


class ThreadRunner:
    """Runs fn after delay on a daemon timer thread."""

    def later(self, delay_s: float, fn: Callable, *args: Any) -> None:
        t = threading.Timer(max(0.0, delay_s), fn, args)
        t.daemon = True
        t.start()


class InlineRunner:
    """Tests: collects scheduled work; run_all() executes it in order (delays ignored)."""

    def __init__(self) -> None:
        self.pending: list[tuple[float, Callable, tuple]] = []

    def later(self, delay_s: float, fn: Callable, *args: Any) -> None:
        self.pending.append((delay_s, fn, args))

    def run_all(self, limit: int = 1000) -> int:
        n = 0
        while self.pending and n < limit:
            _, fn, args = self.pending.pop(0)
            fn(*args)
            n += 1
        return n


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _secs(a: str, b: str) -> float:
    return (datetime.fromisoformat(b) - datetime.fromisoformat(a)).total_seconds()


class VerifyService:
    def __init__(self, resolve: Callable[[str], tuple[dict, Any]], cfg: VerifyConfig, owners: OwnerDirectory,
                 client: BolnaClient | None = None, runner: Any = None, costs: Any = None,
                 webhook_url: str | None = None):
        """resolve(decision_id) -> (decision record, Pipeline that owns it); raises KeyError if unknown."""
        self.resolve = resolve
        self.cfg = cfg
        self.owners = owners
        self.client = client
        self.runner = runner or ThreadRunner()
        self.costs = costs
        self.webhook_url = webhook_url
        self._lock = threading.RLock()
        self._by_decision: dict[str, dict[str, Any]] = {}
        self._by_execution: dict[str, str] = {}
        self._seen: set[tuple[str, str]] = set()
        self._order: list[str] = []
        self._n = 0
        self._sim_n = 0

    # ---------- config facts ----------
    @property
    def real_ready(self) -> bool:
        return self.cfg.mode != "simulated" and self.client is not None and bool(self.cfg.agent_id)

    def yes_likelihood_ratio(self) -> float:
        p_fraud = self.cfg.p_yes_given_fraud
        if p_fraud is None:
            p_fraud = float((getattr(self.costs, "params", None) or {}).get("p_spoof", 0.10))
        return p_fraud / self.cfg.p_yes_given_legit

    def status(self) -> dict[str, Any]:
        return {"enabled": self.cfg.enabled, "mode": self.cfg.mode, "real_calls_ready": self.real_ready,
                "agent_id": self.cfg.agent_id, "api_key_present": self.client is not None,
                "webhook_secret_set": bool(self.cfg.webhook_secret), "webhook_url": self.webhook_url,
                "min_contact_days": self.cfg.min_contact_days, "max_attempts": self.cfg.max_attempts,
                "retry_delay_s": self.cfg.retry_delay_s, "yes_likelihood_ratio": round(self.yes_likelihood_ratio(), 4),
                "release_floor": self.cfg.release_floor, "languages": LANGUAGE_NAMES}

    # ---------- reads ----------
    def get(self, decision_id: str) -> dict[str, Any] | None:
        return self._by_decision.get(decision_id)

    def list(self) -> list[dict[str, Any]]:
        return [self._by_decision[d] for d in reversed(self._order)]

    def cases(self, include_resolved: bool = False) -> list[dict[str, Any]]:
        out = [v for v in self.list() if v.get("case") and (include_resolved or v["case"]["status"] == "open")]
        return sorted(out, key=lambda v: -v["case"]["loss_per_analyst_min"])

    # ---------- audit ----------
    def _audit(self, v: dict, pipe: Any, event: str, payload: dict[str, Any]) -> str:
        rec = pipe.get(v["decision_id"]) or {}
        _, h = pipe.audit.append(event, {"decision_id": v["decision_id"], "booking_id": v["booking_id"],
                                         "verification_id": v["verification_id"],
                                         "decision_audit_hash": rec.get("audit_hash"), **payload})
        v["audit_hashes"].append({"event": event, "hash": h, "at": _now()})
        return h

    def _step(self, v: dict, step: str, detail: str = "", **extra: Any) -> None:
        v["timeline"].append({"at": _now(), "step": step, "detail": detail, **extra})

    # ---------- start ----------
    def maybe_start(self, decision_id: str) -> dict[str, Any] | None:
        """Called after every score: starts verification for stopped bookings when auto_start is on."""
        if not (self.cfg.enabled and self.cfg.auto_start):
            return None
        rec, _ = self.resolve(decision_id)
        if rec.get("action") not in TRIGGER_ACTIONS or decision_id in self._by_decision:
            return None
        return self.start(decision_id)

    def start(self, decision_id: str, sim_answer: dict[str, Any] | None = None,
              force_simulated: bool = False) -> dict[str, Any]:
        rec, pipe = self.resolve(decision_id)
        if rec.get("action") not in TRIGGER_ACTIONS:
            raise ValueError(f"verification applies to held bookings ({', '.join(TRIGGER_ACTIONS)}); "
                             f"this one is {rec.get('action')}")
        b = rec.get("booking") or {}
        with self._lock:
            v = self._by_decision.get(decision_id)
            if v is not None:
                if v["status"] == "awaiting_simulated_answer" and sim_answer is not None:
                    self._place(v, pipe, rec, sim_answer=sim_answer, force_simulated=True)
                return v
            self._n += 1
            owner = self.owners.get(rec["account_id"], b)
            p = float((rec.get("probabilities") or {}).get("misuse", rec.get("gbm_score") or 0.0))
            v = {
                "verification_id": f"vfy_{self._n:06d}", "decision_id": decision_id, "booking_id": rec["booking_id"],
                "account_id": rec["account_id"], "original_action": rec["action"], "p_fraud": round(p, 4),
                "carrier_cost": b.get("carrier_cost"), "booking": _booking_summary(b, rec), "owner": _owner_view(owner),
                "status": "pending_verification", "mode": None, "simulated_reason": None,
                "held_at": _now(), "attempts": [], "transcript": "", "outcome": None, "rescore": None,
                "final_action": None, "final_at": None, "case": None, "sms": None,
                "timeline": [], "audit_hashes": [], "account_locked": False,
            }
            self._by_decision[decision_id] = v
            self._order.append(decision_id)
            self._step(v, "scored", f"{rec['action']} at p={p:.3f}", latency_ms=(rec.get("latency_ms") or {}).get("total"))
            self._step(v, "held", "Verifying with the account owner: they will get a call shortly")
            age = owner.contact_age_days
            if age is None or age < self.cfg.min_contact_days:
                reason = "no_verified_contact" if age is None else "contact_changed_recently"
                detail = ("no verified owner contact on file" if age is None else
                          f"owner number changed {age:.0f} days ago (< {self.cfg.min_contact_days:.0f}): no call, "
                          "a fraudster who changed the number would confirm his own booking")
                self._audit(v, pipe, "verify_started", {"owner_contact_age_days": age, "call": False, "reason": reason})
                self._escalate(v, pipe, rec, reason, detail)
                return v
            self._audit(v, pipe, "verify_started", {"owner_contact_age_days": round(age, 1), "call": True,
                                                    "language": owner.language, "owner_source": owner.source})
            self._place(v, pipe, rec, sim_answer=sim_answer, force_simulated=force_simulated, owner=owner)
            return v

    def _call_variables(self, v: dict, owner: Owner | None) -> dict[str, Any]:
        bk = v["booking"]
        return {"company_name": (owner.company if owner else v["owner"]["company"]),
                "weight_kg": bk["weight_kg"], "service": bk["service"], "origin": bk["origin_label"],
                "destination": bk["dest_label"], "booking_time": bk["time_label"],
                "device_is_new": "yes" if bk["device_is_new"] else "no",
                "device_note": ", from a new device" if bk["device_is_new"] else "",
                "decision_id": v["decision_id"], "booking_id": v["booking_id"],
                "language": v["owner"]["language"]}

    def _place(self, v: dict, pipe: Any, rec: dict, sim_answer: dict | None = None, force_simulated: bool = False,
               owner: Owner | None = None) -> None:
        owner = owner or self.owners.get(rec["account_id"], rec.get("booking") or {})
        n = len(v["attempts"]) + 1
        variables = self._call_variables(v, owner)
        reason = None
        if force_simulated or self.cfg.mode == "simulated":
            reason = (sim_answer or {}).get("reason") or "simulated mode selected"
        elif self.client is None:
            reason = "no BOLNA_API_KEY: simulated call"
        elif not self.cfg.agent_id:
            reason = "no BOLNA_AGENT_ID: simulated call (run python -m fraudshield.voice.setup_agent)"
        elif not owner.phone:
            reason = "no real number on file for this account: simulated call"
        if reason is None:
            try:
                # English-only agent: no agent_data.language (Bolna accepts it only for multilingual agents)
                out = self.client.place_call(self.cfg.agent_id, owner.phone, variables, language=None,
                                             from_phone=self.cfg.from_phone, bypass_guardrails=self.cfg.bypass_guardrails)
                ex = str(out["execution_id"])
                att = {"n": n, "execution_id": ex, "mode": "real", "started_at": _now(), "status": "queued",
                       "statuses": [{"status": "queued", "at": _now()}], "final": False, "ended_at": None,
                       "duration_s": None}
                v["attempts"].append(att)
                v["mode"] = "real"
                v["status"] = "calling"
                self._by_execution[ex] = v["decision_id"]
                self._step(v, "call_placed", f"Bolna call {n} to {owner.masked_phone} in "
                                             f"{LANGUAGE_NAMES.get(owner.language, owner.language)}", execution_id=ex)
                self._audit(v, pipe, "verify_call_event", {"attempt": n, "execution_id": ex, "status": "queued",
                                                           "mode": "real", "language": owner.language,
                                                           "to": owner.masked_phone, "variables": variables})
                self.runner.later(self.cfg.poll_interval_s, self._poll, ex, time.monotonic())
                return
            except BolnaError as e:
                reason = f"Bolna call failed ({e}): simulated call instead"
                self._step(v, "call_failed", str(e))
        # simulated path: labelled, same event handling as a real call
        answer = sim_answer or (rec.get("booking") or {}).get("meta", {}).get("verify_sim") or _truth_answer(rec)
        if answer and answer.get("owner_confirmed") == "yes" and not answer.get("detail_answer"):
            answer = {**answer, "detail_answer": f"from {v['booking']['origin_label']}"}
        v["mode"] = "simulated"
        v["simulated_reason"] = reason
        if not answer:
            v["status"] = "awaiting_simulated_answer"
            self._step(v, "awaiting_simulated_answer", f"{reason}. Pick the owner's simulated answer to place the call")
            return
        with self._lock:
            self._sim_n += 1
            ex = f"sim_{self._sim_n:06d}_{v['decision_id']}"
        att = {"n": n, "execution_id": ex, "mode": "simulated", "started_at": _now(), "status": "queued",
               "statuses": [], "final": False, "ended_at": None, "duration_s": None, "simulated_answer": answer}
        v["attempts"].append(att)
        v["status"] = "calling"
        self._by_execution[ex] = v["decision_id"]
        self._step(v, "call_placed", f"SIMULATED call {n} in {LANGUAGE_NAMES.get(owner.language, owner.language)} "
                                     f"({reason})", execution_id=ex)
        self._simulate(ex, v, owner, variables, answer)

    # ---------- simulated call ----------
    def _simulate(self, ex: str, v: dict, owner: Owner, variables: dict, answer: dict) -> None:
        s = max(0.0, self.cfg.sim_speed)
        base = {"id": ex, "agent_id": "simulated", "context_details": {"recipient_data": variables},
                "telephony_data": {"to_number": owner.masked_phone or "simulated", "provider": "simulated"}}
        if answer.get("owner_confirmed") == "no_answer":
            seq = [(0.5, "queued", {}), (1.5, "ringing", {}), (6.0, "no-answer", {"conversation_duration": 0})]
        else:
            lines = simulated_transcript(owner.language, variables, answer)
            seq = [(0.5, "queued", {}), (1.5, "ringing", {}), (2.0, "in-progress", {"transcript": ""})]
            for i in range(1, len(lines) + 1):
                seq.append((1.6, "in-progress", {"transcript": "\n".join(lines[:i])}))
            dur = round(4.0 + 1.6 * len(lines), 1)
            seq.append((1.0, "completed", {"transcript": "\n".join(lines), "conversation_duration": dur,
                                           "extracted_data": {"Tracd Verify": {
                                               k: {"objective" if k in ("owner_confirmed", "language") else "subjective": answer.get(k)}
                                               for k in ("owner_confirmed", "detail_answer", "change_reason", "language")}}}))
        t = 0.0
        for delay, status, extra in seq:
            t += delay * s
            self.runner.later(t, self.handle_event, {**base, "status": status, **extra}, "simulator")

    # ---------- real call poller ----------
    def _poll(self, ex: str, started: float) -> None:
        did = self._by_execution.get(ex)
        v = self._by_decision.get(did) if did else None
        if v is None or self.client is None:
            return
        att = next((a for a in v["attempts"] if a["execution_id"] == ex), None)
        if att is None or att["final"]:
            return
        try:
            payload = self.client.get_execution(ex)
            if att.get("disconnected_at") is not None:
                att["last_payload"] = payload
            self.handle_event(payload, "poll")
        except BolnaError as e:
            self._step(v, "poll_error", str(e))
        wait = time.monotonic() - att["disconnected_at"] if att.get("disconnected_at") is not None else 0.0
        if not att["final"] and wait > self.cfg.extraction_wait_s:
            # Bolna never sent "completed": decide from the transcript we have (outcome says transcript_fallback)
            self.handle_event({**(att.get("last_payload") or {"id": ex}), "id": ex, "status": "completed",
                               "extracted_data": None}, "poll_timeout")
        if not att["final"] and time.monotonic() - started < self.cfg.poll_timeout_s:
            self.runner.later(self.cfg.poll_interval_s, self._poll, ex, started)

    # ---------- call events (webhook, poller, simulator) ----------
    def handle_event(self, payload: dict[str, Any], source: str = "webhook") -> dict[str, Any]:
        """Idempotent: the same (execution id, status) applies once; transcript growth updates without a new record."""
        ex = str(payload.get("id") or payload.get("execution_id") or "")
        status = str(payload.get("status") or "").strip().lower()
        if not ex or not status:
            return {"accepted": False, "reason": "payload needs id (execution id) and status"}
        with self._lock:
            did = self._by_execution.get(ex)
            if did is None:
                ctx = payload.get("context_details") or {}
                did = (ctx.get("recipient_data") or {}).get("decision_id") or ctx.get("decision_id") or \
                      (payload.get("user_data") or {}).get("decision_id")
            v = self._by_decision.get(did) if did else None
            if v is None:
                return {"accepted": False, "reason": f"unknown execution {ex}"}
            att = next((a for a in v["attempts"] if a["execution_id"] == ex), None)
            if att is None:
                return {"accepted": False, "reason": f"execution {ex} is not an attempt of {did}"}
            transcript = payload.get("transcript")
            if isinstance(transcript, str) and len(transcript) > len(v["transcript"]) and not att["final"]:
                v["transcript"] = transcript
            key = (ex, status)
            if key in self._seen or att["final"]:
                return {"accepted": True, "duplicate": True, "decision_id": did, "status": v["status"]}
            self._seen.add(key)
            rec, pipe = self.resolve(did)
            att["status"] = status
            att["statuses"].append({"status": status, "at": _now(), "source": source})
            if status == DISCONNECTED:
                # the line closed; Bolna's extraction comes with "completed". Keep polling; _poll falls back to the
                # transcript if "completed" does not come within extraction_wait_s.
                att["disconnected_at"] = time.monotonic()
                att["last_payload"] = payload
                self._step(v, "call_ended", "line closed; reading the owner's answer", execution_id=ex)
                self._audit(v, pipe, "verify_call_event", {"attempt": att["n"], "execution_id": ex, "status": status,
                                                           "mode": att["mode"], "source": source})
                return {"accepted": True, "decision_id": did, "status": v["status"]}
            if status in ACTIVE_STATUSES:
                v["status"] = {"ringing": "ringing", "in-progress": "in_call"}.get(status, "calling")
                self._step(v, f"call_{status.replace('-', '_')}", f"attempt {att['n']}", execution_id=ex)
                self._audit(v, pipe, "verify_call_event", {"attempt": att["n"], "execution_id": ex, "status": status,
                                                           "mode": att["mode"], "source": source})
                return {"accepted": True, "decision_id": did, "status": v["status"]}
            if status not in TERMINAL_STATUSES:
                return {"accepted": True, "ignored_status": status, "decision_id": did}
            att["final"] = True
            att["ended_at"] = _now()
            dur = payload.get("conversation_duration") or (payload.get("telephony_data") or {}).get("duration")
            att["duration_s"] = float(dur) if dur not in (None, "") else None
            if not att["duration_s"]:
                # Bolna reported 0 duration on a real call: time it from our own status stamps (talk start -> line closed)
                talk = next((s["at"] for s in att["statuses"] if s["status"] == "in-progress"), None)
                end = next((s["at"] for s in att["statuses"] if s["status"] == DISCONNECTED), att["ended_at"])
                att["duration_s"] = round(_secs(talk, end), 1) if talk else att["duration_s"]
            # Cost of a real call. Bolna's total_cost (cents) left out telephony on the first real call (3.23 cents
            # reported, $0.06 taken from the balance), so the cost shown is minutes x the dashboard's per-minute price;
            # Bolna's own figure is kept next to it. Simulated calls cost nothing.
            att["recording_url"] = (payload.get("telephony_data") or {}).get("recording_url")
            cost = payload.get("total_cost")
            att["cost_reported_usd"] = round(float(cost) / 100.0, 4) if cost not in (None, "") else None
            if att["mode"] != "real":
                att["cost_usd"] = 0.0
            elif self.cfg.price_per_min_usd and att["duration_s"]:
                att["cost_usd"] = round(att["duration_s"] / 60.0 * self.cfg.price_per_min_usd, 4)
            else:
                att["cost_usd"] = att["cost_reported_usd"] or 0.0
            self._audit(v, pipe, "verify_call_event", {
                "attempt": att["n"], "execution_id": ex, "status": status, "mode": att["mode"], "source": source,
                "duration_s": att["duration_s"], "termination_reason": payload.get("termination_reason"),
                "transcript": v["transcript"] if status in ANSWERED_TERMINAL else None,
                "extracted_data": payload.get("extracted_data"),
                "recording_url": (payload.get("telephony_data") or {}).get("recording_url")})
            if status in ANSWERED_TERMINAL and (v["transcript"] or payload.get("extracted_data")):
                self._step(v, "call_ended", f"answered, {att['duration_s'] or 0:.0f} s", execution_id=ex)
                self._on_answered(v, pipe, rec, payload, att)
            else:
                self._step(v, "call_unanswered", f"attempt {att['n']}: {status}", execution_id=ex)
                self._on_unanswered(v, pipe, rec, status)
            return {"accepted": True, "decision_id": did, "status": v["status"], "final_action": v["final_action"]}

    # ---------- outcomes ----------
    def _on_answered(self, v: dict, pipe: Any, rec: dict, payload: dict, att: dict) -> None:
        b = rec.get("booking") or {}
        expected = [v["booking"]["origin_label"], *(v["owner"].get("pickup_addresses") or []),
                    str((b.get("meta") or {}).get("pickup_address") or "")]
        out = parse_outcome({**payload, "transcript": v["transcript"] or payload.get("transcript")},
                            [e for e in expected if e], simulated=att["mode"] == "simulated")
        v["outcome"] = out.to_dict()
        simulated = att["mode"] == "simulated"
        self._step(v, "outcome", f"owner said {out.owner_confirmed}"
                                 + (f"; pickup answer {'matches' if out.detail_match else 'does not match'}"
                                    if out.detail_match is not None else "")
                                 + (f"; change: {out.change_reason}" if out.change_reason else ""))
        if out.branch == "deny":
            self._finalize(v, pipe, "block", "owner said the booking was not theirs", out=out)
            pipe.locked_accounts.add(rec["account_id"])
            v["account_locked"] = True
            self._audit(v, pipe, "account_locked", {"account_id": rec["account_id"], "by": "owner_call",
                                                    "simulated": simulated})
            self._step(v, "account_locked", "label creation locked for this account")
            pipe.analyst_feedback(v["decision_id"], "fraud",
                                  note=f"owner said no on the verification call ({v['verification_id']})",
                                  # an owner's "no" is a confirmed outcome, in minutes instead of weeks
                                  source="simulated_outcome" if simulated else "outcome")
            self._open_case(v, pipe, rec, "owner_denied", "confirm the fraud and contact the owner", status="open")
            return
        if out.branch in ("unsure", "detail_mismatch"):
            why = ("owner was unsure" if out.branch == "unsure" else
                   "owner said yes but the pickup answer did not match the booking" if out.detail_match is False else
                   "owner said yes but gave no pickup detail")
            self._audit(v, pipe, "verify_outcome", {**out.to_dict(), "final_action": None, "kept": rec["action"],
                                                    "simulated": simulated})
            self._escalate(v, pipe, rec, out.branch, why)
            return
        groups = groups_for_reason(out.change_reason)
        rs = pipe.rescore_after_owner(v["decision_id"], groups, self.yes_likelihood_ratio(), self.cfg.release_floor)
        v["rescore"] = rs
        self._step(v, "rescored", f"p {rs['p_before']:.3f} -> {rs['p_model_after_change']:.3f} (stated change) -> "
                                  f"{rs['p_after']:.3f} (yes factor {rs['likelihood_ratio']:.3f}); policy: {rs['action']}")
        if groups:
            ch = {"groups": groups, "reason": out.change_reason, "decision_id": v["decision_id"],
                  "scope": change_scope(b, groups), "until": until_from(datetime.fromisoformat(str(b.get("booked_at"))))}
            pipe.confirmed_changes.setdefault(rec["account_id"], []).append(ch)
            v["confirmed_change"] = ch
            self._step(v, "change_recorded", f"{out.change_reason}: related warnings damped until {ch['until'][:10]}")
        if rs["hard_signal"]:
            self._audit(v, pipe, "verify_outcome", {**out.to_dict(), "rescore": rs, "final_action": None,
                                                    "kept": "hold", "simulated": simulated})
            self._escalate(v, pipe, rec, "hard_signal", "a hard signal stays held: a yes never overrides it")
        elif rs["action"] in RELEASE_ACTIONS:
            self._finalize(v, pipe, rs["action"], "owner confirmed on the call", out=out, rescore=rs)
        else:
            self._audit(v, pipe, "verify_outcome", {**out.to_dict(), "rescore": rs, "final_action": None,
                                                    "kept": rec["action"], "simulated": simulated})
            self._escalate(v, pipe, rec, "still_risky", f"still {rs['action']} after the yes (p={rs['p_after']:.3f})")

    def _on_unanswered(self, v: dict, pipe: Any, rec: dict, status: str) -> None:
        if len(v["attempts"]) < self.cfg.max_attempts:
            v["status"] = "retry_scheduled"
            self._step(v, "retry_scheduled", f"{status}: calling again in {self.cfg.retry_delay_s:.0f} s")
            self.runner.later(self.cfg.retry_delay_s, self._retry, v["decision_id"])
            return
        v["sms"] = {"at": _now(), "to": v["owner"]["masked_phone"], "text": self.cfg.sms_text, "sent": False,
                    "stub": True}
        self._step(v, "sms_stub", "SMS stub logged (not sent: no SMS provider wired in the prototype)")
        self._audit(v, pipe, "verify_escalated", {"step": "sms", "stub": True, "sent": False, "text": self.cfg.sms_text})
        self._escalate(v, pipe, rec, "no_answer", f"no answer after {len(v['attempts'])} calls: never auto-released")

    def _retry(self, decision_id: str) -> None:
        with self._lock:
            v = self._by_decision.get(decision_id)
            if v is None or v["status"] != "retry_scheduled":
                return
            rec, pipe = self.resolve(decision_id)
            last = v["attempts"][-1] if v["attempts"] else {}
            ans = last.get("simulated_answer")
            if last.get("mode") == "simulated":
                ans = (ans or {}).get("retry_answer") or {"owner_confirmed": "no_answer"}
            self._place(v, pipe, rec, sim_answer=ans, force_simulated=last.get("mode") == "simulated")

    def _finalize(self, v: dict, pipe: Any, action: str, why: str, out: Any = None, rescore: dict | None = None,
                  source: str = "owner_call") -> None:
        v["final_action"] = action
        v["final_at"] = _now()
        v["status"] = "blocked" if action == "block" else "released" if action in RELEASE_ACTIONS else "analyst_review"
        v["minutes_to_final"] = round(_secs(v["held_at"], v["final_at"]) / 60.0, 3)
        self._audit(v, pipe, "verify_outcome", {**(out.to_dict() if out else {}), "rescore": rescore,
                                                "final_action": action, "why": why, "source": source,
                                                "simulated": v["mode"] == "simulated",
                                                "minutes_to_final": v["minutes_to_final"]})
        self._step(v, "final", f"{action}: {why}")
        rec = pipe.get(v["decision_id"])
        if rec is not None:  # a new field next to the audited decision (the decision itself is never edited)
            rec["owner_verification"] = {"final_action": action, "at": v["final_at"], "why": why,
                                         "verification_id": v["verification_id"], "simulated": v["mode"] == "simulated"}

    def _open_case(self, v: dict, pipe: Any, rec: dict, reason: str, todo: str, status: str = "open") -> None:
        p = (v.get("rescore") or {}).get("p_after", v["p_fraud"])
        params = getattr(self.costs, "params", None) or {}
        review_min = float(params.get("review_min", 6.0))
        loss = float(p) * (float(v.get("carrier_cost") or 0.0) + float(params.get("c_dispute", 30.0)))
        v["case"] = {"opened_at": _now(), "reason": reason, "todo": todo, "status": status,
                     "expected_loss": round(loss, 2), "review_min": review_min,
                     "loss_per_analyst_min": round(loss / review_min, 3), "resolution": None}

    def _escalate(self, v: dict, pipe: Any, rec: dict, reason: str, detail: str) -> None:
        v["status"] = "analyst_review"
        self._open_case(v, pipe, rec, reason, detail)
        self._audit(v, pipe, "verify_escalated", {"step": "analyst_queue", "reason": reason, "detail": detail,
                                                  "kept": rec.get("action")})
        self._step(v, "analyst_queue", detail)

    def resolve_case(self, decision_id: str, label: str, note: str = "", analyst: str = "analyst") -> dict[str, Any]:
        if label not in ("fraud", "legit"):
            raise ValueError("label must be fraud or legit")
        with self._lock:
            v = self._by_decision[decision_id]
            if not v.get("case"):
                raise ValueError("this verification has no analyst case")
            rec, pipe = self.resolve(decision_id)
            already_final = v["final_action"] is not None
            action = "block" if label == "fraud" else self.cfg.release_floor
            v["case"].update(status="resolved", resolution={"label": label, "note": note, "by": analyst, "at": _now()})
            self._audit(v, pipe, "verify_analyst_resolved", {"label": label, "note": note, "by": analyst,
                                                             "case_reason": v["case"]["reason"]})
            if label == "fraud" and rec["account_id"] not in pipe.locked_accounts:
                pipe.locked_accounts.add(rec["account_id"])
                v["account_locked"] = True
                self._audit(v, pipe, "account_locked", {"account_id": rec["account_id"], "by": "analyst"})
            if not already_final or v["final_action"] != action:
                self._finalize(v, pipe, action, f"analyst marked {label}", source="analyst")
            if not (rec.get("analyst") or {}).get("label") == label:
                pipe.analyst_feedback(decision_id, label, note=note or f"verify case {v['verification_id']}")
            return v

    # ---------- metrics (this run only; nothing here is invented) ----------
    def metrics(self, bookings_scored: int) -> dict[str, Any]:
        vs = self.list()
        atts = [a for v in vs for a in v["attempts"] if a["final"]]
        answered = [a for a in atts if a["status"] in ANSWERED_TERMINAL]
        released = [v for v in vs if v["status"] == "released"]
        released_by_call = [v for v in released if v["outcome"] and v["outcome"]["branch"] == "confirm"]
        blocked_by_call = [v for v in vs if v["final_action"] == "block" and v["outcome"]
                           and v["outcome"]["branch"] == "deny"]
        resolved_by_call = released_by_call + blocked_by_call
        honest_held = [v for v in vs if v["outcome"] and v["outcome"]["branch"] == "confirm"] + \
                      [v for v in vs if (v.get("case") or {}).get("resolution", {}) and
                       v["case"]["resolution"]["label"] == "legit"]
        fraud_caught = blocked_by_call + [v for v in vs if (v.get("case") or {}).get("resolution")
                                          and v["case"]["resolution"]["label"] == "fraud" and v not in blocked_by_call]
        review_min = float((getattr(self.costs, "params", None) or {}).get("review_min", 6.0))
        mins = [v["minutes_to_final"] for v in released_by_call if v.get("minutes_to_final") is not None]
        n_sim = sum(1 for v in vs if v["mode"] == "simulated")
        said = [v["outcome"]["owner_confirmed"] for v in vs if v.get("outcome")]
        all_atts = [a for v in vs for a in v["attempts"]]
        return {
            "calls_made": len(all_atts),
            "calls_live": sum(1 for a in all_atts if not a["final"]),
            "real_calls": sum(1 for a in all_atts if a["mode"] == "real"),
            "simulated_calls": sum(1 for a in all_atts if a["mode"] == "simulated"),
            "call_minutes": round(sum(a.get("duration_s") or 0.0 for a in atts) / 60.0, 2),
            "call_cost_usd": round(sum(a.get("cost_usd") or 0.0 for a in atts), 4),
            "owner_said": {"no": said.count("no"), "yes": said.count("yes"), "unsure": said.count("unsure")},
            "no_answer": sum(1 for v in vs if (v.get("case") or {}).get("reason") == "no_answer"),
            "not_called": sum(1 for v in vs if not v["attempts"] and v.get("case")),
            "source": "this run of the app (in memory)",
            "bookings_scored": bookings_scored, "verifications": len(vs),
            "simulated_verifications": n_sim, "real_verifications": sum(1 for v in vs if v["mode"] == "real"),
            "calls_finished": len(atts), "calls_answered": len(answered),
            "call_answer_rate": round(len(answered) / len(atts), 4) if atts else None,
            "median_minutes_to_release_honest": round(statistics.median(mins), 3) if mins else None,
            "released_by_call": len(released_by_call), "blocked_by_call": len(blocked_by_call),
            "analyst_minutes_saved_per_1000": round(len(resolved_by_call) * review_min / bookings_scored * 1000, 1)
            if bookings_scored else None,
            "review_min_assumption": review_min,
            "honest_held": len(honest_held),
            "false_positive_rate": round(len(honest_held) / bookings_scored, 4) if bookings_scored else None,
            "fraud_caught_at_booking": len(fraud_caught),
            "open_cases": len(self.cases()),
            "labels_note": "honest / fraud here = what the owner said on the call or the analyst decided; "
                           "simulated calls count as simulated",
        }


def _truth_answer(rec: dict) -> dict[str, Any] | None:
    """Simulated owner answer from the replayed dataset's own label (stream bookings only)."""
    meta = (rec.get("booking") or {}).get("meta") or {}
    if "is_fraud" not in meta or meta.get("is_fraud") is None:
        return None
    if meta["is_fraud"]:
        return {"owner_confirmed": "no", "language": "en"}
    return {"owner_confirmed": "yes", "change_reason": None, "language": "en"}


def _owner_view(o: Owner) -> dict[str, Any]:
    return {"company": o.company, "masked_phone": o.masked_phone, "has_number": bool(o.phone), "language": o.language,
            "language_name": LANGUAGE_NAMES.get(o.language, o.language), "contact_age_days":
                None if o.contact_age_days is None else round(o.contact_age_days, 1),
            "source": o.source, "pickup_addresses": list(o.pickup_addresses)}


def _booking_summary(b: dict[str, Any], rec: dict[str, Any]) -> dict[str, Any]:
    meta = b.get("meta") or {}
    disp = meta.get("display") or {}
    at = str(b.get("booked_at") or "")
    try:
        t = datetime.fromisoformat(at)
        time_label = t.strftime("%I:%M %p").lstrip("0").lower()
    except ValueError:
        time_label = at
    return {
        "weight_kg": round(float(b.get("weight_kg") or 0.0), 1), "service": b.get("service"),
        "origin_label": disp.get("origin_city") or f"{b.get('origin_uf')} {b.get('origin_zip3')}",
        "dest_label": disp.get("dest_city") or f"{b.get('dest_uf')} {b.get('dest_zip3')}",
        "lane_data": f"{b.get('origin_uf')} {b.get('origin_zip3')} -> {b.get('dest_uf')} {b.get('dest_zip3')}",
        "display_labels": bool(disp), "time_label": time_label, "booked_at": at,
        "device_is_new": float(b.get("login_device_age_days") or 99) <= 1.0,
        "carrier_cost": b.get("carrier_cost"), "scenario": meta.get("scenario"),
        "reasons": list(rec.get("reasons") or [])[:4],
    }
