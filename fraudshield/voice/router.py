"""HTTP routes for Tracd Verify. Mounted by fraudshield/api/app.py at / and /api.

POST /voice/webhook?token=...      Bolna execution webhook (one POST per call status). Bolna documents no signature,
                                   so the URL carries VERIFY_WEBHOOK_SECRET; optional source-IP allowlist.
GET  /verify/status | /verify/feed | /verify/verifications/{id} | /verify/cases | /verify/metrics | /verify/scenarios
POST /verify/start/{id}            start (or re-start with a simulated answer)
POST /verify/cases/{id}            analyst: fraud / legit
POST /verify/demo/{scenario}       score a seeded scenario booking and start its verification
GET  /audit/records                audit record viewer (filter by decision id)
"""
from __future__ import annotations

import hmac
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Literal

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from pydantic import BaseModel, ConfigDict

from fraudshield.contracts import Booking
from fraudshield.voice.laya_explain import explain as laya_explain
from fraudshield.voice.laya_explain import why_not_live as laya_why_not
from fraudshield.voice.service import TRIGGER_ACTIONS, VerifyService

BOLNA_WEBHOOK_IPS = ("13.203.39.153", "13.126.9.249", "13.202.133.53")


class SimAnswer(BaseModel):
    owner_confirmed: Literal["yes", "no", "unsure", "no_answer"]
    detail_answer: str | None = None
    change_reason: str | None = None
    language: str | None = None


class StartIn(BaseModel):
    simulated: bool = False
    answer: SimAnswer | None = None


class CaseIn(BaseModel):
    label: Literal["fraud", "legit"]
    note: str = ""
    analyst: str = "analyst"


class ExecutionEvent(BaseModel):
    """Bolna's execution object; only id and status are required, everything else passes through."""
    model_config = ConfigDict(extra="allow")
    id: str | None = None
    execution_id: str | None = None
    status: str


def load_real_bookings(svc: VerifyService, pipe: Any, source: Callable[[], list], n_first: int = 30,
                       n_fraud: int = 15, n_hard: int = 10) -> dict[str, Any]:
    """Scores real bookings from the held-out test window (Olist rows + the injected fraud rows, never trained on)
    through the real pipeline. Picked in time order: the first n_first bookings, plus the first n_fraud injected
    frauds and the first n_hard hard honest cases, so the page has held bookings to call about. Held ones get a
    SIMULATED call whose owner answer is the dataset's own label (fraud -> "no", honest -> "yes")."""
    items = source()
    first = items[:n_first]
    fraud = [x for x in items if x.is_fraud][:n_fraud]
    hard = [x for x in items if x.hard_negative][:n_hard]
    seen: set[str] = set()
    picked = []
    for x in sorted(first + fraud + hard, key=lambda x: (x.booking.booked_at, x.booking.booking_id)):
        if x.booking.booking_id not in seen:
            seen.add(x.booking.booking_id)
            picked.append(x)
    held = 0
    for x in picked:
        b = Booking(**{**x.booking.__dict__, "meta": {**(x.booking.meta or {}), "scenario": "real-test-window",
                                                      "is_fraud_label": bool(x.is_fraud)}})
        out = pipe.score(b)
        if out["action"] in TRIGGER_ACTIONS and svc.get(out["decision_id"]) is None:
            held += 1
            why = "dataset accounts have no real phone numbers"
            ans = ({"owner_confirmed": "no", "language": "en", "source": "dataset label (injected fraud)", "reason": why}
                   if x.is_fraud else
                   {"owner_confirmed": "yes", "language": "en", "change_reason": None,
                    "source": "dataset label (honest booking)", "reason": why})
            svc.start(out["decision_id"], sim_answer=ans, force_simulated=True)
    fraud_ids = {x.booking.booking_id for x in picked if x.is_fraud}
    stopped = sum(1 for x in picked if x.is_fraud and (pipe.get_by_booking(x.booking.booking_id) or {}).get("action")
                  not in ("allow", "allow_scan_gated"))
    return {"scored": len(picked), "fraud_in_sample": len(fraud_ids), "fraud_stopped_at_booking": stopped,
            "honest_in_sample": len(picked) - len(fraud_ids), "held": held,
            "window": "test window 2018-05-15 .. 2018-08-31 (never used for training)"}


def laya_status(root: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    """Where Laya stands on this computer, and its fine-tuning record (read from the repo's real result files)."""
    def js(p: str) -> Any:
        f = root / p
        return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None

    def first_line(p: str, prefix: str) -> str | None:
        f = root / p
        if not f.exists():
            return None
        return next((ln.strip() for ln in f.read_text(encoding="utf-8").splitlines() if ln.startswith(prefix)), None)

    v2 = js("artifacts/results_laya_v2.json") or {}
    prep = js("artifacts/laya/prepare.json") or {}
    pilot = (js("artifacts/laya/pilot.json") or [{}])[0]
    cache = js("artifacts/laya_cache.json") or {}
    lc = cfg.get("laya") or {}
    return {
        "connected": bool(os.environ.get(lc.get("api_key_env") or "LAYA_API_KEY") and lc.get("http_url")),
        "why_not": laya_why_not(cfg),
        "decides": bool(lc.get("decide", True)),
        "cache_entries": len([k for k in cache if not str(k).startswith("_")]),
        "model": "Laya (ModernBERT-large, about 421M parameters), fine-tuned to answer yes/no fraud questions",
        "runs": [
            {"name": "Laya v1", "where": "local GPU run (artifacts/laya/train.log)",
             "data": f"{prep.get('bookings', 'n/a')} bookings ({prep.get('fraud_unique', 'n/a')} distinct fraud) "
                     f"turned into {prep.get('items', 'n/a')} question examples",
             "questions": prep.get("questions"), "trainable_params": pilot.get("trainable_params"),
             "result_file": "artifacts/results_laya.md"},
            {"name": "Laya v2", "where": "Kaggle GPU (scripts/laya_train.py, docs/LAYA_V2.md)",
             "data": (first_line("artifacts/results_laya_v2.md", "Training:") or "").removeprefix("Training: "),
             "questions": ["misuse", "foreign_senders", "payoff_max", "action"], "trainable_params": None,
             "result_file": "artifacts/results_laya_v2.md"},
        ],
        "pr_auc": v2.get("pr_auc_mean"),
        "verdict": v2.get("decision"),
        "win_condition_met": v2.get("W1"),
        "files": ["artifacts/laya/prepare.json", "artifacts/laya/train.log", "artifacts/laya/train.jsonl",
                  "artifacts/results_laya.md", "artifacts/results_laya_v2.md", "docs/LAYA_V2.md"],
    }


def build_verify_router(svc: VerifyService, pipe: Any, audit_path: Path, scenarios_path: Path | None,
                        explain: Callable[[str], None] | None = None,
                        real_source: Callable[[], list] | None = None, cfg: dict[str, Any] | None = None) -> APIRouter:
    r = APIRouter()
    real_state: dict[str, Any] = {"loaded": None, "error": None, "running": False}
    cfg = cfg or {}
    laya_memo: dict[str, dict[str, Any]] = {}

    def laya_for(rec: dict[str, Any]) -> dict[str, Any]:
        did = rec["decision_id"]
        if did not in laya_memo:
            laya_memo[did] = laya_explain(rec, cfg)
        return laya_memo[did]

    @r.get("/decisions/{decision_id}/laya")
    def decision_laya(decision_id: str):
        rec = pipe.get(decision_id)
        if rec is None:
            raise HTTPException(404, "decision not found")
        v = svc.get(decision_id)
        return {**laya_for(rec), "verification": None if v is None else
                {"status": v["status"], "final_action": v["final_action"], "mode": v["mode"]}}

    @r.get("/laya/status")
    def laya_page():
        return laya_status(Path(__file__).resolve().parents[2], cfg)

    def load_real() -> dict[str, Any]:
        if real_source is None:
            raise HTTPException(503, "no real booking source configured")
        if real_state["running"]:
            return {"running": True}
        real_state["running"] = True
        try:
            real_state["loaded"] = load_real_bookings(svc, pipe, real_source)
            real_state["error"] = None
        except Exception as e:  # missing data files: the page says so instead of failing
            real_state["error"] = f"{type(e).__name__}: {e}"
        finally:
            real_state["running"] = False
        return {**(real_state["loaded"] or {}), "error": real_state["error"]}

    r.load_real = load_real  # type: ignore[attr-defined]  # the app runs it once at startup

    @r.post("/verify/real")
    def real():
        return load_real()

    @r.get("/verify/real")
    def real_status():
        return real_state
    allow_ips = [x.strip() for x in (os.environ.get("VERIFY_WEBHOOK_ALLOW_IPS") or "").split(",") if x.strip()]

    def _v(decision_id: str) -> dict:
        v = svc.get(decision_id)
        if v is None:
            raise HTTPException(404, "no verification for this decision")
        return v

    @r.post("/voice/webhook")
    async def webhook(req: Request, token: str | None = None):
        secret = svc.cfg.webhook_secret
        given = token or req.headers.get("x-tracd-token") or ""
        if secret and not hmac.compare_digest(given.encode(), secret.encode()):
            raise HTTPException(401, "bad webhook token")
        if allow_ips and (req.client is None or req.client.host not in allow_ips):
            raise HTTPException(403, "source address not allowed")
        try:
            body = await req.json()
        except (ValueError, json.JSONDecodeError):
            raise HTTPException(422, "body must be JSON")
        try:
            ev = ExecutionEvent.model_validate(body)
        except Exception as e:  # pydantic.ValidationError
            raise HTTPException(422, f"not an execution payload: {e}")
        if not (ev.id or ev.execution_id):
            raise HTTPException(422, "payload needs id (the execution id)")
        out = svc.handle_event(body, "webhook")
        if not out.get("accepted"):
            raise HTTPException(404, out.get("reason", "not accepted"))
        return out

    @r.get("/verify/status")
    def status():
        return {**svc.status(), "bolna_webhook_ips": BOLNA_WEBHOOK_IPS}

    @r.get("/verify/feed")
    def feed(limit: int = 60):
        out = []
        for rec in pipe.list()[:limit]:
            b = rec.get("booking") or {}
            v = svc.get(rec["decision_id"])
            meta = b.get("meta") or {}
            disp = meta.get("display") or {}
            out.append({
                "decision_id": rec["decision_id"], "booking_id": rec["booking_id"],
                "account_id": rec.get("account_id"), "company": (v or {}).get("owner", {}).get("company")
                or disp.get("company"), "booked_at": rec.get("booked_at") or b.get("booked_at"),
                "origin": disp.get("origin_city") or f"{b.get('origin_uf', '')} {b.get('origin_zip3', '')}".strip(),
                "dest": disp.get("dest_city") or f"{b.get('dest_uf', '')} {b.get('dest_zip3', '')}".strip(),
                "display_labels": bool(disp), "weight_kg": b.get("weight_kg"), "service": b.get("service"),
                "carrier_cost": b.get("carrier_cost"),
                "risk": (rec.get("probabilities") or {}).get("misuse", rec.get("gbm_score")),
                "action": rec.get("action"), "latency_ms": (rec.get("latency_ms") or {}).get("total"),
                "verify_status": v["status"] if v else None, "verify_mode": v["mode"] if v else None,
                "final_action": v["final_action"] if v else None, "scenario": meta.get("scenario"),
                "source": rec.get("source"),
                "kind": "demo" if str(meta.get("scenario") or "").startswith("verify-") else "real",
                "laya_summary": laya_for(rec)["summary"] if v else None,
                "label": None if "is_fraud_label" not in meta else ("fraud" if meta["is_fraud_label"] else "honest"),
            })
        return out

    @r.get("/verify/verifications")
    def verifications(limit: int = 50):
        return svc.list()[:limit]

    @r.get("/verify/verifications/{decision_id}")
    def verification(decision_id: str):
        v = _v(decision_id)
        rec = pipe.get(decision_id) or {}
        return {**v, "laya": laya_for(rec) if rec else None, "decision": {"expected_costs": rec.get("expected_costs"), "top_features": rec.get("top_features"),
                                  "reasons": rec.get("reasons"), "probabilities": rec.get("probabilities"),
                                  "latency_ms": rec.get("latency_ms"), "audit_hash": rec.get("audit_hash"),
                                  "model_versions": rec.get("model_versions"), "policy_trace": rec.get("policy_trace"),
                                  "explanation": rec.get("explanation"), "hard_signal": rec.get("hard_signal")}}

    @r.post("/verify/start/{decision_id}")
    def start(decision_id: str, body: StartIn):
        if pipe.get(decision_id) is None:
            raise HTTPException(404, "decision not found")
        try:
            return svc.start(decision_id, sim_answer=body.answer.model_dump() if body.answer else None,
                             force_simulated=body.simulated or body.answer is not None)
        except ValueError as e:
            raise HTTPException(409, str(e))

    @r.get("/verify/cases")
    def cases(include_resolved: bool = False):
        return svc.cases(include_resolved)

    @r.post("/verify/cases/{decision_id}")
    def resolve(decision_id: str, body: CaseIn):
        _v(decision_id)
        try:
            return svc.resolve_case(decision_id, body.label, body.note, body.analyst)
        except ValueError as e:
            raise HTTPException(409, str(e))

    @r.get("/verify/metrics")
    def metrics():
        live = [x for x in pipe.list() if x.get("source") == "live"]
        return svc.metrics(len(live))

    def _scenarios() -> list[dict]:
        if not (scenarios_path and scenarios_path.exists()):
            return []
        return json.loads(scenarios_path.read_text(encoding="utf-8"))

    @r.get("/verify/scenarios")
    def scenarios():
        return [{k: s.get(k) for k in ("scenario", "title", "description", "expected", "say")} for s in _scenarios()]

    @r.post("/verify/demo/{scenario}")
    def demo(scenario: str, bg: BackgroundTasks, simulated: bool = False):
        s = next((x for x in _scenarios() if x["scenario"] == scenario), None)
        if s is None:
            raise HTTPException(404, f"unknown scenario {scenario}")
        b = dict(s["booking"])
        b["booking_id"] = f"{b['booking_id']}-{datetime.now().strftime('%H%M%S%f')[:9]}"
        # A re-run starts from the account's seeded state: an earlier run's lock or confirmed change would otherwise
        # decide the booking before the call (correct in production, wrong for a repeatable demo). Audited.
        acct = b["account_id"]
        if acct in pipe.locked_accounts or pipe.confirmed_changes.get(acct):
            pipe.audit.append("config_change", {"demo_reset": acct, "scenario": scenario,
                                                "unlocked": acct in pipe.locked_accounts,
                                                "cleared_changes": len(pipe.confirmed_changes.get(acct) or [])})
            pipe.locked_accounts.discard(acct)
            pipe.confirmed_changes.pop(acct, None)
        t0 = time.perf_counter()
        out = pipe.score(Booking(**b))
        ms = round((time.perf_counter() - t0) * 1000, 2)
        if explain is not None and out.get("explanation_status") == "pending":
            bg.add_task(explain, out["decision_id"])
        v = None
        if out["action"] in TRIGGER_ACTIONS:
            v = svc.start(out["decision_id"], force_simulated=simulated)
        return {"decision_id": out["decision_id"], "action": out["action"], "score_ms": ms,
                "probabilities": out.get("probabilities"), "verification": v}

    @r.get("/audit/records")
    def audit_records(limit: int = 100, decision_id: str | None = None, event_type: str | None = None):
        if not audit_path.exists():
            return {"records": [], "total": 0}
        rows = []
        with audit_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    rows.append({"seq": None, "corrupt": True, "raw": line[:200]})
                    continue
                p = rec.get("payload") or {}
                if decision_id and p.get("decision_id") != decision_id:
                    continue
                if event_type and rec.get("event_type") != event_type:
                    continue
                rows.append(rec)
        total = len(rows)
        out = rows[-limit:][::-1]
        for x in out:  # keep the viewer light: the full state text stays in the file
            pl = x.get("payload") or {}
            if isinstance(pl, dict) and "state_text" in pl:
                x["payload"] = {**pl, "state_text": (pl["state_text"] or "")[:160] + "..."}
            if isinstance(pl, dict) and "feature_values" in pl:
                x["payload"] = {**x["payload"], "feature_values": f"{len(pl['feature_values'])} values"}
        return {"records": out, "total": total}

    return r
