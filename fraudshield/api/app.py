"""FastAPI app implementing docs/API.md. Single wiring point: build_app(config, components).

Real components are plugged in through `components` (objects) or config `components:` import
strings ("package.module:callable"). Anything missing falls back to a clearly labelled fallback,
reported in GET /health. torch/laya are optional: without them Laya runs in `cached` mode.

Run:  uvicorn fraudshield.api.app:create_app --factory --port 8080
"""
from __future__ import annotations

import importlib
import json
import math
import os
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Literal

import yaml
from fastapi import APIRouter, BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from fraudshield.api.metrics import dashboard_metrics
from fraudshield.api.pipeline import Pipeline, public_view
from fraudshield.audit.log import AuditLog, verify_file
from fraudshield.contracts import Booking, FeatureVector
from fraudshield.explain.llm import DEFAULT_MODEL_ID
from fraudshield.learning.service import LearningUnavailable, NotEnoughLabels, RetrainBusy
from fraudshield.learning.wiring import build_learning
from fraudshield.models.calibration import load_calibration
from fraudshield.models.laya_client import LayaClient, LayaUnavailable, cuda_available
from fraudshield.policy.costs import load_costs
from fraudshield.policy.decide import PolicyConfig
from fraudshield.policy.reasons import booking_signals, reason_catalog

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG: dict[str, Any] = {
    "laya": {"mode": "auto", "model_path": "convaiinnovations/laya", "device": "cuda", "timeout_s": 0.18,
             "http_url": "http://localhost:8001", "api_key_env": "LAYA_API_KEY",
             "cache_path": "artifacts/laya_cache.json", "record_cache": None},
    "calibration_path": "artifacts/calibration.json",
    "costs_path": "config/costs.yaml",
    "policy": {"explore_eps": 0.05, "seed": 0},
    "explain": {"model_id": DEFAULT_MODEL_ID, "llm_for_allow": False},
    "replay_path": "data/processed/replay_decisions.jsonl",
    "demo_bookings_path": "data/processed/demo_bookings.json",
    "audit_path": "artifacts/audit/audit.jsonl",
    "web_dist": "web/dist",
    "cors_origins": ["http://localhost:5173", "http://127.0.0.1:5173"],
    "components": {"featurizer": None, "serializer": None, "gbm": None},
    "versions": {},
    "learning": {},  # continuous learning (fraudshield/learning/wiring.py DEFAULTS)
}


# ---------------- request models ----------------
class BookingIn(BaseModel):
    booking_id: str
    account_id: str
    booked_at: str
    channel: Literal["web", "api", "counter"]
    login_device_age_days: float
    payment_method: Literal["account_billing", "card", "ach"]
    sender_id: str
    origin_uf: str
    origin_zip3: str
    dest_uf: str
    dest_zip3: str
    consignee_id: str
    weight_kg: float
    length_cm: float
    width_cm: float
    height_cm: float
    service: Literal["standard", "express"]
    category: str
    declared_value: float
    carrier_cost: float
    owner_contact_age_days: float | None = None
    meta: dict[str, Any] = {}


class AnalystIn(BaseModel):
    label: Literal["fraud", "legit"]
    note: str = ""


class AskIn(BaseModel):
    instructions: str
    yes: str = "yes"
    no: str = "no"


class SimulateIn(BaseModel):
    mode: Literal["realistic", "uniform_noisy"] = "realistic"
    n: int = Field(200, ge=0, le=25000)
    advance_days: float = Field(0.0, ge=0.0, le=365.0)
    seed: int = 1
    error_rate: float = Field(0.05, ge=0.0, le=1.0)
    injected_share: float = Field(0.5, ge=0.0, le=1.0)


class RetrainIn(BaseModel):
    min_new_labels: int | None = Field(None, ge=0)
    gate: Literal["noninferiority", "strict"] | None = None


class RollbackIn(BaseModel):
    version: str


# ---------------- fallbacks (used only when real components are missing) ----------------
def fallback_featurize(b: Booking) -> FeatureVector:
    """Booking-request signals only; no account history. Labelled fallback in /health."""
    return FeatureVector(b.booking_id, b.booked_at, booking_signals(b))


def fallback_serialize(b: Booking, fv: FeatureVector) -> str:
    """Fixed-order state from request fields only (no meta, no free text)."""
    return "\n".join([
        f"BOOKING {b.booked_at[:16].replace('T', ' ')} | channel {b.channel} | login device seen "
        f"{b.login_device_age_days:.0f}d | payment {b.payment_method.replace('_', ' ')}",
        f"SHIPMENT origin {b.origin_uf} {b.origin_zip3} -> dest {b.dest_uf} {b.dest_zip3} | {b.weight_kg:.1f} kg | "
        f"{b.length_cm:.0f}x{b.width_cm:.0f}x{b.height_cm:.0f} cm | service {b.service} | category {b.category}",
        f"COST carrier cost R${b.carrier_cost:.2f}",
        f"SENDER {'differs from account' if b.sender_id != b.account_id else 'is the account'}",
    ])


def fallback_gbm(fv: FeatureVector) -> float:
    """Transparent heuristic used ONLY when no GBM artifact is wired (reported in /health)."""
    v = fv.values
    z = -4.0 + 2.0 * (float(v.get("login_device_age_days", 99)) <= 1) + 1.0 * float(v.get("sender_differs", 0))
    return 1 / (1 + math.exp(-z))


def _import(spec: str):
    mod, _, attr = spec.partition(":")
    obj = importlib.import_module(mod)
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in (over or {}).items():
        out[k] = _merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


def _path(p: str | None) -> Path | None:
    if not p:
        return None
    q = Path(p)
    return q if q.is_absolute() else ROOT / q


def load_config(config: dict | str | Path | None) -> dict[str, Any]:
    if config is None:
        config = ROOT / "config" / "app.yaml"
    if isinstance(config, (str, Path)):
        p = Path(config)
        config = yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else {}
    return _merge(DEFAULT_CONFIG, config or {})


def resolve_model_path(p: str, root: Path | None = None) -> str:
    """A relative laya.model_path that exists under the repo root (a local fine-tuned checkpoint) is
    made absolute, so the app works from any working directory; Hub ids and absolute paths pass through."""
    root = ROOT if root is None else root
    q = Path(p)
    if not q.is_absolute() and (root / q).exists():
        return str(root / q)
    return p


def _build_laya(cfg: dict, warnings: list[str]):
    lc = dict(cfg["laya"])
    lc["model_path"] = resolve_model_path(lc.get("model_path", "convaiinnovations/laya"))
    mode = lc.get("mode", "auto")
    cache = _path(lc.get("cache_path"))
    try:
        if mode == "auto":
            return LayaClient.auto(lc["model_path"], cache, device=lc.get("device", "cuda"),
                                   timeout_s=lc.get("timeout_s", 0.18), record_cache=_path(lc.get("record_cache")))
        if mode == "local":
            return LayaClient.local(lc["model_path"], lc.get("device", "cuda"), timeout_s=lc.get("timeout_s", 0.18),
                                    record_cache=_path(lc.get("record_cache")))
        if mode == "http":
            return LayaClient.http(lc["http_url"], os.environ.get(lc.get("api_key_env") or "", None),
                                   timeout_s=lc.get("timeout_s", 0.18))
        return LayaClient.cached(cache, mode_reason="configured cached mode")
    except Exception as e:
        warnings.append(f"laya {mode} failed ({e}); using cached answers")
        return LayaClient.cached(cache, mode_reason=f"{mode} failed: {e}")


def _component(name: str, comps: dict, cfg: dict, fallback, report: dict, warnings: list[str]):
    if name in comps and comps[name] is not None:
        report[name] = "injected"
        return comps[name]
    spec = (cfg.get("components") or {}).get({"featurize": "featurizer", "serialize": "serializer"}.get(name, name))
    if spec:
        try:
            obj = _import(spec)
            report[name] = f"real ({spec})"
            return obj
        except Exception as e:
            warnings.append(f"{name}: could not import {spec} ({e}); using fallback")
    report[name] = "fallback (no real component wired)"
    warnings.append(f"{name}: fallback in use, scores are not from the trained model")
    return fallback


# ---------------- app ----------------
def build_app(config: dict | str | Path | None = None, components: dict[str, Any] | None = None) -> FastAPI:
    cfg = load_config(config)
    comps = dict(components or {})
    warnings: list[str] = []
    report: dict[str, str] = {}

    featurize = _component("featurize", comps, cfg, fallback_featurize, report, warnings)
    serialize = _component("serialize", comps, cfg, fallback_serialize, report, warnings)
    gbm = _component("gbm", comps, cfg, fallback_gbm, report, warnings)
    if "laya" in comps:
        laya = comps["laya"]
        report["laya"] = "injected"
    else:
        laya = _build_laya(cfg, warnings)
        report["laya"] = laya.mode

    calibration = load_calibration(_path(cfg["calibration_path"]))
    if calibration is None:
        warnings.append("no calibration file: Laya probabilities are uncalibrated (identity)")
    costs = load_costs(_path(cfg["costs_path"]))
    pol = cfg.get("policy") or {}
    policy = PolicyConfig(**{k: v for k, v in pol.items() if k in {f.name for f in fields(PolicyConfig)}})

    explain_cfg = cfg.get("explain") or {}
    llm_model_id = explain_cfg.get("model_id", DEFAULT_MODEL_ID)
    if "llm_client" in comps:
        llm_client = comps["llm_client"]
        provider = "injected"
    else:
        from fraudshield.explain.groq_client import pick_llm  # noqa: PLC0415
        llm_client, picked_model, provider = pick_llm(explain_cfg)
        llm_model_id = picked_model or llm_model_id
        if llm_client is None:
            warnings.append("no ANTHROPIC_API_KEY or GROQ_API_KEY: explanations use the template")
    report["explainer"] = f"llm+validator ({provider}: {llm_model_id})" if llm_client is not None else "template"

    audit_path = _path(cfg["audit_path"])
    audit = AuditLog(audit_path)
    versions = {"gbm": "fallback-heuristic" if report["gbm"].startswith("fallback") else "gbm",
                "serializer": "fallback-serializer" if report["serialize"].startswith("fallback") else "serializer",
                "features": "fallback-booking-only" if report["featurize"].startswith("fallback") else "features"}
    versions.update(cfg.get("versions") or {})
    pipe = Pipeline(featurize, serialize, gbm, laya, audit, calibration=calibration, costs=costs, policy=policy,
                    seed=int(pol.get("seed", 0)), llm_client=llm_client,
                    llm_model_id=llm_model_id,
                    llm_for_allow=bool(explain_cfg.get("llm_for_allow", False)), versions=versions)

    replay = {"path": str(_path(cfg.get("replay_path"))), "loaded": 0, "error": None}
    rp = _path(cfg.get("replay_path"))
    if rp and rp.exists():
        try:
            for line in rp.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    pipe.add_record(json.loads(line))
                    replay["loaded"] += 1
        except Exception as e:
            replay["error"] = str(e)
            warnings.append(f"replay file could not be fully loaded: {e}")

    learning = None
    lcfg = cfg.get("learning") or {}
    if lcfg.get("enabled", True):
        try:
            if report["gbm"] == "injected":  # never replace an injected scorer at startup
                cfg = {**cfg, "learning": {**lcfg, "wire_active": False}}
            learning = build_learning(cfg, pipe, audit, audit_path, _path)
            if pipe.versions.get("gbm") != versions.get("gbm"):
                report["gbm"] = f"learning registry ({pipe.versions['gbm']})"
        except Exception as e:  # learning must never stop the decision service
            warnings.append(f"continuous learning disabled: {type(e).__name__}: {e}")

    app = FastAPI(title="FraudShield decision service", version="1")
    app.state.pipeline = pipe
    app.state.learning = learning
    app.add_middleware(CORSMiddleware, allow_origins=cfg.get("cors_origins") or [], allow_methods=["*"],
                       allow_headers=["*"])
    scheduled: set[str] = set()
    r = APIRouter()

    def _get(did: str) -> dict:
        rec = pipe.get(did)
        if rec is None:
            raise HTTPException(404, "decision not found")
        return rec

    @r.post("/score")
    def score(body: BookingIn, bg: BackgroundTasks):
        b = Booking(**body.model_dump())
        try:
            out = pipe.score(b)
        except LayaUnavailable as e:  # pragma: no cover - pipeline already degrades
            raise HTTPException(503, str(e))
        except OSError as e:
            raise HTTPException(503, f"audit write failed, no decision issued: {e}")
        if out["explanation_status"] == "pending" and out["decision_id"] not in scheduled:
            scheduled.add(out["decision_id"])
            bg.add_task(pipe.explain_decision, out["decision_id"])
        return out

    @r.get("/decisions")
    def list_decisions(limit: int = 50, action: str | None = None):
        out = []
        for rec in pipe.list():
            if action and rec.get("action") != action:
                continue
            bk = rec.get("booking") or {}
            probs = rec.get("probabilities") or {}
            out.append({"decision_id": rec["decision_id"], "booking_id": rec["booking_id"],
                        "account_id": rec.get("account_id", bk.get("account_id")),
                        "booked_at": rec.get("booked_at", bk.get("booked_at")), "action": rec.get("action"),
                        "misuse": probs.get("misuse", rec.get("gbm_score")),
                        "carrier_cost": bk.get("carrier_cost", rec.get("carrier_cost")),
                        "analyst_label": (rec.get("analyst") or {}).get("label"),
                        "scenario": (bk.get("meta") or {}).get("scenario")})
            if len(out) >= limit:
                break
        return out

    @r.get("/decisions/{decision_id}")
    def get_decision(decision_id: str):
        rec = _get(decision_id)
        return {**public_view(rec), "booking": rec.get("booking"), "explanation": rec.get("explanation"),
                "analyst": rec.get("analyst"), "policy_trace": rec.get("policy_trace")}

    @r.post("/decisions/{decision_id}/analyst")
    def analyst(decision_id: str, body: AnalystIn):
        _get(decision_id)
        return {"ok": True, "audit_hash": pipe.analyst_feedback(decision_id, body.label, body.note)}

    @r.post("/decisions/{decision_id}/ask")
    def ask(decision_id: str, body: AskIn):
        _get(decision_id)
        try:
            return pipe.ask(decision_id, body.instructions, body.yes, body.no)
        except LayaUnavailable as e:
            raise HTTPException(503, f"Laya unavailable: {e}")

    def _learning():
        if learning is None:
            raise HTTPException(503, "continuous learning is disabled (see /health warnings)")
        return learning

    @r.get("/learning/status")
    def learning_status():
        return _learning().status()

    @r.post("/learning/simulate_feedback")
    def learning_simulate(body: SimulateIn | None = None):
        body = body or SimulateIn()
        try:
            return _learning().simulate(n=body.n, seed=body.seed, error_rate=body.error_rate,
                                        injected_share=body.injected_share, mode=body.mode,
                                        advance_days=body.advance_days)
        except LearningUnavailable as e:
            raise HTTPException(503, str(e))

    @r.post("/learning/retrain")
    def learning_retrain(body: RetrainIn | None = None):
        body = body or RetrainIn()
        try:
            return _learning().retrain(min_new_labels=body.min_new_labels, gate=body.gate)
        except (NotEnoughLabels, RetrainBusy) as e:
            raise HTTPException(409, str(e))
        except LearningUnavailable as e:
            raise HTTPException(503, str(e))

    @r.post("/learning/rollback")
    def learning_rollback(body: RollbackIn):
        try:
            return _learning().rollback(body.version)
        except KeyError:
            raise HTTPException(404, f"unknown version {body.version}")
        except ValueError as e:
            raise HTTPException(409, str(e))

    @r.get("/dashboard/metrics")
    def metrics():
        return dashboard_metrics(pipe.list(), costs)

    @r.get("/audit/verify")
    def audit_verify():
        return verify_file(audit_path)

    @r.get("/demo/bookings")
    def demo_bookings():
        p = _path(cfg.get("demo_bookings_path"))
        if not (p and p.exists()):
            return []
        return [_wrap_demo(x) for x in json.loads(p.read_text(encoding="utf-8"))]

    @r.get("/reason-codes")
    def reason_codes_list():
        return reason_catalog()

    @r.get("/health")
    def health():
        lh = laya.health() if hasattr(laya, "health") else {}
        if pipe.last_degraded is not None:
            degraded = pipe.last_degraded
        else:  # nothing scored yet: degraded if Laya is cached with an empty cache
            degraded = lh.get("mode") == "cached" and not lh.get("cache_entries")
        return {"ok": True, "degraded": bool(degraded), "laya_mode": lh.get("mode", getattr(laya, "mode", None)),
                "laya_reason": lh.get("reason"), "gpu": cuda_available(),
                "versions": {**versions, "gbm": pipe.versions.get("gbm", versions.get("gbm")),
                             "calibration": (pipe.calibration or {}).get("version", "uncalibrated"),
                             "policy": policy.version, "costs": costs.version, "laya": lh.get("revision")},
                "components": report, "calibrated": calibration is not None, "warnings": warnings,
                "decisions": len(pipe.list()), "replay": replay, "audit_path": str(audit_path)}

    app.include_router(r)
    app.include_router(r, prefix="/api")

    dist = _path(cfg.get("web_dist"))
    if dist and (dist / "index.html").exists():
        app.mount("/", SPAStaticFiles(directory=str(dist), html=True), name="web")
    return app


def _wrap_demo(x: dict[str, Any]) -> dict[str, Any]:
    """Demo entries: pass through wrapped ones, wrap bare Booking dicts (scenario from meta)."""
    if "booking" in x:
        return {"scenario": x.get("scenario"), "title": x.get("title"), "description": x.get("description"),
                "expected": x.get("expected"), "booking": x["booking"]}
    meta = x.get("meta") or {}
    return {"scenario": meta.get("scenario"), "title": meta.get("title") or x.get("booking_id"),
            "description": meta.get("description", ""), "expected": meta.get("expected"), "booking": x}


class SPAStaticFiles(StaticFiles):
    """Static files with single-page-app fallback to index.html for unknown paths."""

    async def get_response(self, path, scope):
        try:
            resp = await super().get_response(path, scope)
        except StarletteHTTPException as e:
            if e.status_code != 404:
                raise
            return await super().get_response("index.html", scope)
        if resp.status_code == 404:
            return await super().get_response("index.html", scope)
        return resp


def create_app() -> FastAPI:
    """uvicorn factory: reads FS_CONFIG or config/app.yaml."""
    from fraudshield.explain.groq_client import load_dotenv  # noqa: PLC0415
    load_dotenv(ROOT / ".env")
    return build_app(os.environ.get("FS_CONFIG") or ROOT / "config" / "app.yaml")
