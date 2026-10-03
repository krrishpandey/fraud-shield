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
import threading
import time
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
from fraudshield.explain.llm import DEFAULT_MODEL_ID, check_text
from fraudshield.learning.service import LearningUnavailable, NotEnoughLabels, RetrainBusy
from fraudshield.learning.wiring import build_learning
from fraudshield.models.calibration import load_calibration
from fraudshield.models.laya_client import LayaClient, LayaUnavailable, cuda_available
from fraudshield.policy.costs import load_costs
from fraudshield.policy.decide import PolicyConfig
from fraudshield.policy.reasons import booking_signals, reason_catalog
from fraudshield.sim.stream import StreamMetrics, StreamRunner, offline_metrics
from fraudshield.monitor.live import live_estimate, load_results as load_monitor_results

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG: dict[str, Any] = {
    "first_scan": {"level": "standard", "dial_path": "artifacts/first_scan_dial.json"},
    "stream": {"audit_path": "artifacts/audit/stream_audit.jsonl", "explain_llm_per_min": 5, "max_rate": 200,
               "explain_llm_always": ["hold", "block"]},
    "laya_ask": {"enabled": False, "model_path": "convaiinnovations/laya", "device": "cuda", "timeout_s": 3.0},
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


class StreamStartIn(BaseModel):
    rate: float = 20.0
    concurrency: int = 4
    seed: int = 0
    limit: int | None = None


class DialIn(BaseModel):
    level: str


class ScanIn(BaseModel):
    measured_weight_kg: float = Field(gt=0)


class CheckIn(BaseModel):
    text: str


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


def _json_safe(x):
    """NaN and infinity are not valid JSON: send them as null."""
    if isinstance(x, float):
        return None if (math.isnan(x) or math.isinf(x)) else x
    if isinstance(x, dict):
        return {k: _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    return x


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
    if lc.get("decide", True) is False:
        return LayaClient("cached", None, model="none", cache={},
                          mode_reason="laya.decide is false: decisions use the LightGBM backup score; Laya answers "
                                      "questions only (laya_ask)")
    lc["model_path"] = resolve_model_path(lc.get("model_path", "convaiinnovations/laya"))
    mode = lc.get("mode", "auto")
    cache = _path(lc.get("cache_path"))
    if mode == "auto" and not Path(lc["model_path"]).is_dir():
        return LayaClient.cached(cache, mode_reason="no fine-tuned Laya imported yet (laya.model_path is the stock "
                                 "model), so decisions use cached answers or the backup model")
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

    story = comps.get("story")
    story_spec = (cfg.get("components") or {}).get("story")
    if story is None and story_spec:
        try:
            story = _import(story_spec)
        except Exception as e:
            warnings.append(f"story: could not import {story_spec} ({e}); account story unavailable")
    report["story"] = "injected" if "story" in comps else (f"real ({story_spec})" if story else "none")
    stream_source = comps.get("stream_source")
    src_spec = (cfg.get("components") or {}).get("stream_source")
    if stream_source is None and src_spec:
        try:
            stream_source = _import(src_spec)
        except Exception as e:
            warnings.append(f"stream_source: could not import {src_spec} ({e}); live stream unavailable")

    ask_laya = comps.get("ask_laya")
    ac = cfg.get("laya_ask") or {}
    if ask_laya is None and ac.get("enabled") and getattr(laya, "mode", None) not in ("local", "http") and cuda_available():
        from fraudshield.models.laya_client import LazyLaya  # noqa: PLC0415
        ask_path = resolve_model_path(ac.get("model_path", "convaiinnovations/laya"))
        if str(ac.get("model_path", "")).startswith(("artifacts/", "./", "/")) and not Path(ask_path).is_dir():
            warnings.append(f"laya_ask: {ask_path} not found (fine-tuned model not imported here); using stock Laya")
            ask_path = "convaiinnovations/laya"
        ask_label = "fine-tuned Laya" if Path(ask_path).is_dir() else "stock Laya (not fine-tuned)"
        ask_laya = LazyLaya(lambda: LayaClient.local(ask_path, ac.get("device", "cuda"),
                                                     timeout_s=float(ac.get("timeout_s", 3.0))),
                            label=ask_label)
        ask_laya.warm()

    # Laya v2 decides with its action head (docs/LAYA_V2.md); only when Laya is the decision model
    laya_action = bool((cfg.get("laya") or {}).get("action_head", False)) and (cfg.get("laya") or {}).get("decide", True) is not False
    report["decider"] = ("laya (action head, cost auditor)" if laya_action else
                         "laya probabilities + cost rule" if (cfg.get("laya") or {}).get("decide", True) is not False
                         else "lightgbm + cost rule (laya answers questions)")
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
                    llm_for_allow=bool(explain_cfg.get("llm_for_allow", False)), versions=versions,
                    ask_laya=ask_laya, laya_action=laya_action)

    fs_cfg = cfg.get("first_scan") or {}
    dial_path = _path(fs_cfg.get("dial_path"))
    dial = json.loads(dial_path.read_text(encoding="utf-8")) if dial_path and dial_path.exists() else {"levels": []}
    dial_state = {"current": "standard"}

    def _set_dial(level: str) -> None:
        lv = next((x for x in dial.get("levels", []) if x["name"] == level), None)
        if lv is None and level != "standard":
            raise ValueError(level)
        dial_state["current"] = level
        pipe.scan_threshold = lv.get("threshold") if lv else None  # the stream service copies it (routes below)

    try:
        _set_dial(fs_cfg.get("level", "standard"))
    except ValueError:
        warnings.append(f"first_scan.level {fs_cfg.get('level')!r} not in the dial; using standard")
        _set_dial("standard")

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

    app = FastAPI(title="tracd decision service", version="1")
    app.state.pipeline = pipe
    app.state.learning = learning
    app.add_middleware(CORSMiddleware, allow_origins=cfg.get("cors_origins") or [], allow_methods=["*"],
                       allow_headers=["*"])
    scheduled: set[str] = set()
    stream: dict[str, Any] = {}
    app.state.stream = stream
    if learning is not None:  # a deploy or rollback also reaches a running live stream, with its version label
        learning.swap_listeners.append(lambda scorer, v: stream["pipe"].swap_gbm(scorer, v) if stream.get("pipe") else None)
    r = APIRouter()

    def _owner(did: str) -> Pipeline:
        sp = stream.get("pipe")
        return sp if (did.startswith("str_") and sp is not None) else pipe

    def _get(did: str) -> dict:
        rec = _owner(did).get(did)
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
                "analyst": rec.get("analyst"), "policy_trace": rec.get("policy_trace"),
                "first_scan": rec.get("first_scan"), "owner_confirmation": rec.get("owner_confirmation")}

    @r.post("/decisions/{decision_id}/analyst")
    def analyst(decision_id: str, body: AnalystIn):
        _get(decision_id)
        return {"ok": True, "audit_hash": _owner(decision_id).analyst_feedback(decision_id, body.label, body.note)}

    @r.post("/decisions/{decision_id}/ask")
    def ask(decision_id: str, body: AskIn):
        _get(decision_id)
        try:
            return _owner(decision_id).ask(decision_id, body.instructions, body.yes, body.no)
        except LayaUnavailable as e:
            raise HTTPException(503, f"Laya unavailable: {e}")

    @r.get("/decisions/{decision_id}/account-story")
    def account_story(decision_id: str):
        rec = _get(decision_id)
        if story is None:
            raise HTTPException(503, "account history is not available in this configuration")
        return story(Booking(**rec["booking"]))

    @r.get("/decisions/{decision_id}/counterfactual")
    def counterfactual(decision_id: str):
        """ANALYST-ONLY (docs/API.md, "What would change this decision"): the smallest changes to booker-controlled
        fields that would soften this decision. These are evasion hints: never shown to the booker, and every view is
        audited before anything is returned."""
        from fraudshield.redteam.search import Outcome, WhatIf, counterfactual_payload, search  # noqa: PLC0415
        rec = _get(decision_id)
        if not rec.get("booking"):
            raise HTTPException(409, "this decision has no stored booking (replayed record), so there is nothing to vary")
        owner = _owner(decision_id)
        probs = rec.get("probabilities") or {}
        current = Outcome(rec["action"], float(probs.get("misuse", rec.get("gbm_score") or 0.0)),
                          float(rec.get("gbm_score") or 0.0), rec.get("greedy_action") or rec["action"])
        out = counterfactual_payload(search(WhatIf.for_pipeline(owner), Booking(**rec["booking"]), current))
        try:
            _, h = owner.audit.append("counterfactual_view", {
                "decision_id": decision_id, "booking_id": rec["booking_id"], "viewer": "analyst",
                "found": out["found"], "shown": [c["summary"] for c in out["counterfactuals"]],
                "evaluations": out["evaluations"], "latency_ms": out["latency_ms"]})
        except OSError as e:
            raise HTTPException(503, f"audit write failed, counterfactual not shown: {e}")
        return _json_safe({"decision_id": decision_id, **out, "audit_hash": h})

    @r.post("/decisions/{decision_id}/first-scan")
    def first_scan(decision_id: str, body: ScanIn):
        """Depot scale reading for a booking (integration point for the carrier's scanners)."""
        _get(decision_id)
        return _owner(decision_id).record_first_scan(decision_id, body.measured_weight_kg)

    @r.get("/first-scan/dial")
    def first_scan_dial():
        """Depot weighing levels (fitted by scripts/fit_first_scan.py) and the one in use."""
        return {"current": dial_state["current"], "levels": dial.get("levels", []), "version": dial.get("version")}

    @r.post("/first-scan/dial")
    def set_first_scan_dial(body: DialIn):
        names = {x["name"] for x in dial.get("levels", [])} | {"standard"}
        if body.level not in names:
            raise HTTPException(422, f"unknown level {body.level!r}; choose one of {sorted(names)}")
        old = dial_state["current"]
        _set_dial(body.level)
        sp = stream.get("pipe")
        if sp is not None:
            sp.scan_threshold = pipe.scan_threshold
        _, h = audit.append("config_change", {"setting": "first_scan.level", "from": old, "to": body.level,
                                              "threshold": pipe.scan_threshold})
        return {**first_scan_dial(), "audit_hash": h}

    @r.post("/decisions/{decision_id}/explanation/check")
    def check_explanation(decision_id: str, body: CheckIn):
        return check_text(_get(decision_id), body.text)

    # ---------- live booking stream ----------
    def _stream_status() -> dict:
        r = stream.get("runner")
        if r is None:
            return {"state": "idle", "available": stream_source is not None}
        return {**r.status(), "seed": stream["seed"], "available": True}

    def _stream_send_factory(spipe: Pipeline, metrics: StreamMetrics, cap: int, always: set[str]):
        """Explanations for streamed decisions. Actions in `always` (held, blocked: rare, and the ones an analyst
        reviews) always get a language-model explanation, one at a time so the provider never sees a burst.
        Other non-allow actions share `cap` LLM explanations per minute; everything else uses the template.
        Every LLM text still passes the validator, or the template is shown."""
        import collections  # noqa: PLC0415
        from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415
        llm_times: collections.deque = collections.deque()
        lock = threading.Lock()
        pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="stream-explain")
        priority_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stream-explain-held")
        stream["explain_pool"], stream["explain_priority_pool"] = pool, priority_pool

        def explain(did: str, use_llm: bool) -> None:
            spipe.explain_decision(did, use_llm=use_llm)

        def send(b: Booking) -> dict:
            out = spipe.score(b)
            if out["action"] == "allow_scan_gated":
                # SIMULATED depot scan: the replayed dataset knows the parcel's true weight
                it = (stream.get("truth") or {}).get(b.booking_id)
                true_w = getattr(it, "true_weight_kg", None) or b.weight_kg
                scan = spipe.record_first_scan(out["decision_id"], true_w, source="simulated depot scan")
                with lock:
                    metrics.scans["checked"] += 1
                    metrics.scans["mismatch"] += int(scan["mismatch"])
            if out.get("explanation_status") == "pending":
                if out["action"] in always and spipe.llm_client is not None:
                    with lock:
                        metrics.explanations["llm"] += 1
                        metrics.explanations["llm_held_blocked"] += 1
                    priority_pool.submit(explain, out["decision_id"], True)
                    return out
                now = time.monotonic()
                with lock:
                    while llm_times and now - llm_times[0] > 60:
                        llm_times.popleft()
                    use_llm = spipe.llm_client is not None and out["action"] != "allow" and len(llm_times) < cap
                    if use_llm:
                        llm_times.append(now)
                    metrics.explanations["llm" if use_llm else "template"] += 1
                pool.submit(explain, out["decision_id"], use_llm)
            return out
        return send

    @r.post("/stream/start")
    def stream_start(body: StreamStartIn):
        if stream_source is None:
            raise HTTPException(503, "the live stream needs the dataset (components.stream_source is not configured)")
        cur = stream.get("runner")
        if cur is not None and cur.status()["state"] in ("running", "paused"):
            raise HTTPException(409, "a stream is already running; stop it first")
        scfg = cfg.get("stream") or {}
        if not (0 < body.rate <= float(scfg.get("max_rate", 200))) or not (1 <= body.concurrency <= 16):
            raise HTTPException(422, "rate must be above 0 and at most max_rate; concurrency between 1 and 16")
        try:
            items = stream_source(body.seed)
        except ValueError as e:
            raise HTTPException(400, str(e))
        if body.limit:
            items = items[: body.limit]
        # the stream scores with the main pipeline's GBM; learning deploys and rollbacks swap it (listener below)
        spipe = Pipeline(featurize, serialize, pipe.gbm, laya, AuditLog(_path(scfg["audit_path"])),
                         calibration=calibration, costs=costs, policy=policy, seed=int(pol.get("seed", 0)),
                         llm_client=llm_client, llm_model_id=llm_model_id,
                         llm_for_allow=bool(explain_cfg.get("llm_for_allow", False)), versions=dict(pipe.versions),
                         ask_laya=ask_laya, id_prefix="str_", laya_action=laya_action)
        spipe.scan_threshold = pipe.scan_threshold
        metrics = StreamMetrics()
        metrics.offline_full = offline_metrics(items)
        cap = int(scfg.get("explain_llm_per_min", 5))
        metrics.explanations["llm_cap_per_min"] = cap
        metrics.explanations["llm_held_blocked"] = 0
        always = set(scfg.get("explain_llm_always", ["hold", "block"]) or [])
        metrics.explanations["llm_always_for"] = sorted(always)
        runner = StreamRunner(items, _stream_send_factory(spipe, metrics, cap, always), metrics,
                              rate=body.rate, concurrency=body.concurrency)
        stream.update({"runner": runner, "metrics": metrics, "pipe": spipe, "seed": body.seed,
                       "truth": {it.booking.booking_id: it for it in items}})
        runner.start()
        return _stream_status()

    def _runner():
        r = stream.get("runner")
        if r is None:
            raise HTTPException(404, "no stream has been started")
        return r

    @r.post("/stream/pause")
    def stream_pause():
        _runner().pause()
        return _stream_status()

    @r.post("/stream/resume")
    def stream_resume():
        _runner().resume()
        return _stream_status()

    @r.post("/stream/stop")
    def stream_stop():
        _runner().stop()
        return _stream_status()

    @r.get("/stream/status")
    def stream_status():
        return _stream_status()

    @r.get("/stream/metrics")
    def stream_metrics():
        m = stream.get("metrics")
        if m is None:
            return {"status": _stream_status()}
        return _json_safe({**m.snapshot(), "status": _stream_status()})

    @r.get("/stream/flagged")
    def stream_flagged(actions: str = "hold,block", limit: int = 2000):
        """Every streamed booking that was held or blocked, newest first, for the analyst panel."""
        want = {a.strip() for a in actions.split(",") if a.strip()}
        sp = stream.get("pipe")
        rows, counts = [], {"hold": 0, "block": 0, "unreviewed": 0}
        for rec in (sp.list() if sp is not None else []):
            act = rec.get("action")
            if act not in ("hold", "block"):
                continue
            counts[act] += 1
            label = (rec.get("analyst") or {}).get("label")
            counts["unreviewed"] += label is None
            if act not in want or len(rows) >= max(1, min(limit, 5000)):
                continue
            b = rec.get("booking") or {}
            rows.append({"decision_id": rec["decision_id"], "booking_id": rec["booking_id"],
                         "booked_at": b.get("booked_at"), "account_id": b.get("account_id"),
                         "route": f"{b.get('origin_uf')} {b.get('origin_zip3')} to {b.get('dest_uf')} {b.get('dest_zip3')}",
                         "carrier_cost": b.get("carrier_cost"), "action": act,
                         "score": (rec.get("probabilities") or {}).get("misuse", rec.get("gbm_score")),
                         "reasons": list(rec.get("reasons") or [])[:3], "reviewed": label,
                         "explanation_status": rec.get("explanation_status"), "explanation": rec.get("explanation")})
        return _json_safe({"rows": rows, "counts": counts})

    @r.get("/stream/feed")
    def stream_feed(limit: int = 30):
        m = stream.get("metrics")
        return [] if m is None else _json_safe(m.feed(max(1, min(limit, 200))))

    # ---------- label-free monitor (fraudshield/monitor) ----------
    monitor_results = load_monitor_results(_path(cfg.get("monitor_results_path") or "artifacts/results_monitor.json"))

    @r.get("/monitor/estimate")
    def monitor_estimate():
        """Estimated precision of stops and missed fraud over the stream's decisions so far, without labels."""
        m = stream.get("metrics")
        return _json_safe({**live_estimate(m.rows() if m is not None else [], monitor_results),
                           "source": "stream", "status": _stream_status()})

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

    def _stream_records_with_truth() -> list[dict]:
        """Streamed decisions for the dashboard. The replayed dataset's truth is attached to these copies only (the
        decision records and the model never see it); an analyst label on a streamed decision still wins."""
        sp, truth = stream.get("pipe"), stream.get("truth") or {}
        out = []
        for rec in (sp.list() if sp is not None else []):
            it = truth.get(rec["booking_id"])
            if it is None:
                out.append(rec)
                continue
            b = dict(rec.get("booking") or {})
            b["meta"] = {**(b.get("meta") or {}), "typology": it.typology if it.is_fraud else None,
                         "hard_negative": bool(getattr(it, "hard_negative", False))}
            out.append({**rec, "booking": b, "label": "fraud" if it.is_fraud else "legit",
                        "label_source": "replayed dataset"})
        return out

    @r.get("/dashboard/metrics")
    def metrics(source: Literal["app", "stream", "all"] = "app"):
        recs = (pipe.list() if source in ("app", "all") else []) + (_stream_records_with_truth() if source in ("stream", "all") else [])
        return {**dashboard_metrics(recs, costs), "source": source,
                "stream_bookings": len(stream["pipe"].list()) if stream.get("pipe") is not None else 0}

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
        ask_model = pipe.ask_model()
        return {"ok": True, "degraded": bool(degraded), "laya_mode": lh.get("mode", getattr(laya, "mode", None)),
                "ask_model": ask_model,
                "laya_reason": lh.get("reason"), "gpu": cuda_available(),
                "versions": {**versions, "gbm": pipe.versions.get("gbm", versions.get("gbm")),
                             "calibration": (pipe.calibration or {}).get("version", "uncalibrated"),
                             "policy": policy.version, "costs": costs.version, "laya": lh.get("revision")},
                "components": report, "calibrated": calibration is not None, "warnings": warnings,
                "decisions": len(pipe.list()), "replay": replay, "audit_path": str(audit_path)}

    app.include_router(r)
    app.include_router(r, prefix="/api")
    # owner passkey "was this you?" (fraudshield/api/passkey.py, docs/PASSKEY.md); credentials in artifacts/passkeys/
    from fraudshield.api.passkey import build_passkey_router  # noqa: PLC0415
    pk = build_passkey_router(cfg.get("passkey"), lambda did: (_get(did), _owner(did).audit), audit,
                              (audit_path.parent.parent if audit_path.parent.name == "audit" else audit_path.parent)
                              / "passkeys")
    app.include_router(pk)
    app.include_router(pk, prefix="/api")

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
