"""LearningService: the continuous-learning loop behind /learning/* (docs/API.md v1.2).

analyst label -> LabelStore (+ Laya export row) -> retrain (candidate vs active on an eval set
neither trained on) -> gate -> deploy (registry + atomic swap of the pipeline's GBM scorer) ->
audit "retrain"; rollback swaps back and audits "rollback". Retraining runs synchronously in the
request (one at a time); the original train window can be subsampled to keep it under a minute.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from fraudshield.learning.calibration_refresh import refresh_misuse_calibration
from fraudshield.learning.laya_export import LIVE_UPDATE_NOTE, LayaExport
from fraudshield.learning.metrics import GateConfig
from fraudshield.learning.registry import ModelRegistry
from fraudshield.learning.retrain import fit_weighted, run_retrain
from fraudshield.learning.simulate import RealisticFeedbackSim, simulate_feedback
from fraudshield.policy.costs import CostConfig

METRIC_KEYS = ("pr_auc", "ece", "cost_per_1k_brl", "fpr_hard_negative", "recall_new_pattern")


def _count_sources(labels) -> dict[str, int]:
    c: dict[str, int] = {}
    for x in labels:
        c[x["source"]] = c.get(x["source"], 0) + 1
    return c


class NotEnoughLabels(Exception):
    pass


class RetrainBusy(Exception):
    pass


class LearningUnavailable(Exception):
    pass


def gbm_scorer(model) -> Callable:
    def score(fv) -> float:
        return model.score_values(dict(fv.values))
    score.model = model
    return score


class LearningService:
    def __init__(self, pipeline, audit, registry: ModelRegistry, label_store, laya_export: LayaExport,
                 reference: Callable[[], pd.DataFrame], pool: Callable[[], pd.DataFrame] | None = None,
                 costs: CostConfig | None = None, policy=None, gate_cfg: GateConfig | None = None,
                 fit_fn: Callable = fit_weighted, max_train_rows: int | None = None, B: int = 200,
                 calibration_dir: str | Path | None = None, wire_active: bool = False,
                 new_pattern: Callable[[], pd.DataFrame] | None = None, train_options: dict | None = None):
        self.pipeline, self.audit, self.registry = pipeline, audit, registry
        self.label_store, self.laya_export = label_store, laya_export
        self._reference_fn, self._pool_fn = reference, pool
        self._reference: pd.DataFrame | None = None
        self._pool: pd.DataFrame | None = None
        self._new_pattern_fn = new_pattern
        # training-side options (docs/LEARNING_GATE.md): max_feedback_share, ipw_clip, lgb_params
        self.train_options = dict(train_options or {})
        self._new_pattern: pd.DataFrame | None = None
        self._sim: RealisticFeedbackSim | None = None
        self.costs = costs or pipeline.costs
        self.policy = policy or pipeline.policy
        self.gate_cfg = gate_cfg or GateConfig()
        self.fit_fn, self.max_train_rows, self.B = fit_fn, max_train_rows, B
        self.calibration_dir = Path(calibration_dir) if calibration_dir else None
        self._run_lock = threading.Lock()
        pipeline.label_store = label_store
        pipeline.on_label = laya_export.add
        if wire_active and registry.path.exists() and registry.active_version:
            pipeline.swap_gbm(gbm_scorer(registry.load_model(registry.active_version)), registry.active_version)
            cal = (registry.data.get("calibration") or {}).get("path")
            if cal and Path(cal).exists():
                pipeline.calibration = json.loads(Path(cal).read_text(encoding="utf-8"))

    # ---------- data ----------
    def reference(self) -> pd.DataFrame:
        if self._reference is None:
            self._reference = self._reference_fn()
        return self._reference

    def pool(self) -> pd.DataFrame:
        if self._pool_fn is None:
            raise LearningUnavailable("no ground-truth pool configured for simulated feedback")
        if self._pool is None:
            self._pool = self._pool_fn()
        return self._pool

    def new_pattern_pool(self) -> pd.DataFrame | None:
        if self._new_pattern_fn is None:
            return None
        if self._new_pattern is None:
            self._new_pattern = self._new_pattern_fn()
        return self._new_pattern

    # ---------- status ----------
    def _new_labels(self) -> list[dict[str, Any]]:
        return self.label_store.all()[int(self.registry.data.get("last_retrain_label_count", 0)):]

    def status(self) -> dict[str, Any]:
        cal = self.pipeline.calibration or {}
        runs = self.registry.data.get("runs") or []
        return {
            "active_version": self.registry.active_version,
            "labels_since_last_retrain": len(self._new_labels()),
            "label_sources": self.label_store.counts_by_source(),
            "versions": [{k: v.get(k) for k in ("version", "created_at", "parent", "n_train", "n_feedback_labels",
                                                 "metrics", "deployed", "ever_deployed", "gate_passed", "note")}
                         for v in self.registry.versions()],
            "calibration_version": cal.get("version", "uncalibrated"),
            "laya_export": {"path": str(self.laya_export.path), "rows": self.laya_export.rows()},
            "laya_weights": {"updated_live": False, "note": LIVE_UPDATE_NOTE},
            "retrain_mode": "synchronous (one run at a time)",
            "last_run": runs[-1] if runs else None,
            "min_new_labels_default": self.gate_cfg.min_new_labels,
            "gate_mode_default": self.gate_cfg.mode,
            "simulation": None if self._sim is None else {
                "mode": "realistic", "simulated": True,
                "now": self._sim.now.isoformat(timespec="seconds") if self._sim.now else None,
                "cursor": self._sim.cursor, "pool_size": len(self._sim.pool), "pending": len(self._sim.pending)},
        }

    # ---------- simulate ----------
    def simulate(self, n: int = 200, seed: int = 1, error_rate: float = 0.05, injected_share: float = 0.5,
                 mode: str = "realistic", advance_days: float = 0.0) -> dict[str, Any]:
        """mode realistic (default): stream bookings in time order with delayed, action-dependent labels.
        mode uniform_noisy: the original demo (random bookings, every one analyst-labelled with error)."""
        if mode == "uniform_noisy":
            out = simulate_feedback(self.pipeline, self.pool(), n=n, seed=seed, error_rate=error_rate,
                                    injected_share=injected_share)
            return {"mode": "uniform_noisy", **out}
        if mode != "realistic":
            raise ValueError(f"unknown simulation mode {mode}")
        if self._sim is None:
            self._sim = RealisticFeedbackSim(self.pool(), seed=seed, error_rate=error_rate)
        self._sim.error_rate = float(error_rate)
        return self._sim.step(self.pipeline, n=n, advance_days=advance_days)

    # ---------- retrain ----------
    def retrain(self, min_new_labels: int | None = None, gate: str | None = None) -> dict[str, Any]:
        if self.registry.active_version is None:
            raise LearningUnavailable("no active GBM version (artifacts/models/<system>_<tier>.lgb missing)")
        if not self._run_lock.acquire(blocking=False):
            raise RetrainBusy("a retrain is already running")
        try:
            return self._retrain(min_new_labels, gate)
        finally:
            self._run_lock.release()

    def _retrain(self, min_new_labels: int | None, gate_mode: str | None = None) -> dict[str, Any]:
        need = self.gate_cfg.min_new_labels if min_new_labels is None else int(min_new_labels)
        new = self._new_labels()
        if len(new) < need:
            raise NotEnoughLabels(f"Need at least {need} new labels, have {len(new)}")
        gate_cfg = GateConfig(**{**self.gate_cfg.__dict__, "min_new_labels": need,
                                 "mode": gate_mode or self.gate_cfg.mode})
        labels = self.label_store.latest_by_decision()
        res = run_retrain(self.registry, self.reference(), labels, n_new_labels=len(new), costs=self.costs,
                          policy=self.policy, gate_cfg=gate_cfg, fit_fn=self.fit_fn,
                          max_train_rows=self.max_train_rows, B=self.B, new_pattern_pool=self.new_pattern_pool(),
                          **self.train_options)
        rep = res.report
        runs = self.registry.data.setdefault("runs", [])
        run_id = f"rt_{len(runs) + 1:04d}"
        cur = self.registry.active_version
        passed = bool(rep["gate"]["passed"])
        cand_metrics = {k: res.candidate_metrics.get(k) for k in METRIC_KEYS}
        cand_version = self.registry.add_version(
            res.candidate, parent=cur, n_train=rep["training_set"]["n"],
            n_feedback_labels=rep["training_set"]["n_feedback"], metrics=cand_metrics, gate_passed=passed,
            trained_typologies=res.trained_typologies, run_id=run_id)
        self.registry.update(cur, metrics={k: res.current_metrics.get(k) for k in METRIC_KEYS},
                             metrics_run_id=run_id)
        if passed:
            self.registry.activate(cand_version)
            self.pipeline.swap_gbm(gbm_scorer(res.candidate), cand_version)
        deployed = self.registry.active_version
        cal = self._refresh_calibration(labels)
        rep["candidate"]["version"] = cand_version
        out = {
            "run_id": run_id, "n_new_labels": len(new),
            "n_fraud": sum(x["label"] == "fraud" for x in new), "n_legit": sum(x["label"] == "legit" for x in new),
            "n_simulated_labels": sum(bool(x.get("simulated")) for x in labels),
            "eval_set": rep["eval_set"], "current": rep["current"], "candidate": rep["candidate"],
            "gate": rep["gate"], "deployed": passed, "deployed_version": deployed,
            "training_set": rep["training_set"], "bootstrap": rep["bootstrap"],
            "operating_point": rep["operating_point"], "new_pattern_typologies": rep["new_pattern_typologies"],
            "new_pattern_eval": rep["new_pattern_eval"], "gate_mode": gate_cfg.mode,
            "label_sources_used": _count_sources(labels),
            "calibration_refresh": cal,
            "laya": {"export_path": str(self.laya_export.path), "export_rows": self.laya_export.rows(),
                     "weights_updated": False, "note": LIVE_UPDATE_NOTE},
            "notes": rep["notes"], "duration_s": rep["duration_s"],
        }
        payload = {k: v for k, v in out.items() if k != "audit_hash"}
        payload["calibration_refresh"] = {k: v for k, v in cal.items() if k != "calibration"}
        _, h = self.audit.append("retrain", payload)
        out["audit_hash"] = h
        self.registry.data["last_retrain_label_count"] = len(self.label_store)
        runs.append({"run_id": run_id, "at": datetime.now().isoformat(timespec="seconds"),
                     "candidate": cand_version, "gate_passed": passed, "deployed_version": deployed,
                     "audit_hash": h, "duration_s": rep["duration_s"]})
        self.registry.save()
        return out

    def _refresh_calibration(self, labels) -> dict[str, Any]:
        r = refresh_misuse_calibration(labels, self.pipeline.calibration)
        if r["status"] == "updated":
            new = r["calibration"]
            path = None
            if self.calibration_dir is not None:
                self.calibration_dir.mkdir(parents=True, exist_ok=True)
                path = self.calibration_dir / f"{new['version']}.json"
                path.write_text(json.dumps(new, indent=2), encoding="utf-8")
            with self.pipeline._lock:
                self.pipeline.calibration = new
            self.registry.data["calibration"] = {"version": new["version"], "path": str(path) if path else None,
                                                 "parent": new.get("parent")}
        return {k: v for k, v in r.items() if k != "calibration"}

    # ---------- rollback ----------
    def rollback(self, version: str) -> dict[str, Any]:
        with self._run_lock:
            prev = self.registry.active_version
            self.registry.get(version)  # KeyError if unknown
            self.registry.activate(version, require_ever_deployed=True)
            self.pipeline.swap_gbm(gbm_scorer(self.registry.load_model(version)), version)
            _, h = self.audit.append("rollback", {"from": prev, "to": version,
                                                  "at": datetime.now().isoformat(timespec="seconds")})
            return {"active_version": version, "previous_version": prev, "audit_hash": h}
