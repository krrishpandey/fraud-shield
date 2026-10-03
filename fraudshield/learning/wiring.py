"""Builds the LearningService for the app from config (`learning:` section) and data outputs.

Reference data = the GBM's own train + calibration windows from the data pipeline's feature table
(data/processed/features/seed_0.parquet, the seed the saved models were trained on). The simulate
pool = test-window rows of the same table, plus sender/consignee ids from bookings_all.parquet.
Both load lazily on first use, so the app starts even when the data outputs are missing.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pandas as pd

from fraudshield.learning.label_store import LabelStore
from fraudshield.learning.laya_export import LayaExport
from fraudshield.learning.metrics import GateConfig
from fraudshield.learning.registry import ModelRegistry
from fraudshield.learning.service import LearningService, LearningUnavailable

DEFAULTS: dict[str, Any] = {
    "enabled": True, "system": "B2", "tier": "F", "models_dir": "artifacts/models", "registry_path": None,
    "versions_dir": None, "feedback_dir": None, "calibration_dir": None,
    "features_path": "data/processed/features/seed_0.parquet",
    "features_fallback_path": "data/processed/features.parquet",
    "bookings_path": "data/processed/bookings_all.parquet",
    "max_train_rows": None, "bootstrap_B": 200, "min_new_labels": 20, "wire_active": False,
    "cost_tolerance_rel": 0.02, "max_ece_increase": 0.02, "max_fpr_hn_increase": 0.005,
    "gate_mode": "noninferiority", "cost_margin_rel": 0.05, "pr_auc_margin": 0.02,
    "new_pattern_glob": "data/processed/features/seed_*.parquet",
    # training-side options (docs/LEARNING_GATE.md "Training changes after run ...")
    "max_feedback_share": None, "ipw_clip": 20.0, "lgb_params": None,
}


def _features_path(lc: dict, resolve: Callable) -> Path:
    for key in ("features_path", "features_fallback_path"):
        p = resolve(lc.get(key))
        if p is not None and p.exists():
            return p
    raise LearningUnavailable(f"feature table not found ({lc.get('features_path')}); run scripts/build_dataset.py")


def load_reference(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df[df.split.isin(["train", "cal"])].reset_index(drop=True)


def load_pool(path: Path, bookings_path: Path | None) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df[df.split == "test"].reset_index(drop=True)
    if bookings_path is not None and bookings_path.exists():
        b = pd.read_parquet(bookings_path, columns=["booking_id", "sender_id", "consignee_id"])
        df = df.merge(b.drop_duplicates("booking_id"), on="booking_id", how="left")
    for c, default in (("sender_id", None), ("consignee_id", "unknown")):
        if c not in df:
            df[c] = df["account_id"] if default is None else default
    return df


def load_new_pattern_pool(paths: list[Path]) -> pd.DataFrame | None:
    """Test-window fraud rows from every injection seed (eval set b). Filtering to typologies the active
    model never trained on happens in run_retrain."""
    parts = [pd.read_parquet(p, filters=[("split", "==", "test"), ("is_fraud", "==", True)]) for p in paths]
    return pd.concat(parts, ignore_index=True) if parts else None


def build_learning(cfg: dict, pipeline, audit, audit_path: Path, resolve: Callable) -> LearningService:
    lc = {**DEFAULTS, **(cfg.get("learning") or {})}
    models_dir = resolve(lc["models_dir"])
    if lc.get("feedback_dir"):
        fb = resolve(lc["feedback_dir"])
    else:  # next to the audit dir: artifacts/audit/audit.jsonl -> artifacts/feedback
        fb = (audit_path.parent.parent if audit_path.parent.name == "audit" else audit_path.parent) / "feedback"
    registry = ModelRegistry(resolve(lc["registry_path"]) if lc.get("registry_path") else models_dir / "registry.json",
                             models_dir, lc["system"], lc["tier"],
                             versions_dir=resolve(lc["versions_dir"]) if lc.get("versions_dir") else None)
    gate = GateConfig(min_new_labels=int(lc["min_new_labels"]), cost_tolerance_rel=float(lc["cost_tolerance_rel"]),
                      max_ece_increase=float(lc["max_ece_increase"]),
                      max_fpr_hn_increase=float(lc["max_fpr_hn_increase"]), mode=str(lc["gate_mode"]),
                      cost_margin_rel=float(lc["cost_margin_rel"]), pr_auc_margin=float(lc["pr_auc_margin"]))
    np_paths: list[Path] = []
    if lc.get("new_pattern_glob"):
        g = resolve(lc["new_pattern_glob"])
        np_paths = sorted(g.parent.glob(g.name))
    return LearningService(
        pipeline=pipeline, audit=audit, registry=registry, label_store=LabelStore(fb / "labels.jsonl"),
        laya_export=LayaExport(fb / "laya_feedback.jsonl"),
        reference=lambda: load_reference(_features_path(lc, resolve)),
        pool=lambda: load_pool(_features_path(lc, resolve), resolve(lc.get("bookings_path"))),
        gate_cfg=gate, max_train_rows=lc.get("max_train_rows"), B=int(lc["bootstrap_B"]),
        calibration_dir=resolve(lc["calibration_dir"]) if lc.get("calibration_dir") else fb.parent / "calibration",
        wire_active=bool(lc.get("wire_active")),
        new_pattern=(lambda: load_new_pattern_pool(np_paths)) if np_paths else None,
        train_options={"max_feedback_share": None if lc.get("max_feedback_share") is None
                       else float(lc["max_feedback_share"]),
                       "ipw_clip": float(lc["ipw_clip"]), "lgb_params": lc.get("lgb_params") or None})
