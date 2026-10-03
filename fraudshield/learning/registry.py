"""GBM version registry (artifacts/models/registry.json).

Version names: gbm-<system>-<tier>-vN with a parent pointer. v1 is the GBM trained by the data
pipeline (artifacts/models/<system>_<tier>.lgb); it is listed virtually until the first change, so
reading status never writes files. Retrained versions are saved under versions_dir/<version>/.
Writes are atomic (temp file + os.replace).
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from fraudshield.models.gbm import GBMModel


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class ModelRegistry:
    def __init__(self, path: str | Path, models_dir: str | Path, system: str, tier: str,
                 versions_dir: str | Path | None = None):
        self.path = Path(path)
        self.models_dir = Path(models_dir)
        self.versions_dir = Path(versions_dir) if versions_dir else self.models_dir / "versions"
        self.system, self.tier = system, tier
        self._lock = threading.RLock()
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            self.data = {"system": system, "tier": tier, "active": None, "versions": [],
                         "last_retrain_label_count": 0, "calibration": None, "runs": []}
            if (self.models_dir / f"{system}_{tier}.lgb").exists():
                meta = self._base_meta()
                v1 = self.name(1)
                self.data["versions"].append({
                    "version": v1, "created_at": meta.get("created_at"), "parent": None,
                    "path": str(self.models_dir), "n_train": meta.get("n_train"), "n_feedback_labels": 0,
                    "metrics": None, "deployed": True, "ever_deployed": True, "gate": None,
                    "trained_typologies": None, "note": "trained by the data pipeline (scripts/train_gbm.py)"})
                self.data["active"] = v1

    def _base_meta(self) -> dict:
        p = self.models_dir / f"{self.system}_{self.tier}.json"
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def name(self, n: int) -> str:
        return f"gbm-{self.system}-{self.tier}-v{n}"

    @property
    def active_version(self) -> str | None:
        return self.data["active"]

    def versions(self) -> list[dict[str, Any]]:
        return [dict(v) for v in self.data["versions"]]

    def get(self, version: str) -> dict[str, Any]:
        for v in self.data["versions"]:
            if v["version"] == version:
                return v
        raise KeyError(version)

    def load_model(self, version: str) -> GBMModel:
        return GBMModel.load(self.get(version)["path"], self.system, self.tier)

    def update(self, version: str, **fields) -> None:
        with self._lock:
            self.get(version).update(fields)
            self.save()

    def add_version(self, model: GBMModel, parent: str | None, n_train: int, n_feedback_labels: int,
                    metrics: dict | None = None, **extra) -> str:
        with self._lock:
            version = self.name(len(self.data["versions"]) + 1)
            d = self.versions_dir / version
            model.metadata = {**model.metadata, "version": version, "parent": parent}
            model.save(d, extra={"registry_version": version, "parent": parent})
            self.data["versions"].append({
                "version": version, "created_at": _now(), "parent": parent, "path": str(d), "n_train": int(n_train),
                "n_feedback_labels": int(n_feedback_labels), "metrics": metrics, "deployed": False,
                "ever_deployed": False, **extra})
            self.save()
            return version

    def activate(self, version: str, require_ever_deployed: bool = False) -> None:
        with self._lock:
            target = self.get(version)
            if require_ever_deployed and not target.get("ever_deployed"):
                raise ValueError(f"{version} never passed the deployment gate; rollback only goes to a version "
                                 f"that was deployed before")
            for v in self.data["versions"]:
                v["deployed"] = v["version"] == version
            target["ever_deployed"] = True
            self.data["active"] = version
            self.save()

    def save(self) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(self.data, indent=2, default=str), encoding="utf-8")
            os.replace(tmp, self.path)
