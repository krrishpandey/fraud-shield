"""Append-only store of feedback labels (artifacts/feedback/labels.jsonl).

Each label carries the FeatureVector values snapshot taken AT DECISION TIME (copied from the stored
decision record), never recomputed later, so training on feedback cannot leak future information.
`typology` comes from Booking.meta and is kept ONLY for evaluation reporting, never as a feature.
Relabelling appends a new line; the latest line per decision wins when building training sets.
"""
from __future__ import annotations

import copy
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

LABELS = ("fraud", "legit")
SOURCES = ("analyst", "simulated_analyst", "outcome", "simulated_outcome")
SIMULATED_SOURCES = ("simulated_analyst", "simulated_outcome")


def label_from_decision(rec: dict[str, Any], label: str, source: str, labelled_at: str | None = None,
                        note: str = "") -> dict[str, Any]:
    if label not in LABELS:
        raise ValueError(f"label must be one of {LABELS}")
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}")
    booking = rec.get("booking") or {}
    meta = booking.get("meta") or {}
    fv = rec.get("feature_values")
    raw = (rec.get("raw_probabilities") or {}).get("misuse")
    return {
        "decision_id": rec["decision_id"], "booking_id": rec["booking_id"],
        "account_id": rec.get("account_id", booking.get("account_id")),
        "label": label, "source": source, "simulated": source in SIMULATED_SOURCES,
        "labelled_at": labelled_at or datetime.now().isoformat(timespec="seconds"),
        "booked_at": rec.get("booked_at", booking.get("booked_at")),
        "features": copy.deepcopy(fv) if fv is not None else None,
        "action": rec.get("action"), "propensity": rec.get("propensity"), "explored": bool(rec.get("explored")),
        "model_version": (rec.get("model_versions") or {}).get("gbm"),
        "raw_laya_misuse": raw, "carrier_cost": booking.get("carrier_cost", rec.get("carrier_cost")),
        "state_text": rec.get("state_text"),
        "typology": meta.get("typology", meta.get("scenario")),  # evaluation reporting only
        "campaign_id": meta.get("campaign_id"),  # bootstrap clusters only
        "note": note,
    }


class LabelStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._rows: list[dict[str, Any]] = []
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self._rows.append(json.loads(line))

    def append(self, lab: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            lab = {**lab, "seq": len(self._rows)}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(lab, default=str, ensure_ascii=False) + "\n")
            self._rows.append(lab)
            return lab

    def all(self) -> list[dict[str, Any]]:
        return list(self._rows)

    def __len__(self) -> int:
        return len(self._rows)

    def latest_by_decision(self) -> list[dict[str, Any]]:
        """Latest label per decision, in first-labelled order."""
        out: dict[str, dict] = {}
        for r in self._rows:
            out[r["decision_id"]] = r
        return list(out.values())

    def counts_by_source(self) -> dict[str, int]:
        c: dict[str, int] = {}
        for r in self._rows:
            c[r["source"]] = c.get(r["source"], 0) + 1
        return c
