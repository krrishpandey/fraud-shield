"""Hardening from live red-team attacks (docs/API.md, "Red team you can watch").

Each live attack (POST /decisions/{id}/redteam) keeps its minimal evading variants, with their own feature snapshots,
in an append-only store next to the label store. A variant becomes a "fraud" training label (source "redteam") only
once the ORIGINAL booking's truth is known to be fraud: an analyst confirmed it, or the booking carries injected ground
truth (Booking.meta scenario; flagged simulated). A legit original adds nothing (a softer action is then correct).
Retraining is the learning service's own retrain and pre-registered gate, unchanged (POST /redteam/retrain).
"""
from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from fraudshield.contracts import ACTIONS
from fraudshield.redteam.search import SearchResult, WhatIf, apply_changes, describe

MAX_PER_ATTACK = 3  # as the offline hardening run (scripts/redteam.py): at most 3 evasive variants per booking
SOURCE = "redteam"
TYPOLOGIES = ("T1", "T2", "T3", "T4", "T5", "T6", "T7")


def original_truth(rec: dict[str, Any], label_store) -> dict[str, Any] | None:
    """Truth of the attacked booking: the latest analyst label wins, else injected ground truth, else unknown."""
    did = rec["decision_id"]
    analyst = [r for r in label_store.all() if r.get("decision_id") == did and r.get("source") == "analyst"]
    if analyst:
        return {"label": analyst[-1]["label"], "source": "analyst", "simulated": False}
    meta = (rec.get("booking") or {}).get("meta") or {}
    sc = meta.get("typology", meta.get("scenario"))
    if sc in TYPOLOGIES:
        return {"label": "fraud", "source": f"injected ground truth ({sc})", "simulated": True}
    return None


def _minimal_evasions(res: SearchResult) -> list[tuple[int, Any]]:
    cur = ACTIONS.index(res.original.action)
    hits = [(n, t) for n, t in enumerate(res.tried, start=1) if ACTIONS.index(t.outcome.action) < cur]
    hits.sort(key=lambda x: (ACTIONS.index(x[1].outcome.action), len(x[1].changes), x[1].size))
    return hits[:MAX_PER_ATTACK]


class EvasionStore:
    """Append-only JSONL of evasion candidates; `promoted` lines record which ones became labels."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._rows: list[dict[str, Any]] = []
        if self.path.exists():
            self._rows = [json.loads(x) for x in self.path.read_text(encoding="utf-8").splitlines() if x.strip()]

    def _write(self, row: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(row, default=str, ensure_ascii=False) + "\n")
        self._rows.append(row)

    def harvest(self, rec: dict[str, Any], res: SearchResult, whatif: WhatIf, label_store) -> dict[str, Any]:
        """Keep this attack's minimal evasions; label them now if the original's truth is already known."""
        view = whatif.view(res.booking)
        did, at = rec["decision_id"], datetime.now().isoformat(timespec="seconds")
        meta = (rec.get("booking") or {}).get("meta") or {}
        with self._lock:
            self._write({"kind": "attack", "decision_id": did, "at": at, "queries": len(res.tried)})
            for n, t in _minimal_evasions(res):
                variant = apply_changes(res.booking, t.changes)
                fv = whatif.featurize(variant, view)
                self._write({"kind": "evasion", "id": f"{did}~rt{n}", "decision_id": did, "at": at, "try": n,
                             "action": t.outcome.action, "booking_id": f"{res.booking.booking_id}~rt{n}",
                             "account_id": variant.account_id, "booked_at": variant.booked_at,
                             "carrier_cost": variant.carrier_cost, "features": dict(fv.values),
                             "changes": [describe(c, res.booking)["text"] for c in t.changes],
                             "model_version": (rec.get("model_versions") or {}).get("gbm"),
                             "typology": meta.get("typology", meta.get("scenario")),
                             "campaign_id": meta.get("campaign_id")})
        added = self.promote(label_store)
        evasions = sum(1 for r in self._rows if r["kind"] == "evasion" and r["decision_id"] == did and r["at"] == at)
        truth = original_truth(rec, label_store)
        if truth is None:
            note = "Confirm the original booking as fraud to turn these evasions into training labels."
        elif truth["label"] == "legit":
            note = "The original booking is legit, so a softer action is correct: no labels added."
        else:
            note = f"Labelled as fraud from {truth['source']}."
        return {"evasions": evasions, "labelled": added, "needs_confirmation": truth is None and evasions > 0,
                "truth": truth, "simulated": bool(truth and truth["simulated"]), "note": note}

    def promote(self, label_store) -> int:
        """Turn every evasion whose original is now known fraud into a label, once. Returns how many were added."""
        with self._lock:
            done = {r["id"] for r in self._rows if r["kind"] == "promoted"}
            n = 0
            for r in [x for x in self._rows if x["kind"] == "evasion" and x["id"] not in done]:
                truth = self._truth(r, label_store)
                if not truth or truth["label"] != "fraud":
                    continue
                label_store.append({
                    "decision_id": r["id"], "booking_id": r["booking_id"], "account_id": r["account_id"],
                    "label": "fraud", "source": SOURCE, "simulated": truth["simulated"],
                    "labelled_at": datetime.now().isoformat(timespec="seconds"), "booked_at": r["booked_at"],
                    "features": r["features"], "action": r["action"], "propensity": None, "explored": False,
                    "model_version": r.get("model_version"), "raw_laya_misuse": None,
                    "carrier_cost": r["carrier_cost"], "state_text": None, "typology": r.get("typology"),
                    "campaign_id": r.get("campaign_id"),
                    "note": f"red-team evasion of {r['decision_id']} (try {r['try']}; truth: {truth['source']})"})
                self._write({"kind": "promoted", "id": r["id"], "at": datetime.now().isoformat(timespec="seconds")})
                n += 1
            return n

    @staticmethod
    def _truth(r: dict[str, Any], label_store) -> dict[str, Any] | None:
        """Truth of the evasion's original booking (analyst label, else the injected scenario it carried)."""
        return original_truth({"decision_id": r["decision_id"], "booking": {"meta": {"scenario": r.get("typology")}}},
                              label_store)

    def summary(self, label_store, since_seq: int = 0) -> dict[str, Any]:
        ev = [r for r in self._rows if r["kind"] == "evasion"]
        promoted = {r["id"] for r in self._rows if r["kind"] == "promoted"}
        rt = [r for r in label_store.all() if r.get("source") == SOURCE]
        pending = sorted({r["decision_id"] for r in ev if r["id"] not in promoted and self._truth(r, label_store) is None})
        return {"attacks": sum(1 for r in self._rows if r["kind"] == "attack"), "evasions": len(ev),
                "labelled": len(rt), "labelled_since_last_retrain": sum(1 for r in rt if r.get("seq", 0) >= since_seq),
                "simulated_labels": sum(1 for r in rt if r.get("simulated")), "pending_decisions": pending}
