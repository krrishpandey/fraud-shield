"""Feedback labels -> rows in the official Laya fine-tuning JSONL schema (DESIGN 7,
docs/reference/phase3_finetune.md section 11): id, workflow, state, questions, gold, where state,
questions and gold are JSON-encoded strings. Only the misuse gold is known from a label; the other
question golds are omitted, not invented. Rows are for the NEXT OFFLINE fine-tune: Laya weights are
never updated live in the app (6 GB GPU; a fine-tune takes minutes to hours).
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from fraudshield.contracts import QUESTIONS

WORKFLOW = "fraudshield_booking"
LIVE_UPDATE_NOTE = ("Laya weights are not updated in the app: fine-tuning a 421M-parameter model needs minutes to "
                    "hours of GPU time and does not fit next to serving on a 6 GB GPU. Feedback rows are exported "
                    "for the next offline fine-tune.")


def laya_row(lab: dict[str, Any]) -> dict[str, Any] | None:
    state = lab.get("state_text")
    if not state:
        return None
    fraud = lab["label"] == "fraud"
    gold = {"misuse": {"label": "a" if fraud else "b", "probabilities": {"a": 1.0 if fraud else 0.0,
                                                                         "b": 0.0 if fraud else 1.0}}}
    return {
        "id": f"fb_{lab['decision_id']}_{lab.get('seq', 0)}", "workflow": WORKFLOW, "split": "feedback",
        "booking_ts": lab.get("booked_at"), "account_id": lab.get("account_id"),
        "typology": lab.get("typology"), "label_source": lab["source"], "simulated": bool(lab.get("simulated")),
        "labelled_at": lab.get("labelled_at"),
        "state": json.dumps(state, ensure_ascii=False),
        "questions": json.dumps({"misuse": QUESTIONS["misuse"]}, ensure_ascii=False),
        "gold": json.dumps(gold),
    }


class LayaExport:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self.skipped = 0
        self._n = sum(1 for x in self.path.read_text(encoding="utf-8").splitlines() if x.strip()) \
            if self.path.exists() else 0

    def add(self, lab: dict[str, Any]) -> bool:
        row = laya_row(lab)
        with self._lock:
            if row is None:
                self.skipped += 1
                return False
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            self._n += 1
            return True

    def rows(self) -> int:
        return self._n
