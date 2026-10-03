"""Append-only JSONL audit log with a sha256 hash chain.

Each line is one record: {seq, ts, event_type, payload, prev_hash, record_hash}.
record_hash = sha256(canonical JSON of the record without record_hash).
Single writer: one lock per AuditLog instance; one instance per file in the app.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EVENT_TYPES = ("decision", "explanation", "analyst_feedback", "outcome", "ask", "retrain", "rollback",
               "config_change")
GENESIS = "0" * 64


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def canonical_hash(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


class AuditLog:
    def __init__(self, path: str | Path, fsync: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fsync = fsync
        self._lock = threading.Lock()
        self._seq, self._head = self._scan_tail()

    def _scan_tail(self) -> tuple[int, str]:
        if not self.path.exists():
            return 0, GENESIS
        n, head = 0, GENESIS
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    n += 1
                    try:
                        head = json.loads(line)["record_hash"]
                    except (ValueError, KeyError):
                        head = "corrupt"
        return n, head

    def append(self, event_type: str, payload: dict[str, Any]) -> tuple[int, str]:
        if event_type not in EVENT_TYPES:
            raise ValueError(f"unknown audit event type: {event_type}")
        with self._lock:
            rec = {
                "seq": self._seq,
                "ts": datetime.now(timezone.utc).isoformat(),
                "event_type": event_type,
                "payload": payload,
                "prev_hash": self._head,
            }
            # round-trip through JSON so the hash covers exactly what is written
            rec = json.loads(canonical_json(rec))
            rec["record_hash"] = canonical_hash(rec)
            with self.path.open("a", encoding="utf-8", newline="\n") as f:
                f.write(canonical_json(rec) + "\n")
                f.flush()
                if self.fsync:
                    os.fsync(f.fileno())
            seq = self._seq
            self._seq += 1
            self._head = rec["record_hash"]
            return seq, rec["record_hash"]

    @property
    def head_hash(self) -> str:
        return self._head

    def verify(self) -> dict[str, Any]:
        return verify_file(self.path)


def verify_file(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    prev, n = GENESIS, 0
    if not path.exists():
        return {"ok": True, "records": 0, "head_hash": None, "first_bad_index": None}
    with path.open("rb") as f:
        for i, raw in enumerate(x for x in f.read().split(b"\n") if x.strip()):
            try:
                rec = json.loads(raw.decode("utf-8"))
                body = {k: v for k, v in rec.items() if k != "record_hash"}
                good = rec["prev_hash"] == prev and canonical_hash(body) == rec["record_hash"]
            except (ValueError, KeyError, UnicodeDecodeError, AttributeError):
                good = False
            if not good:
                return {"ok": False, "records": i, "head_hash": prev, "first_bad_index": i}
            prev, n = rec["record_hash"], n + 1
    return {"ok": True, "records": n, "head_hash": prev if n else None, "first_bad_index": None}
