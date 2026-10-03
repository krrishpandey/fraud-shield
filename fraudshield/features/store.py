"""In-memory feature store (implements contracts.FeatureStore).

Holds booking rows (Booking fields only; meta and any label columns are dropped on the way in),
plus two simulated event streams: verified (pre-announced) account changes and confirmed-fraud
feedback. Features only ever see rows strictly before the booking time.
"""
from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from fraudshield.contracts import Booking

ROW_COLUMNS: tuple[str, ...] = tuple(f.name for f in fields(Booking) if f.name != "meta")
_NUM = ("login_device_age_days", "weight_kg", "length_cm", "width_cm", "height_cm", "declared_value",
        "carrier_cost", "owner_contact_age_days")


def to_ts(x) -> pd.Timestamp:
    t = pd.Timestamp(x)
    if t.tzinfo is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    return t


def booking_to_row(b: Booking) -> dict[str, Any]:
    row = {c: getattr(b, c) for c in ROW_COLUMNS}
    row["booked_at"] = to_ts(b.booked_at)
    row["owner_contact_age_days"] = np.nan if b.owner_contact_age_days is None else float(b.owner_contact_age_days)
    return row


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.loc[:, [c for c in ROW_COLUMNS if c in df.columns]].copy()
    missing = [c for c in ROW_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"frame lacks booking columns: {missing}")
    out["booked_at"] = pd.to_datetime(out["booked_at"])
    for c in _NUM:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype(float)
    for c in ROW_COLUMNS:
        if c not in _NUM and c != "booked_at":
            out[c] = out[c].astype(object)
    return out.reset_index(drop=True)


class FeatureStore:
    def __init__(self) -> None:
        self._df = _normalize(pd.DataFrame({c: [] for c in ROW_COLUMNS}))
        self._pending: list[dict[str, Any]] = []
        self._acc: dict[str, list[int]] = {}
        self._cons: dict[str, list[int]] = {}
        self._n = 0
        self.changes: list[dict[str, Any]] = []
        self.confirmed: list[dict[str, Any]] = []

    # construction ---------------------------------------------------------------------------
    @classmethod
    def from_frame(cls, df: pd.DataFrame, changes: pd.DataFrame | None = None,
                   confirmed: Iterable[dict] | None = None) -> "FeatureStore":
        s = cls()
        s._df = _normalize(df)
        s._n = len(s._df)
        s._acc = {k: list(v) for k, v in s._df.groupby("account_id", sort=False).indices.items()}
        s._cons = {k: list(v) for k, v in s._df.groupby("consignee_id", sort=False).indices.items()}
        if changes is not None and len(changes):
            for r in changes.to_dict("records"):
                if r.get("announced", True):
                    s.register_change(r["account_id"], r["kind"], r["at"], r.get("announced_at", r["at"]))
        for c in confirmed or []:
            s.confirm_fraud(c["account_id"], c["confirmed_at"], list(c.get("entities", [])))
        return s

    def append(self, booking: Booking) -> None:
        row = booking_to_row(booking)
        self._pending.append(row)
        self._acc.setdefault(row["account_id"], []).append(self._n)
        self._cons.setdefault(row["consignee_id"], []).append(self._n)
        self._n += 1

    def register_change(self, account_id: str, kind: str, at, announced_at=None) -> None:
        """A change the owner pre-announced through the verified change flow (simulated)."""
        self.changes.append({"account_id": account_id, "kind": kind, "at": to_ts(at),
                             "announced_at": to_ts(announced_at if announced_at is not None else at)})

    def confirm_fraud(self, account_id: str, confirmed_at, entities: list[str]) -> None:
        """Confirmed-fraud feedback (simulated delay offline; analyst label online)."""
        self.confirmed.append({"account_id": account_id, "confirmed_at": to_ts(confirmed_at),
                               "entities": list(entities)})

    # queries --------------------------------------------------------------------------------
    @property
    def frame(self) -> pd.DataFrame:
        if self._pending:
            add = _normalize(pd.DataFrame(self._pending))
            self._df = add if len(self._df) == 0 else pd.concat([self._df, add], ignore_index=True)
            self._pending = []
        return self._df

    def _rows(self, idx: list[int] | None, before) -> pd.DataFrame:
        df = self.frame
        if not idx:
            return df.iloc[0:0]
        h = df.iloc[idx]
        h = h[h.booked_at < to_ts(before)]
        return h.sort_values(["booked_at", "booking_id"], kind="mergesort")

    def history(self, account_id: str, before: str) -> pd.DataFrame:
        return self._rows(self._acc.get(account_id), before)

    def consignee_history(self, consignee_id: str, before: str) -> pd.DataFrame:
        return self._rows(self._cons.get(consignee_id), before)

    def __len__(self) -> int:
        return self._n

    @staticmethod
    def booking_from_row(r) -> Booking:
        d = {c: r[c] for c in ROW_COLUMNS}
        d["booked_at"] = to_ts(d["booked_at"]).isoformat()
        oc = d["owner_contact_age_days"]
        d["owner_contact_age_days"] = None if oc is None or (isinstance(oc, float) and np.isnan(oc)) else float(oc)
        for c in _NUM:
            if c != "owner_contact_age_days":
                d[c] = float(d[c])
        return Booking(**d)

    # persistence ----------------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        self.frame.to_parquet(path, index=False)
        side = {"changes": [{**c, "at": c["at"].isoformat(), "announced_at": c["announced_at"].isoformat()}
                            for c in self.changes],
                "confirmed": [{**c, "confirmed_at": c["confirmed_at"].isoformat()} for c in self.confirmed]}
        path.with_suffix(".events.json").write_text(json.dumps(side))

    @classmethod
    def load(cls, path: str | Path) -> "FeatureStore":
        path = Path(path)
        s = cls.from_frame(pd.read_parquet(path))
        ev = path.with_suffix(".events.json")
        if ev.exists():
            side = json.loads(ev.read_text())
            for c in side["changes"]:
                s.register_change(c["account_id"], c["kind"], c["at"], c["announced_at"])
            for c in side["confirmed"]:
                s.confirm_fraud(c["account_id"], c["confirmed_at"], c["entities"])
        return s
