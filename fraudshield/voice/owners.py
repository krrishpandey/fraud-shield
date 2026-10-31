"""Owner directory: who to call for an account, in which language, and since when that number is on file.

config/owners.yaml lists the demo owners. Phone numbers never live in the repo: each entry names an environment
variable (.env) that holds the number. Accounts not in the directory fall back to the booking's own
owner_contact_age_days and have no real number, so only a simulated call is possible for them.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

LANGUAGES = ("en",)  # English only
LANGUAGE_NAMES = {"en": "English"}


@dataclass(frozen=True)
class Owner:
    account_id: str
    company: str
    phone: str | None            # E.164; None = no real number on file (simulated call only)
    language: str                # en | hi | ta
    contact_age_days: float | None
    pickup_addresses: tuple[str, ...] = ()
    source: str = "directory"    # directory | booking (fallback)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def masked_phone(self) -> str | None:
        if not self.phone:
            return None
        return self.phone[:3] + " " + "x" * max(0, len(self.phone) - 7) + self.phone[-4:]


class OwnerDirectory:
    def __init__(self, entries: dict[str, dict[str, Any]] | None = None, env: dict[str, str] | None = None):
        self.entries = entries or {}
        self.env = env if env is not None else os.environ

    @classmethod
    def load(cls, path: str | Path | None, env: dict[str, str] | None = None) -> "OwnerDirectory":
        if path is None or not Path(path).exists():
            return cls({}, env)
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls(raw.get("owners") or {}, env)

    def get(self, account_id: str, booking: dict[str, Any] | None = None, now: datetime | None = None) -> Owner:
        e = self.entries.get(account_id)
        b = booking or {}
        if e is None:
            age = b.get("owner_contact_age_days")
            return Owner(account_id, company=str((b.get("meta") or {}).get("company") or f"Account {account_id[:8]}"),
                         phone=None, language="en", contact_age_days=None if age is None else float(age),
                         source="booking")
        phone = (self.env.get(e["phone_env"]) or "").strip() or None if e.get("phone_env") else e.get("phone")
        lang = (self.env.get(e.get("language_env") or "") or e.get("language") or "en").strip().lower()
        if lang not in LANGUAGES:
            lang = "en"
        age = e.get("phone_changed_days_ago")
        if e.get("phone_changed_at"):
            when = datetime.fromisoformat(str(e["phone_changed_at"]))
            age = ((now or datetime.now()) - when).total_seconds() / 86400.0
        return Owner(account_id, company=str(e.get("company") or account_id[:8]), phone=phone, language=lang,
                     contact_age_days=None if age is None else float(age),
                     pickup_addresses=tuple(e.get("pickup_addresses") or ()), source="directory",
                     meta={k: v for k, v in e.items() if k not in ("phone", "phone_env")})
