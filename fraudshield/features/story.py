"""Account story: the account's own bookings as of one booking, for the decision page.

Answers "who is this account paying for?" from stored booking rows only (no labels, no meta).
Same definitions as the features (featurize.py, change block), so the page, the explanation and the
model input agree:
- "new" sender or receiver: the value appears for the first time in the account's history;
- last 10: the 10 bookings BEFORE this one (new_*_l10);
- usual: the rate per 10 bookings over the 90 days before that window (base_*_l10), or None if empty.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from fraudshield.contracts import Booking
from fraudshield.features.store import booking_to_row, to_ts

WINDOW = 10
BASE_DAYS = 90


def account_story(history: pd.DataFrame, booking: Booking, n: int = 40) -> dict[str, Any]:
    rows = history.to_dict("records") + [booking_to_row(booking)]
    seen_senders: set[str] = set()
    seen_receivers: set[str] = set()
    out: list[dict[str, Any]] = []
    for i, r in enumerate(rows):
        sender, receiver = str(r["sender_id"]), str(r["consignee_id"])
        out.append({
            "booking_id": str(r["booking_id"]),
            "booked_at": pd.Timestamp(r["booked_at"]).isoformat(),
            "own_goods": sender == str(r["account_id"]),
            "new_sender": sender not in seen_senders,
            "new_receiver": receiver not in seen_receivers,
            "origin": f"{r['origin_uf']} {r['origin_zip3']}",
            "dest": f"{r['dest_uf']} {r['dest_zip3']}",
            "carrier_cost": float(r["carrier_cost"]),
            "current": i == len(rows) - 1,
        })
        seen_senders.add(sender)
        seen_receivers.add(receiver)

    prior = out[:-1]
    window = prior[-WINDOW:]
    since = to_ts(booking.booked_at) - pd.Timedelta(days=BASE_DAYS)
    base = [b for b in prior[:len(prior) - len(window)] if pd.Timestamp(b["booked_at"]) >= since]
    usual = None
    if base:
        usual = {
            "new_senders_per10": round(WINDOW * sum(b["new_sender"] for b in base) / len(base), 2),
            "new_receivers_per10": round(WINDOW * sum(b["new_receiver"] for b in base) / len(base), 2),
            "based_on": len(base),
        }
    return {
        "account_id": str(booking.account_id),
        "n_prior": len(history),
        "bookings": out[-n:],
        "last10": {"size": len(window), "new_senders": sum(b["new_sender"] for b in window),
                   "new_receivers": sum(b["new_receiver"] for b in window)},
        "usual": usual,
    }
