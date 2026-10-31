"""Laya's four fraud questions, answered for one booking and written as plain English for the user.

LightGBM decides (Laya v2 lost its pre-registered gate, artifacts/results_laya_v2.md). Laya's job here is to say in
words WHAT looks wrong. Where the answers come from, in this order:
  1. live: the fine-tuned Laya over HTTP (laya.http_url + LAYA_API_KEY in .env)
  2. cached: real Laya answers recorded earlier (artifacts/laya_cache.json; few entries)
  3. demo: Laya is not reachable on this computer (no weights, no GPU, no key), so a DEMO TEXT answers each question
     from the booking's own measured facts. Always labelled demo; never presented as Laya's output.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fraudshield.models.calibration import p_yes
from fraudshield.models.laya_client import LayaClient, LayaUnavailable

ROOT = Path(__file__).resolve().parents[2]

# question -> (plain question, sentence if yes, sentence if no)
PLAIN: dict[str, tuple[str, str, str]] = {
    "misuse": ("Is someone other than the owner using this account?",
               "Someone other than the owner may be using this account.",
               "It looks like the owner's own booking."),
    "foreign_senders": ("Is the account paying for senders it never used before?",
                        "The account is paying for senders or pickup places it has never used before.",
                        "The sender and pickup place are normal for this account."),
    "payoff_max": ("Is this parcel much more expensive than the account's usual ones?",
                   "This parcel is much heavier or more expensive than the account's usual parcels.",
                   "The weight and price are normal for this account."),
    "drop_consignee": ("Does the receiver look like a reshipping drop?",
                       "The receiver gets parcels from many unrelated accounts, like a reshipping drop.",
                       "The receiver looks ordinary."),
}
QUESTION_IDS = tuple(PLAIN)


def _f(values: dict[str, Any], k: str, default: float = 0.0) -> float:
    try:
        v = values.get(k)
        return default if v is None or v != v else float(v)
    except (TypeError, ValueError):
        return default


def demo_answers(rec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """DEMO TEXT: each question answered from the booking's measured facts (fixed thresholds, shown with the fact)."""
    v = rec.get("feature_values") or {}
    p = float((rec.get("probabilities") or {}).get("misuse", rec.get("gbm_score") or 0.0))
    senders = max(_f(v, "new_senders_l10"), _f(v, "new_origins_l10"))
    facts = {
        "misuse": (p >= 0.5, f"LightGBM risk {p:.2f}"),
        "foreign_senders": (_f(v, "sender_differs") >= 1 or senders >= 3 or _f(v, "origin_seen", 1) == 0,
                            f"{int(senders)} new senders or origins in the last 10 bookings; sender "
                            f"{'differs from' if _f(v, 'sender_differs') >= 1 else 'is'} the account"),
        "payoff_max": (_f(v, "cost_vs_median", 1) >= 3 or _f(v, "weight_z") >= 3,
                       f"cost {_f(v, 'cost_vs_median', 1):.1f}x the usual, weight {_f(v, 'weight_z'):.1f} "
                       f"standard units from the usual"),
        "drop_consignee": (_f(v, "drop_pattern") >= 1 or _f(v, "consignee_other_accts_30d") >= 3,
                           f"receiver shared with {int(_f(v, 'consignee_other_accts_30d'))} other accounts in 30 days"),
    }
    return {q: {"yes": yes, "probability": None, "fact": fact} for q, (yes, fact) in facts.items()}


def _client(cfg: dict[str, Any]) -> tuple[LayaClient | None, str]:
    lc = cfg.get("laya") or {}
    key = os.environ.get(lc.get("api_key_env") or "LAYA_API_KEY")
    if key and lc.get("http_url"):
        return LayaClient.http(lc["http_url"], key, timeout_s=float(lc.get("timeout_s", 0.18)) * 10), "live"
    cache = lc.get("cache_path") or "artifacts/laya_cache.json"
    path = Path(cache) if Path(cache).is_absolute() else ROOT / cache
    return (LayaClient.cached(path), "cached") if path.exists() else (None, "")


def why_not_live(cfg: dict[str, Any]) -> str:
    lc = cfg.get("laya") or {}
    bits = []
    if not os.environ.get(lc.get("api_key_env") or "LAYA_API_KEY"):
        bits.append("no LAYA_API_KEY in .env")
    mp = lc.get("model_path") or "artifacts/laya/fraudshield-laya"
    if not (ROOT / mp).is_dir() and not Path(mp).is_dir():
        bits.append("the fine-tuned weights are not on this computer (trained on Kaggle)")
    try:
        import torch  # noqa: F401, PLC0415
    except ImportError:
        bits.append("the Laya runtime (PyTorch) is not installed")
    return "; ".join(bits) or "Laya did not answer in time"


def explain(rec: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    """Plain-English 'what is wrong' for one decision record."""
    answers: dict[str, dict[str, Any]] | None = None
    source, note = "demo", ""
    client, kind = _client(cfg)
    if client is not None and rec.get("state_text"):
        try:
            res = client.predict(rec["state_text"], QUESTION_IDS)
            answers = {q: {"yes": p_yes(res.raw[q]) >= 0.5, "probability": round(p_yes(res.raw[q]), 3), "fact": None}
                       for q in QUESTION_IDS if q in res.raw}
            source = kind
        except (LayaUnavailable, KeyError, Exception):  # any failure: fall back to the labelled demo text
            answers = None
    if not answers:
        answers = demo_answers(rec)
        note = f"Demo text: Laya is not connected ({why_not_live(cfg)}). Each answer is read from the booking's own facts."
    items = [{"id": q, "question": PLAIN[q][0], "yes": a["yes"], "probability": a.get("probability"),
              "fact": a.get("fact"), "text": PLAIN[q][1] if a["yes"] else PLAIN[q][2]} for q, a in answers.items()]
    wrong = [i["text"] for i in items if i["yes"]]
    summary = " ".join(wrong) if wrong else "Laya found nothing unusual about this booking."
    return {"source": source, "note": note, "summary": summary, "items": items,
            "decides": "LightGBM decides; Laya explains (Laya v2 lost to LightGBM on held-out fraud, PR-AUC 0.356 vs 0.777)"}
