"""Apply calibration.json (schema in fraudshield/contracts.py) to raw Laya probabilities.

Per question: temperature, softmax(log p / T). For misuse: then Platt, sigmoid(a*logit(p_T) + b) on
the yes key "a". No calibration file -> identity, flagged calibrated=False.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

EPS = 1e-6
YES = "a"


def load_calibration(path: str | Path | None) -> dict[str, Any] | None:
    if path is None or not Path(path).exists():
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _temper(probs: dict[str, float], T: float) -> dict[str, float]:
    logs = {k: math.log(min(max(v, EPS), 1.0)) / T for k, v in probs.items()}
    m = max(logs.values())
    ex = {k: math.exp(v - m) for k, v in logs.items()}
    s = sum(ex.values())
    return {k: v / s for k, v in ex.items()}


def _platt(probs: dict[str, float], a: float, b: float) -> dict[str, float]:
    p = min(max(probs[YES], EPS), 1 - EPS)
    z = a * math.log(p / (1 - p)) + b
    py = 1.0 / (1.0 + math.exp(-z))
    others = [k for k in probs if k != YES]
    out = {YES: py}
    rest = sum(probs[k] for k in others)
    for k in others:
        out[k] = (1 - py) * (probs[k] / rest if rest > 0 else 1.0 / len(others))
    return out


def apply_calibration(raw: dict[str, dict[str, float]], cal: dict[str, Any] | None):
    """Returns (calibrated probabilities per qid, info {calibrated, version})."""
    if not cal:
        return {q: dict(p) for q, p in raw.items()}, {"calibrated": False, "version": "uncalibrated"}
    temps = cal.get("temperatures", {}) or {}
    platt = cal.get("platt", {}) or {}
    out = {}
    for q, probs in raw.items():
        p = dict(probs)
        if q in temps:
            p = _temper(p, float(temps[q]))
        if q in platt and YES in p:
            p = _platt(p, float(platt[q]["a"]), float(platt[q]["b"]))
        out[q] = p
    return out, {"calibrated": True, "version": cal.get("version", "unknown")}


def p_yes(probs: dict[str, float]) -> float:
    return float(probs.get(YES, 0.0))
