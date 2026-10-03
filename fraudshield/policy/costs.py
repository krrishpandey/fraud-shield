"""Cost matrix in BRL. C[a] = (cost if legit, cost if fraud). F = booking carrier_cost.

All parameters are assumptions (see config/costs.yaml). Fraud cost can depend on the fraud
mode m (takeover / payoff / drop / other) because the scan gate catches some modes better.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from fraudshield.contracts import ACTIONS

DEFAULT_PARAMS: dict[str, float] = dict(
    c_dispute=30.0, margin=0.30, goodwill=15.0,
    c_scan=0.50, q_scan=0.40, q_scan_takeover=0.40, q_scan_payoff=0.60, q_scan_drop=0.30,
    q_scan_other=0.40, c_intercept=8.0, ab_scan=0.02,
    c_msg=0.20, fatigue=2.0, p_unreach=0.30, ab_wait=0.30, p_spoof=0.10,
    analyst_per_min=1.00, review_min=6.0, ab_review=0.05, analyst_miss=0.10,
    c_hold=3.0, ab_hold=0.25, c_block_support=5.0,
)

# served question -> fraud mode it signals
MODE_OF_QUESTION = {"foreign_senders": "takeover", "payoff_max": "payoff", "drop_consignee": "drop"}


@dataclass(frozen=True)
class CostConfig:
    version: str
    params: dict[str, float]
    notes: dict[str, str] = field(default_factory=dict)

    @classmethod
    def default(cls) -> "CostConfig":
        return cls("costs-default", dict(DEFAULT_PARAMS), {k: "ASSUMPTION: built-in default" for k in DEFAULT_PARAMS})


def load_costs(path: str | Path | None) -> CostConfig:
    if path is None or not Path(path).exists():
        return CostConfig.default()
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    params = dict(DEFAULT_PARAMS)
    notes = {}
    for k, v in raw["params"].items():
        params[k] = float(v["value"])
        notes[k] = str(v.get("note", ""))
    return CostConfig(raw["version"], params, notes)


def cost_matrix(F: float, p: dict[str, float], mode: str | None = None) -> dict[str, tuple[float, float]]:
    lost = p["margin"] * F + p["goodwill"]
    leak = F + p["c_dispute"]
    q_scan = p.get(f"q_scan_{mode}", p["q_scan"]) if mode else p["q_scan"]
    rev = p["analyst_per_min"] * p["review_min"]
    return {
        "allow": (0.0, leak),
        "allow_scan_gated": (p["c_scan"] + p["ab_scan"] * lost,
                             p["c_scan"] + q_scan * p["c_intercept"] + (1 - q_scan) * leak),
        "owner_confirm": (p["c_msg"] + p["fatigue"] + p["p_unreach"] * p["ab_wait"] * lost,
                          p["c_msg"] + p["p_unreach"] * p["c_hold"] + (1 - p["p_unreach"]) * p["p_spoof"] * leak),
        "review": (rev + p["ab_review"] * lost, rev + p["analyst_miss"] * leak),
        "hold": (p["c_hold"] + p["ab_hold"] * lost, p["c_hold"]),
        "block": (lost + p["c_block_support"], 0.0),
    }


def mode_weights(probs: dict[str, float]) -> dict[str, float]:
    """P(mode | fraud) from the served mode questions, normalised. Empty -> unexplained."""
    w = {MODE_OF_QUESTION[q]: float(v) for q, v in probs.items() if q in MODE_OF_QUESTION and v > 0}
    s = sum(w.values())
    if s <= 1e-9:
        return {"other": 1.0}
    return {m: v / s for m, v in w.items()}


def expected_costs(p_fraud: float, F: float, params: dict[str, float], allowed=None,
                   modes: dict[str, float] | None = None) -> dict[str, float]:
    """(1-p) c_legit(a) + p * sum_m P(m|fraud) c_fraud(a, m)."""
    modes = modes or {None: 1.0}
    mats = {m: cost_matrix(F, params, m) for m in modes}
    out = {}
    for a in ACTIONS:
        if allowed is not None and a not in allowed:
            continue
        legit = mats[next(iter(mats))][a][0]
        fraud = sum(w * mats[m][a][1] for m, w in modes.items())
        out[a] = (1 - p_fraud) * legit + p_fraud * fraud
    return out
