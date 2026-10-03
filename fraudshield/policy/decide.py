"""Decision rule: argmin expected cost over the allowed action set, plus logged exploration.

Allowed set = ACTIONS minus: actions below the rules floor; owner_confirm when the owner is not
trusted (account < 30 d old or no verified contact older than 30 d); plain allow when the
conformal allow-guard fails (p > lambda_allow); review when the queue is full.
Block preconditions: hard signal and per-account cap not exceeded, otherwise hold.
Degraded mode (Laya down, p from calibrated GBM): stricter allow guard, high-risk floor, no exploration.
Exploration: near the allow_scan_gated boundary, with probability eps route to allow_scan_gated
(never plain allow); the propensity of the taken action is logged for OPE.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from fraudshield.contracts import ACTIONS, Decision
from fraudshield.policy.costs import CostConfig, expected_costs, mode_weights

SCAN = "allow_scan_gated"


@dataclass(frozen=True)
class PolicyContext:
    carrier_cost: float
    account_tenure_days: float | None = None
    owner_contact_age_days: float | None = None
    rule_floor: str = "allow"
    hard_signal: bool = False
    blocks_last_24h: int = 0
    degraded: bool = False
    review_available: bool = True


@dataclass(frozen=True)
class PolicyConfig:
    version: str = "policy-v1"
    lambda_allow: float | None = None      # conformal allow guard from calibration.json
    explore_eps: float = 0.0               # 0.05 in the logging policy; 0 = deterministic
    explore_delta_brl: float = 5.0         # ASSUMPTION: max extra expected cost of exploring
    explore_p_max: float = 0.6             # ASSUMPTION
    block_cap_24h: int = 1
    degraded_lambda_allow: float = 0.02    # ASSUMPTION: stricter guard when Laya is down
    degraded_high_risk: float = 0.30       # ASSUMPTION: p at or above -> at least review
    degraded_floor: str = "review"
    owner_min_tenure_days: float = 30.0
    owner_min_contact_days: float = 30.0


def owner_trusted(ctx: PolicyContext, cfg: PolicyConfig) -> bool:
    return (
        ctx.account_tenure_days is not None
        and ctx.account_tenure_days >= cfg.owner_min_tenure_days
        and ctx.owner_contact_age_days is not None
        and ctx.owner_contact_age_days > cfg.owner_min_contact_days
    )


def _floor_index(ctx: PolicyContext, cfg: PolicyConfig, p: float) -> int:
    floor = ACTIONS.index(ctx.rule_floor) if ctx.rule_floor in ACTIONS else 0
    if ctx.degraded and p >= cfg.degraded_high_risk:
        floor = max(floor, ACTIONS.index(cfg.degraded_floor))
    return floor


def decide_with_trace(probs: dict[str, float], ctx: PolicyContext, costs: CostConfig, cfg: PolicyConfig,
                      rng: random.Random, reasons: list[str] | None = None) -> tuple[Decision, dict[str, Any]]:
    p = float(probs["misuse"])
    modes = mode_weights({k: v for k, v in probs.items() if k != "misuse"})
    ec_all = expected_costs(p, ctx.carrier_cost, costs.params, modes=modes)

    floor = _floor_index(ctx, cfg, p)
    allowed = [a for a in ACTIONS if ACTIONS.index(a) >= floor]
    if not owner_trusted(ctx, cfg):
        allowed = [a for a in allowed if a != "owner_confirm"]
    if not ctx.review_available and len(allowed) > 1:
        allowed = [a for a in allowed if a != "review"]

    lam = cfg.lambda_allow
    if ctx.degraded:
        lam = cfg.degraded_lambda_allow if lam is None else min(lam, cfg.degraded_lambda_allow)
    guard_passed = lam is None or p <= lam
    if not guard_passed:
        allowed = [a for a in allowed if a != "allow"]

    greedy = min(allowed, key=lambda a: ec_all[a])
    block_downgraded = False
    if greedy == "block" and (not ctx.hard_signal or ctx.blocks_last_24h >= cfg.block_cap_24h):
        greedy, block_downgraded = "hold", True

    action, propensity, explored = greedy, 1.0, False
    in_band = (
        cfg.explore_eps > 0
        and not ctx.degraded
        and greedy != SCAN
        and SCAN in allowed
        and p <= cfg.explore_p_max
        and ec_all[SCAN] - ec_all[greedy] <= cfg.explore_delta_brl
    )
    if in_band:
        if rng.random() < cfg.explore_eps:
            action, propensity, explored = SCAN, cfg.explore_eps, True
        else:
            propensity = 1.0 - cfg.explore_eps

    d = Decision(
        action=action, propensity=propensity, greedy_action=greedy,
        expected_costs={a: round(v, 4) for a, v in ec_all.items()},
        explored=explored, degraded=ctx.degraded, reasons=list(reasons or []),
    )
    trace = {
        "p_fraud": p, "modes": modes, "allowed": allowed, "allow_guard_passed": guard_passed,
        "lambda_allow": lam, "block_downgraded": block_downgraded, "in_explore_band": in_band,
        "policy_version": cfg.version, "cost_version": costs.version,
    }
    return d, trace


def decide(probs, ctx, costs, cfg, rng, reasons=None) -> Decision:
    return decide_with_trace(probs, ctx, costs, cfg, rng, reasons)[0]
