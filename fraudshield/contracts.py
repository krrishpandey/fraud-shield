"""Shared contracts between build agents. Changing anything here needs the orchestrator.

Data/ML agent implements: FeatureStore, featurize, serialize, GBM scoring, fit_calibration.
Decision-service agent implements: LayaClient, apply_calibration, decide, explain, AuditLog, API.
Both code against these types; tests use fakes where the other side is not built yet.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

ACTIONS: tuple[str, ...] = (
    "allow",
    "allow_scan_gated",
    "owner_confirm",
    "review",
    "hold",
    "block",
)
Action = Literal["allow", "allow_scan_gated", "owner_confirm", "review", "hold", "block"]

# Yes/no questions are two-option Choices with neutral keys "a" (yes) / "b" (no),
# the workaround for Laya issue #156 (Noul can follow its labels instead of the state).
SERVED_QUESTIONS: tuple[str, ...] = ("misuse", "foreign_senders", "payoff_max", "drop_consignee")
TRAINED_QUESTIONS: tuple[str, ...] = ("misuse", "foreign_senders", "payoff_max", "risk_level", "action")
ZERO_SHOT_QUESTIONS: tuple[str, ...] = ("drop_consignee",)


def yes_no(instructions: str, yes: str, no: str) -> dict[str, Any]:
    return {"type": "choice", "instructions": instructions, "criteria": {"a": yes, "b": no}}


QUESTIONS: dict[str, dict[str, Any]] = {
    "misuse": yes_no(
        "Is this booking being made by someone misusing this account, rather than the legitimate "
        "account holder shipping for its own business?",
        "yes: a third party or a fraudulent holder is using the account",
        "no: the holder is shipping for its own business",
    ),
    "foreign_senders": yes_no(
        "Is this account now paying for parcels from senders or origins outside its own customer "
        "base, beyond its normal growth?",
        "yes: the account is serving parties it never served before, a break from its pattern",
        "no: the parties fit the account's existing base or its normal growth",
    ),
    "payoff_max": yes_no(
        "Was this booking chosen to maximize the shipping cost charged to the account, relative to "
        "the account's norm (heavier, longer, faster)?",
        "yes: far above the account's usual cost per parcel",
        "no: cost is in line with the account's usual parcels",
    ),
    "drop_consignee": yes_no(
        "Does the consignee look like a reshipping drop: a recent address receiving parcels from "
        "many unrelated senders?",
        "yes: likely a drop address",
        "no: an ordinary consignee",
    ),
    "risk_level": {
        "type": "score",
        "instructions": "What is the fraud risk level of this booking, combining whether it is misuse "
        "and how much the carrier would lose?",
        "criteria": [
            "legitimate booking by the account holder",
            "misuse, low loss: cost near the account usual",
            "misuse, moderate loss: up to 3x the account median cost",
            "misuse, high loss: 3x to 8x the account median cost",
            "misuse, severe loss: over 8x median or part of a burst",
        ],
    },
    "action": {
        "type": "choice",
        "instructions": "Which action should the carrier take on this booking before it enters the network?",
        "criteria": {
            "allow": "accept normally",
            "allow_scan_gated": "accept, but hold at first scan if weight, size or drop-off point differ",
            "owner_confirm": "ask the account owner to confirm out of band before accepting",
            "review": "send to a fraud analyst before accepting",
            "hold": "do not issue a label until verified",
            "block": "refuse the booking and lock label creation",
        },
    },
}


@dataclass(frozen=True)
class Booking:
    """A booking request as received from the booking system. No labels in here."""

    booking_id: str
    account_id: str
    booked_at: str  # ISO 8601
    channel: Literal["web", "api", "counter"]
    login_device_age_days: float
    payment_method: Literal["account_billing", "card", "ach"]
    sender_id: str  # == account_id when the account ships for itself
    origin_uf: str
    origin_zip3: str
    dest_uf: str
    dest_zip3: str
    consignee_id: str
    weight_kg: float
    length_cm: float
    width_cm: float
    height_cm: float
    service: Literal["standard", "express"]
    category: str
    declared_value: float
    carrier_cost: float  # BRL, F in the cost matrix
    owner_contact_age_days: float | None = None  # None = no verified contact on file
    meta: dict[str, Any] = field(default_factory=dict)  # demo/eval only (typology etc). NEVER used as a feature.


@dataclass(frozen=True)
class FeatureVector:
    booking_id: str
    as_of: str
    values: dict[str, float | int | str]  # named features, all computed from history strictly before as_of


class FeatureStore(Protocol):
    def append(self, booking: Booking) -> None: ...
    def history(self, account_id: str, before: str) -> Any: ...  # pandas DataFrame


@dataclass(frozen=True)
class Answer:
    qid: str
    type: str
    probabilities: dict[str, float]  # raw (uncalibrated) from Laya


@dataclass(frozen=True)
class Decision:
    action: Action
    propensity: float  # P(action | context) under the logging policy, for OPE
    greedy_action: Action
    expected_costs: dict[str, float]
    explored: bool
    degraded: bool
    reasons: list[str]  # fixed reason codes, not LLM text


@dataclass(frozen=True)
class Explanation:
    text: str
    source: Literal["llm", "template"]
    valid: bool
    model_id: str | None = None
    prompt_hash: str | None = None


# calibration.json schema (written by fit_calibration, read by apply_calibration):
# {
#   "version": "cal-YYYYMMDD-<hash8>",
#   "model_revision": "<laya checkpoint id or local path hash>",
#   "temperatures": {"<qid>": float},          # applied as softmax(log p / T)
#   "platt": {"misuse": {"a": float, "b": float}},  # p_cal = sigmoid(a * logit(p_T) + b)
#   "conformal": {"lambda_allow": float, "alpha": float, "n_fraud": int},
#   "prevalence": float
# }
