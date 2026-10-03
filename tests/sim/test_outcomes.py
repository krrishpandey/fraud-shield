import random

import pytest

from fraudshield.contracts import ACTIONS
from fraudshield.policy.costs import CostConfig, cost_matrix
from fraudshield.sim.outcomes import simulate_outcome

P = CostConfig.default().params


@pytest.mark.parametrize("action", ACTIONS)
@pytest.mark.parametrize("is_fraud", [False, True])
def test_mean_realized_cost_matches_cost_matrix(action, is_fraud):
    rng = random.Random(0)
    n = 20000
    tot = sum(simulate_outcome(action, is_fraud, 45.0, rng, P)["realized_cost_brl"] for _ in range(n))
    expect = cost_matrix(45.0, P)[action][1 if is_fraud else 0]
    assert tot / n == pytest.approx(expect, abs=max(0.6, 0.03 * expect))


def test_outcomes_are_labelled_simulated_and_reveal_the_right_signal():
    rng = random.Random(1)
    o = simulate_outcome("owner_confirm", True, 45.0, rng, P)
    assert o["simulated"] is True
    assert o["owner_reply"] in {"yes_me", "not_me", "no_answer"}
    s = simulate_outcome("allow_scan_gated", False, 45.0, rng, P)
    assert s["scan_result"] in {"match", "mismatch"}
    b = simulate_outcome("block", True, 45.0, rng, P)
    assert b["label"] is None  # a block reveals nothing
    a = simulate_outcome("allow", True, 45.0, rng, P)
    assert a["label"] == "fraud" and a["label_delay_days"] > 0


def test_deterministic_with_seed():
    a = [simulate_outcome("review", True, 80.0, random.Random(5), P) for _ in range(3)]
    assert a[0] == a[1] == a[2]
