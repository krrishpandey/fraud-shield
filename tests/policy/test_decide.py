import random

import pytest

from fraudshield.contracts import ACTIONS, Decision
from fraudshield.policy.costs import CostConfig
from fraudshield.policy.decide import PolicyConfig, PolicyContext, decide, decide_with_trace

COSTS = CostConfig.default()


def ctx(**kw):
    base = dict(carrier_cost=45.0, account_tenure_days=400.0, owner_contact_age_days=400.0)
    base.update(kw)
    return PolicyContext(**base)


def run(p, c=None, cfg=None, seed=0, **probs):
    return decide({"misuse": p, **probs}, c or ctx(), COSTS, cfg or PolicyConfig(), random.Random(seed))


def test_returns_contract_decision_with_all_expected_costs():
    d = run(0.0)
    assert isinstance(d, Decision)
    assert set(d.expected_costs) == set(ACTIONS)
    assert d.action == "allow" and d.greedy_action == "allow"
    assert d.propensity == 1.0 and d.explored is False and d.degraded is False


def test_certain_fraud_with_hard_signal_blocks():
    assert run(1.0, ctx(hard_signal=True)).action == "block"


def test_block_without_hard_signal_becomes_hold():
    assert run(1.0, ctx(hard_signal=False)).action == "hold"


def test_block_over_account_cap_becomes_hold():
    assert run(1.0, ctx(hard_signal=True, blocks_last_24h=1)).action == "hold"


@pytest.mark.parametrize("tenure,contact", [(10.0, 400.0), (400.0, None), (400.0, 20.0)])
def test_owner_confirm_never_offered_without_trusted_owner(tenure, contact):
    d = run(0.3, ctx(account_tenure_days=tenure, owner_contact_age_days=contact))
    assert d.action != "owner_confirm"
    assert "owner_confirm" not in _allowed(tenure, contact)


def _allowed(tenure, contact):
    _, tr = decide_with_trace({"misuse": 0.3}, ctx(account_tenure_days=tenure, owner_contact_age_days=contact),
                              COSTS, PolicyConfig(), random.Random(0))
    return tr["allowed"]


def test_owner_confirm_chosen_mid_band_with_trusted_owner():
    assert run(0.3).action == "owner_confirm"


def test_rule_floor_is_a_minimum_action():
    d = run(0.0, ctx(rule_floor="review"))
    assert ACTIONS.index(d.action) >= ACTIONS.index("review")


def test_conformal_guard_removes_plain_allow():
    assert run(0.02).action == "allow"
    d = run(0.02, cfg=PolicyConfig(lambda_allow=0.01))
    assert d.action == "allow_scan_gated"
    _, tr = decide_with_trace({"misuse": 0.02}, ctx(), COSTS, PolicyConfig(lambda_allow=0.01), random.Random(0))
    assert tr["allow_guard_passed"] is False


def test_higher_payoff_lowers_the_allow_band():
    assert run(0.03, ctx(carrier_cost=15.0)).action == "allow"
    assert run(0.03, ctx(carrier_cost=120.0)).action != "allow"


def test_mode_probabilities_change_expected_costs():
    a = run(0.5, payoff_max=0.9, foreign_senders=0.05).expected_costs["allow_scan_gated"]
    b = run(0.5, drop_consignee=0.9, foreign_senders=0.05).expected_costs["allow_scan_gated"]
    assert a < b


def test_degraded_never_plain_allow_for_high_risk_and_no_exploration():
    cfg = PolicyConfig(explore_eps=1.0, degraded_high_risk=0.3)
    d = run(0.35, ctx(degraded=True), cfg=cfg)
    assert d.degraded is True and d.explored is False
    assert ACTIONS.index(d.action) >= ACTIONS.index("review")


def test_degraded_uses_stricter_allow_guard():
    cfg = PolicyConfig(lambda_allow=0.05, degraded_lambda_allow=0.01)
    assert run(0.02, cfg=cfg).action == "allow"
    assert run(0.02, ctx(degraded=True), cfg=cfg).action != "allow"


def test_deterministic_when_exploration_disabled():
    ds = [run(0.09, seed=s) for s in range(20)]
    assert len({d.action for d in ds}) == 1
    assert all(d.propensity == 1.0 and not d.explored for d in ds)


def test_exploration_only_to_scan_gated_with_logged_propensity():
    cfg = PolicyConfig(explore_eps=0.05)
    rng = random.Random(7)
    ds = [decide({"misuse": 0.09}, ctx(), COSTS, cfg, rng) for _ in range(4000)]
    explored = [d for d in ds if d.explored]
    assert 0.03 < len(explored) / len(ds) < 0.07
    assert all(d.action == "allow_scan_gated" and d.propensity == pytest.approx(0.05) for d in explored)
    assert all(d.greedy_action == "owner_confirm" for d in ds)
    assert all(d.propensity == pytest.approx(0.95) for d in ds if not d.explored)
    assert not any(d.action == "allow" for d in ds)


def test_no_exploration_far_from_boundary():
    cfg = PolicyConfig(explore_eps=0.05)
    rng = random.Random(1)
    ds = [decide({"misuse": 0.95}, ctx(hard_signal=True), COSTS, cfg, rng) for _ in range(500)]
    assert not any(d.explored for d in ds)
    assert all(d.propensity == 1.0 for d in ds)


def test_reasons_passed_through():
    d = decide({"misuse": 0.5}, ctx(), COSTS, PolicyConfig(), random.Random(0), reasons=["NEW_LOGIN_DEVICE"])
    assert d.reasons == ["NEW_LOGIN_DEVICE"]


def test_laya_allow_guard_does_not_apply_when_the_backup_model_decides():
    # The conformal guard in calibration.json is fitted on Laya's misuse probability. When Laya is not deciding
    # (degraded: the LightGBM score decides), a Laya guard of 0.0 must not rule out "allow" for every booking.
    cfg = PolicyConfig(lambda_allow=0.0, degraded_lambda_allow=0.02)
    assert run(0.001, ctx(degraded=True), cfg=cfg).action == "allow"
    assert run(0.05, ctx(degraded=True), cfg=cfg).action != "allow"  # the stricter backup guard still applies
    assert run(0.001, cfg=cfg).action != "allow"  # when Laya decides, its own guard applies
