from pathlib import Path

import pytest

from fraudshield.contracts import ACTIONS
from fraudshield.policy.costs import CostConfig, cost_matrix, expected_costs, load_costs, mode_weights

ROOT = Path(__file__).resolve().parents[2]


def test_load_costs_yaml_marks_every_number_as_assumption():
    cfg = load_costs(ROOT / "config" / "costs.yaml")
    assert cfg.version.startswith("costs-")
    assert cfg.params["margin"] == pytest.approx(0.30)
    # every param has a source note, and all are assumptions (only F comes from data)
    assert set(cfg.notes) == set(cfg.params)
    assert all("ASSUMPTION" in n for n in cfg.notes.values())


def test_cost_matrix_matches_phase3_table_at_F45():
    C = cost_matrix(45.0, CostConfig.default().params)
    expect = {
        "allow": (0.00, 75.00),
        "allow_scan_gated": (1.07, 48.70),
        "owner_confirm": (4.77, 6.35),
        "review": (7.42, 13.50),
        "hold": (10.12, 3.00),
        "block": (33.50, 0.00),
    }
    assert list(C) == list(ACTIONS)
    for a, (leg, fr) in expect.items():
        assert C[a][0] == pytest.approx(leg, abs=0.01)
        assert C[a][1] == pytest.approx(fr, abs=0.01)


def test_loss_scales_with_carrier_cost():
    p = CostConfig.default().params
    assert cost_matrix(120.0, p)["allow"][1] > cost_matrix(45.0, p)["allow"][1]
    assert cost_matrix(120.0, p)["allow"][1] == pytest.approx(150.0)


def test_payoff_mode_makes_scan_gate_more_effective():
    p = CostConfig.default().params
    payoff = cost_matrix(45.0, p, mode="payoff")["allow_scan_gated"][1]
    drop = cost_matrix(45.0, p, mode="drop")["allow_scan_gated"][1]
    assert payoff < drop


def test_mode_weights_normalise_and_fallback():
    w = mode_weights({"foreign_senders": 0.8, "payoff_max": 0.2, "drop_consignee": 0.0})
    assert sum(w.values()) == pytest.approx(1.0)
    assert w["takeover"] == pytest.approx(0.8)
    w0 = mode_weights({})
    assert w0 == {"other": 1.0}


def test_expected_cost_is_linear_in_p_and_restricted():
    p = CostConfig.default().params
    e0 = expected_costs(0.0, 45.0, p)
    e1 = expected_costs(1.0, 45.0, p)
    eh = expected_costs(0.5, 45.0, p)
    for a in ACTIONS:
        assert eh[a] == pytest.approx((e0[a] + e1[a]) / 2)
    sub = expected_costs(0.5, 45.0, p, allowed=["hold", "block"])
    assert set(sub) == {"hold", "block"}
