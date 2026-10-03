import pytest

from fraudshield.contracts import ACTIONS
from fraudshield.data.labels import action_costs, risk_level


def test_cost_vector_matches_reference_matrix_at_45():
    legit = action_costs(45.0, fraud=False, typology="none", owner_ok=True)
    fraud = action_costs(45.0, fraud=True, typology="T1", owner_ok=True)
    assert list(legit) == list(ACTIONS)
    # reference values from docs/reference/phase3_rl.md (cost_matrix(45))
    assert legit == pytest.approx({"allow": 0.0, "allow_scan_gated": 1.07, "owner_confirm": 4.77,
                                   "review": 7.42, "hold": 10.12, "block": 33.5}, abs=0.01)
    assert fraud == pytest.approx({"allow": 75.0, "allow_scan_gated": 48.7, "owner_confirm": 6.35,
                                   "review": 13.5, "hold": 3.0, "block": 0.0}, abs=0.01)


def test_mode_adjustments():
    t6 = action_costs(45.0, fraud=True, typology="T6", owner_ok=True)
    t1 = action_costs(45.0, fraud=True, typology="T1", owner_ok=True)
    assert t6["allow_scan_gated"] < t1["allow_scan_gated"]  # scan gate catches weight manipulation
    t2 = action_costs(45.0, fraud=True, typology="T2", owner_ok=True)
    assert t2["owner_confirm"] > t1["owner_confirm"]  # the owner is the fraudster
    no_owner = action_costs(45.0, fraud=False, typology="none", owner_ok=False)
    assert no_owner["owner_confirm"] >= no_owner["review"]


def test_risk_level():
    assert risk_level(False, 10.0) == 0
    assert risk_level(True, 1.2) == 1
    assert risk_level(True, 2.5) == 2
    assert risk_level(True, 5.0) == 3
    assert risk_level(True, 9.0) == 4
    assert risk_level(True, 1.0, burst=True) == 4
