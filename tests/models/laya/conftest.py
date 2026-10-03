import pytest


def _row(i=0, typology="none", misuse=0, fs=0, pm=0, dc=0, rl=0, hn=(), split="train", seed=0, cost=None):
    cost = cost or {"allow": 0.0, "allow_scan_gated": 1.0, "owner_confirm": 4.0, "review": 7.0, "hold": 8.0,
                    "block": 25.0}
    return {
        "id": f"bk_{i}", "workflow": "fraudshield_booking", "seed": seed, "split": split,
        "booking_ts": f"2018-01-{1 + i % 28:02d}T10:00:00", "account_id": f"acc{i % 7}", "typology": typology,
        "is_injected": typology != "none", "scenario_id": None, "campaign_id": None if typology == "none" else f"c{i % 5}",
        "camouflage_level": -1, "state": f"BOOKING 2018-01-01 Mon 10:00 | channel web | row {i}",
        "labels": {"misuse": misuse, "foreign_senders": fs, "payoff_max": pm, "drop_consignee": dc, "risk_level": rl},
        "action_cost": cost, "hard_negative": list(hn), "gbm_b2f": None,
    }


@pytest.fixture
def make_row():
    return _row


@pytest.fixture(scope="session")
def tok():
    pytest.importorskip("transformers")
    from fraudshield.models.laya_train.modeling import stock_model_dir, load_tokenizer
    try:
        d = stock_model_dir()
    except Exception as e:  # no network and no cache
        pytest.skip(f"stock laya snapshot not available: {e}")
    return load_tokenizer(d)
