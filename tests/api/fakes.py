"""Fakes for decision-service tests (the real featurizer/GBM/Laya are wired by the orchestrator)."""
from types import SimpleNamespace

from fraudshield.contracts import SERVED_QUESTIONS, FeatureVector
from fraudshield.models.laya_client import LayaClient


class FakeAgent:
    def __init__(self, p=None, delay=0.0):
        self.p = p or {"misuse": 0.97, "foreign_senders": 0.99, "payoff_max": 0.93, "drop_consignee": 0.12}
        self.calls = 0

    def predict(self, state, questions):
        self.calls += 1
        return {"answers": {q: {"probabilities": {"a": self.p.get(q, 0.5), "b": 1 - self.p.get(q, 0.5)}}
                            for q in questions}}


def fake_featurize(booking):
    return FeatureVector(booking.booking_id, booking.booked_at,
                         {"cost_vs_median": 11.6, "new_senders_l10": 8, "tenure_days": 412.0})


def fake_serialize(booking, fv):
    return f"BOOKING {booking.booked_at} | channel {booking.channel} | cost x{fv.values['cost_vs_median']}"


def fake_gbm(fv):
    return 0.71


def laya(p=None):
    return LayaClient.local(agent=FakeAgent(p), revision="fake-laya")


class DownLaya:
    mode = "local"
    mode_reason = "test"

    def predict(self, state, *a, **k):
        from fraudshield.models.laya_client import LayaUnavailable
        raise LayaUnavailable("down")

    ask = predict

    def health(self):
        return {"mode": "local", "reason": "test"}


def llm_client(text):
    return SimpleNamespace(messages=SimpleNamespace(
        create=lambda **kw: SimpleNamespace(content=[SimpleNamespace(type="text", text=text)], stop_reason="end_turn")))
