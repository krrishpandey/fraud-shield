"""Laya v2 decides (docs/LAYA_V2.md): LightGBM's score is a witness inside Laya's state, Laya proposes the
action, and the cost rule audits it. LightGBM decides only when Laya is unavailable."""
from fraudshield.api.pipeline import Pipeline
from fraudshield.audit.log import AuditLog
from fraudshield.contracts import ACTIONS
from fraudshield.models.laya_client import LayaClient
from tests.api import fakes
from tests.policy.fixtures import make_booking


class ActionAgent(fakes.FakeAgent):
    """Answers the yes/no questions like FakeAgent and the action question with one preferred action."""

    def __init__(self, p=None, action="hold"):
        super().__init__(p)
        self.action = action
        self.asked: list[tuple] = []

    def predict(self, state, questions):
        self.asked.append(tuple(questions))
        out = super().predict(state, {q: v for q, v in questions.items() if q != "action"})
        if "action" in questions:
            out["answers"]["action"] = {"probabilities": {a: (0.75 if a == self.action else 0.05) for a in ACTIONS}}
        return out


def serialize_with_gbm(booking, fv, gbm_risk=None):
    return fakes.fake_serialize(booking, fv) + f"\nEVIDENCE LightGBM risk {gbm_risk:.2f}"


serialize_with_gbm.wants_gbm = True


def make(tmp_path, agent=None, laya=None, **kw):
    return Pipeline(featurize=fakes.fake_featurize, serialize=serialize_with_gbm, gbm=fakes.fake_gbm,
                    laya=laya or LayaClient.local(agent=agent or ActionAgent(), revision="fake-v2"),
                    audit=AuditLog(tmp_path / "audit.jsonl"), **kw)


def test_gbm_score_is_evidence_in_lays_state(tmp_path):
    r = make(tmp_path).score(make_booking())
    assert r["state_text"].endswith("EVIDENCE LightGBM risk 0.71")


def test_laya_action_stands_when_the_auditor_agrees(tmp_path):
    agent = ActionAgent(action="hold")
    r = make(tmp_path, agent, laya_action=True).score(make_booking())
    assert "action" in agent.asked[0]
    assert r["action"] == "hold"
    assert r["decider"] == "laya"
    la = r["laya_action"]
    assert la["proposed"] == "hold" and la["accepted"] is True and la["overrule_reason"] is None
    assert "action" not in r["probabilities"] and "action" not in r["raw_probabilities"]


def test_auditor_overrules_a_costly_laya_action_and_says_so(tmp_path):
    agent = ActionAgent(p={"misuse": 0.97, "foreign_senders": 0.99, "payoff_max": 0.93, "drop_consignee": 0.1},
                        action="allow")
    pipe = make(tmp_path, agent, laya_action=True)
    r = pipe.score(make_booking())
    assert r["action"] != "allow"
    assert r["decider"] == "cost auditor (overruled laya)"
    assert r["laya_action"]["proposed"] == "allow" and r["laya_action"]["accepted"] is False
    assert r["laya_action"]["overrule_reason"]
    assert "AUDITOR_OVERRULED_LAYA" in r["reasons"]
    rec = pipe.audit.verify()
    assert rec["records"] == 1


def test_without_action_head_laya_probabilities_decide(tmp_path):
    agent = ActionAgent()
    r = make(tmp_path, agent).score(make_booking())
    assert "action" not in agent.asked[0]
    assert r["decider"] == "laya" and r["laya_action"] is None


def test_lightgbm_decides_only_when_laya_is_down(tmp_path):
    r = make(tmp_path, laya=fakes.DownLaya(), laya_action=True).score(make_booking())
    assert r["degraded"] is True
    assert r["decider"] == "lightgbm (laya unavailable)"
    assert r["laya_action"] is None
