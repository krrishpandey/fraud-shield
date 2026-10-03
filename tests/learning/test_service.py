import json

import numpy as np
import pytest

from fraudshield.audit.log import AuditLog, verify_file
from fraudshield.learning.label_store import LabelStore
from fraudshield.learning.laya_export import LayaExport
from fraudshield.learning.metrics import GateConfig
from fraudshield.learning.registry import ModelRegistry
from fraudshield.learning.retrain import fit_weighted
from fraudshield.learning.service import LearningService, NotEnoughLabels
from tests.learning.helpers import SYSTEM, TIER, fit_base, make_pipeline, make_pool, make_reference


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    d = tmp_path_factory.mktemp("models")
    df = make_reference()
    fit_base(df, d)
    return d, df


def _svc(data, tmp_path, **kw):
    d, df = data
    pipe = make_pipeline(tmp_path)
    reg = ModelRegistry(tmp_path / "registry.json", d, SYSTEM, TIER, versions_dir=tmp_path / "versions")
    svc = LearningService(pipeline=pipe, audit=pipe.audit, registry=reg,
                          label_store=LabelStore(tmp_path / "fb" / "labels.jsonl"),
                          laya_export=LayaExport(tmp_path / "fb" / "laya_feedback.jsonl"),
                          reference=lambda: df, pool=lambda: make_pool(df[df.split == "test"]),
                          gate_cfg=GateConfig(min_new_labels=20), B=40, calibration_dir=tmp_path / "cal", **kw)
    return pipe, svc


def _events(tmp_path):
    return [json.loads(x) for x in (tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()]


def test_status_before_any_retrain(data, tmp_path):
    _, svc = _svc(data, tmp_path)
    st = svc.status()
    assert st["active_version"] == "gbm-B1-R-v1" and st["labels_since_last_retrain"] == 0
    assert st["label_sources"] == {} and st["versions"][0]["deployed"] is True
    assert st["laya_export"]["rows"] == 0 and "not updated" in st["laya_weights"]["note"]
    assert st["versions"][0]["ever_deployed"] is True  # console offers rollback only to such versions


def test_too_few_labels_is_refused(data, tmp_path):
    _, svc = _svc(data, tmp_path)
    svc.simulate(n=5, seed=1, mode="uniform_noisy")
    with pytest.raises(NotEnoughLabels, match="Need at least 20 new labels, have 5"):
        svc.retrain(min_new_labels=20)


def test_retrain_deploys_swaps_and_audits_then_rollback(data, tmp_path):
    pipe, svc = _svc(data, tmp_path)
    sim = svc.simulate(n=900, seed=1, error_rate=0.0, mode="uniform_noisy")
    assert sim["simulated"] is True
    st = svc.status()
    assert st["labels_since_last_retrain"] == 900 and st["label_sources"] == {"simulated_analyst": 900}
    assert st["laya_export"]["rows"] == 900
    out = svc.retrain(min_new_labels=20)
    assert out["gate"]["passed"] is True, out["gate"]
    assert out["deployed_version"] == "gbm-B1-R-v2" and out["candidate"]["version"] == "gbm-B1-R-v2"
    assert out["n_new_labels"] == 900 and out["n_fraud"] + out["n_legit"] == 900
    assert out["run_id"] == "rt_0001"
    # the fake Laya says 0.97 for everything, so the misuse Platt refit lowers ECE and is deployed
    cr = out["calibration_refresh"]
    assert cr["status"] == "updated" and cr["ece_after"] < cr["ece_before"] and cr["n_rows"] == 900
    assert pipe.calibration["version"] == cr["version"] and svc.status()["calibration_version"] == cr["version"]
    assert (tmp_path / "cal" / f"{cr['version']}.json").exists()
    assert "not updated" in out["laya"]["note"]
    assert pipe.versions["gbm"] == "gbm-B1-R-v2"
    ev = _events(tmp_path)[-1]
    assert ev["event_type"] == "retrain" and ev["record_hash"] == out["audit_hash"]
    assert ev["payload"]["current"]["pr_auc"] == out["current"]["pr_auc"]
    assert ev["payload"]["candidate"]["version"] == "gbm-B1-R-v2" and ev["payload"]["gate"]["passed"] is True
    st = svc.status()
    assert st["active_version"] == "gbm-B1-R-v2" and st["labels_since_last_retrain"] == 0
    assert [v["parent"] for v in st["versions"]] == [None, "gbm-B1-R-v1"]
    assert st["versions"][1]["metrics"]["pr_auc"] == out["candidate"]["pr_auc"]
    # new decisions are scored by the deployed version
    r = svc.simulate(n=1, seed=9, mode="uniform_noisy")
    assert r["added"] == 1
    rb = svc.rollback("gbm-B1-R-v1")
    assert rb["active_version"] == "gbm-B1-R-v1" and pipe.versions["gbm"] == "gbm-B1-R-v1"
    ev = _events(tmp_path)[-1]
    assert ev["event_type"] == "rollback" and ev["payload"]["from"] == "gbm-B1-R-v2"
    assert verify_file(tmp_path / "audit.jsonl")["ok"] is True


def test_worse_candidate_rejected_not_deployed(data, tmp_path):
    def shuffled(system, tier, X, y, w, cal, y_cal):
        rng = np.random.default_rng(0)
        return fit_weighted(system, tier, X, rng.permutation(np.asarray(y)), w, cal, rng.permutation(np.asarray(y_cal)))
    pipe, svc = _svc(data, tmp_path, fit_fn=shuffled)
    pipe.laya = __import__("tests.api.fakes", fromlist=["DownLaya"]).DownLaya()  # degraded: no raw Laya probs
    svc.simulate(n=300, seed=1, error_rate=0.0, mode="uniform_noisy")
    before = pipe.gbm
    out = svc.retrain(min_new_labels=20)
    assert out["gate"]["passed"] is False
    assert out["deployed_version"] == "gbm-B1-R-v1" and out["candidate"]["version"] == "gbm-B1-R-v2"
    assert pipe.gbm is before and svc.registry.active_version == "gbm-B1-R-v1"
    assert svc.status()["versions"][1]["deployed"] is False
    with pytest.raises(ValueError):
        svc.rollback("gbm-B1-R-v2")  # never passed the gate
    ev = _events(tmp_path)[-1]
    assert ev["event_type"] == "retrain" and ev["payload"]["gate"]["passed"] is False
    assert ev["payload"]["deployed_version"] == "gbm-B1-R-v1"
    assert out["calibration_refresh"]["status"] == "skipped" and out["calibration_refresh"]["n_rows"] == 0
    assert out["training_set"]["n_explored_feedback"] == 0  # degraded mode never explores
    assert any("no explored" in n.lower() for n in out["notes"])


def test_realistic_simulation_is_default_and_reported_in_status(data, tmp_path):
    _, svc = _svc(data, tmp_path)
    out = svc.simulate(n=400, seed=1)
    assert out["mode"] == "realistic" and out["simulated"] is True and out["scored"] == 400
    st = svc.status()["simulation"]
    assert st["mode"] == "realistic" and st["cursor"] == 400 and st["pending"] == out["pending"]
    out2 = svc.simulate(n=0, advance_days=61)
    assert out2["pending"] == 0 and set(svc.status()["label_sources"]) <= {"simulated_analyst", "simulated_outcome"}


def test_retrain_reports_new_pattern_eval_and_gate_override(data, tmp_path):
    from tests.learning.helpers import make_new_pattern_pool
    _, svc = _svc(data, tmp_path, new_pattern=lambda: make_new_pattern_pool())
    svc.simulate(n=900, seed=1, error_rate=0.0, mode="uniform_noisy")
    out = svc.retrain(min_new_labels=20, gate="strict")
    assert out["gate"]["mode"] == "strict" and out["new_pattern_eval"]["typologies"] == ["T5"]
    assert out["new_pattern_eval"]["n"] > 0


def test_training_options_reach_retrain(data, tmp_path):
    _, svc = _svc(data, tmp_path, train_options={"max_feedback_share": 0.05, "ipw_clip": 5.0,
                                                 "lgb_params": {"min_child_samples": 50}})
    svc.simulate(n=900, seed=1, error_rate=0.0, mode="uniform_noisy")
    ts = svc.retrain(min_new_labels=20)["training_set"]
    assert ts["feedback_weight_share"] <= 0.05 + 1e-9 and ts["ipw_clip"] == 5.0
    assert ts["lgb_params_override"] == {"min_child_samples": 50}
