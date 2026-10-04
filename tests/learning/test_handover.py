"""Model handover: after a retrain or rollback, status and response say which GBM scores new bookings from now on."""
import json

import numpy as np
import pytest

from fraudshield.learning.handover import KEPT, NEW_IN_USE, ROLLED_BACK, plain_check
from fraudshield.learning.label_store import LabelStore
from fraudshield.learning.laya_export import LayaExport
from fraudshield.learning.metrics import GateConfig
from fraudshield.learning.registry import ModelRegistry
from fraudshield.learning.retrain import fit_weighted
from fraudshield.learning.service import LearningService
from fraudshield.learning.simulate import booking_from_row, feature_vector_from_row
from tests.learning.helpers import SYSTEM, TIER, fit_base, make_pipeline, make_pool, make_reference

V1, V2 = f"gbm-{SYSTEM}-{TIER}-v1", f"gbm-{SYSTEM}-{TIER}-v2"


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


def _score_fresh(pipe, df, tag):
    """Score one new booking (fresh id) and return its record and its audit event."""
    r = make_pool(df[df.split == "test"]).iloc[0].to_dict()
    r["booking_id"] = f"handover-{tag}"
    pipe.score(booking_from_row(r), fv=feature_vector_from_row(r))
    rec = pipe.get_by_booking(r["booking_id"])
    lines = [json.loads(x) for x in pipe.audit.path.read_text(encoding="utf-8").splitlines()]
    ev = [e for e in lines if e["event_type"] == "decision" and e["payload"]["booking_id"] == r["booking_id"]][-1]
    return rec, ev


def _shuffled(system, tier, X, y, w, cal, y_cal):
    rng = np.random.default_rng(0)
    return fit_weighted(system, tier, X, rng.permutation(np.asarray(y)), w, cal, rng.permutation(np.asarray(y_cal)))


def test_status_names_the_model_in_use_before_any_retrain(data, tmp_path):
    _, svc = _svc(data, tmp_path)
    m = svc.status()["model_in_use"]
    assert m["registry_active_version"] == V1 and m["since"] == svc.started_at
    assert m["since_reason"].startswith("service start") and m["rollback_target"] is None
    assert svc.status()["last_handover"] is None


def test_wire_active_labels_v1_exactly_even_without_a_registry_file(data, tmp_path):
    pipe, svc = _svc(data, tmp_path, wire_active=True)
    assert not svc.registry.path.exists()
    assert pipe.versions["gbm"] == V1 and svc.status()["model_in_use"]["matches_registry"] is True
    rec, ev = _score_fresh(pipe, data[1], "v1")
    assert rec["model_versions"]["gbm"] == V1 and ev["payload"]["model_versions"]["gbm"] == V1


def test_status_reports_a_running_retrain_and_keeps_the_last_result(data, tmp_path):
    """A console page left mid-retrain must find the run on return and then its result (the response went elsewhere)."""
    import threading

    from fraudshield.learning.service import RetrainBusy

    _, svc = _svc(data, tmp_path)
    svc.simulate(n=300, seed=1, error_rate=0.0, mode="uniform_noisy")
    st = svc.status()
    assert st["retrain_running"] is None and st["last_result"] is None and st["last_retrain_error"] is None

    gate, seen = threading.Event(), {}
    orig = svc._retrain

    def slow(*a, **kw):
        seen["running"] = svc.status()["retrain_running"]
        try:
            svc.retrain(min_new_labels=20)
        except RetrainBusy as e:  # a second click while the first run is going
            seen["busy"] = str(e)
        gate.set()
        return orig(*a, **kw)

    svc._retrain = slow
    out = svc.retrain(min_new_labels=20)
    assert gate.is_set() and seen["running"]["started_at"] and "already running" in seen["busy"]
    st = svc.status()
    assert st["retrain_running"] is None and st["last_result"]["run_id"] == out["run_id"]
    assert st["last_result"]["handover"]["verdict"] == out["handover"]["verdict"]


def test_passing_gate_new_model_in_use_from_now_on_then_rollback(data, tmp_path):
    pipe, svc = _svc(data, tmp_path, wire_active=True)
    seen = []
    svc.swap_listeners.append(lambda scorer, v: seen.append((v, scorer.version)))
    svc.simulate(n=900, seed=1, error_rate=0.0, mode="uniform_noisy")
    out = svc.retrain(min_new_labels=20)
    assert out["gate"]["passed"] is True
    h = out["handover"]
    assert h["verdict"] == NEW_IN_USE and h["event"] == "retrain" and h["run_id"] == out["run_id"]
    assert (h["active_version"], h["previous_version"], h["candidate_version"]) == (V2, V1, V2)
    assert h["rollback_target"] == V1 and h["failed_checks"] == []
    assert h["active_since"] == svc.registry.active_since == svc.registry.get(V2)["activated_at"]
    assert h["message"] == (f"From now on, new bookings are scored by {V2} (since {h['active_since']}). "
                            f"Previous model {V1} is kept for rollback.")
    st = svc.status()
    assert st["model_in_use"] == {"version": V2, "since": h["active_since"], "since_reason": "activated",
                                  "registry_active_version": V2, "matches_registry": True, "rollback_target": V1}
    assert st["last_handover"]["verdict"] == NEW_IN_USE and st["last_handover"]["audit_hash"] == out["audit_hash"]
    assert seen == [(V2, V2)]  # listeners (the live stream) get the same scorer and label
    # the swap really takes effect: a new booking is scored and audited with v2
    rec, ev = _score_fresh(pipe, data[1], "after-deploy")
    assert rec["model_versions"]["gbm"] == V2 and ev["payload"]["model_versions"]["gbm"] == V2
    audit = [json.loads(x) for x in pipe.audit.path.read_text(encoding="utf-8").splitlines()]
    assert [e for e in audit if e["event_type"] == "retrain"][-1]["payload"]["handover"]["verdict"] == NEW_IN_USE

    rb = svc.rollback(V1)
    hb = rb["handover"]
    assert hb["verdict"] == ROLLED_BACK and hb["active_version"] == V1 and hb["previous_version"] == V2
    assert hb["rollback_target"] == V2 and hb["active_since"] == svc.registry.active_since
    assert hb["message"].startswith(f"Rolled back. From now on, new bookings are scored by {V1}")
    st = svc.status()
    assert st["model_in_use"]["version"] == V1 and st["last_handover"]["verdict"] == ROLLED_BACK
    assert st["last_handover"]["audit_hash"] == rb["audit_hash"]
    rec, ev = _score_fresh(pipe, data[1], "after-rollback")
    assert rec["model_versions"]["gbm"] == V1 and ev["payload"]["model_versions"]["gbm"] == V1
    assert seen[-1] == (V1, V1)


def test_failing_gate_previous_model_kept_with_reasons_in_plain_words(data, tmp_path):
    pipe, svc = _svc(data, tmp_path, fit_fn=_shuffled, wire_active=True)
    svc.simulate(n=300, seed=1, error_rate=0.0, mode="uniform_noisy")
    out = svc.retrain(min_new_labels=20)
    assert out["gate"]["passed"] is False
    h = out["handover"]
    assert h["verdict"] == KEPT
    assert (h["active_version"], h["previous_version"], h["candidate_version"]) == (V1, V1, V2)
    assert h["active_since"] == svc.started_at and h["rollback_target"] is None
    failed = {c["name"] for c in out["gate"]["checks"] if not c["passed"]}
    assert {f["name"] for f in h["failed_checks"]} == failed and failed
    assert all(f["plain"] and f["plain"] != f["detail"] for f in h["failed_checks"])
    assert h["message"].startswith(f"Still using {V1} (since {svc.started_at}). Candidate {V2}")
    assert "was not deployed because: " in h["message"]
    for f in h["failed_checks"]:
        assert f["plain"] in h["message"]
    st = svc.status()
    assert st["model_in_use"]["version"] == V1 and st["last_handover"]["verdict"] == KEPT
    rec, ev = _score_fresh(pipe, data[1], "after-reject")
    assert rec["model_versions"]["gbm"] == V1 and ev["payload"]["model_versions"]["gbm"] == V1


def test_plain_words_for_each_gate_check():
    rep = {"current": {"pr_auc": 0.811, "ece": 0.001, "cost_per_1k_brl": 122.2, "fpr_hard_negative": 0.0051},
           "candidate": {"pr_auc": 0.78, "ece": 0.03, "cost_per_1k_brl": 125.8, "fpr_hard_negative": 0.0248},
           "bootstrap": {"cost_improvement_ci95": [-20.46, 8.42], "pr_auc_diff_ci95": [-0.05, -0.01]},
           "new_pattern_eval": {"recall_diff_ci95": [-0.01, 0.05], "current": {"recall": 0.5},
                                "candidate": {"recall": 0.52}}}
    cfg = GateConfig(min_new_labels=20)

    def p(name):
        return plain_check({"name": name, "detail": "x"}, rep, cfg, 7)
    assert p("fpr_hard_negative_noninferior") == ("honest hard-case false positives would rise from 0.51% to 2.48%; "
                                                  "the limit is +0.5 points")
    assert p("cost_noninferior") == ("the cost per 1,000 bookings could get worse by up to R$20.46 (95% range of the "
                                     "saving: R$-20.46 to R$8.42); the limit is R$6.11, 5% of the current R$122.20")
    assert p("pr_auc_noninferior") == "ranking quality (PR-AUC) would fall from 0.811 to 0.780; the limit is 0.791"
    assert p("ece_noninferior") == "calibration error would rise from 0.001 to 0.030; the limit is +0.02"
    assert "no proven improvement" in p("superiority") and "-1.0 to +5.0 points" in p("superiority")
    assert p("min_new_labels") == "only 7 new labels since the last retrain; at least 20 are needed"
    assert "2%" in p("cost_not_worse") and "below zero" in p("pr_auc_not_worse")
