import numpy as np
import pytest

from fraudshield.learning.metrics import GateConfig
from fraudshield.learning.registry import ModelRegistry
from fraudshield.learning.retrain import fit_weighted, run_retrain
from fraudshield.policy.costs import CostConfig
from fraudshield.policy.decide import PolicyConfig
from tests.learning.helpers import SYSTEM, TIER, fit_base, labels_from_frame, make_reference


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    d = tmp_path_factory.mktemp("models")
    df = make_reference()
    fit_base(df, d)
    labels = labels_from_frame(df[df.split == "test"].iloc[:1200])
    return d, df, labels


def _run(world, tmp_path, **kw):
    d, df, labels = world
    reg = ModelRegistry(tmp_path / "registry.json", d, SYSTEM, TIER, versions_dir=tmp_path / "v")
    args = dict(registry=reg, reference=df, labels=labels, n_new_labels=len(labels), costs=CostConfig.default(),
                policy=PolicyConfig(), gate_cfg=GateConfig(min_new_labels=20), B=60)
    args.update(kw)
    return reg, run_retrain(**args)


def test_good_candidate_learns_new_pattern_and_passes(world, tmp_path):
    reg, res = _run(world, tmp_path)
    rep = res.report
    assert rep["current"]["version"] == "gbm-B1-R-v1"
    assert rep["candidate"]["recall_new_pattern"] > rep["current"]["recall_new_pattern"]
    assert rep["candidate"]["pr_auc"] > rep["current"]["pr_auc"]
    assert rep["gate"]["passed"] is True, rep["gate"]
    assert rep["new_pattern_typologies"] == ["T5"]
    ts = rep["training_set"]
    assert ts["n_feedback"] == 840 and ts["n_reference"] == 1500 and ts["n_explored_feedback"] > 0
    assert any("biased sample" in n for n in rep["notes"])
    # eval set is disjoint from everything the candidate trained on
    assert not (set(res.train_ids) & set(res.eval_ids))
    assert rep["eval_set"]["n"] == 800 + 360
    assert res.candidate is not None and reg.active_version == "gbm-B1-R-v1"  # run_retrain never deploys


def test_deliberately_worse_candidate_is_rejected(world, tmp_path):
    def shuffled_fit(system, tier, X, y, w, cal, y_cal):
        rng = np.random.default_rng(0)
        return fit_weighted(system, tier, X, rng.permutation(np.asarray(y)), w, cal, rng.permutation(np.asarray(y_cal)))
    _, res = _run(world, tmp_path, fit_fn=shuffled_fit)
    assert res.report["gate"]["passed"] is False
    failed = {c["name"] for c in res.report["gate"]["checks"] if not c["passed"]}
    assert "pr_auc_noninferior" in failed


def test_fit_weighted_uses_weights():
    from tests.learning.helpers import make_frame
    tr = make_frame(600, "train", "2017-01-01", 3)
    cal = make_frame(300, "cal", "2017-06-01", 4)
    w0 = np.ones(len(tr))
    w1 = np.where(tr.is_fraud, 5.0, 1.0)
    a = fit_weighted(SYSTEM, TIER, tr, tr.is_fraud, w0, cal, cal.is_fraud).predict_raw(cal)
    b = fit_weighted(SYSTEM, TIER, tr, tr.is_fraud, w1, cal, cal.is_fraud).predict_raw(cal)
    assert b.mean() > a.mean()  # upweighting fraud raises raw scores


def test_hard_negative_typology_mislabelled_fraud_is_not_a_new_pattern(world, tmp_path):
    d, df, labels = world
    labels = [dict(x) for x in labels]
    for x in labels[-20:]:  # latest by time -> eval part; an analyst error marks hard negatives as fraud
        if x["label"] == "legit":
            x.update(label="fraud", typology="HN_3pl")
    _, res = _run((d, df, labels), tmp_path)
    assert res.report["new_pattern_typologies"] == ["T5"]


def test_new_pattern_eval_set_separate_and_campaign_disjoint(world, tmp_path):
    from tests.learning.helpers import make_new_pattern_pool
    d, df, labels = world
    pool = make_new_pattern_pool()
    train_camp = next(x["campaign_id"] for x in labels[:100] if x["campaign_id"] and x["label"] == "fraud")
    pool.loc[pool.index[:7], "campaign_id"] = train_camp  # a campaign the candidate trained on
    _, res = _run(world, tmp_path, new_pattern_pool=pool)
    npe = res.report["new_pattern_eval"]
    assert npe["typologies"] == ["T5"]
    t5 = pool[pool.typology == "T5"]
    assert npe["n"] == int((t5.campaign_id != train_camp).sum()) and npe["excluded_rows"] == int((t5.campaign_id == train_camp).sum())
    assert npe["campaigns"] == t5.loc[t5.campaign_id != train_camp, "campaign_id"].nunique()
    assert npe["candidate"]["recall"] > npe["current"]["recall"]
    assert npe["recall_diff_ci95"][0] > 0
    assert set(npe["current"]["by_typology"]) == {"T5"}
    names = {c["name"]: c for c in res.report["gate"]["checks"]}
    assert names["superiority"]["passed"] is True and "new-pattern recall" in names["superiority"]["detail"]


def test_no_new_pattern_pool_reports_null(world, tmp_path):
    _, res = _run(world, tmp_path)
    assert res.report["new_pattern_eval"] is None
