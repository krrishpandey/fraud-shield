"""Training-side fixes after the realistic run (docs/LEARNING_GATE.md, 'Training changes after run')."""
import numpy as np
import pytest

from fraudshield.learning.retrain import cap_feedback_weight, feedback_weights, fit_weighted
from tests.learning.helpers import SYSTEM, TIER, make_frame
from tests.learning.test_retrain import _run, world  # noqa: F401  (module fixture)


def test_cap_scales_feedback_to_max_share():
    w = np.full(500, 2.0)  # 1000 total vs 1000 reference -> 50% share
    out = cap_feedback_weight(w, ref_total=1000.0, max_share=0.3)
    assert out.sum() / (out.sum() + 1000.0) == pytest.approx(0.3)
    np.testing.assert_allclose(out / out[0], np.ones(500))  # relative weights unchanged


def test_cap_leaves_small_feedback_alone_and_none_disables():
    w = np.ones(100)
    np.testing.assert_array_equal(cap_feedback_weight(w, 1000.0, 0.3), w)
    np.testing.assert_array_equal(cap_feedback_weight(np.full(5000, 1.0), 1000.0, None), np.full(5000, 1.0))


def test_lower_clip_changes_relative_explored_weights():
    labs = [{"explored": True, "propensity": 0.05}, {"explored": True, "propensity": 0.5}]
    w20, w5 = feedback_weights(labs, clip=20.0), feedback_weights(labs, clip=5.0)
    assert w20[0] / w20[1] == pytest.approx(10.0) and w5[0] / w5[1] == pytest.approx(2.5)


def test_run_retrain_reports_and_respects_share(world, tmp_path):  # noqa: F811
    _, res = _run(world, tmp_path, max_feedback_share=0.1, ipw_clip=5.0)
    ts = res.report["training_set"]
    assert ts["feedback_weight_share"] <= 0.1 + 1e-9 and ts["ipw_clip"] == 5.0
    assert ts["max_feedback_share"] == 0.1 and ts["explored_weight"] >= 0


def test_subsampling_keeps_every_hard_negative(world, tmp_path):  # noqa: F811
    d, df, labels = world
    n_hn = int((df[df.split == "train"].hn_injected).sum())
    _, res = _run(world, tmp_path, max_train_rows=600)
    ts = res.report["training_set"]
    assert ts["n_reference"] <= 600 + n_hn and ts["n_reference_hard_negatives"] == n_hn


def test_lgb_param_overrides_applied():
    tr = make_frame(400, "train", "2017-01-01", 3)
    cal = make_frame(200, "cal", "2017-06-01", 4)
    m = fit_weighted(SYSTEM, TIER, tr, tr.is_fraud, np.ones(len(tr)), cal, cal.is_fraud,
                     params={"min_child_samples": 100, "lambda_l2": 10.0})
    assert m.metadata["params"]["min_child_samples"] == 100 and m.metadata["params"]["lambda_l2"] == 10.0
