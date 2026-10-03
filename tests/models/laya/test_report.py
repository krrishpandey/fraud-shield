from fraudshield.models.laya_train.report import verdict_text


def _res(diff_t3, diff_t5, auc_diff):
    a = []
    for t, d in (("T3", diff_t3), ("T5", diff_t5)):
        a.append({"typology": t, "fpr": 0.01, "B2": 0.5, "M1": 0.5 + sum(d) / 2, "B3": 0.2, "M1_minus_B2_ci": d})
    return {"headline_a": a, "headline_b": {"B3": {"auc": 0.6}, "M1": {"auc": 0.6 + sum(auc_diff) / 2},
                                            "M1_minus_B3_auc_ci": auc_diff}}


def test_gbm_wins_is_said_plainly():
    v = verdict_text(_res((-0.3, -0.1), (-0.5, -0.4), (-0.02, 0.03)))
    assert "GBM" in v and "not met" in v
    assert "evidence layer" in v


def test_m1_win_requires_ci_above_zero():
    v = verdict_text(_res((0.05, 0.2), (-0.1, 0.1), (-0.02, 0.03)))
    assert "met for T3" in v and "T5" in v


def test_zero_shot_win():
    v = verdict_text(_res((-0.3, -0.1), (-0.5, -0.4), (0.05, 0.12)))
    assert "drop_consignee" in v and "beats stock" in v
