import json

import pytest

from fraudshield.learning.registry import ModelRegistry
from tests.learning.helpers import SYSTEM, TIER, fit_base, make_reference


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    d = tmp_path_factory.mktemp("models")
    df = make_reference()
    return d, fit_base(df, d)


def test_virtual_base_version_without_registry_file(base, tmp_path):
    d, _ = base
    r = ModelRegistry(tmp_path / "registry.json", d, SYSTEM, TIER)
    assert r.active_version == "gbm-B1-R-v1"
    assert r.versions()[0]["parent"] is None and r.versions()[0]["deployed"] is True
    assert not (tmp_path / "registry.json").exists()  # nothing written until a change


def test_no_base_model_means_no_active_version(tmp_path):
    r = ModelRegistry(tmp_path / "registry.json", tmp_path / "empty", SYSTEM, TIER)
    assert r.active_version is None and r.versions() == []


def test_add_activate_rollback_persisted(base, tmp_path):
    d, m = base
    r = ModelRegistry(tmp_path / "registry.json", d, SYSTEM, TIER, versions_dir=tmp_path / "versions")
    v2 = r.add_version(m, parent="gbm-B1-R-v1", n_train=100, n_feedback_labels=7, metrics={"pr_auc": 0.5})
    assert v2 == "gbm-B1-R-v2" and r.active_version == "gbm-B1-R-v1"  # adding does not deploy
    with pytest.raises(ValueError):
        r.activate("gbm-B1-R-v2", require_ever_deployed=True)  # never passed the gate
    r.activate(v2)
    assert r.active_version == v2
    data = json.loads((tmp_path / "registry.json").read_text())
    assert data["active"] == v2 and [v["parent"] for v in data["versions"]] == [None, "gbm-B1-R-v1"]
    r2 = ModelRegistry(tmp_path / "registry.json", d, SYSTEM, TIER, versions_dir=tmp_path / "versions")
    assert r2.active_version == v2
    assert r2.load_model(v2).score_values({"weight_kg": 0.0}) == pytest.approx(m.score_values({"weight_kg": 0.0}))
    r2.activate("gbm-B1-R-v1", require_ever_deployed=True)
    assert [v["deployed"] for v in r2.versions()] == [True, False]
