import hashlib
import json
import zipfile

import pytest

APP_YAML = """# config
laya:
  mode: auto                     # auto | local
  model_path: convaiinnovations/laya   # or a local fine-tuned checkpoint dir
  device: cuda
calibration_path: artifacts/old.json
costs_path: config/costs.yaml
"""


def _fake_zip(tmp_path, *, bad_revision=False, drop=None):
    w = b"fake weights"
    rev = hashlib.sha256(w).hexdigest()[:12]
    files = {
        "fraudshield-laya/model.safetensors": w,
        "fraudshield-laya/rl_agent_config.json": json.dumps({"temperature": [1, 1, 1]}).encode(),
        "fraudshield-laya/encoder/config.json": b"{}",
        "fraudshield-laya/tokenizer/tokenizer.json": b"{}",
        "calibration.json": json.dumps({"version": "cal-x", "model_revision": "000000000000" if bad_revision else rev,
                                        "temperatures": {}, "platt": {}, "conformal": {}, "prevalence": 0.01}).encode(),
        "laya_cache.json": json.dumps({"_revision": rev, "k": {}}).encode(),
        "results_laya.md": b"# results",
        "results_laya.json": b"{}",
        "laya/scores_ft_test.parquet": b"pq",
        "laya/train.log": b"log",
        "versions.json": b"{}",
    }
    if drop:
        files.pop(drop)
    z = tmp_path / "fraudshield_laya_output.zip"
    with zipfile.ZipFile(z, "w") as f:
        for k, v in files.items():
            f.writestr(k, v)
    return z


def _repo(tmp_path):
    r = tmp_path / "repo"
    (r / "config").mkdir(parents=True)
    (r / "artifacts").mkdir()
    (r / "config" / "app.yaml").write_text(APP_YAML)
    return r


def test_import_copies_files_and_updates_config(tmp_path, import_mod):
    r = _repo(tmp_path)
    rep = import_mod.import_output(_fake_zip(tmp_path), root=r, run_eval=False)
    assert (r / "artifacts/laya/fraudshield-laya/model.safetensors").read_bytes() == b"fake weights"
    assert (r / "artifacts/laya/fraudshield-laya/tokenizer/tokenizer.json").exists()
    for f in ("calibration.json", "laya_cache.json", "results_laya.md", "results_laya.json"):
        assert (r / "artifacts" / f).exists()
    assert (r / "artifacts/laya/scores_ft_test.parquet").exists()
    y = (r / "config/app.yaml").read_text()
    assert "  model_path: artifacts/laya/fraudshield-laya" in y
    assert "calibration_path: artifacts/calibration.json" in y
    assert "# config" in y and "costs_path: config/costs.yaml" in y and "  device: cuda" in y   # rest untouched
    assert rep["model_revision"] == hashlib.sha256(b"fake weights").hexdigest()[:12]


def test_import_refuses_missing_files(tmp_path, import_mod):
    with pytest.raises(ValueError, match="calibration.json"):
        import_mod.import_output(_fake_zip(tmp_path, drop="calibration.json"), root=_repo(tmp_path), run_eval=False)


def test_import_refuses_revision_mismatch(tmp_path, import_mod):
    with pytest.raises(ValueError, match="revision"):
        import_mod.import_output(_fake_zip(tmp_path, bad_revision=True), root=_repo(tmp_path), run_eval=False)


def test_import_is_idempotent(tmp_path, import_mod):
    r = _repo(tmp_path)
    z = _fake_zip(tmp_path)
    import_mod.import_output(z, root=r, run_eval=False)
    import_mod.import_output(z, root=r, run_eval=False)
    assert (r / "config/app.yaml").read_text().count("model_path:") == 1


def test_update_config_keeps_crlf(tmp_path, import_mod):
    p = tmp_path / "app.yaml"
    p.write_bytes(APP_YAML.replace("\n", "\r\n").encode())
    import_mod.update_config(p)
    b = p.read_bytes()
    assert b.count(b"\r\n") == APP_YAML.count("\n") and b.count(b"\n") == b.count(b"\r\n")
