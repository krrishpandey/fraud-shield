"""laya.model_path: a relative path to an existing checkpoint resolves against the repo root."""
from pathlib import Path

from fraudshield.api import app as A


def test_relative_existing_dir_resolves_against_root(tmp_path):
    (tmp_path / "artifacts" / "laya" / "ck").mkdir(parents=True)
    out = A.resolve_model_path("artifacts/laya/ck", root=tmp_path)
    assert Path(out) == tmp_path / "artifacts" / "laya" / "ck"


def test_hub_id_and_missing_paths_pass_through(tmp_path):
    assert A.resolve_model_path("convaiinnovations/laya", root=tmp_path) == "convaiinnovations/laya"
    assert A.resolve_model_path("artifacts/laya/missing", root=tmp_path) == "artifacts/laya/missing"


def test_absolute_path_unchanged(tmp_path):
    assert A.resolve_model_path(str(tmp_path), root=Path("/elsewhere")) == str(tmp_path)


def test_build_laya_uses_resolved_path(monkeypatch, tmp_path):
    seen = {}

    def fake_local(path, device, **kw):
        seen["path"] = path
        return "client"

    monkeypatch.setattr(A, "ROOT", tmp_path)
    (tmp_path / "ck").mkdir()
    monkeypatch.setattr(A.LayaClient, "local", staticmethod(fake_local))
    cfg = {"laya": {"mode": "local", "model_path": "ck", "device": "cuda", "cache_path": None}}
    assert A._build_laya(cfg, []) == "client"
    assert Path(seen["path"]) == tmp_path / "ck"
