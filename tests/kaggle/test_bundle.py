import zipfile

import pytest


def _repo(tmp_path):
    r = tmp_path / "repo"
    for p, txt in {
        "fraudshield/__init__.py": "", "fraudshield/contracts.py": "X=1",
        "fraudshield/features/zip3_centroids.csv": "a,b", "fraudshield/__pycache__/x.cpython-312.pyc": "bin",
        "fraudshield/data/notes.md": "skip me",
        "scripts/laya_train.py": "", "scripts/laya_eval.py": "", "scripts/train_gbm.py": "not needed",
        "artifacts/laya/train.jsonl": "{}", "artifacts/laya/prepare.json": "{}",
        "artifacts/laya/evalset_test.parquet": "p", "artifacts/laya/evalset_cal.parquet": "p",
        "artifacts/laya/demo_states.json": "[]", "artifacts/laya/train_items.pt": "pickle",
        "artifacts/laya/resume_state.pt": "big", "kaggle/requirements.txt": "laya==0.3.23\n",
        "kaggle/fraudshield_laya_kaggle.ipynb": "{}", ".env": "GROQ_API_KEY=secret",
        "data/raw/olist_orders_dataset.csv.gz": "raw",
    }.items():
        f = r / p
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(txt)
    return r


def test_collect_only_needed_files(tmp_path, bundle_mod):
    names = {arc for _, arc in bundle_mod.collect_files(_repo(tmp_path))}
    assert {"fraudshield/contracts.py", "fraudshield/features/zip3_centroids.csv", "scripts/laya_train.py",
            "scripts/laya_eval.py", "artifacts/laya/train.jsonl", "artifacts/laya/prepare.json",
            "artifacts/laya/evalset_test.parquet", "artifacts/laya/evalset_cal.parquet",
            "artifacts/laya/demo_states.json", "requirements.txt", "fraudshield_laya_kaggle.ipynb"} <= names
    for bad in ("scripts/train_gbm.py", "artifacts/laya/train_items.pt", "artifacts/laya/resume_state.pt",
                ".env", "fraudshield/data/notes.md"):
        assert bad not in names
    assert not any("__pycache__" in n or n.startswith("data/") for n in names)


def test_safety_check_rejects_env_raw_and_secrets(tmp_path, bundle_mod):
    r = _repo(tmp_path)
    with pytest.raises(ValueError, match="env"):
        bundle_mod.check_safe([(r / ".env", ".env")])
    with pytest.raises(ValueError, match="raw"):
        bundle_mod.check_safe([(r / "data/raw/olist_orders_dataset.csv.gz", "data/raw/olist_orders_dataset.csv.gz")])
    leak = r / "scripts" / "laya_leak.py"
    leak.write_text('KEY = "sk-ant-api03-abcdefghijklmnopqrstuvwxyz"')
    with pytest.raises(ValueError, match="secret"):
        bundle_mod.check_safe([(leak, "scripts/laya_leak.py")])


def test_missing_required_file_fails(tmp_path, bundle_mod):
    r = _repo(tmp_path)
    (r / "artifacts/laya/evalset_cal.parquet").unlink()
    with pytest.raises(FileNotFoundError):
        bundle_mod.collect_files(r)


def test_build_writes_zip_with_files_at_root(tmp_path, bundle_mod):
    out = bundle_mod.build(_repo(tmp_path), tmp_path / "dist" / "b.zip")
    with zipfile.ZipFile(out) as z:
        names = z.namelist()
    assert "scripts/laya_train.py" in names and ".env" not in names
