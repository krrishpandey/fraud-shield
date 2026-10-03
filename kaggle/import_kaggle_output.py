"""Import the Kaggle run's output zip into the repo.

Usage: uv run python kaggle/import_kaggle_output.py path/to/fraudshield_laya_output.zip

- verifies the expected files are present and that calibration.json and laya_cache.json were made for
  exactly these weights (model_revision == sha256(model.safetensors)[:12]);
- copies the checkpoint to artifacts/laya/fraudshield-laya/, calibration.json, laya_cache.json and
  results_laya.{md,json} to artifacts/, scores and logs to artifacts/laya/;
- points config/app.yaml at the checkpoint (laya.model_path) and calibration (calibration_path), editing
  only those two lines (comments, order and line endings kept);
- runs scripts/laya_eval.py locally (CPU only, GPU hidden) only if results_laya.md is missing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CKPT_REL = "artifacts/laya/fraudshield-laya"
REQUIRED = ["fraudshield-laya/model.safetensors", "fraudshield-laya/rl_agent_config.json",
            "fraudshield-laya/encoder/config.json", "calibration.json", "laya_cache.json"]


def _sha12(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def update_config(path: Path) -> None:
    """Edit only laya.model_path and calibration_path; keep comments, order and line endings."""
    lines = path.read_bytes().decode("utf-8").splitlines(keepends=True)
    in_laya = False
    for i, line in enumerate(lines):
        body = line.rstrip("\r\n")
        eol = line[len(body):] or "\n"
        if re.match(r"^\S", body):
            in_laya = body.startswith("laya:")
        if in_laya and re.match(r"^\s+model_path:", body):
            indent = re.match(r"^(\s+)", body).group(1)
            lines[i] = (f"{indent}model_path: {CKPT_REL}   # fine-tuned Laya (imported from Kaggle); "
                        f"stock: convaiinnovations/laya{eol}")
        if re.match(r"^calibration_path:", body):
            lines[i] = f"calibration_path: artifacts/calibration.json{eol}"
    path.write_bytes("".join(lines).encode("utf-8"))


def import_output(zip_path: str | Path, root: Path = ROOT, run_eval: bool = True) -> dict:
    root = Path(root)
    art = root / "artifacts"
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        with zipfile.ZipFile(zip_path) as z:
            names = set(z.namelist())
            missing = [r for r in REQUIRED if r not in names]
            if missing:
                raise ValueError(f"output zip is missing: {', '.join(missing)}")
            if not any(n.startswith("fraudshield-laya/tokenizer/") for n in names):
                raise ValueError("output zip is missing: fraudshield-laya/tokenizer/")
            z.extractall(td)
        rev = _sha12(td / "fraudshield-laya" / "model.safetensors")
        cal = json.loads((td / "calibration.json").read_text())
        cache = json.loads((td / "laya_cache.json").read_text())
        if cal.get("model_revision") != rev or cache.get("_revision") != rev:
            raise ValueError(f"revision mismatch: weights {rev}, calibration {cal.get('model_revision')}, "
                             f"cache {cache.get('_revision')}")
        dst = root / CKPT_REL
        if dst.exists():
            shutil.rmtree(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(td / "fraudshield-laya", dst)
        copied = [CKPT_REL + "/"]
        for f in ("calibration.json", "laya_cache.json", "results_laya.md", "results_laya.json"):
            if (td / f).exists():
                shutil.copy(td / f, art / f)
                copied.append(f"artifacts/{f}")
        if (td / "laya").is_dir():
            for p in (td / "laya").iterdir():
                if p.is_file():
                    shutil.copy(p, art / "laya" / p.name)
                    copied.append(f"artifacts/laya/{p.name}")
        if (td / "versions.json").exists():
            shutil.copy(td / "versions.json", art / "laya" / "kaggle_versions.json")
    update_config(root / "config" / "app.yaml")
    ran_eval = False
    if run_eval and not (art / "results_laya.md").exists():
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1")      # CPU only
        subprocess.run([sys.executable, str(root / "scripts" / "laya_eval.py")], cwd=root, env=env, check=True)
        ran_eval = True
    return {"model_revision": rev, "calibration_version": cal.get("version"), "copied": copied,
            "config": "config/app.yaml: laya.model_path, calibration_path", "ran_eval": ran_eval}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("zip")
    ap.add_argument("--no-eval", action="store_true")
    a = ap.parse_args()
    print(json.dumps(import_output(a.zip, run_eval=not a.no_eval), indent=2))
