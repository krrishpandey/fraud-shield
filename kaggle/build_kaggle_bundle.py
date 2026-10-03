"""Build kaggle/dist/fraudshield_laya_bundle.zip: everything the Kaggle notebook needs, nothing else.

Contents (files at the zip root, so Kaggle's auto-extracted folder looks like the repo):
  fraudshield/**/*.py (+ the zip3 centroid csv)   code only, no __pycache__
  scripts/laya_*.py                                prepare / train / score / eval / cache
  requirements.txt                                 laya==0.3.23 etc. (torch is NOT pinned: Kaggle's is used)
  fraudshield_laya_kaggle.ipynb                    the notebook (also imported separately)
  artifacts/laya/train.jsonl                       portable training records (items are rebuilt on Kaggle)
  artifacts/laya/prepare.json                      sampling manifest
  artifacts/laya/evalset_{test,cal}.parquet        evaluation states + labels (+ precomputed GBM score)
  artifacts/laya/demo_states.json                  demo booking states for the cache
Never included: .env or any secret, data/raw (Olist), pickled tensors, resume states. check_safe enforces it.

Usage: uv run python kaggle/build_kaggle_bundle.py
"""
from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "kaggle" / "dist" / "fraudshield_laya_bundle.zip"
DATA = ["artifacts/laya/train.jsonl", "artifacts/laya/prepare.json", "artifacts/laya/evalset_test.parquet",
        "artifacts/laya/evalset_cal.parquet", "artifacts/laya/demo_states.json"]
EXTRA = {"kaggle/requirements.txt": "requirements.txt",
         "kaggle/fraudshield_laya_kaggle.ipynb": "fraudshield_laya_kaggle.ipynb"}
# v2 ("Laya decides", docs/LAYA_V2.md): uv run python kaggle/build_kaggle_bundle.py --v2
DIST_V2 = ROOT / "kaggle" / "dist" / "fraudshield_laya_v2_bundle.zip"
DATA_V2 = ["artifacts/laya_v2/train.jsonl", "artifacts/laya_v2/prepare.json", "artifacts/laya_v2/evalset_test.parquet",
           "artifacts/laya_v2/evalset_cal.parquet"]
EXTRA_V2 = {"kaggle/requirements.txt": "requirements.txt",
            "kaggle/fraudshield_laya_v2_kaggle.ipynb": "fraudshield_laya_v2_kaggle.ipynb",
            "docs/LAYA_V2.md": "docs/LAYA_V2.md"}
SECRET = re.compile(r"(sk-ant-[A-Za-z0-9_\-]{10,}|gsk_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{30,}|"
                    r"(API_KEY|SECRET|TOKEN)\s*=\s*['\"]?[A-Za-z0-9_\-]{16,})")
TEXT_SUFFIX = {".py", ".json", ".jsonl", ".txt", ".csv", ".ipynb", ".md", ".yaml", ".yml"}


def collect_files(root: Path = ROOT, data=None, extra=None) -> list[tuple[Path, str]]:
    data, extra = data or DATA, extra or EXTRA
    root = Path(root)
    files: list[tuple[Path, str]] = []
    for p in sorted((root / "fraudshield").rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts and p.suffix in (".py", ".csv"):
            files.append((p, p.relative_to(root).as_posix()))
    for p in sorted((root / "scripts").glob("laya_*.py")):
        files.append((p, p.relative_to(root).as_posix()))
    for rel in data:
        if not (root / rel).exists():
            raise FileNotFoundError(f"{rel} missing; run scripts/laya_prepare.py and scripts/laya_score.py's evalset step first")
        files.append((root / rel, rel))
    for rel, arc in extra.items():
        if not (root / rel).exists():
            raise FileNotFoundError(rel)
        files.append((root / rel, arc))
    return files


def check_safe(files: list[tuple[Path, str]]) -> None:
    for src, arc in files:
        name = Path(arc).name
        if name == ".env" or name.endswith(".env") or name.startswith(".env"):
            raise ValueError(f"refusing to bundle an env file: {arc}")
        parts = Path(arc).parts
        if "raw" in parts or arc.endswith(".csv.gz") or "olist_" in name:
            raise ValueError(f"refusing to bundle raw Olist data: {arc}")
        if Path(arc).suffix in (".pt", ".pkl", ".lgb"):
            raise ValueError(f"refusing to bundle a pickle/model file: {arc}")
        if Path(arc).suffix in TEXT_SUFFIX and Path(src).stat().st_size < 50_000_000:
            m = SECRET.search(Path(src).read_text(encoding="utf-8", errors="ignore"))
            if m:
                raise ValueError(f"possible secret in {arc}: {m.group(0)[:12]}...")


def build(root: Path = ROOT, out: Path = DIST, v2: bool = False) -> Path:
    files = collect_files(root, DATA_V2, EXTRA_V2) if v2 else collect_files(root)
    check_safe(files)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for src, arc in files:
            z.write(src, arc)
    return out


if __name__ == "__main__":
    v2 = "--v2" in sys.argv
    p = build(out=DIST_V2, v2=True) if v2 else build()
    with zipfile.ZipFile(p) as z:
        n = len(z.namelist())
    print(f"wrote {p} ({p.stat().st_size / 1e6:.1f} MB, {n} files)")
    sys.exit(0)
