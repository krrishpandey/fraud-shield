"""Build a zip of the full project for the team (includes .env with the API key: share privately only).

Usage: python scripts/make_team_zip.py [--out ../FraudShield_team_share.zip]
"""
from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", "node_modules", ".venv", "__pycache__", ".pytest_cache", ".tmp", "test-results",
             "playwright-report", "dist_kaggle_tmp"}
SKIP_FILES = {"states.jsonl"}  # 100 MB, rebuilt by scripts/build_dataset.py
SKIP_PREFIXES = ("_cache_",)    # loader caches, rebuilt automatically
SKIP_PATHS = {Path("artifacts/laya/train_items.pt")}  # rebuilt by scripts/laya_prepare.py

NOTES = """FraudShield: team copy
=======================

HOW TO RUN (Windows): double-click FraudShield.bat
- First run installs uv and Python packages (a few minutes, needs internet).
- With an NVIDIA GPU the Laya model runs live (light use, ~2.5 GB VRAM). Without one the app uses cached answers.
- Do NOT run model training on a laptop GPU. Fine-tuning runs on Kaggle: see kaggle/KAGGLE_STEPS.md.

SECRET INSIDE: .env contains the team's GROQ_API_KEY (used for the plain-language explanations).
- Share this zip only inside the team (no public GitHub, Drive "anyone with link", Discord servers, etc.).
- If it leaks, rotate the key at https://console.groq.com/keys and update .env.
- git ignores .env, so it will not be committed by accident.

DATA LICENSE: data/raw (Olist Brazilian E-Commerce) is CC BY-NC-SA 4.0: non-commercial, share-alike.

START HERE: PROJECT_HISTORY.md (what we built and why), then ../DESIGN.md if included, docs/API.md, DATA_CARD.md.
"""


def include(rel: Path) -> bool:
    if any(part in SKIP_DIRS for part in rel.parts):
        return False
    if rel.name in SKIP_FILES or rel.name.startswith(SKIP_PREFIXES) or rel in SKIP_PATHS:
        return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT.parent / "FraudShield_team_share.zip"))
    args = ap.parse_args()
    out = Path(args.out).resolve()
    if ROOT in out.parents:
        raise SystemExit("write the zip outside the repo folder")
    n, raw = 0, 0
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for dirpath, dirnames, filenames in os.walk(ROOT):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for f in filenames:
                p = Path(dirpath) / f
                rel = p.relative_to(ROOT)
                if include(rel):
                    z.write(p, Path("fraudshield") / rel)
                    n += 1
                    raw += p.stat().st_size
        design = ROOT.parent / "DESIGN.md"
        if design.exists():
            z.write(design, "DESIGN.md")
        z.writestr("SHARE_NOTES.txt", NOTES)
        names = set(z.namelist())
    assert "fraudshield/.env" in names, ".env missing from zip"
    print(f"files: {n}, uncompressed: {raw / 1e6:.0f} MB, zip: {out.stat().st_size / 1e6:.0f} MB -> {out}")


if __name__ == "__main__":
    main()
