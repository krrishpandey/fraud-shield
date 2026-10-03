"""Import the Laya v2 Kaggle output and evaluate it (CPU).

Usage: uv run python kaggle/import_kaggle_v2_output.py <path to fraudshield_laya_v2_output.zip>
1. Extracts laya_v2/* into artifacts/laya_v2/ (checkpoint, score files, logs). Never overwrites the v1 model.
2. Runs scripts/laya_v2_eval.py (final model; and the epoch-1 checkpoint's scores if present).
3. Prints the pre-registered decision. Switching the app to "Laya decides" is a separate, explicit step
   (config/app.yaml), taken only if the win condition in docs/LAYA_V2.md is met.
"""
from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "artifacts"


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    zp = Path(sys.argv[1])
    with zipfile.ZipFile(zp) as z:
        names = [n for n in z.namelist() if n.startswith("laya_v2/")]
        if not names:
            raise SystemExit(f"{zp} has no laya_v2/ folder: is this the v2 output?")
        for n in names:
            out = (DEST / n).resolve()
            if DEST.resolve() not in out.parents:
                raise SystemExit(f"refusing to extract outside artifacts/: {n}")
            z.extract(n, DEST)
    la = DEST / "laya_v2"
    print("imported:", sorted(p.name for p in la.iterdir()))
    ck = la / "fraudshield-laya" / "model.safetensors"
    print("checkpoint:", ck, "OK" if ck.exists() else "MISSING (training did not finish; using epoch-1 scores if present)")
    if (la / "scores_ft_test.parquet").exists() and (la / "scores_ft_cal.parquet").exists():
        subprocess.run([sys.executable, str(ROOT / "scripts" / "laya_v2_eval.py")], check=True, cwd=ROOT)
        print("\nwrote artifacts/results_laya_v2.md")
    if (la / "scores_ft_ep1_test.parquet").exists() and (la / "scores_ft_ep1_cal.parquet").exists():
        subprocess.run([sys.executable, str(ROOT / "scripts" / "laya_v2_eval.py"), "--tag", "_ep1"], check=True, cwd=ROOT)
    elif (la / "scores_ft_ep1_test.parquet").exists():
        print("epoch-1 test scores present (no epoch-1 calibration-set scores, so no epoch-1 report)")


if __name__ == "__main__":
    main()
