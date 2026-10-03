"""Add the account mix features (acct_hv_share, acct_far_share, acct_mean_dist) to the existing feature files
without rebuilding the dataset. Uses fraudshield.features.mix.account_mix_frame, the same definition as the live
featurizer (tests/features/test_mix.py and test_real_equivalence.py check that they agree).

Usage: uv run python scripts/add_account_mix.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fraudshield.features.mix import MIX_FEATURES, account_mix_frame  # noqa: E402

PROC = ROOT / "data" / "processed"


def augment(path: Path) -> None:
    df = pd.read_parquet(path)
    mix = account_mix_frame(df).set_index("booking_id")
    for k in MIX_FEATURES:
        df[k] = df.booking_id.map(mix[k])
    df.to_parquet(path, index=False)
    print(f"{path.name}: {len(df):,} rows, acct_hv_share filled {df.acct_hv_share.notna().mean():.1%}", flush=True)


if __name__ == "__main__":
    for p in sorted((PROC / "features").glob("seed_*.parquet")):
        augment(p)
    if (PROC / "features.parquet").exists():
        augment(PROC / "features.parquet")
