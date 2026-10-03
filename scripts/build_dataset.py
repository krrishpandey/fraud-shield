"""Build the FraudShield dataset: real Olist bookings + synthetic billing layer + injected campaigns
for N seeds, features for every booking of every seed.

Usage: uv run python scripts/build_dataset.py [--seeds 10] [--workers 5]
Outputs (data/processed/): bookings_all.parquet, features/seed_K.parquet, features.parquet (seed 0),
events/seed_K.json, build_manifest.json. See docs/DATA_OUTPUTS.md.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fraudshield.data.dataset import assemble, hard_negative_slices  # noqa: E402
from fraudshield.data.inject import InjectionContext, inject  # noqa: E402
from fraudshield.data.pipeline import PROCESSED, real_with_billing  # noqa: E402
from fraudshield.features import featurize_frame  # noqa: E402

KEEP = ["booking_id", "account_id", "booked_at", "split", "is_injected", "is_fraud", "typology", "scenario_id",
        "campaign_id", "camouflage_level", "mimic", "seed", "label_misuse", "label_foreign_senders",
        "label_payoff_max", "label_drop_consignee", "risk_level", "phase", "cost_ratio", "hn_change"]


def run_seed(seed: int) -> dict:
    t0 = time.time()
    real, _, legit_changes, pool = real_with_billing()
    ctx = InjectionContext(real, pool)
    res = inject(ctx, seed)
    df, changes, confirmed = assemble(real, res, legit_changes)
    feats = featurize_frame(df, changes=changes, confirmed=confirmed)
    hn = hard_negative_slices(df, feats)
    out = df[KEEP].reset_index(drop=True).join(hn.reset_index(drop=True))
    out = out.merge(feats, on="booking_id", how="left", suffixes=("", "_f"))
    out["seed"] = seed
    (PROCESSED / "features").mkdir(exist_ok=True)
    (PROCESSED / "events").mkdir(exist_ok=True)
    out.to_parquet(PROCESSED / "features" / f"seed_{seed}.parquet", index=False)
    res.rows.to_parquet(PROCESSED / "features" / f"injected_seed_{seed}.parquet", index=False)
    ev = {"changes": [{**c, "at": str(c["at"]), "announced_at": str(c["announced_at"])} for c in changes.to_dict("records")],
          "confirmed": [{**c, "confirmed_at": str(c["confirmed_at"])} for c in confirmed]}
    (PROCESSED / "events" / f"seed_{seed}.json").write_text(json.dumps(ev, default=str))
    inj = res.rows
    summ = {
        "seed": seed, "seconds": round(time.time() - t0, 1),
        "injected_rows": int(len(inj)), "fraud_rows": int(inj.is_fraud.sum()),
        "campaigns": inj[inj.is_fraud].groupby("split").campaign_id.nunique().to_dict(),
        "hn_campaigns": inj[~inj.is_fraud].groupby("split").campaign_id.nunique().to_dict(),
        "prevalence": {s: round(float(out[out.split == s].is_fraud.mean()), 5) for s in ("train", "cal", "test")},
    }
    print(json.dumps(summ), flush=True)
    return summ


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--force-real", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    real, accounts, changes, pool = real_with_billing(force=a.force_real)
    print(f"real bookings {len(real):,}, accounts {real.account_id.nunique():,}, legit changes {len(changes)}", flush=True)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        summaries = list(ex.map(run_seed, range(a.seeds)))
    # bookings_all: real rows once (seed -1 = present in every seed) + injected rows of every seed
    from fraudshield.data.dataset import label_real
    inj = pd.concat([pd.read_parquet(PROCESSED / "features" / f"injected_seed_{s}.parquet") for s in range(a.seeds)],
                    ignore_index=True)
    allb = pd.concat([label_real(real), inj], ignore_index=True)
    allb.to_parquet(PROCESSED / "bookings_all.parquet", index=False)
    pd.read_parquet(PROCESSED / "features" / "seed_0.parquet").to_parquet(PROCESSED / "features.parquet", index=False)
    accounts.to_parquet(PROCESSED / "accounts_synthetic.parquet", index=False)
    manifest = {"seeds": a.seeds, "real_bookings": int(len(real)), "real_accounts": int(real.account_id.nunique()),
                "legit_changes": changes.kind.value_counts().to_dict(),
                "real_by_split": real.split.value_counts().to_dict(), "per_seed": summaries,
                "test_fraud_campaigns_total": int(sum(s["campaigns"].get("test", 0) for s in summaries)),
                "seconds": round(time.time() - t0, 1)}
    (PROCESSED / "build_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(json.dumps({k: v for k, v in manifest.items() if k != "per_seed"}, default=str))


if __name__ == "__main__":
    main()
