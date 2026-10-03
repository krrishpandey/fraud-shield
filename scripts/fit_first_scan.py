"""Fit the depot weighing dial (first-scan check for under-declared parcels) and measure each level.

Score = how far below the account's usual size and weight a parcel is declared: -(dims_z + weight_z), for
accounts with 5+ earlier bookings (fraudshield.features.mix.under_score). A level weighs the parcels flagged by the
standard rule (mix.rule_flags) plus every parcel whose score reaches the level's threshold.

Thresholds are chosen on the CALIBRATION split (honest parcels only) of each seed, and the median over seeds is
used; each level is then measured on the TEST split of every seed, with the depot follow-up (a failed scan sends
the account's later parcels to the scale). Writes artifacts/first_scan_dial.json.

Usage: uv run python scripts/fit_first_scan.py [--seeds 10]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fraudshield.features.mix import rule_flags_frame, under_score_frame  # noqa: E402

LEVELS = [("standard", 0.0), ("3%", 0.03), ("5%", 0.05), ("10%", 0.10)]
COLS = ["booking_id", "account_id", "booked_at", "split", "is_fraud", "typology", "dims_z", "weight_z", "n_prior",
        "consignee_first_seen_days", "consignee_bookings_30d", "consignee_other_accts_30d", "carrier_cost"]


def threshold_for(cal: pd.DataFrame, budget: float) -> float | None:
    """Lowest score threshold whose union with the standard rule weighs at most `budget` of honest parcels."""
    if budget <= 0:
        return None
    h = cal[~cal.is_fraud]
    base = rule_flags_frame(h).under_declared.to_numpy().astype(bool)
    s = under_score_frame(h)
    cand = np.sort(np.unique(s[np.isfinite(s)]))[::-1]
    best = None
    lo, hi = 0, len(cand) - 1
    while lo <= hi:  # the weighed share grows as the threshold falls: binary search
        mid = (lo + hi) // 2
        rate = (base | (s >= cand[mid])).mean()
        if rate <= budget:
            best, lo = cand[mid], mid + 1
        else:
            hi = mid - 1
    return float(best) if best is not None else None


def weighed(te: pd.DataFrame, thr: float | None) -> np.ndarray:
    flag = rule_flags_frame(te).under_declared.to_numpy().astype(bool)
    if thr is not None:
        flag |= under_score_frame(te) >= thr
    t6 = (te.typology == "T6").to_numpy()
    out = np.zeros(len(te), bool)
    proven: set = set()
    for i, a in enumerate(te.account_id.to_numpy()):
        out[i] = flag[i] or a in proven
        if out[i] and t6[i]:
            proven.add(a)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    a = ap.parse_args()
    frames = [pd.read_parquet(ROOT / "data" / "processed" / "features" / f"seed_{s}.parquet", columns=COLS)
              for s in range(a.seeds)]
    thresholds = {}
    for name, budget in LEVELS:
        ts = [threshold_for(f[f.split == "cal"], budget) for f in frames]
        ts = [t for t in ts if t is not None]
        thresholds[name] = float(np.median(ts)) if ts else None
    rows = []
    for seed, f in enumerate(frames):
        te = f[f.split == "test"].sort_values(["booked_at", "booking_id"], kind="mergesort").reset_index(drop=True)
        t6, legit = (te.typology == "T6").to_numpy(), ~te.is_fraud.to_numpy()
        freight = te.carrier_cost.to_numpy()
        for name, budget in LEVELS:
            w = weighed(te, thresholds[name])
            rows.append({"seed": seed, "level": name, "t6_caught": w[t6].mean(), "honest_weighed": w[legit].mean(),
                         "t6_freight_per_1k": 1000 * freight[t6 & w].sum() / len(te),
                         "weighs_per_1k": 1000 * w.mean()})
    r = pd.DataFrame(rows)
    levels = []
    for name, budget in LEVELS:
        g = r[r.level == name]
        levels.append({"name": name, "target_honest_share": budget, "threshold": thresholds[name],
                       "t6_caught": round(float(g.t6_caught.mean()), 4), "t6_caught_sd": round(float(g.t6_caught.std()), 4),
                       "honest_weighed": round(float(g.honest_weighed.mean()), 4),
                       "weighs_per_1k_bookings": round(float(g.weighs_per_1k.mean()), 2),
                       "t6_freight_checked_per_1k_brl": round(float(g.t6_freight_per_1k.mean()), 2)})
    out = {"version": "first-scan-dial-v1", "seeds": a.seeds,
           "score": "-(dims_z + weight_z) for accounts with 5+ earlier bookings, plus the standard rule",
           "chosen_on": "calibration split, honest parcels; median over seeds",
           "measured_on": "test split, with the depot follow-up", "levels": levels}
    (ROOT / "artifacts" / "first_scan_dial.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(pd.DataFrame(levels).to_string())


if __name__ == "__main__":
    main()
