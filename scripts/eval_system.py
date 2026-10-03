"""Whole-system evaluation per fraud type: LightGBM, plus the drop-address rule (T3), plus the first-scan check for
under-declared parcels (T6). Test window, 10 injection seeds, both tiers. Writes artifacts/results_system.md.

- Model only: the B2 LightGBM score (same training as scripts/train_gbm.py).
- + drop rule: misuse = 1 - (1 - p)(1 - DROP_PRIOR) when the drop pattern fires (pipeline.default_rules). It uses no
  T3 labels; T3 and T5 stay out of training.
- Caught by the system: flagged in the top 1% per day by the system score, or sent to a first-scan check by the
  under-declared rule. A first-scan check catches a T6 parcel because its true weight is 30-70% above the declared
  one (the dataset keeps the true weight). It also costs honest shippers a scan (about R$1, an assumption).

Usage: uv run python scripts/eval_system.py [--seeds 10]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fraudshield.api.pipeline import DROP_PRIOR  # noqa: E402
from fraudshield.data.metrics import flag_top_per_day, pr_auc, roc_auc  # noqa: E402
from fraudshield.features.mix import rule_flags_frame  # noqa: E402
from fraudshield.models.gbm import NUM_ROUNDS, PARAMS, system_features  # noqa: E402

TYPES = ("T1", "T2", "T3", "T4", "T5", "T6", "T7")


def evaluate(te: pd.DataFrame, s_model: np.ndarray) -> list[dict]:
    flags = rule_flags_frame(te)
    drop, under = flags.drop_pattern.to_numpy().astype(bool), flags.under_declared.to_numpy().astype(bool)
    s_sys = 1 - (1 - s_model) * (1 - DROP_PRIOR * drop)
    y = te.is_fraud.to_numpy()
    t = te.typology.astype(str).to_numpy()
    legit = ~y
    out = []
    for name, s, scan in (("model only", s_model, np.zeros_like(under)), ("model + drop rule", s_sys, np.zeros_like(under)),
                          ("model + drop rule + first-scan check", s_sys, under)):
        f = flag_top_per_day(te.booked_at, s, 0.01)
        caught = f | scan
        te_sorted_ok = te.booked_at.is_monotonic_increasing
        out.append(_row(name, te, s, f, caught, y, t, legit))
    # + depot follow-up: once a scan proves under-declaration, the account's later parcels are weighed too.
    # Honest parcels never fail the scale, so this adds no honest checks (bookings are in booked_at order).
    f = flag_top_per_day(te.booked_at, s_sys, 0.01)
    scanned = np.zeros(len(te), bool)
    proven: set = set()
    for i, acc in enumerate(te.account_id.to_numpy()):
        scanned[i] = under[i] or acc in proven
        if scanned[i] and t[i] == "T6":
            proven.add(acc)
    out.append(_row("model + drop rule + first-scan check + depot follow-up", te, s_sys, f, f | scanned, y, t, legit,
                    scan=scanned))
    return out


def _row(name, te, s, f, caught, y, t, legit, scan=None):
    if True:
        row = {"system": name, "pr_auc": pr_auc(y, s), "roc_auc": roc_auc(y, s),
               "caught_all": caught[y].mean(), "honest_stopped": f[legit].mean(),
               "honest_scan_checked": ((caught & ~f) if scan is None else (scan & ~f))[legit].mean()}
        for k in TYPES:
            m = legit | (t == k)
            row[f"pr_{k}"] = pr_auc(y[m], s[m])
            row[f"caught_{k}"] = caught[t == k].mean() if (t == k).any() else np.nan
        return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    a = ap.parse_args()
    rows = []
    for seed in range(a.seeds):
        df = pd.read_parquet(ROOT / "data" / "processed" / "features" / f"seed_{seed}.parquet")
        tr = df[df.split == "train"]
        te = df[df.split == "test"].sort_values(["booked_at", "booking_id"], kind="mergesort").reset_index(drop=True)
        for tier in ("R", "F"):
            feats = system_features("B2", tier)
            b = lgb.train(PARAMS, lgb.Dataset(tr[feats].to_numpy(float), label=tr.is_fraud.astype(int)), NUM_ROUNDS)
            for r in evaluate(te, b.predict(te[feats].to_numpy(float))):
                rows.append({"seed": seed, "tier": tier, **r})
        print(f"seed {seed} done", flush=True)
    res = pd.DataFrame(rows)
    res.to_json(ROOT / "artifacts" / "results_system.json", orient="records", indent=1)
    write(res, a.seeds)


def _ms(g, c, pct=False):
    m, s = g[c].mean(), g[c].std()
    return f"{100 * m:.1f} ± {100 * s:.1f}%" if pct else f"{m:.3f} ± {s:.3f}"


def write(res: pd.DataFrame, n: int) -> None:
    L = ["# Whole-system results per fraud type", "",
         f"Test window 2018-05-15..2018-08-31, {n} injection seeds, mean ± sd. Real run of scripts/eval_system.py.",
         "Model = B2 LightGBM with the account mix features (fraudshield/features/mix.py). The drop rule and the",
         "first-scan rule use fixed thresholds (drop rule from Hao et al., CCS 2015; first-scan threshold chosen on the",
         "train window with honest scan checks <= 2%). T3 and T5 are never in training. 'Caught' = flagged in the top 1%",
         "of bookings per day by the system score, or sent to a first-scan check, where the depot scale reveals a",
         "T6 parcel's true weight (simulated from the dataset's true weights).", ""]
    for tier, label in (("R", "real columns only"), ("F", "all columns")):
        g = res[res.tier == tier]
        L += [f"## Tier {tier} ({label})", "", "| System | PR-AUC | ROC-AUC | Fraud caught | Honest stopped | Honest scan-checked |",
              "|---|---|---|---|---|---|"]
        for sysname, h in g.groupby("system", sort=False):
            L.append(f"| {sysname} | {_ms(h, 'pr_auc')} | {_ms(h, 'roc_auc')} | {_ms(h, 'caught_all', True)} | "
                     f"{_ms(h, 'honest_stopped', True)} | {_ms(h, 'honest_scan_checked', True)} |")
        L += ["", "PR-AUC per type (held-out marked *):", "", "| System | " + " | ".join(
            k + ("*" if k in ("T3", "T5") else "") for k in TYPES) + " |", "|---" * (len(TYPES) + 1) + "|"]
        for sysname, h in g.groupby("system", sort=False):
            if "first-scan" in sysname:
                continue  # same score as the row above; the scan changes who is caught, not the ranking
            L.append(f"| {sysname} | " + " | ".join(f"{h['pr_' + k].mean():.3f}" for k in TYPES) + " |")
        L += ["", "Share of each type caught (top 1% per day, or first-scan check):", "",
              "| System | " + " | ".join(TYPES) + " |", "|---" * (len(TYPES) + 1) + "|"]
        for sysname, h in g.groupby("system", sort=False):
            L.append(f"| {sysname} | " + " | ".join(f"{100 * h['caught_' + k].mean():.0f}%" for k in TYPES) + " |")
        L.append("")
    (ROOT / "artifacts" / "results_system.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
