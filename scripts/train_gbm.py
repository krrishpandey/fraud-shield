"""Train B0/B1/B2 x tier R/F per injection seed, evaluate on the test window, write
artifacts/models/ (seed 0 models), artifacts/results_gbm.md and artifacts/results_gbm.json.

Usage: uv run python scripts/train_gbm.py [--seeds 10]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fraudshield.data.inject import FRAUD_TYPOLOGIES, HELD_OUT  # noqa: E402
from fraudshield.data.metrics import flag_top_per_day, pr_auc, precision_recall_at_top_per_day  # noqa: E402
from fraudshield.models.gbm import GBMModel, rules_score  # noqa: E402

PROC = ROOT / "data" / "processed"
ART = ROOT / "artifacts"
HN = ("hn_new_state", "hn_new_seller", "hn_billing_change", "hn_injected", "hn_multi_account_consignee")
SYSTEMS = [("B0", "R"), ("B0", "F"), ("B1", "R"), ("B1", "F"), ("B2", "R"), ("B2", "F")]


def evaluate(te: pd.DataFrame, s: np.ndarray) -> dict:
    y = te.is_fraud.to_numpy()
    legit = ~y
    flags = flag_top_per_day(te.booked_at, s, 0.01)
    p, r = precision_recall_at_top_per_day(te.booked_at, s, y, 0.01)
    m = {"pr_auc": pr_auc(y, s), "prevalence": float(y.mean()), "prec_top1pct_day": p, "rec_top1pct_day": r,
         "fpr_legit": float(flags[legit].mean())}
    for h in HN:
        mask = te[h].to_numpy() & legit
        m[f"fpr_{h}"] = float(flags[mask].mean()) if mask.any() else float("nan")
        m[f"n_{h}"] = int(mask.sum())
    for t in FRAUD_TYPOLOGIES:
        mask = legit | (te.typology.to_numpy() == t)
        m[f"pr_auc_{t}"] = pr_auc(y[mask], s[mask])
        tm = te.typology.to_numpy() == t
        m[f"rec_{t}"] = float(flags[tm].mean()) if tm.any() else float("nan")
    for c in (0, 1, 2):
        mask = legit | (te.camouflage_level.to_numpy() == c)
        m[f"pr_auc_cam{c}"] = pr_auc(y[mask], s[mask])
    camp = te[y].assign(f=flags[y]).groupby("campaign_id").f.any()
    m["campaign_recall"] = float(camp.mean())
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    a = ap.parse_args()
    t0 = time.time()
    per_seed = []
    for seed in range(a.seeds):
        df = pd.read_parquet(PROC / "features" / f"seed_{seed}.parquet")
        tr, cal, te = (df[df.split == s].reset_index(drop=True) for s in ("train", "cal", "test"))
        for system, tier in SYSTEMS:
            if system == "B0":
                s = rules_score(te, tier)
            else:
                m = GBMModel(system, tier).fit(tr, tr.is_fraud, cal, cal.is_fraud)
                s = m.predict(te)
                if seed == 0:
                    m.save(ART / "models", extra={"seed": 0, "test_metrics": evaluate(te, s)})
                    if (system, tier) == ("B2", "F"):
                        all_s = m.predict(df)
                        df[["booking_id", "split"]].assign(gbm_b2f=all_s, gbm_b2r=GBMModel.load(ART / "models", "B2", "R").predict(df)
                                                           if (ART / "models" / "B2_R.lgb").exists() else np.nan
                                                           ).to_parquet(PROC / "gbm_scores_seed0.parquet", index=False)
            per_seed.append({"seed": seed, "system": system, "tier": tier, **evaluate(te, s)})
        print(f"seed {seed} done ({time.time() - t0:.0f}s)", flush=True)
    res = pd.DataFrame(per_seed)
    res.to_json(ART / "results_gbm.json", orient="records", indent=1)
    write_markdown(res, a.seeds)
    print((ART / "results_gbm.md").read_text())


def _ms(g: pd.Series, pct: bool = False) -> str:
    v = g.dropna()
    if v.empty:
        return "n/a"
    k = 100 if pct else 1
    return f"{v.mean() * k:.{1 if pct else 3}f} ± {v.std(ddof=1) * k:.{1 if pct else 3}f}" if len(v) > 1 else f"{v.mean() * k:.3f}"


def write_markdown(res: pd.DataFrame, n_seeds: int) -> None:
    g = res.groupby(["system", "tier"], sort=False)
    prev = res.prevalence.mean()
    L = [f"# GBM baseline results (test window 2018-05-15..2018-08-31, {n_seeds} injection seeds)", "",
         f"Real run of scripts/train_gbm.py. Values are mean ± sd across seeds. Test prevalence {prev * 100:.2f}% of bookings "
         f"(injected fraud on real Olist histories; real rows assumed legit). Operating point for precision, recall and FPR: "
         f"flag the top 1% of bookings per day. FPR legit = flagged legit / all legit bookings; hard-negative FPR = flagged / "
         f"bookings in that slice. T3 and T5 are HELD OUT (never in train or calibration). Tier R = real columns only; "
         f"tier F adds the synthetic billing/login layer.", "",
         "## Headline", "",
         "| System | Tier | PR-AUC | Prec@1%/day | Rec@1%/day | Campaign recall | FPR legit (%) |",
         "|---|---|---|---|---|---|---|"]
    for (s, t), x in g:
        L.append(f"| {s} | {t} | {_ms(x.pr_auc)} | {_ms(x.prec_top1pct_day)} | {_ms(x.rec_top1pct_day)} | "
                 f"{_ms(x.campaign_recall)} | {_ms(x.fpr_legit, True)} |")
    L += ["", "## FPR on hard-negative slices (% flagged at top 1%/day)", "",
          "| System | Tier | " + " | ".join(h.removeprefix("hn_") for h in HN) + " |", "|---|---|" + "---|" * len(HN)]
    for (s, t), x in g:
        L.append(f"| {s} | {t} | " + " | ".join(_ms(x[f"fpr_{h}"], True) for h in HN) + " |")
    L.append("")
    L.append("Slice sizes (test, seed 0): " + ", ".join(f"{h.removeprefix('hn_')} {int(res[f'n_{h}'].iloc[0]):,}" for h in HN))
    L += ["", "## PR-AUC per typology (typology vs all legit; held-out marked *)", "",
          "| System | Tier | " + " | ".join(t + ("*" if t in HELD_OUT else "") for t in FRAUD_TYPOLOGIES) + " |",
          "|---|---|" + "---|" * len(FRAUD_TYPOLOGIES)]
    for (s, t), x in g:
        L.append(f"| {s} | {t} | " + " | ".join(_ms(x[f"pr_auc_{ty}"]) for ty in FRAUD_TYPOLOGIES) + " |")
    L += ["", "## Recall at top 1%/day per typology", "",
          "| System | Tier | " + " | ".join(t + ("*" if t in HELD_OUT else "") for t in FRAUD_TYPOLOGIES) + " |",
          "|---|---|" + "---|" * len(FRAUD_TYPOLOGIES)]
    for (s, t), x in g:
        L.append(f"| {s} | {t} | " + " | ".join(_ms(x[f"rec_{ty}"]) for ty in FRAUD_TYPOLOGIES) + " |")
    L += ["", "## PR-AUC per camouflage level", "", "| System | Tier | cam 0 | cam 1 | cam 2 |", "|---|---|---|---|---|"]
    for (s, t), x in g:
        L.append(f"| {s} | {t} | {_ms(x.pr_auc_cam0)} | {_ms(x.pr_auc_cam1)} | {_ms(x.pr_auc_cam2)} |")
    L += ["", "Notes: B0 is a fixed rule sum (ties are common, which lowers its PR-AUC). GBM calibrated with Platt scaling on "
          "the calibration window 2018-02-15..2018-04-30. Models saved for seed 0 in artifacts/models/."]
    (ART / "results_gbm.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
