"""Too-easy checks (DESIGN section 10). Writes artifacts/sanity_checks.md and .json.

1. Rules-only (B0) PR-AUC at camouflage 2 must be < 0.8 (read from artifacts/results_gbm.json, a real run).
2. Artifact classifier: injected-legit rows vs real rows on columns with a real counterpart, AUC <= 0.6.
   Also all injected rows vs real on non-semantic columns only (formatting, residuals, lags), AUC <= 0.6.
3. Shuffled-label run: B2 tier F trained on shuffled labels must fall to PR-AUC ~ prevalence.
Usage: uv run python scripts/sanity_checks.py [--seeds 10]
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

from fraudshield.data.inject import InjectionContext  # noqa: E402
from fraudshield.data.metrics import pr_auc  # noqa: E402
from fraudshield.data.pipeline import real_with_billing  # noqa: E402
from fraudshield.data.sanity import artifact_auc, artifact_frame  # noqa: E402,F401
from fraudshield.models.gbm import GBMModel  # noqa: E402

PROC, ART = ROOT / "data" / "processed", ROOT / "artifacts"
NON_SEMANTIC = ["second", "cost_cents", "freight_residual", "scan_lag_h", "scan_missing"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    a = ap.parse_args()
    out: dict = {}
    res = pd.DataFrame(json.loads((ART / "results_gbm.json").read_text()))
    b0 = res[res.system == "B0"].groupby("tier").pr_auc_cam2.agg(["mean", "std"])
    out["b0_pr_auc_cam2"] = {t: {"mean": float(r["mean"]), "sd": float(r["std"])} for t, r in b0.iterrows()}
    out["check1_pass"] = bool((b0["mean"] < 0.8).all())

    real, _, _, pool = real_with_billing()
    ctx = InjectionContext(real, pool)
    med = {k: float(np.median(v)) if len(v) else np.nan for k, v in ctx.fine.items()}
    iqr = {k: float(np.subtract(*np.percentile(v, [75, 25]))) if len(v) else 1.0 for k, v in ctx.fine.items()}

    def frame(d):
        wb = np.digitize(d.weight_kg.to_numpy(float), ctx.fw_edges)
        db = np.digitize(d.distance_km.fillna(0).to_numpy(float), ctx.fd_edges)
        keys = list(zip(wb, db))
        return artifact_frame(d, pred=np.array([med[k] for k in keys]), scale=np.array([iqr[k] for k in keys]))

    inj = pd.concat([pd.read_parquet(PROC / "features" / f"injected_seed_{s}.parquet") for s in range(a.seeds)],
                    ignore_index=True)
    rs = real[real.split.isin(["train", "cal", "test"])]
    rng = np.random.default_rng(0)

    def check(side, cols, name):
        smp = rs.iloc[rng.choice(len(rs), size=min(len(rs), 5 * len(side)), replace=False)]
        fa, fb = frame(side)[cols], frame(smp)[cols]
        out[f"{name}_plain"] = artifact_auc(fa, fb)
        out[f"{name}_grouped"] = artifact_auc(fa, fb, groups_a=side.campaign_id.to_numpy(), groups_b=smp.account_id.to_numpy())
        out[f"{name}_per_column_grouped"] = {c: round(artifact_auc(fa[[c]], fb[[c]], folds=3, groups_a=side.campaign_id.to_numpy(),
                                                                   groups_b=smp.account_id.to_numpy()), 3) for c in cols}

    allcols = list(frame(inj.head(5)).columns)
    check(inj[~inj.is_fraud], allcols, "artifact_injected_legit")
    check(inj, NON_SEMANTIC, "artifact_all_injected_non_semantic")
    # placebo: REAL campaign-like groups (contiguous runs of ~25 bookings of one real account) vs random real rows.
    # It shows the floor of this test when the positive side is clustered by account, with no injection at all.
    hn_n = int((~inj.is_fraud).sum())
    runs, gid = [], []
    accs = rs.groupby("account_id").indices
    big = [a_ for a_, ix in accs.items() if len(ix) >= 30]
    while sum(len(r) for r in runs) < hn_n:
        a_ = big[rng.integers(len(big))]
        ix = accs[a_]
        st = rng.integers(0, len(ix) - 25)
        runs.append(ix[st:st + 25])
        gid += [f"plc{len(runs)}"] * 25
    plc = rs.iloc[np.concatenate(runs)].assign(campaign_id=gid)
    smp = rs[~rs.index.isin(plc.index)]
    smp = smp.iloc[rng.choice(len(smp), size=5 * len(plc), replace=False)]
    out["placebo_real_groups_grouped"] = artifact_auc(frame(plc)[allcols], frame(smp)[allcols],
                                                      groups_a=plc.campaign_id.to_numpy(), groups_b=smp.account_id.to_numpy())
    fr = inj[inj.is_fraud]
    smp = rs.iloc[rng.choice(len(rs), size=min(len(rs), 5 * len(fr)), replace=False)]
    out["info_auc_fraud_vs_real_all_columns_grouped"] = artifact_auc(frame(fr), frame(smp), groups_a=fr.campaign_id.to_numpy(),
                                                                     groups_b=smp.account_id.to_numpy())
    out["check2_pass_grouped"] = bool(out["artifact_injected_legit_grouped"] <= 0.6 and out["artifact_all_injected_non_semantic_grouped"] <= 0.6)
    out["check2_pass_plain"] = bool(out["artifact_injected_legit_plain"] <= 0.6 and out["artifact_all_injected_non_semantic_plain"] <= 0.6)

    df = pd.read_parquet(PROC / "features" / "seed_0.parquet")
    tr, cal, te = (df[df.split == s].reset_index(drop=True) for s in ("train", "cal", "test"))
    shuf = []
    for k in range(5):
        m = GBMModel("B2", "F").fit(tr, rng.permutation(tr.is_fraud.to_numpy()), cal, rng.permutation(cal.is_fraud.to_numpy()))
        shuf.append(pr_auc(te.is_fraud, m.predict(te)))
    out["shuffled_label_pr_auc_runs"] = shuf
    out["shuffled_label_pr_auc"] = float(np.mean(shuf))
    out["test_prevalence"] = float(te.is_fraud.mean())
    out["check3_pass"] = bool(out["shuffled_label_pr_auc"] < 3 * out["test_prevalence"])

    (ART / "sanity_checks.json").write_text(json.dumps(out, indent=2))
    L = ["# Too-easy checks (real run of scripts/sanity_checks.py)", "",
         f"1. Rules-only B0 PR-AUC at camouflage 2 (target < 0.8): tier R {out['b0_pr_auc_cam2']['R']['mean']:.3f}, "
         f"tier F {out['b0_pr_auc_cam2']['F']['mean']:.3f} -> {'PASS' if out['check1_pass'] else 'FAIL'}",
         f"2. Artifact classifier (target AUC <= 0.6). Folds grouped by campaign (injected) / account (real), so the "
         f"classifier cannot win by memorizing one campaign's repeated values; plain row-level CV shown too.",
         f"   a) injected-legit vs real, all columns with a real counterpart: grouped {out['artifact_injected_legit_grouped']:.3f}, "
         f"plain {out['artifact_injected_legit_plain']:.3f}",
         f"   b) all injected vs real, non-semantic columns only: grouped {out['artifact_all_injected_non_semantic_grouped']:.3f}, "
         f"plain {out['artifact_all_injected_non_semantic_plain']:.3f}",
         f"   -> grouped {'PASS' if out['check2_pass_grouped'] else 'FAIL'}, plain {'PASS' if out['check2_pass_plain'] else 'FAIL'}",
         "   Per-column grouped AUC (a): " + ", ".join(f"{k} {v}" for k, v in sorted(out["artifact_injected_legit_per_column_grouped"].items(), key=lambda kv: -kv[1])),
         "   Per-column grouped AUC (b): " + ", ".join(f"{k} {v}" for k, v in sorted(out["artifact_all_injected_non_semantic_per_column_grouped"].items(), key=lambda kv: -kv[1])),
         f"   Placebo (no injection): real campaign-like groups of 25 bookings from one account vs random real rows, grouped AUC "
         f"{out['placebo_real_groups_grouped']:.3f}. This is the floor of the test for clustered data.",
         f"   For information (not a check, fraud is meant to differ): injected fraud vs real, all columns, grouped AUC "
         f"{out['info_auc_fraud_vs_real_all_columns_grouped']:.3f}",
         f"3. Shuffled-label B2 tier F PR-AUC, mean of 5 shuffles {out['shuffled_label_pr_auc']:.4f} "
         f"(runs {', '.join(f'{x:.4f}' for x in out['shuffled_label_pr_auc_runs'])}) vs prevalence {out['test_prevalence']:.4f} "
         f"-> {'PASS' if out['check3_pass'] else 'FAIL'}"]
    (ART / "sanity_checks.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
