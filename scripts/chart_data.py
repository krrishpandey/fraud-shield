"""Compute the data behind docs/figures/*.html from the saved models and results files. CPU only.

Writes docs/figures/chart_data.json. The "app score" is what the decision service uses: the LightGBM score with
the drop-address rule (fraudshield.api.pipeline.DROP_PRIOR, fraudshield.features.mix.rule_flags_frame).
Usage: uv run python scripts/chart_data.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fraudshield.api.pipeline import DROP_PRIOR  # noqa: E402
from fraudshield.data.metrics import flag_top_per_day, pr_auc, roc_auc  # noqa: E402
from fraudshield.features.mix import rule_flags_frame  # noqa: E402
from fraudshield.models.gbm import NUM_ROUNDS, PARAMS, GBMModel, system_features  # noqa: E402

ART, PROC = ROOT / "artifacts", ROOT / "data" / "processed"


def app_score(df, s):
    drop = rule_flags_frame(df).drop_pattern.to_numpy()
    return 1 - (1 - s) * (1 - DROP_PRIOR * drop)


def main() -> None:
    out: dict = {}
    s0 = pd.read_parquet(PROC / "features" / "seed_0.parquet")
    sp = {k: s0[s0.split == k].reset_index(drop=True) for k in ("train", "cal", "test")}
    te = sp["test"]
    y = te.is_fraud.to_numpy()
    # learning curve (seed 0 model, real columns), scored with the first k trees
    m = GBMModel.load(ART / "models", "B2", "R")
    X = {k: m._X(v) for k, v in sp.items()}
    out["learning"] = [{"trees": k, **{s: round(pr_auc(sp[s].is_fraud.to_numpy(), m.booster.predict(X[s], num_iteration=k)), 4)
                                       for s in sp}} for k in (5, 10, 20, 30, 40, 60, 80, 100, 130, 160, 200, 250, 300, 350, 400)]
    # ROC and precision-recall of the app score (seed 0)
    out["roc"], out["prc"] = {}, {}
    for tier in ("R", "F"):
        sc = app_score(te, GBMModel.load(ART / "models", "B2", tier).predict(te))
        fpr, tpr, _ = roc_curve(y, sc)
        grid = np.concatenate([np.linspace(0, 0.05, 51), np.linspace(0.05, 1, 96)[1:]])
        out["roc"][tier] = {"fpr": [round(float(g), 4) for g in grid], "tpr": [round(float(np.interp(g, fpr, tpr)), 4) for g in grid],
                            "auc": round(roc_auc(y, sc), 4)}
        if tier == "R":
            p, r, _ = precision_recall_curve(y, sc)
            pts = [(round(float(t), 2), round(float(p[r >= t - 1e-12].max()), 4)) for t in np.linspace(0, 1, 101)]
            f = flag_top_per_day(te.booked_at, sc, 0.01)
            out["prc"] = {"recall": [a for a, _ in pts], "precision": [b for _, b in pts], "pr_auc": round(pr_auc(y, sc), 4),
                          "op_precision": round(float(y[f].mean()), 4), "op_recall": round(float(f[y].mean()), 4),
                          "prevalence": round(float(y.mean()), 4), "n_test": int(len(y)), "n_fraud": int(y.sum())}
    # model comparison (results_gbm.json) and whole system (results_system.json)
    g = pd.read_json(ART / "results_gbm.json")
    cols = ["pr_auc", "prec_top1pct_day", "rec_top1pct_day", "campaign_recall", "fpr_legit"] + [f"pr_auc_T{i}" for i in range(1, 8)]
    out["summary"] = {f"{a}_{b}": {c: [round(float(d[c].mean()), 4), round(float(d[c].std()), 4)] for c in cols}
                      for (a, b), d in g.groupby(["system", "tier"])}
    out["n_seeds"] = int(g.seed.nunique())
    out["prevalence_mean"] = round(float(g.prevalence.mean()), 4)
    sy = pd.read_json(ART / "results_system.json")
    full = "model + drop rule + first-scan check + depot follow-up"
    for tier in ("R", "F"):
        d = sy[(sy.tier == tier) & (sy.system == "model + drop rule")]
        out["summary"][f"APP_{tier}"] = {"pr_auc": [round(float(d.pr_auc.mean()), 4), round(float(d.pr_auc.std()), 4)]}
    types = ("T1", "T2", "T3", "T4", "T5", "T6", "T7")
    r = sy[sy.tier == "R"]
    out["caught_by_type"] = {k: {"model": round(float(r[r.system == "model only"][f"caught_{k}"].mean()), 4),
                                 "system": round(float(r[r.system == full][f"caught_{k}"].mean()), 4)} for k in types}
    out["system_r"] = {"caught_all": round(float(r[r.system == full].caught_all.mean()), 4),
                       "honest_stopped": round(float(r[r.system == full].honest_stopped.mean()), 4),
                       "honest_scan": round(float(r[r.system == full].honest_scan_checked.mean()), 4)}
    # headline ranges over 10 seeds (real columns, app score)
    rows = []
    feats = system_features("B2", "R")
    for seed in range(out["n_seeds"]):
        df = pd.read_parquet(PROC / "features" / f"seed_{seed}.parquet")
        tr, cal, t = (df[df.split == k].reset_index(drop=True) for k in ("train", "cal", "test"))
        b = lgb.train(PARAMS, lgb.Dataset(tr[feats].to_numpy(float), label=tr.is_fraud.astype(int)), NUM_ROUNDS)
        s = app_score(t, b.predict(t[feats].to_numpy(float)))
        sc = app_score(cal, b.predict(cal[feats].to_numpy(float)))
        p, rc, th = precision_recall_curve(cal.is_fraud.to_numpy(), sc)
        ok = np.where(p[:-1] >= 0.90)[0]
        thr = th[ok[0]] if len(ok) else 0.5
        yy, typ = t.is_fraud.to_numpy(), t.typology.astype(str).to_numpy()
        f = flag_top_per_day(t.booked_at, s, 0.01)
        conf = s >= thr
        vis = ~(yy & np.isin(typ, ["T3", "T5", "T6"]))
        rows.append({"ROC-AUC": roc_auc(yy, s), "Precision when confident": yy[conf].mean() if conf.any() else np.nan,
                     "Honest shippers untouched": (~f[~yy]).mean(), "PR-AUC, fraud visible at booking": pr_auc(yy[vis], s[vis]),
                     "Fraud campaigns caught": t[yy].assign(c=f[yy]).groupby("campaign_id").c.any().mean(),
                     "PR-AUC, all fraud": pr_auc(yy, s)})
        print("seed", seed, flush=True)
    rr = pd.DataFrame(rows)
    strong = {"ROC-AUC", "Precision when confident", "Honest shippers untouched"}
    out["ranges"] = [{"label": c, "mean": round(100 * rr[c].mean(), 1), "low": round(100 * rr[c].min(), 1),
                      "high": round(100 * rr[c].max(), 1), "strong": c in strong} for c in rr.columns]
    # live stream (artifacts/stream_consistency.md, stream_load.md)
    txt = (ART / "stream_consistency.md").read_text(encoding="utf-8")
    vals = {}
    for line in txt.splitlines():
        if line.startswith("| ") and not line.startswith("| Metric") and not line.startswith("|---"):
            c = [x.strip() for x in line.strip("|").split("|")]
            vals[c[0]] = (float(c[1]), float(c[2]))
    out["live_vs_offline"] = [{"label": k.replace(" per day", "/day"), "live": v[0], "offline": v[1]} for k, v in vals.items()]
    load = []
    for line in (ART / "stream_load.md").read_text(encoding="utf-8").splitlines():
        c = [x.strip() for x in line.strip("|").split("|")]
        if line.startswith("| ") and c[0].replace(".", "").isdigit():
            load.append({"target": float(c[0]), "achieved": float(c[2]), "p99": float(c[6]), "errors": int(c[3])})
    out["load"] = load
    (ROOT / "docs" / "figures" / "chart_data.json").write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("caught_by_type", "system_r")}, indent=1))
    print("ranges", [(x["label"], x["mean"], x["low"], x["high"]) for x in out["ranges"]])
    print("roc", out["roc"]["R"]["auc"], out["roc"]["F"]["auc"], "prc", out["prc"]["pr_auc"], out["prc"]["op_precision"], out["prc"]["op_recall"])


if __name__ == "__main__":
    main()
