"""Evaluate Laya v2 ("Laya decides") against LightGBM on the same rows, apply the pre-registered win
condition (docs/LAYA_V2.md), and fit the v2 calibration. CPU only, a few minutes.

Reads artifacts/laya_v2/{evalset_*.parquet, scores_ft[_nogbm|_ep1]_*.parquet}, artifacts/laya/scores_ft_test.parquet (v1).
Writes artifacts/laya_v2/calibration_v2.json, artifacts/results_laya_v2.md, artifacts/results_laya_v2.json.
Usage: uv run python scripts/laya_v2_eval.py [--tag _ep1]   (--tag scores an epoch-1 checkpoint's files instead)

Cost rule used for every system (so the comparison is like for like): the drop-rule prior of the live pipeline
(p -> 1 - (1 - p)(1 - 0.5) on a drop-rule hit), argmin expected cost over all six actions with config/costs.yaml,
block capped to hold (no hard-signal bookkeeping offline). Realized cost = the booking's action-cost vector
(fraudshield/data/labels.py). The conformal allow guard is reported separately, not applied here.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from fraudshield.contracts import ACTIONS  # noqa: E402
from fraudshield.models import calibrate_fit as C  # noqa: E402
from fraudshield.models.laya_train import metrics as M  # noqa: E402
from fraudshield.models.laya_train.data import seed_view  # noqa: E402
from fraudshield.policy.costs import expected_costs, load_costs, mode_weights  # noqa: E402
from laya_eval import HELD, SLICES, TYP, ci, f3, ms, system_metrics  # noqa: E402

ART = ROOT / "artifacts"
LA = Path(os.environ.get("FS_LAYA_ART", ART / "laya_v2"))
OUT = Path(os.environ.get("FS_LAYA_OUT", ART))
DROP_PRIOR = 0.5
AUDIT_MARGIN_BRL = 2.0   # the auditor overrules Laya's action only if it costs > 2 BRL more in expectation
B = 500
N_SEEDS = 10


def _flag(states: pd.Series, name: str) -> np.ndarray:
    return states.str.contains(f"{name} rule yes", regex=False).to_numpy()


def load(name: str, tag: str) -> pd.DataFrame:
    s = pd.read_parquet(LA / f"evalset_{name}.parquet")
    lab = s.labels.map(json.loads)
    for q in ("misuse", "foreign_senders", "payoff_max", "drop_consignee"):
        s[f"y_{q}"] = lab.map(lambda l, q=q: int(l[q]))
    s["hn"] = s.hard_negative.map(json.loads)
    s["cluster"] = np.where(s.y_misuse == 1, s.seed.astype(str) + ":" + s.campaign_id.astype(str), "legit")
    s["drop"] = _flag(s.state, "drop")
    s["under"] = _flag(s.state, "under-declared")
    for prefix, f in (("m2", f"scores_ft{tag}_{name}.parquet"), ("solo", f"scores_ft_nogbm_{name}.parquet")):
        p = LA / f
        if p.exists():
            sc = pd.read_parquet(p).set_index("id")
            sc.columns = [f"{prefix}_{c[2:]}" for c in sc.columns]
            s = s.join(sc, on="id")
    v1 = ART / "laya" / f"scores_ft_{name}.parquet"
    if v1.exists():
        s = s.join(pd.read_parquet(v1).set_index("id")[["p_misuse"]].rename(columns={"p_misuse": "m1_misuse"}), on="id")
    return s


# ---------------------------------------------------------------- calibration (v2 misuse) and the S0 stack
def fit_v2_calibration(cal: pd.DataFrame) -> dict:
    from laya_eval import account_split  # noqa: PLC0415
    c1 = account_split(cal.account_id)
    w, y = cal.weight.to_numpy(), cal.y_misuse.to_numpy()
    a, b = C.fit_platt(cal.m2_misuse.to_numpy()[c1], y[c1], w[c1])
    pc = C.platt(cal.m2_misuse.to_numpy(), a, b)
    c2f = (~c1) & (y == 1)
    lam = C.conformal_lambda(pc[c2f], 0.05)
    temps = {"misuse": 1.0, "drop_consignee": 1.0}
    for q in ("foreign_senders", "payoff_max"):
        p = cal[f"m2_{q}"].to_numpy()
        yy = cal[f"y_{q}"].to_numpy()
        temps[q] = C.fit_temperature(np.stack([p, 1 - p], 1)[c1], np.stack([yy, 1 - yy], 1)[c1], w[c1])
    prev = float((w * y).sum() / w.sum())
    ck = LA / "fraudshield-laya" / "model.safetensors"
    info = {"window": "calibration split, seed 0", "split": "by account hash, 60/40", "c1_fraud": int((c1 & (y == 1)).sum()),
            "c2_fraud": int(c2f.sum()), "serializer": "ser-v2", "platt_on": "raw misuse (temperature 1)"}
    if ck.exists():
        return C.calibration_json(ck, temps, {"misuse": {"a": a, "b": b}}, lam, 0.05, int(c2f.sum()), prev, extra=info)
    return {"version": "cal-v2-nockpt", "temperatures": temps, "platt": {"misuse": {"a": a, "b": b}},
            "conformal": {"lambda_allow": lam, "alpha": 0.05, "n_fraud": int(c2f.sum())}, "prevalence": prev, "fit": info}


def fit_stack(cal: pd.DataFrame):
    """S0: logistic regression on (LightGBM logit, drop rule, under-declared rule), fitted on the calibration set."""
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415

    def X(d):
        g = np.clip(d.gbm.to_numpy(float), 1e-6, 1 - 1e-6)
        return np.column_stack([np.log(g / (1 - g)), d["drop"].astype(float), d["under"].astype(float)])
    lr = LogisticRegression(C=1.0, max_iter=1000).fit(X(cal), cal.y_misuse, sample_weight=cal.weight)
    return lambda d: lr.predict_proba(X(d))[:, 1]


# ---------------------------------------------------------------- decisions and realized cost
def decide(p: np.ndarray, d: pd.DataFrame, params: dict, modes: pd.DataFrame | None) -> np.ndarray:
    p = np.where(d["drop"].to_numpy(), 1 - (1 - p) * (1 - DROP_PRIOR), p)
    out = []
    for i, (pi, F) in enumerate(zip(p, d.carrier_cost.to_numpy(float))):
        mw = mode_weights(modes.iloc[i].to_dict()) if modes is not None else None
        ec = expected_costs(float(pi), F, params, modes=mw)
        a = min(ACTIONS, key=lambda k: ec[k])
        out.append("hold" if a == "block" else a)
    return np.array(out)


def audited(laya_action: np.ndarray, p: np.ndarray, d: pd.DataFrame, params: dict, modes: pd.DataFrame):
    """Laya proposes; the cost rule overrules only an action that is > AUDIT_MARGIN_BRL costlier in expectation."""
    p = np.where(d["drop"].to_numpy(), 1 - (1 - p) * (1 - DROP_PRIOR), p)
    final, over = [], []
    for i, (pi, F) in enumerate(zip(p, d.carrier_cost.to_numpy(float))):
        ec = expected_costs(float(pi), F, params, modes=mode_weights(modes.iloc[i].to_dict()))
        best = min(ACTIONS, key=lambda k: ec[k])
        la = "hold" if laya_action[i] == "block" else laya_action[i]
        o = ec[la] - ec[best] > AUDIT_MARGIN_BRL
        final.append(("hold" if best == "block" else best) if o else la)
        over.append(o)
    return np.array(final), np.array(over)


def realized(actions: np.ndarray, d: pd.DataFrame) -> np.ndarray:
    return np.array([json.loads(c)[a] for c, a in zip(d.action_cost, actions)])


def pooled_weights(d: pd.DataFrame) -> np.ndarray:
    """Seed views share the real legit rows; pooled, legit counts N_SEEDS times so the mix matches a seed view."""
    return np.where(d.is_injected.astype(bool), 1.0, N_SEEDS) * d.weight.to_numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="", help="_ep1 to evaluate the epoch-1 checkpoint's score files")
    a = ap.parse_args()
    test, cal = load("test", a.tag), load("cal", a.tag)
    assert "m2_misuse" in test and "m2_misuse" in cal, f"v2 score files missing in {LA}"
    params = load_costs(ROOT / "config" / "costs.yaml").params
    doc = fit_v2_calibration(cal)
    (LA / f"calibration_v2{a.tag}.json").write_text(json.dumps(doc, indent=2))
    pl = doc["platt"]["misuse"]
    test["m2_cal"] = C.platt(test.m2_misuse.to_numpy(), pl["a"], pl["b"])
    test["s0"] = fit_stack(cal)(test)

    systems = {"B2 LightGBM (current decider)": "gbm", "S0 3-input stack (control)": "s0"}
    if "m1_misuse" in test:
        systems["M1 Laya v1"] = "m1_misuse"
    systems["M2 Laya v2 decider"] = "m2_misuse"
    if "solo_misuse" in test:
        systems["M2-solo (LightGBM line withheld)"] = "solo_misuse"
    per = {k: [] for k in systems}
    for sd in range(N_SEEDS):
        v = seed_view(test, sd)
        for name, col in systems.items():
            per[name].append(system_metrics(v, v[col].to_numpy()))
    mean = {n: float(np.mean([x["pr_auc"] for x in per[n]])) for n in systems}

    # held-out typologies at equal legit friction, pooled seeds, campaign bootstrap
    y = test.y_misuse.to_numpy().astype(bool)
    w = test.weight.to_numpy()
    cl = test.cluster.to_numpy()
    held = []
    for t in HELD:
        mask = (~y) | (test.typology.to_numpy() == t)
        yy, ww, cc = y[mask], w[mask], cl[mask]
        g, m = test.gbm.to_numpy()[mask], test.m2_misuse.to_numpy()[mask]
        for fpr in (0.005, 0.01):
            row = {"typology": t, "fpr": fpr, "n": int(yy.sum()), "campaigns": int(len(set(cc[yy])))}
            for col, lab in (("gbm", "B2"), ("m2_misuse", "M2"), ("solo_misuse", "solo"), ("s0", "S0")):
                if col in test:
                    row[lab] = M.recall_at_fpr(yy, test[col].to_numpy()[mask], ww, fpr)
            row["M2_minus_B2_ci"] = M.bootstrap_ci(
                lambda i: M.recall_at_fpr(yy[i], m[i], ww[i], fpr) - M.recall_at_fpr(yy[i], g[i], ww[i], fpr),
                yy, cc, B=B, seed=2)
            held.append(row)

    # decisions: cost per booking
    pw = pooled_weights(test)
    modes = test[["m2_foreign_senders", "m2_payoff_max", "m2_drop_consignee"]].rename(columns=lambda c: c[3:])
    acts = {"B2 + cost rule": decide(test.gbm.to_numpy(), test, params, None),
            "M2 probabilities + cost rule": decide(test.m2_cal.to_numpy(), test, params, modes)}
    act_cols = [f"m2_act_{k}" for k in ACTIONS]
    overrule = None
    if all(c in test for c in act_cols):
        head = np.array(ACTIONS)[test[act_cols].to_numpy().argmax(1)]
        acts["M2 action head (Laya alone)"] = np.where(head == "block", "hold", head)
        acts["M2 action head + cost auditor"], overrule = audited(head, test.m2_cal.to_numpy(), test, params, modes)
    cost = {k: realized(v, test) for k, v in acts.items()}
    cost_rows = {}
    for k, c in cost.items():
        seeds = []
        for sd in range(N_SEEDS):
            m_ = (~test.is_injected.astype(bool) | (test.seed == sd)).to_numpy()
            seeds.append(float((w[m_] * c[m_]).sum() / w[m_].sum()))
        cost_rows[k] = {"brl_per_booking": seeds, "fraud_leak_brl": float((pw * c * y).sum() / pw.sum()),
                        "legit_friction_brl": float((pw * c * ~y).sum() / pw.sum()),
                        "actions": pd.Series(acts[k]).value_counts().to_dict()}
    cb, cm = cost["B2 + cost rule"], cost["M2 probabilities + cost rule"]
    saved_ci = M.bootstrap_ci(lambda i: float((pw[i] * (cb[i] - cm[i])).sum() / pw[i].sum()), y, cl, B=B, seed=7)
    saved = float((pw * (cb - cm)).sum() / pw.sum())

    # zero-shot drop_consignee on T3
    mask = (~y) | (test.typology.to_numpy() == "T3")
    zs = {lab: M.roc_auc_w(y[mask], test[col].to_numpy()[mask], w[mask])
          for col, lab in (("m2_drop_consignee", "M2"), ("solo_drop_consignee", "solo"), ("gbm", "B2_general"))
          if col in test}

    # pre-registered verdict
    w1 = mean["M2 Laya v2 decider"] >= mean["B2 LightGBM (current decider)"] - 0.02
    w2 = {f"{r['typology']}_recall_fpr1": r["M2_minus_B2_ci"][0] > 0 for r in held if r["fpr"] == 0.01}
    w2["cost_saved"] = saved_ci[0] > 0
    decision = ("Laya decides (W1 and W2 met)" if w1 and any(w2.values()) else
                "LightGBM keeps deciding; Laya matches it and adds named fraud modes (W1 met, W2 not met)" if w1 else
                "LightGBM keeps deciding (W1 not met)")
    res = {"tag": a.tag, "pr_auc_mean": mean, "per_seed": per, "held_out": held, "cost": cost_rows,
           "cost_saved_brl_per_booking": saved, "cost_saved_ci": saved_ci, "zero_shot_drop_t3_auc": zs,
           "auditor_overrule_rate": float((pw * overrule).sum() / pw.sum()) if overrule is not None else None,
           "calibration": doc, "W1": bool(w1), "W2": {k: bool(v) for k, v in w2.items()}, "decision": decision}
    (OUT / f"results_laya_v2{a.tag}.json").write_text(json.dumps(res, indent=2, default=float))
    write_md(res, systems, a.tag)
    print(json.dumps({"pr_auc_mean": mean, "W1": w1, "W2": w2, "cost_saved": [saved, saved_ci],
                      "decision": decision}, indent=1, default=float))


def write_md(res: dict, systems: dict, tag: str) -> None:
    prep = json.loads((LA / "prepare.json").read_text())
    per = res["per_seed"]
    L = [f"# Laya v2 results: Laya decides, LightGBM testifies{' (epoch-1 checkpoint)' if tag else ''}", "",
         "Design and the pre-registered win condition: `docs/LAYA_V2.md` (written before training). Same test rows as "
         "v1 (`results_laya.md`): seed-0 real Olist legit (5,000 sampled + all real hard negatives, weighted) plus the "
         "injected rows of 10 seeds. T3 and T5 never appear in training or calibration.", "",
         f"Training: {prep['bookings']} bookings, {prep['fraud_unique']} distinct fraud from {prep['injection_seeds']} seeds "
         f"(v1: 386), LightGBM line withheld on {prep['gbm_withheld_share']:.0%} of states, out-of-fold LightGBM risk on "
         f"training states, questions {', '.join(prep['questions'])}.", "",
         "## Verdict (pre-registered)", "",
         f"- W1, no loss in ranking (M2 PR-AUC >= B2 - 0.02): **{'met' if res['W1'] else 'not met'}** "
         f"({res['pr_auc_mean']['M2 Laya v2 decider']:.3f} vs {res['pr_auc_mean']['B2 LightGBM (current decider)']:.3f})",
         "- W2, adds something (95% CI of M2 - B2 above 0): " + ", ".join(f"{k} {'met' if v else 'not met'}" for k, v in res["W2"].items()),
         f"- **Decision: {res['decision']}**", "",
         "## Ranking (mean ± sd over 10 seeds; top 1% of bookings per day)", "",
         "| System | PR-AUC | Prec@1%/day | Rec@1%/day | FPR legit (%) | " + " | ".join(f"FPR {s} (%)" for s in SLICES) + " |",
         "|---|---|---|---|---|" + "---|" * len(SLICES)]
    for n in systems:
        p = per[n]
        L.append(f"| {n} | {ms([x['pr_auc'] for x in p])} | {ms([x['prec_top1pct_day'] for x in p])} | "
                 f"{ms([x['rec_top1pct_day'] for x in p])} | {ms([100 * x['fpr_legit'] for x in p])} | "
                 + " | ".join(ms([100 * x[f'fpr_{s}'] for x in p]) for s in SLICES) + " |")
    L += ["", "PR-AUC per typology (held-out marked *):", "",
          "| System | " + " | ".join(t + ("*" if t in HELD else "") for t in TYP) + " |", "|---|" + "---|" * len(TYP)]
    for n in systems:
        L.append(f"| {n} | " + " | ".join(ms([x[f'pr_auc_{t}'] for x in per[n]]) for t in TYP) + " |")
    L += ["", "## Held-out fraud types at equal legit friction (pooled seeds, 95% campaign-bootstrap CI)", "",
          "| Type | Legit FPR | n / campaigns | B2 | M2 | M2-solo | S0 | M2 - B2 CI |", "|---|---|---|---|---|---|---|---|"]
    for r in res["held_out"]:
        L.append(f"| {r['typology']} | {r['fpr']:.1%} | {r['n']} / {r['campaigns']} | {f3(r.get('B2'))} | {f3(r.get('M2'))} | "
                 f"{f3(r.get('solo'))} | {f3(r.get('S0'))} | {ci(r['M2_minus_B2_ci'])} |")
    L += ["", "## Decisions: realized cost per booking (BRL, cost rule of the live pipeline; mean ± sd over seeds)", "",
          "| Decider | BRL / booking | fraud leak | legit friction | actions taken (pooled rows) |", "|---|---|---|---|---|"]
    for k, r in res["cost"].items():
        L.append(f"| {k} | {ms(r['brl_per_booking'])} | {r['fraud_leak_brl']:.3f} | {r['legit_friction_brl']:.3f} | "
                 + ", ".join(f"{a} {n}" for a, n in sorted(r["actions"].items(), key=lambda x: -x[1])) + " |")
    L.append(f"\nCost saved by M2 vs B2 (same cost rule): {res['cost_saved_brl_per_booking']:.3f} BRL per booking, "
             f"95% CI {ci(res['cost_saved_ci'])}.")
    if res["auditor_overrule_rate"] is not None:
        L.append(f"The cost auditor overruled Laya's own action on {res['auditor_overrule_rate']:.2%} of bookings "
                 f"(margin {AUDIT_MARGIN_BRL} BRL).")
    L += ["", "## Zero-shot `drop_consignee` on T3 (never trained), ROC-AUC", "",
          ", ".join(f"{k} {v:.3f}" for k, v in res["zero_shot_drop_t3_auc"].items()) + " (v1 fine-tuned: 0.603, stock: 0.488)", "",
          "## Calibration", "", f"`{res['calibration']['version']}`: Platt misuse a={res['calibration']['platt']['misuse']['a']:.3f} "
          f"b={res['calibration']['platt']['misuse']['b']:.3f}; conformal lambda_allow {res['calibration']['conformal']['lambda_allow']:.4f} "
          f"from {res['calibration']['conformal']['n_fraud']} calibration fraud.", ""]
    (OUT / f"results_laya_v2{tag}.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
