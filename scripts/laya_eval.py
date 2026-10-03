"""Fit calibration.json for the fine-tuned Laya and evaluate B2 (GBM) vs B3 (stock Laya) vs M1
(fine-tuned Laya). Reads artifacts/laya/scores_*.parquet (scripts/laya_score.py). CPU only.

Writes artifacts/calibration.json, artifacts/results_laya.md, artifacts/results_laya.json.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fraudshield.models import calibrate_fit as C  # noqa: E402
from fraudshield.models.laya_train import metrics as M  # noqa: E402
from fraudshield.models.laya_train.data import seed_view  # noqa: E402

import os  # noqa: E402

ART = ROOT / "artifacts"
LA = Path(os.environ.get("FS_LAYA_DIR") or os.environ.get("FS_LAYA_ART") or ART / "laya")          # score files (override for dry runs)
CKPT = Path(os.environ.get("FS_LAYA_CKPT", LA / "fraudshield-laya"))
OUT = Path(os.environ.get("FS_LAYA_OUT", ART))                   # calibration.json / results_laya.*
TYP = ("T1", "T2", "T3", "T4", "T5", "T6", "T7")
HELD = ("T3", "T5")
SLICES = ("new_state", "new_seller", "billing_change", "injected", "multi_account_consignee")
PROB_Q = ("misuse", "foreign_senders", "payoff_max")
ALPHA = 0.05
B = 500


def load_set(name: str) -> pd.DataFrame:
    s = pd.read_parquet(LA / f"evalset_{name}.parquet")
    lab = s.labels.map(json.loads)
    for q in ("misuse", "foreign_senders", "payoff_max", "drop_consignee"):
        s[f"y_{q}"] = lab.map(lambda l: int(l[q]))
    s["hn"] = s.hard_negative.map(json.loads)
    s["cluster"] = np.where(s.y_misuse == 1, s.seed.astype(str) + ":" + s.campaign_id.astype(str), "legit")
    for m in ("ft", "stock"):
        p = LA / f"scores_{m}_{name}.parquet"
        if p.exists():
            sc = pd.read_parquet(p).set_index("id")
            sc.columns = [f"{m}_{c[2:]}" for c in sc.columns]
            s = s.join(sc, on="id")
    return s


def gbm_scores(s: pd.DataFrame) -> pd.Series:
    """B2 tier F (seed-0 model, Platt-calibrated) for every row: states.jsonl value for seed 0, and the
    same saved model applied to features/seed_K.parquet for injected rows of seeds 1-9."""
    if "gbm" in s.columns and s.gbm.notna().all():      # precomputed locally (Kaggle bundle)
        return s.gbm.astype(float)
    from fraudshield.models.gbm import GBMModel  # noqa: PLC0415
    g = s.gbm_b2f.astype(float).copy()
    model = GBMModel.load(ART / "models", "B2", "F")
    for k in sorted(s.loc[g.isna(), "seed"].unique()):
        ids = s.loc[g.isna() & (s.seed == k), "id"]
        f = pd.read_parquet(ROOT / "data/processed/features" / f"seed_{k}.parquet")
        f = f[f.booking_id.isin(ids)].set_index("booking_id").loc[ids]
        g.loc[ids.index] = model.predict(f.reset_index())
    assert g.notna().all()
    return g


# ---------------------------------------------------------------- calibration
def account_split(acc: pd.Series, frac: float = 0.6) -> np.ndarray:
    h = acc.map(lambda a: int(hashlib.sha256(str(a).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF)
    return (h < frac).to_numpy()


def fit_calibration(cal: pd.DataFrame) -> tuple[dict, dict]:
    c1 = account_split(cal.account_id)
    w = cal.weight.to_numpy()
    temps = {}
    for q in PROB_Q:
        p = cal[f"ft_{q}"].to_numpy()
        y = cal[f"y_{q}"].to_numpy()
        temps[q] = C.fit_temperature(np.stack([p, 1 - p], 1)[c1], np.stack([y, 1 - y], 1)[c1], w[c1])
    temps["drop_consignee"] = 1.0       # zero-shot, no positives in the calibration window (T3 held out)
    pT = C.temper(np.stack([cal.ft_misuse, 1 - cal.ft_misuse], 1), temps["misuse"])[:, 0]
    y = cal.y_misuse.to_numpy()
    a, b = C.fit_platt(pT[c1], y[c1], w[c1])
    pc = C.platt(pT, a, b)
    c2f = (~c1) & (y == 1)
    lam = C.conformal_lambda(pc[c2f], ALPHA)
    prev = float((w * y).sum() / w.sum())
    info = {"c1_bookings": int(c1.sum()), "c2_bookings": int((~c1).sum()), "c1_fraud": int((c1 & (y == 1)).sum()),
            "c2_fraud": int(c2f.sum()), "split": "by account hash, 60/40", "window": "calibration split, seed 0",
            "weights": "random legit rows weighted back to the full calibration window",
            "drop_consignee": "T=1.0, zero-shot question with no positive labels in the calibration window",
            "platt_on": "misuse after its temperature"}
    doc = C.calibration_json(CKPT / "model.safetensors", temps, {"misuse": {"a": a, "b": b}}, lam, ALPHA,
                             int(c2f.sum()), prev, extra=info)
    return doc, info


def calibrated_misuse(p, doc):
    pT = C.temper(np.stack([p, 1 - p], 1), doc["temperatures"]["misuse"])[:, 0]
    return C.platt(pT, doc["platt"]["misuse"]["a"], doc["platt"]["misuse"]["b"])


# ---------------------------------------------------------------- metrics
def system_metrics(v: pd.DataFrame, s: np.ndarray) -> dict:
    y = v.y_misuse.to_numpy().astype(bool)
    w = v.weight.to_numpy()
    flags = M.flag_top_per_day_w(v.booking_ts, s, w, 0.01)
    legit = ~y
    m = {"pr_auc": M.pr_auc_w(y, s, w),
         "prec_top1pct_day": float((w[flags] * y[flags]).sum() / w[flags].sum()),
         "rec_top1pct_day": float(flags[y].mean()),
         "fpr_legit": float((w * (flags & legit)).sum() / (w * legit).sum())}
    for sl in SLICES:
        mask = legit & v.hn.map(lambda h: sl in h).to_numpy()
        m[f"fpr_{sl}"] = float((w[mask] * flags[mask]).sum() / w[mask].sum()) if mask.any() else float("nan")
    for t in TYP:
        tm = v.typology.to_numpy() == t
        mask = legit | tm
        m[f"pr_auc_{t}"] = M.pr_auc_w(y[mask], s[mask], w[mask])
        m[f"rec_{t}"] = float(flags[tm].mean()) if tm.any() else float("nan")
    return m


def ms(vals) -> str:
    a = np.array([x for x in vals if not np.isnan(x)])
    return f"{a.mean():.3f} ± {a.std(ddof=1) if len(a) > 1 else 0:.3f}" if len(a) else "n/a"


def main() -> None:
    test, cal = load_set("test"), load_set("cal")
    test["gbm"] = gbm_scores(test)
    doc, cinfo = fit_calibration(cal)
    (OUT / "calibration.json").write_text(json.dumps(doc, indent=2))
    test["m1_cal"] = calibrated_misuse(test.ft_misuse.to_numpy(), doc)
    systems = {"B2 GBM+graph (F)": "gbm", "B3 Laya stock zero-shot": "stock_misuse", "M1 Laya fine-tuned": "ft_misuse"}

    # ---- per-seed metrics (seed view = real rows + that seed's injected rows)
    per = {k: [] for k in systems}
    for sd in range(10):
        v = seed_view(test, sd)
        for name, col in systems.items():
            per[name].append(system_metrics(v, v[col].to_numpy()))

    # ---- calibration before / after on test, seed-0 view (natural prevalence)
    v0 = seed_view(test, 0)
    w0 = v0.weight.to_numpy()
    calrows = []
    for q in PROB_Q:
        y = v0[f"y_{q}"].to_numpy()
        raw = v0[f"ft_{q}"].to_numpy()
        after = (calibrated_misuse(raw, doc) if q == "misuse"
                 else C.temper(np.stack([raw, 1 - raw], 1), doc["temperatures"][q])[:, 0])
        st = v0[f"stock_{q}"].to_numpy()
        calrows.append({"q": q, "ece_stock": C.ece(st, y, w0), "brier_stock": C.brier(st, y, w0),
                        "ece_raw": C.ece(raw, y, w0), "brier_raw": C.brier(raw, y, w0),
                        "ece_cal": C.ece(after, y, w0), "brier_cal": C.brier(after, y, w0),
                        "mean_p_raw": float((w0 * raw).sum() / w0.sum()),
                        "mean_p_cal": float((w0 * after).sum() / w0.sum()), "rate": float((w0 * y).sum() / w0.sum())})

    # ---- conformal check on test: allowed-fraud rate (calibrated misuse <= lambda)
    lam = doc["conformal"]["lambda_allow"]
    fr = test[test.y_misuse == 1]
    allowed = fr.m1_cal <= lam
    conf = {"lambda_allow": lam, "test_fraud": int(len(fr)), "allowed_fraud_rate_all": float(allowed.mean()),
            **{f"allowed_fraud_rate_{t}": float(allowed[fr.typology == t].mean()) for t in TYP},
            "legit_blocked_from_allow_rate": float(
                (test.weight * ((test.y_misuse == 0) & (test.m1_cal > lam))).sum()
                / (test.weight * (test.y_misuse == 0)).sum())}

    # ---- headline (a): held-out T3/T5 recall at equal legit friction, pooled over seeds 0-9
    y = test.y_misuse.to_numpy().astype(bool)
    w = test.weight.to_numpy()
    cl = test.cluster.to_numpy()
    head_a = []
    for t in HELD:
        mask = (~y) | (test.typology.to_numpy() == t)
        yy, ww, cc = y[mask], w[mask], cl[mask]
        for fpr in (0.005, 0.01):
            row = {"typology": t, "fpr": fpr, "n_bookings": int(yy.sum()), "n_campaigns": int(len(set(cc[yy])))}
            for col, lab in (("gbm", "B2"), ("ft_misuse", "M1"), ("stock_misuse", "B3")):
                sc = test[col].to_numpy()[mask]
                row[lab] = M.recall_at_fpr(yy, sc, ww, fpr)
                row[lab + "_ci"] = M.bootstrap_ci(lambda i, sc=sc: M.recall_at_fpr(yy[i], sc[i], ww[i], fpr),
                                                   yy, cc, B=B, seed=1)
            g, f = test.gbm.to_numpy()[mask], test.ft_misuse.to_numpy()[mask]
            row["M1_minus_B2_ci"] = M.bootstrap_ci(
                lambda i: M.recall_at_fpr(yy[i], f[i], ww[i], fpr) - M.recall_at_fpr(yy[i], g[i], ww[i], fpr),
                yy, cc, B=B, seed=2)
            head_a.append(row)
        row = {"typology": t, "metric": "pr_auc"}
        for col, lab in (("gbm", "B2"), ("ft_misuse", "M1"), ("stock_misuse", "B3")):
            sc = test[col].to_numpy()[mask]
            row[lab] = M.pr_auc_w(yy, sc, ww)
            row[lab + "_ci"] = M.bootstrap_ci(lambda i, sc=sc: M.pr_auc_w(yy[i], sc[i], ww[i]), yy, cc, B=B, seed=3)
        head_a.append(row)

    # ---- headline (b): zero-shot drop_consignee, T3 bookings vs legit
    mask = (~y) | (test.typology.to_numpy() == "T3")
    yy, ww, cc = y[mask], w[mask], cl[mask]
    head_b = {"n_t3": int(yy.sum()), "n_campaigns": int(len(set(cc[yy])))}
    for col, lab in (("stock_drop_consignee", "B3"), ("ft_drop_consignee", "M1"), ("gbm", "B2_general_score")):
        sc = test[col].to_numpy()[mask]
        head_b[lab] = {"auc": M.roc_auc_w(yy, sc, ww),
                       "auc_ci": M.bootstrap_ci(lambda i, sc=sc: M.roc_auc_w(yy[i], sc[i], ww[i]), yy, cc, B=B, seed=4),
                       "pr_auc": M.pr_auc_w(yy, sc, ww),
                       "rec_fpr1": M.recall_at_fpr(yy, sc, ww, 0.01),
                       "rec_fpr1_ci": M.bootstrap_ci(lambda i, sc=sc: M.recall_at_fpr(yy[i], sc[i], ww[i], 0.01),
                                                     yy, cc, B=B, seed=5),
                       "rec_fpr5": M.recall_at_fpr(yy, sc, ww, 0.05)}
    s_, f_ = test.stock_drop_consignee.to_numpy()[mask], test.ft_drop_consignee.to_numpy()[mask]
    head_b["M1_minus_B3_auc_ci"] = M.bootstrap_ci(
        lambda i: M.roc_auc_w(yy[i], f_[i], ww[i]) - M.roc_auc_w(yy[i], s_[i], ww[i]), yy, cc, B=B, seed=6)
    # drop_consignee also on legit vs ALL fraud and on the multi-account-consignee hard negatives
    hn_mac = test.hn.map(lambda h: "multi_account_consignee" in h).to_numpy()
    for col, lab in (("stock_drop_consignee", "B3"), ("ft_drop_consignee", "M1")):
        sc = test[col].to_numpy()
        thr = M.threshold_at_friction(sc[~y], w[~y], 0.01)
        head_b[lab]["fpr_multi_account_consignee_at_fpr1"] = float((sc[hn_mac & ~y] >= thr).mean())

    res = {"per_seed": per, "calibration": calrows, "calibration_doc": doc, "conformal_test": conf,
           "headline_a": head_a, "headline_b": head_b,
           "eval_set": {"bookings": int(len(test)), "fraud": int(y.sum()),
                        "weighted_bookings_seed0": float(seed_view(test, 0).weight.sum())}}
    from fraudshield.models.laya_train.report import verdict_text  # noqa: PLC0415
    res["verdict"] = verdict_text(res)
    (OUT / "results_laya.json").write_text(json.dumps(res, indent=2, default=float))
    write_md(res, per, systems)
    print(json.dumps({"calibration": calrows, "conformal": conf, "headline_b": head_b}, indent=1, default=float))


def f3(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def ci(c):
    return f"[{c[0]:.3f}, {c[1]:.3f}]"


def write_md(res, per, systems) -> None:
    prep = json.loads((LA / "prepare.json").read_text())
    pilots = json.loads((LA / "pilot.json").read_text()) if (LA / "pilot.json").exists() else []
    run = json.loads((CKPT / "rl_agent_config.json").read_text()).get("fraudshield", {})
    extra = json.loads((LA / "runtime.json").read_text()) if (LA / "runtime.json").exists() else {}
    runcfg = json.loads((LA / "run_config.json").read_text()) if (LA / "run_config.json").exists() else {}
    L = ["# Laya fine-tune results (FraudShield)", "",
         f"Real runs. Training GPU: {run.get('gpu', 'unknown')}, precision {run.get('precision', 'unknown')}. Test window = seed-0 real Olist rows (legit, shared by every "
         "seed) plus the injected rows of 10 injection seeds. Legit rows are a 5,000-row random sample plus every real "
         "hard negative, weighted back to the full window (22,925 bookings per seed view, prevalence about 1.2%). "
         "All three systems score exactly the same rows. T3 and T5 are held out (never in training or calibration). "
         "B2 = the data agent's LightGBM + change/graph features, tier F, seed-0 model, Platt-calibrated.", ""]
    L += ["## Training", "",
          f"- Data: {prep['bookings']} training bookings ({prep['unique_bookings']} unique) from the seed-0 train "
          f"window: all {prep['fraud_unique']} eligible train fraud bookings repeated {prep['fraud_repeat']}x "
          f"({prep['fraud_share']:.1%} fraud), 1,800 hard negatives, 3,000 random legit; T3/T5 excluded. "
          f"{prep['items']} items per epoch (questions: {', '.join(prep['questions'])}). Only 386 train fraud "
          "bookings exist in the seed-0 train window, so the DESIGN target of 1,200 distinct fraud bookings was "
          "met by repetition, not by new data.",
          f"- Token budget: max state {prep['all_states_max_tokens']} tokens over all {64200} states, max sequence "
          f"{prep['max_seq_len_train']} of 512; 0 items dropped by the marker check, 0 truncated.",
          f"- Pilot (200 micro-steps each): " + "; ".join(
              f"K={p['k_top']} micro={p['micro']}: {p['items_per_s']} items/s, peak VRAM {p['peak_vram_alloc_gb']} GB "
              f"allocated / {p['peak_vram_reserved_gb']} GB reserved, {p['trainable_params'] / 1e6:.0f}M trainable, "
              f"est {p['est_hours_per_epoch_full']} h/epoch" for p in pilots) + ".",
          f"- Run: top {run.get('k_top')} of 28 encoder layers + decision head trained (embeddings and lower layers "
          f"frozen, in bf16), bf16 autocast, gradient checkpointing, micro-batch {run.get('micro')} x accumulation "
          f"{run.get('accum')} = {(run.get('micro') or 0) * (run.get('accum') or 0)}, {run.get('epochs')} epochs, "
          f"LR encoder 2.5e-5 / head 1e-4 cosine to 1e-6, sigma 0.4 -> 0.1, group 4, misuse weight 2.0. "
          f"Wall time {run.get('train_minutes')} min. Run config: {runcfg.get('note', 'planned (no fallback)')}.",
          "- Objective: notebook recipe (log + spherical proper reward, RPS for risk_level, group-mean REINFORCE on "
          f"Gaussian logit noise, plus soft CE) on probability heads; action head reward -(q . cost)/C with "
          f"C = {prep['cost_scale_C']:.1f} BRL and CE weight 0.", ""]
    L += ["## Main table (mean ± sd over 10 injection seeds; top 1% of bookings per day)", "",
          "| System | PR-AUC | Prec@1%/day | Rec@1%/day | FPR legit (%) | " + " | ".join(f"FPR {s} (%)" for s in SLICES) + " |",
          "|---|---|---|---|---|" + "---|" * len(SLICES)]
    for name in systems:
        p = per[name]
        L.append(f"| {name} | {ms([x['pr_auc'] for x in p])} | {ms([x['prec_top1pct_day'] for x in p])} | "
                 f"{ms([x['rec_top1pct_day'] for x in p])} | {ms([100 * x['fpr_legit'] for x in p])} | " +
                 " | ".join(ms([100 * x[f'fpr_{s}'] for x in p]) for s in SLICES) + " |")
    L += ["", "PR-AUC per typology (typology vs all legit; held-out marked *):", "",
          "| System | " + " | ".join(t + ("*" if t in HELD else "") for t in TYP) + " |", "|---|" + "---|" * len(TYP)]
    for name in systems:
        L.append(f"| {name} | " + " | ".join(ms([x[f'pr_auc_{t}'] for x in per[name]]) for t in TYP) + " |")
    L += ["", "Recall at top 1%/day per typology:", "",
          "| System | " + " | ".join(t + ("*" if t in HELD else "") for t in TYP) + " |", "|---|" + "---|" * len(TYP)]
    for name in systems:
        L.append(f"| {name} | " + " | ".join(ms([x[f'rec_{t}'] for x in per[name]]) for t in TYP) + " |")
    L += ["", "## Headline (a): held-out T3 and T5 at equal legit friction (pooled seeds 0-9, 95% bootstrap CI by campaign)", "",
          "| Typology | Legit FPR | n bookings / campaigns | B2 GBM recall | M1 misuse recall | B3 stock recall | M1 - B2 |",
          "|---|---|---|---|---|---|---|"]
    for r in res["headline_a"]:
        if "fpr" in r:
            L.append(f"| {r['typology']} | {r['fpr']:.1%} | {r['n_bookings']} / {r['n_campaigns']} | {f3(r['B2'])} {ci(r['B2_ci'])} | "
                     f"{f3(r['M1'])} {ci(r['M1_ci'])} | {f3(r['B3'])} {ci(r['B3_ci'])} | {ci(r['M1_minus_B2_ci'])} |")
    for r in res["headline_a"]:
        if r.get("metric") == "pr_auc":
            L.append(f"| {r['typology']} PR-AUC | all | | {f3(r['B2'])} {ci(r['B2_ci'])} | {f3(r['M1'])} {ci(r['M1_ci'])} | "
                     f"{f3(r['B3'])} {ci(r['B3_ci'])} | |")
    hb = res["headline_b"]
    L += ["", f"## Headline (b): zero-shot `drop_consignee` on T3 bookings ({hb['n_t3']} bookings, {hb['n_campaigns']} campaigns) vs legit", "",
          "| System | ROC-AUC | PR-AUC | Recall @ 1% legit FPR | Recall @ 5% legit FPR | FPR on multi-account-consignee legit at the 1% threshold |",
          "|---|---|---|---|---|---|"]
    for lab, name in (("B3", "B3 stock Laya, drop_consignee"), ("M1", "M1 fine-tuned Laya, drop_consignee (never trained)"),
                      ("B2_general_score", "B2 GBM general fraud score (reference)")):
        h = hb[lab]
        L.append(f"| {name} | {f3(h['auc'])} {ci(h['auc_ci'])} | {f3(h['pr_auc'])} | {f3(h['rec_fpr1'])} {ci(h['rec_fpr1_ci'])} | "
                 f"{f3(h['rec_fpr5'])} | {f3(h.get('fpr_multi_account_consignee_at_fpr1'))} |")
    L.append(f"\nM1 - B3 ROC-AUC difference, 95% CI: {ci(hb['M1_minus_B3_auc_ci'])}")
    L += ["", "## Calibration (test window, seed-0 view, weighted to true prevalence; ECE = 15 equal-mass bins)", "",
          "| Question | rate | Stock ECE | Stock Brier | M1 raw ECE | M1 raw Brier | M1 calibrated ECE | M1 calibrated Brier | mean p raw -> cal |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in res["calibration"]:
        L.append(f"| {r['q']} | {r['rate']:.4f} | {r['ece_stock']:.4f} | {r['brier_stock']:.4f} | {r['ece_raw']:.4f} | {r['brier_raw']:.4f} | "
                 f"{r['ece_cal']:.4f} | {r['brier_cal']:.4f} | {r['mean_p_raw']:.4f} -> {r['mean_p_cal']:.4f} |")
    d = res["calibration_doc"]
    c = res["conformal_test"]
    L += ["", f"calibration.json `{d['version']}` (model_revision {d['model_revision']}): temperatures "
          + ", ".join(f"{k} {v:.3f}" for k, v in d["temperatures"].items())
          + f"; Platt misuse a={d['platt']['misuse']['a']:.3f} b={d['platt']['misuse']['b']:.3f}; prevalence {d['prevalence']:.4f}. "
          f"Fitted on {d['fit']['c1_bookings']} calibration bookings ({d['fit']['c1_fraud']} fraud, split by account); "
          f"conformal lambda_allow = {d['conformal']['lambda_allow']:.4f} from {d['conformal']['n_fraud']} held-out calibration fraud "
          f"(alpha {d['conformal']['alpha']}).",
          f"- Test check of the allow guard: allowed-fraud rate {c['allowed_fraud_rate_all']:.3f} over {c['test_fraud']} test fraud "
          f"(T3* {c['allowed_fraud_rate_T3']:.3f}, T5* {c['allowed_fraud_rate_T5']:.3f}, T6 {c['allowed_fraud_rate_T6']:.3f}); "
          f"share of legit bookings for which allow is ruled out: {c['legit_blocked_from_allow_rate']:.3f}. "
          "The guarantee covers exchangeable fraud only; campaigns are clustered and T3/T5 are new typologies, so rates above 0.05 "
          "there are a real miss of the guarantee, not noise.", ""]
    if extra:
        L += ["## Serving", "", f"- {extra.get('summary', '')}", ""]
    L += ["## Verdict", "", res.get("verdict", "VERDICT_PLACEHOLDER"), ""]
    (OUT / "results_laya.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
