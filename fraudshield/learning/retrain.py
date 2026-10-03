"""Gated retraining of the GBM from feedback labels (docs/API.md v1.2, DESIGN 7 delayed outcomes).

Training set = the GBM's own train window + the time-earliest 70% of feedback labels (decision-time
feature snapshots). Eval set = calibration window + latest 30% of feedback; neither model trains on it.
run_retrain builds, evaluates and gates a candidate; it never deploys (LearningService does).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from fraudshield.data.metrics import pr_auc
from fraudshield.learning.metrics import (STOP_ACTIONS, GateConfig, gate_checks, paired_cluster_bootstrap,
                                          policy_costs, summary_metrics)
from fraudshield.models.gbm import NUM_ROUNDS, PARAMS, GBMModel, _logit

IPW_CLIP = 20.0
EVAL_DESCRIPTION = "calibration/validation window + latest 30% of feedback labels by time, never trained on"


def feedback_weights(labels: list[dict[str, Any]], clip: float = IPW_CLIP) -> np.ndarray:
    """Explored rows: 1/propensity, clipped at `clip`, self-normalised to mean 1 over explored rows.
    Other feedback: weight 1 (a biased sample: labels only exist where we acted)."""
    w = np.ones(len(labels))
    idx = [i for i, x in enumerate(labels) if x.get("explored") and x.get("propensity")]
    if idx:
        raw = np.minimum(1.0 / np.array([float(labels[i]["propensity"]) for i in idx]), clip)
        w[idx] = raw * len(idx) / raw.sum()
    return w


def cap_feedback_weight(w: np.ndarray, ref_total: float, max_share: float | None) -> np.ndarray:
    """Scale feedback weights so they carry at most `max_share` of the total sample weight
    (relative weights kept). None disables the cap."""
    w = np.asarray(w, dtype=float)
    if max_share is None or w.sum() <= 0:
        return w
    limit = max_share * ref_total / (1.0 - max_share)
    return w * (limit / w.sum()) if w.sum() > limit else w


def split_feedback_by_time(labels: list[dict[str, Any]], train_frac: float = 0.7):
    """Earliest `train_frac` by booking time -> training; latest rest -> evaluation."""
    s = sorted(labels, key=lambda x: (str(x.get("booked_at")), str(x.get("labelled_at")), str(x["decision_id"])))
    k = int(np.floor(train_frac * len(s)))
    return s[:k], s[k:]


def fit_weighted(system: str, tier: str, X: pd.DataFrame, y, w, cal: pd.DataFrame, y_cal,
                 params: dict | None = None) -> GBMModel:
    """Same recipe as GBMModel.fit (params, rounds, Platt on the calibration window) plus sample weights.
    Wraps LightGBM here instead of editing models/gbm.py (owned by the data agent)."""
    m = GBMModel(system, tier)
    ds = lgb.Dataset(m._X(X), label=np.asarray(y).astype(int), weight=np.asarray(w, dtype=float),
                     feature_name=m.features, free_raw_data=True)
    prm = {**PARAMS, **(params or {})}
    m.booster = lgb.train(prm, ds, num_boost_round=NUM_ROUNDS)
    z = _logit(m.predict_raw(cal)).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(z, np.asarray(y_cal).astype(int))
    m.platt = (float(lr.coef_[0, 0]), float(lr.intercept_[0]))
    m.metadata = {"version": None, "system": system, "tier": tier, "features": m.features,
                  "platt": {"a": m.platt[0], "b": m.platt[1]}, "params": prm, "num_rounds": NUM_ROUNDS,
                  "n_train": int(len(X)), "n_train_pos": int(np.asarray(y).sum()), "weighted": True,
                  "n_cal": int(len(cal)), "n_cal_pos": int(np.asarray(y_cal).sum())}
    return m


def _hn_mask(df: pd.DataFrame) -> np.ndarray:
    m = df["typology"].astype(str).str.startswith("HN").to_numpy()
    for c in df.columns:
        if c.startswith("hn_") and df[c].dtype == bool:
            m = m | df[c].to_numpy()
    return m & ~df["is_fraud"].to_numpy().astype(bool)


def _present(v) -> bool:
    return v is not None and not (isinstance(v, float) and np.isnan(v)) and str(v) not in ("", "nan", "None")


def _cluster(lab: dict[str, Any]) -> str:
    for k in ("campaign_id", "account_id", "booking_id"):
        if _present(lab.get(k)):
            return str(lab[k])
    return "unknown"


def feedback_frame(labels: list[dict[str, Any]], features: list[str]) -> pd.DataFrame:
    """Feedback labels -> rows with the decision-time feature snapshot (labels without a snapshot dropped)."""
    labels = [x for x in labels if x.get("features")]
    cols = [*features, "booking_id", "is_fraud", "typology", "cluster", "carrier_cost", "tenure_days",
            "owner_contact_age_days", "explored"]
    if not labels:
        return pd.DataFrame({c: pd.Series(dtype=bool if c in ("is_fraud", "explored") else object)
                             for c in dict.fromkeys(cols)})

    def num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return np.nan

    X = pd.DataFrame([{f: num(x["features"].get(f)) for f in features} for x in labels])
    meta = pd.DataFrame({
        "booking_id": [str(x["booking_id"]) for x in labels],
        "is_fraud": [x["label"] == "fraud" for x in labels],
        "typology": [x.get("typology") or "unknown" for x in labels],
        "cluster": [_cluster(x) for x in labels],
        "_fb_cost": [num(x.get("carrier_cost")) for x in labels],
        "_tenure": [num(x["features"].get("tenure_days")) for x in labels],
        "_owner": [num(x["features"].get("owner_contact_age_days")) for x in labels],
        "explored": [bool(x.get("explored")) for x in labels],
    })
    out = pd.concat([X, meta], axis=1)
    for col, src in (("carrier_cost", "_fb_cost"), ("tenure_days", "_tenure"), ("owner_contact_age_days", "_owner")):
        out[col] = out[col].fillna(out[src]) if col in features else out[src]
    return out.drop(columns=["_fb_cost", "_tenure", "_owner"])


def _eval_frame(ref_cal: pd.DataFrame, fb_eval: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    cal = pd.DataFrame({f: ref_cal[f].to_numpy(float) for f in features})
    cal["booking_id"] = ref_cal["booking_id"].astype(str).to_numpy()
    cal["is_fraud"] = ref_cal["is_fraud"].to_numpy(bool)
    cal["typology"] = ref_cal["typology"].astype(str).to_numpy()
    camp = ref_cal["campaign_id"] if "campaign_id" in ref_cal else pd.Series([None] * len(ref_cal))
    cal["cluster"] = [str(c) if _present(c) else str(acc) for c, acc in zip(camp.to_numpy(), ref_cal["account_id"])]
    for c in ("carrier_cost", "tenure_days", "owner_contact_age_days"):
        cal[c] = ref_cal[c].to_numpy(float) if c in ref_cal else np.nan
    cal["hn"] = _hn_mask(ref_cal)
    cal["origin"] = "cal_window"
    fb = fb_eval.copy()
    fb["is_fraud"] = fb["is_fraud"].astype(bool)
    fb["hn"] = fb["typology"].astype(str).str.startswith("HN").to_numpy() & ~fb["is_fraud"].to_numpy()
    fb["origin"] = "feedback"
    return pd.concat([cal, fb[cal.columns]], ignore_index=True)


NEW_PATTERN_DESCRIPTION = ("test-window bookings of fraud typologies the active model never trained on, pooled "
                           "across injection seeds; campaigns present in the training feedback removed")


def new_pattern_eval(current, candidate, pool: pd.DataFrame, cur_trained, fb_train_labels, costs, policy,
                     B: int = 200, seed: int = 0):
    """Eval set (b): recall at the operating point on typologies unseen by the active model.
    Returns (report, CI of candidate minus current recall, campaign-cluster bootstrap) or (None, None)."""
    fr = pool[pool.is_fraud.astype(bool)]
    typ = fr.typology.astype(str)
    fr = fr[~typ.isin(set(cur_trained)) & ~typ.str.startswith("HN") & ~typ.isin(["none", "unknown"])]
    if fr.empty:
        return None, None
    train_camps = {str(x.get("campaign_id")) for x in fb_train_labels if x.get("campaign_id")}
    camp = fr.campaign_id.astype(str)
    excluded = int(camp.isin(train_camps).sum())
    fr = fr[~camp.isin(train_camps)]
    if fr.empty:
        return None, None
    F = fr.carrier_cost.to_numpy(float)
    y = np.ones(len(fr), bool)
    ten = fr["tenure_days"].to_numpy(float) if "tenure_days" in fr else None
    own = fr["owner_contact_age_days"].to_numpy(float) if "owner_contact_age_days" in fr else None
    s_cur = np.isin(policy_costs(current.predict(fr), y, F, costs, policy, ten, own)[0], STOP_ACTIONS)
    s_cand = np.isin(policy_costs(candidate.predict(fr), y, F, costs, policy, ten, own)[0], STOP_ACTIONS)
    clusters = np.array([str(c) for c in fr.campaign_id])
    lo, hi = paired_cluster_bootstrap(lambda i: float(s_cand[i].mean() - s_cur[i].mean()), clusters, B=B, seed=seed)
    t = fr.typology.astype(str).to_numpy()

    def side(s):
        return {"recall": round(float(s.mean()), 4),
                "by_typology": {k: round(float(s[t == k].mean()), 4) for k in sorted(set(t))}}
    rep = {"description": NEW_PATTERN_DESCRIPTION, "typologies": sorted(set(t)), "n": int(len(fr)),
           "campaigns": int(len(set(clusters))), "excluded_rows": excluded,
           "seeds": sorted(int(x) for x in fr["seed"].unique()) if "seed" in fr else None,
           "current": side(s_cur), "candidate": side(s_cand),
           "recall_diff_ci95": [round(lo, 4), round(hi, 4)], "bootstrap": {"B": B, "unit": "campaign"}}
    return rep, (lo, hi)


@dataclass
class RetrainResult:
    report: dict[str, Any]
    candidate: GBMModel | None
    current_metrics: dict[str, Any]
    candidate_metrics: dict[str, Any]
    trained_typologies: list[str]
    train_ids: list[str] = field(default_factory=list)
    eval_ids: list[str] = field(default_factory=list)


def run_retrain(registry, reference: pd.DataFrame, labels: list[dict[str, Any]], n_new_labels: int, costs,
                policy, gate_cfg: GateConfig, fit_fn: Callable = fit_weighted, max_train_rows: int | None = None,
                B: int = 200, seed: int = 0, train_frac: float = 0.7,
                new_pattern_pool: pd.DataFrame | None = None, max_feedback_share: float | None = None,
                ipw_clip: float = IPW_CLIP, lgb_params: dict | None = None) -> RetrainResult:
    """Build, evaluate and gate a candidate against the active version. Never deploys."""
    t0 = time.perf_counter()
    notes: list[str] = []
    cur_version = registry.active_version
    current = registry.load_model(cur_version)
    feats = current.features
    ref_tr = reference[reference.split == "train"]
    ref_cal = reference[reference.split == "cal"]
    cur_trained = registry.get(cur_version).get("trained_typologies")
    if cur_trained is None:  # base model: the fraud typologies in its own train window
        cur_trained = sorted(set(ref_tr.loc[ref_tr.is_fraud.astype(bool), "typology"].astype(str)))

    fb_train_l, fb_eval_l = split_feedback_by_time(labels, train_frac)
    n_no_snap = sum(1 for x in labels if not x.get("features"))
    if n_no_snap:
        notes.append(f"{n_no_snap} labels had no decision-time feature snapshot and were not used")
    fb_train_l = [x for x in fb_train_l if x.get("features")]
    fb_tr = feedback_frame(fb_train_l, feats)
    fb_ev = feedback_frame(fb_eval_l, feats)
    w_fb = feedback_weights(fb_train_l, clip=ipw_clip)

    # reference rows that also appear in feedback are taken out of both reference sets (no double use)
    fb_ids = set(fb_tr.booking_id) | set(fb_ev.booking_id)
    ref_tr = ref_tr[~ref_tr.booking_id.astype(str).isin(fb_ids)]
    if max_train_rows and len(ref_tr) > max_train_rows:
        # every fraud row and every hard negative is kept; only ordinary legit rows are subsampled
        keep = ref_tr.is_fraud.astype(bool).to_numpy() | _hn_mask(ref_tr)
        pos = ref_tr[keep]
        neg = ref_tr[~keep].sample(max(0, min(int((~keep).sum()), max_train_rows - len(pos))), random_state=seed)
        ref_tr = pd.concat([pos, neg]).sort_values("booked_at")
        notes.append(f"Original train window subsampled to {len(ref_tr):,} rows (all fraud and hard-negative rows "
                     f"kept) to keep retraining under a minute.")
    w_fb = cap_feedback_weight(w_fb, float(len(ref_tr)), max_feedback_share)
    X = pd.concat([ref_tr[feats].astype(float), fb_tr[feats].astype(float)], ignore_index=True)
    y = np.r_[ref_tr.is_fraud.to_numpy(bool), fb_tr.is_fraud.to_numpy(bool)]
    w = np.r_[np.ones(len(ref_tr)), w_fb]
    cal_fit = ref_cal[~ref_cal.booking_id.astype(str).isin(fb_ids)]

    ev = _eval_frame(cal_fit, fb_ev, feats)
    fit_kw = {"params": lgb_params} if lgb_params else {}
    candidate = fit_fn(registry.system, registry.tier, X, y, w, cal_fit[feats], cal_fit.is_fraud.to_numpy(bool),
                       **fit_kw)
    p_cur, p_cand = current.predict(ev), candidate.predict(ev)
    yv = ev.is_fraud.to_numpy(bool)
    # hard-negative typologies (HN_*) are legit by construction; a fraud label on one is an analyst error
    new_typ = sorted(t for t in set(ev.loc[yv, "typology"].astype(str)) - set(cur_trained)
                     if t not in ("unknown", "none", "None") and not t.startswith("HN"))
    new_mask = ev["typology"].astype(str).isin(new_typ).to_numpy()
    F = ev.carrier_cost.to_numpy(float)
    ten, own = ev.tenure_days.to_numpy(float), ev.owner_contact_age_days.to_numpy(float)
    a_cur, c_cur = policy_costs(p_cur, yv, F, costs, policy, ten, own)
    a_cand, c_cand = policy_costs(p_cand, yv, F, costs, policy, ten, own)
    hn = ev.hn.to_numpy(bool)
    m_cur = summary_metrics(yv, p_cur, c_cur, np.isin(a_cur, STOP_ACTIONS), hn, new_mask)
    m_cand = summary_metrics(yv, p_cand, c_cand, np.isin(a_cand, STOP_ACTIONS), hn, new_mask)
    clusters = np.array([str(c) for c in ev.cluster], dtype=object)
    ci = {
        "cost_improvement": paired_cluster_bootstrap(lambda i: float(np.mean(c_cur[i] - c_cand[i]) * 1000),
                                                     clusters, B=B, seed=seed),
        "pr_auc_diff": paired_cluster_bootstrap(lambda i: pr_auc(yv[i], p_cand[i]) - pr_auc(yv[i], p_cur[i]),
                                                clusters, B=B, seed=seed),
    }
    npe = None
    if new_pattern_pool is not None and len(new_pattern_pool):
        npe, np_ci = new_pattern_eval(current, candidate, new_pattern_pool, cur_trained, fb_train_l, costs, policy,
                                      B=B, seed=seed)
        ci["new_pattern_recall_diff"] = np_ci
    gate = gate_checks(m_cur, m_cand, ci, n_new_labels, gate_cfg)
    n_expl = int(sum(1 for x in fb_train_l if x.get("explored")))
    notes += [
        "Feedback that was not explored is a biased sample: labels only exist where we acted or an analyst looked. "
        "It gets weight 1 and cannot by itself estimate performance on all bookings. Explored rows get "
        f"1/propensity (clip {ipw_clip:g}, self-normalised to mean 1).",
        "Both models' 2-parameter Platt maps were fitted on the calibration window, so ECE on that part of the eval "
        "set is slightly optimistic for both; the trees never saw it.",
        "Typology is used only to report recall on new patterns, never as a feature.",
    ]
    if n_expl == 0 and len(fb_tr):
        notes.append("No explored rows in the training feedback (exploration is off when Laya is degraded), so "
                     "no inverse-propensity correction was possible; all feedback has weight 1.")
    if any(x.get("simulated") for x in labels):
        notes.append("Some labels are SIMULATED analyst labels drawn from injected ground truth (demo only).")
    if not new_typ:
        notes.append("No typology unseen by the active model is in the eval set, so recall_new_pattern is null.")
    trained_typ = sorted((set(cur_trained) | set(fb_tr.loc[fb_tr.is_fraud.astype(bool), "typology"].astype(str)))
                         - {"unknown"})
    report = {
        "n_fraud_feedback": int(sum(x["label"] == "fraud" for x in labels)),
        "eval_set": {"n": int(len(ev)), "n_cal_window": int((ev.origin == "cal_window").sum()),
                     "n_feedback": int((ev.origin == "feedback").sum()), "n_fraud": int(yv.sum()),
                     "description": EVAL_DESCRIPTION},
        "training_set": {"n": int(len(X)), "n_reference": int(len(ref_tr)), "n_feedback": int(len(fb_tr)),
                         "n_explored_feedback": n_expl, "n_fraud": int(y.sum()), "ipw_clip": ipw_clip,
                         "max_train_rows": max_train_rows, "max_feedback_share": max_feedback_share,
                         "feedback_weight_share": round(float(w_fb.sum() / max(w.sum(), 1e-9)), 4),
                         "explored_weight": round(float(w_fb[[bool(x.get("explored")) for x in fb_train_l]].sum()), 2)
                         if len(fb_train_l) else 0.0,
                         "n_reference_hard_negatives": int(_hn_mask(ref_tr).sum()),
                         "lgb_params_override": lgb_params or {}},
        "current": {"version": cur_version, **m_cur},
        "candidate": {"version": None, **m_cand},
        "bootstrap": {"B": B, "clusters": int(len(set(clusters))), "unit": "fraud campaign, else account",
                      "cost_improvement_ci95": [round(v, 2) for v in ci["cost_improvement"]],
                      "pr_auc_diff_ci95": [round(v, 4) for v in ci["pr_auc_diff"]]},
        "operating_point": "production cost rule on each model's probability, no exploration; stopped = "
                           + ", ".join(STOP_ACTIONS),
        "new_pattern_typologies": new_typ,
        "new_pattern_eval": npe,
        "gate": gate, "notes": notes,
    }
    report["duration_s"] = round(time.perf_counter() - t0, 2)
    return RetrainResult(report, candidate, m_cur, m_cand, trained_typ,
                         train_ids=[*ref_tr.booking_id.astype(str), *fb_tr.booking_id.astype(str)],
                         eval_ids=list(ev.booking_id.astype(str)))
