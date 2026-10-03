"""Build the Laya-v2 ("Laya decides") training and evaluation data. CPU only, about 10 minutes.

What changes from v1 (artifacts/laya, results_laya.md):
1. State = ser-v2: the v1 lines + MORE (signals the LightGBM used that v1 never showed Laya) + EVIDENCE
   (LightGBM risk and the two published rules). Laya weighs the witnesses; LightGBM no longer decides.
2. The LightGBM risk on TRAINING states is out-of-fold (5 folds grouped by account, per injection seed), so
   Laya sees a witness as noisy as the one it meets in production, not an in-sample one that is always right.
   Calibration and test states carry the served model's score (B2 tier F, seed 0), exactly as at run time.
3. Witness dropout: the LightGBM line is withheld ("not available") on 30% of training states, so Laya
   cannot become a pass-through and still decides when LightGBM is down or being retrained.
4. Ten times more distinct fraud: the train windows of all 10 injection seeds (v1 used seed 0 only and
   repeated its 386 frauds 3x). T3 and T5 stay held out of everything.
5. Questions trained: misuse, foreign_senders, payoff_max, action (risk_level dropped: it duplicated misuse
   and the cost; this keeps the action question's token room). drop_consignee stays zero-shot.
The evaluation rows are the SAME bookings as v1 (artifacts/laya/evalset_*.parquet), re-serialized, so v1, v2
and LightGBM are compared on identical rows. Each eval row gets two states: with and without the GBM line.

Usage: uv run python scripts/laya_v2_build.py [--legit 4500 --hard-neg 2000 --dropout 0.3]
Writes artifacts/laya_v2/{train.jsonl, prepare.json, evalset_test.parquet, evalset_cal.parquet, oof_gbm.json}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fraudshield.contracts import FeatureVector  # noqa: E402
from fraudshield.data.labels import action_costs  # noqa: E402
from fraudshield.features.serialize import SERIALIZER_V2, evidence_for, serialize_v2  # noqa: E402
from fraudshield.features.spec import FEATURES  # noqa: E402
from fraudshield.models.gbm import NUM_ROUNDS, PARAMS, GBMModel  # noqa: E402
from fraudshield.models.laya_train import metrics as M  # noqa: E402
from fraudshield.models.laya_train.data import HELD_OUT_TYPOLOGIES, to_notebook_record  # noqa: E402

PROC = ROOT / "data" / "processed"
V1 = ROOT / "artifacts" / "laya"
OUT = Path(os.environ.get("FS_LAYA_ART", ROOT / "artifacts" / "laya_v2"))
QIDS_V2 = ("misuse", "foreign_senders", "payoff_max", "action")
HN = ("hn_new_state", "hn_new_seller", "hn_billing_change", "hn_injected", "hn_multi_account_consignee")
N_SEEDS = 10


def _fv(r: dict) -> FeatureVector:
    return FeatureVector(booking_id=r["booking_id"], as_of=pd.Timestamp(r["booked_at"]).isoformat(),
                         values={k: r[k] for k in FEATURES})


def state_of(r: dict, gbm: float | None) -> str:
    fv = _fv(r)
    return serialize_v2(fv, evidence_for(fv.values, gbm))


def oof_scores(f: pd.DataFrame, model: GBMModel, k: int = 5) -> pd.Series:
    """Out-of-fold calibrated LightGBM risk for the train window (folds grouped by account; the served
    model's Platt mapping turns the raw fold score into the same scale Laya sees at run time)."""
    tr = f[f.split == "train"]
    X = tr[model.features].to_numpy(dtype=float)
    y = tr.is_fraud.astype(int).to_numpy()
    raw = np.full(len(tr), np.nan)
    params = {**PARAMS, "num_threads": max(1, (os.cpu_count() or 4) - 1)}
    for fit_idx, out_idx in GroupKFold(n_splits=k).split(X, y, groups=tr.account_id.to_numpy()):
        b = lgb.train(params, lgb.Dataset(X[fit_idx], label=y[fit_idx], feature_name=model.features),
                      num_boost_round=NUM_ROUNDS)
        raw[out_idx] = b.predict(X[out_idx])
    a, c = model.platt
    p = np.clip(raw, 1e-6, 1 - 1e-6)
    cal = 1.0 / (1.0 + np.exp(-(a * np.log(p / (1 - p)) + c)))
    return pd.Series(cal, index=tr.booking_id.to_numpy())


def label_row(r: dict) -> dict:
    fraud = bool(r["is_fraud"])
    return {
        "id": r["booking_id"], "workflow": "fraudshield_booking", "seed": int(r["seed"]), "split": r["split"],
        "booking_ts": pd.Timestamp(r["booked_at"]).isoformat(), "account_id": r["account_id"],
        "typology": r["typology"], "is_injected": bool(r["is_injected"]),
        "scenario_id": r["scenario_id"], "campaign_id": r["campaign_id"],
        "camouflage_level": int(r["camouflage_level"]),
        "labels": {"misuse": int(fraud), "foreign_senders": int(r["label_foreign_senders"]),
                   "payoff_max": int(r["label_payoff_max"]), "drop_consignee": int(r["label_drop_consignee"]),
                   "risk_level": int(r["risk_level"])},
        "action_cost": action_costs(float(r["carrier_cost"]), fraud, r["typology"], bool(r["owner_contact_ok"])),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--legit", type=int, default=4500)
    ap.add_argument("--hard-neg", type=int, default=2000)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    served = GBMModel.load(ROOT / "artifacts" / "models", "B2", "F")
    ev_test = pd.read_parquet(V1 / "evalset_test.parquet")
    ev_cal = pd.read_parquet(V1 / "evalset_cal.parquet")

    fraud_parts, hn_parts, legit_pool = [], [], None
    oof_q: dict = {}
    eval_rows: dict[int, pd.DataFrame] = {}
    for k in range(a.seeds):
        f = pd.read_parquet(PROC / "features" / f"seed_{k}.parquet")
        f["seed"] = k
        oof = oof_scores(f, served)
        tr = f[f.split == "train"].assign(gbm=lambda d: d.booking_id.map(oof))
        oof_q[k] = {"pr_auc_oof_train": round(float(M.pr_auc_w(tr.is_fraud.to_numpy().astype(bool), tr.gbm.to_numpy(),
                                                                 np.ones(len(tr)))), 4),
                    "train_fraud": int(tr.is_fraud.sum())}
        tr = tr[~tr.typology.isin(HELD_OUT_TYPOLOGIES)]
        fraud_parts.append(tr[tr.is_fraud.astype(bool)])
        hn_mask = ~tr.is_fraud.astype(bool) & (tr.typology.astype(str).str.startswith("HN_") | tr[list(HN)].any(axis=1))
        hn_parts.append(tr[hn_mask] if k == 0 else tr[hn_mask & tr.is_injected.astype(bool)])
        if k == 0:
            legit_pool = tr[~tr.is_fraud.astype(bool) & ~hn_mask]
        want = set(ev_test.id[ev_test.seed == k]) | set(ev_cal.id[ev_cal.seed == k])
        eval_rows[k] = f[f.booking_id.isin(want)]
        print(f"seed {k}: OOF {oof_q[k]}  ({time.time() - t0:.0f}s)", flush=True)

    fraud = pd.concat(fraud_parts, ignore_index=True)
    hn = pd.concat(hn_parts, ignore_index=True)
    hn = hn.iloc[np.sort(rng.choice(len(hn), size=min(a.hard_neg, len(hn)), replace=False))]
    legit = legit_pool.iloc[np.sort(rng.choice(len(legit_pool), size=min(a.legit, len(legit_pool)), replace=False))]
    train = pd.concat([fraud, hn, legit], ignore_index=True)
    train = train.iloc[rng.permutation(len(train))].reset_index(drop=True)
    withheld = rng.random(len(train)) < a.dropout

    recs = []
    for r, w in zip(train.to_dict("records"), withheld):
        row = label_row(r)
        row["state"] = state_of(r, None if w else float(r["gbm"]))
        rec = to_notebook_record(row, QIDS_V2)
        rec["gbm_withheld"] = bool(w)
        recs.append(rec)
    with open(OUT / "train.jsonl", "w", encoding="utf-8") as fh:
        for rec in recs:
            fh.write(json.dumps(rec) + "\n")

    # evaluation: same rows as v1, re-serialized; gbm = the served model's calibrated score
    feats = pd.concat(eval_rows.values(), ignore_index=True)
    for name, ev in (("test", ev_test), ("cal", ev_cal)):
        g = ev.gbm.astype(float) if "gbm" in ev.columns and ev.gbm.notna().all() else ev.gbm_b2f.astype(float)
        assert g.notna().all(), name
        fx = feats.set_index(["seed", "booking_id"])
        st, st0, costs, fee = [], [], [], []
        for sid, bid, gv in zip(ev.seed, ev.id, g):
            r = {**fx.loc[(sid, bid)].to_dict(), "booking_id": bid}
            st.append(state_of(r, float(gv)))
            st0.append(state_of(r, None))
            costs.append(json.dumps(action_costs(float(r["carrier_cost"]), bool(r["is_fraud"]), r["typology"],
                                                 bool(r["owner_contact_ok"]))))
            fee.append(float(r["carrier_cost"]))
        out = ev.drop(columns=["state"]).assign(gbm=g.to_numpy(), state=st, state_nogbm=st0, action_cost=costs,
                                                carrier_cost=fee)
        out.to_parquet(OUT / f"evalset_{name}.parquet")
        print(f"evalset_{name}: {len(out)} rows", flush=True)

    is_f = train.is_fraud.astype(bool)
    manifest = {
        "version": "laya-v2", "serializer": SERIALIZER_V2,
        "states_sha256": hashlib.sha256((OUT / "train.jsonl").read_bytes()).hexdigest()[:16],
        "seed": a.seed, "fraud_repeat": 1, "questions": list(QIDS_V2), "injection_seeds": a.seeds,
        "bookings": len(train), "unique_bookings": int(train.booking_id.nunique()),
        "fraud_rows": int(is_f.sum()), "fraud_unique": int(train[is_f].booking_id.nunique()),
        "fraud_share": float(is_f.mean()), "hard_neg_rows": len(hn), "legit_rows": len(legit),
        "gbm_withheld_share": float(withheld.mean()), "gbm_dropout": a.dropout,
        "typologies": train.typology.value_counts().to_dict(), "oof_gbm": oof_q,
        "oof_note": "5-fold GroupKFold by account on each seed's train window, served Platt mapping",
        "seconds": round(time.time() - t0, 1),
    }
    (OUT / "prepare.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: v for k, v in manifest.items() if k != "oof_gbm"}, indent=2))


if __name__ == "__main__":
    main()
