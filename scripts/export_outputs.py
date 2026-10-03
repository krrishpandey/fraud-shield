"""Export outputs for other agents (run after build_dataset.py and train_gbm.py):
  data/processed/states.jsonl            serialized states + labels + action cost vectors
  data/processed/feature_store_demo.parquet (+ .events.json)  history the API loads at startup
  data/processed/demo_bookings.json      scripted demo bookings (GET /demo/bookings format)
Usage: uv run python scripts/export_outputs.py [--seeds 10] [--train-legit 15000]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fraudshield.contracts import FeatureVector  # noqa: E402
from fraudshield.data.dataset import label_real  # noqa: E402
from fraudshield.data.demo import DEMO_BOOKINGS_PATH, DEMO_STORE_PATH, booking_dict  # noqa: E402
from fraudshield.data.labels import COST_VERSION, action_costs  # noqa: E402
from fraudshield.data.pipeline import real_with_billing  # noqa: E402
from fraudshield.features.spec import FEATURES  # noqa: E402
from fraudshield.features.serialize import SERIALIZER_VERSION, serialize  # noqa: E402
from fraudshield.features.store import FeatureStore  # noqa: E402

PROC = ROOT / "data" / "processed"
HN = ("hn_new_state", "hn_new_seller", "hn_billing_change", "hn_injected", "hn_multi_account_consignee")


def _num(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else x


def state_records(f: pd.DataFrame, seed: int, scores: pd.Series | None):
    for r in f.to_dict("records"):
        fv = FeatureVector(booking_id=r["booking_id"], as_of=pd.Timestamp(r["booked_at"]).isoformat(),
                           values={k: r[k] for k in FEATURES})
        fraud = bool(r["is_fraud"])
        yield {
            "id": r["booking_id"], "workflow": "fraudshield_booking", "seed": seed, "split": r["split"],
            "booking_ts": pd.Timestamp(r["booked_at"]).isoformat(), "account_id": r["account_id"],
            "typology": r["typology"], "is_injected": bool(r["is_injected"]), "scenario_id": r["scenario_id"],
            "campaign_id": r["campaign_id"], "camouflage_level": int(r["camouflage_level"]),
            "state": serialize(fv),
            "labels": {"misuse": int(fraud), "foreign_senders": int(r["label_foreign_senders"]),
                       "payoff_max": int(r["label_payoff_max"]), "drop_consignee": int(r["label_drop_consignee"]),
                       "risk_level": int(r["risk_level"])},
            "action_cost": action_costs(float(r["carrier_cost"]), fraud, r["typology"], bool(r["owner_contact_ok"])),
            "hard_negative": [h.removeprefix("hn_") for h in HN if r[h]],
            "gbm_b2f": _num(float(scores.get(r["booking_id"], np.nan))) if scores is not None and r["split"] != "train" else None,
        }


def export_states(seeds: int, train_legit: int) -> dict:
    sc = pd.read_parquet(PROC / "gbm_scores_seed0.parquet").set_index("booking_id").gbm_b2f
    rng = np.random.default_rng(0)
    n = 0
    counts: dict = {}
    with open(PROC / "states.jsonl", "w", encoding="utf-8") as fh:
        for seed in range(seeds):
            f = pd.read_parquet(PROC / "features" / f"seed_{seed}.parquet")
            if seed == 0:
                keep = f.split.isin(["cal", "test"]) | ((f.split == "train") & (f.is_injected | f[list(HN)].any(axis=1)))
                legit_train = f.index[(f.split == "train") & ~keep]
                keep.loc[rng.choice(legit_train, size=min(train_legit, len(legit_train)), replace=False)] = True
            else:
                keep = (f.split == "test") & f.is_injected
            sub = f[keep]
            for rec in state_records(sub, seed, sc if seed == 0 else None):
                fh.write(json.dumps(rec) + "\n")
                n += 1
            counts[seed] = sub.groupby("split").size().to_dict()
    return {"rows": n, "per_seed": counts}


def export_demo(seeds_file: int = 0) -> list[dict]:
    real, _, legit_changes, _ = real_with_billing()
    inj = pd.read_parquet(PROC / "features" / f"injected_seed_{seeds_file}.parquet")
    rows = pd.concat([label_real(real), inj], ignore_index=True).sort_values(["booked_at", "booking_id"])
    f = pd.read_parquet(PROC / "features" / f"seed_{seeds_file}.parquet").set_index("booking_id")
    d = rows.join(f[[c for c in f.columns if c not in rows.columns]], on="booking_id")
    test = d[(d.split == "test") & (d.booked_at >= "2018-06-01")]

    legit = test[~test.is_injected & (test.n_prior >= 200) & (test.lane_seen >= 5) & (test.hn_change.fillna("") == "")
                 & test.cost_vs_median.between(0.8, 1.25) & (test.consignee_prior == 0)].iloc[0]
    hn = test[test.hn_new_state & (test.n_prior >= 50) & (test.hn_change.fillna("") == "")].iloc[0]
    t1 = test[(test.typology == "T1") & (test.phase == "B") & (test.camouflage_level == 0)]
    t1 = t1.sort_values(["new_senders_l10", "booked_at"], ascending=[False, True]).iloc[0]
    t3 = test[test.typology == "T3"].sort_values(["consignee_other_accts_30d", "booked_at"], ascending=[False, True]).iloc[0]
    t6 = test[test.typology == "T6"].assign(r=lambda x: x.true_weight_kg / x.weight_kg).sort_values("r", ascending=False).iloc[0]

    items = [
        {"scenario": "legit-tenured", "title": "Established seller, usual lane",
         "description": f"Real Olist booking. The account has {int(legit.n_prior)} earlier bookings and has used this lane "
                        f"{int(legit.lane_seen)} times; the cost is close to its usual cost.",
         "expected": "legit", "row": legit},
        {"scenario": "hard-negative-new-state", "title": "Established seller, first parcel to a new state",
         "description": f"Real Olist booking (a real hard negative). The account has {int(hn.n_prior)} earlier bookings and "
                        f"ships to {hn.dest_uf} for the first time. Nothing else changed.",
         "expected": "legit", "row": hn},
        {"scenario": "takeover", "title": "Label-resale account takeover (injected T1)",
         "description": "Injected campaign on a real seller's history: the account now pays for parcels from senders and "
                        "origins it never used before (label resale after takeover).",
         "expected": "T1", "row": t1},
        {"scenario": "reshipping-drop", "title": "Parcel to a reshipping drop (injected T3, held out)",
         "description": f"Injected campaign: a recently seen consignee that received parcels from "
                        f"{int(t3.consignee_other_accts_30d)} other accounts in the last 30 days. T3 is never used in training.",
         "expected": "T3", "row": t3},
        {"scenario": "weight-manipulation", "title": "Declared weight below the real weight (injected T6)",
         "description": f"Injected campaign: declared {t6.weight_kg:.2f} kg while the parcel really weighs "
                        f"{t6.true_weight_kg:.2f} kg (revealed only at first scan, simulated).",
         "expected": "T6", "row": t6},
    ]
    out = [{k: v for k, v in it.items() if k != "row"} | {"booking": booking_dict(it["row"])} for it in items]
    DEMO_BOOKINGS_PATH.write_text(json.dumps(out, indent=2))

    demo_ids = {it["booking"]["booking_id"] for it in out}
    store = FeatureStore.from_frame(rows[~rows.booking_id.isin(demo_ids)])
    ev = json.loads((PROC / "events" / f"seed_{seeds_file}.json").read_text())
    for c in ev["changes"]:
        store.register_change(c["account_id"], c["kind"], c["at"], c["announced_at"])
    for c in ev["confirmed"]:
        store.confirm_fraud(c["account_id"], c["confirmed_at"], c["entities"])
    store.save(DEMO_STORE_PATH)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--train-legit", type=int, default=15000)
    a = ap.parse_args()
    demo = export_demo()
    print("demo:", [(d["scenario"], d["booking"]["booking_id"], d["booking"]["booked_at"]) for d in demo])
    st = export_states(a.seeds, a.train_legit)
    manifest = {"serializer": SERIALIZER_VERSION, "cost_version": COST_VERSION, **st}
    (PROC / "states_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: v for k, v in manifest.items() if k != "per_seed"}))


if __name__ == "__main__":
    main()
