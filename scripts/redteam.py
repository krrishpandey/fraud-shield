"""We attack our own model. Writes artifacts/results_redteam.md + .json.

Attacker (decision-based black box, Fok et al. arXiv 2508.14699): sees only the returned action, at most 50 queries
and at most 2 changed booker-controlled fields per booking (fraudshield/redteam/search.py, feedback="action").
Target: the deployed system as config/app.yaml runs it (B2_F LightGBM decides, Platt-calibrated, cost rule, rules
floor; Laya not deciding), scored through the real feature path on a per-seed feature store.

Population: test-window fraud bookings (2018-05-15..2018-08-31) of the trained types T1, T2, T4, T6, T7 that the
system STOPS (owner_confirm, review, hold, block). T3 and T5 (never trained) are reported separately and never used
for hardening. Seeds = injection seeds; the deployed model was trained on seed 0's train window only, so every
seed's test window is out of sample. Seeds run in order until the time budget is spent; each seed's population is
subsampled (seeded) to --per-seed bookings.

Also measured (seed 0, test window, in-process, no HTTP): the share of stopped decisions with an analyst
counterfactual of at most 2 changes (feedback="probability", the endpoint's search) and its latency.

Hardening (--harden): evasions generated from TRAIN-window fraud only (seed 0, trained types), labelled fraud
(SIMULATED analyst labels: the label comes from the injected ground truth), run through the existing
LearningService retrain + deployment gate (unchanged code and thresholds) in a scratch directory. The candidate's
test-window flip rate is then measured once (seed 0).

Usage: uv run --no-sync python scripts/redteam.py [--budget-s 600] [--per-seed 40] [--cf-n 60] [--harden]
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fraudshield.api.pipeline import Pipeline  # noqa: E402
from fraudshield.api.real_components import serializer  # noqa: E402
from fraudshield.contracts import ACTIONS, FeatureVector  # noqa: E402
from fraudshield.features.mix import rule_flags, under_score  # noqa: E402
from fraudshield.features.spec import FEATURES  # noqa: E402
from fraudshield.features.store import ROW_COLUMNS, FeatureStore  # noqa: E402
from fraudshield.learning.service import gbm_scorer  # noqa: E402
from fraudshield.models.calibration import load_calibration  # noqa: E402
from fraudshield.models.gbm import GBMModel  # noqa: E402
from fraudshield.models.laya_client import LayaClient  # noqa: E402
from fraudshield.policy.costs import load_costs  # noqa: E402
from fraudshield.policy.decide import PolicyConfig  # noqa: E402
from fraudshield.redteam.search import (BUDGET, MAX_FIELDS, STOP_ACTIONS, WhatIf, apply_changes,  # noqa: E402
                                        batch_gbm, decide_pure, search)

PROC = ROOT / "data" / "processed"
TRAINED = ("T1", "T2", "T4", "T6", "T7")
HELD_OUT = ("T3", "T5")
OUT_MD, OUT_JSON = ROOT / "artifacts" / "results_redteam.md", ROOT / "artifacts" / "results_redteam.json"


# ---------- system under attack ----------
def app_config() -> dict:
    return yaml.safe_load((ROOT / "config" / "app.yaml").read_text(encoding="utf-8"))


def make_pipeline(scorer, cfg: dict) -> Pipeline:
    pol = cfg.get("policy") or {}
    policy = PolicyConfig(**{k: v for k, v in pol.items() if k in PolicyConfig.__dataclass_fields__})
    laya = LayaClient("cached", None, model="none", cache={},
                      mode_reason="laya.decide is false: LightGBM decides (as config/app.yaml)")
    return Pipeline(None, serializer, scorer, laya, audit=None, calibration=load_calibration(ROOT / cfg["calibration_path"]),
                    costs=load_costs(ROOT / cfg["costs_path"]), policy=policy, seed=int(pol.get("seed", 0)))


_REAL: pd.DataFrame | None = None


def seed_store(seed: int) -> FeatureStore:
    """Real bookings + this seed's injected rows + its simulated events (as scripts/export_outputs.py builds seed 0)."""
    global _REAL
    if _REAL is None:
        b = pd.read_parquet(PROC / "bookings_all.parquet")
        _REAL = b[~b.is_injected.astype(bool)][list(ROW_COLUMNS)]
    inj = pd.read_parquet(PROC / "features" / f"injected_seed_{seed}.parquet")
    store = FeatureStore.from_frame(pd.concat([_REAL, inj[list(ROW_COLUMNS)]], ignore_index=True))
    ev = json.loads((PROC / "events" / f"seed_{seed}.json").read_text())
    for c in ev["changes"]:
        store.register_change(c["account_id"], c["kind"], c["at"], c["announced_at"])
    for c in ev["confirmed"]:
        store.confirm_fraud(c["account_id"], c["confirmed_at"], c["entities"])
    return store


def offline_outcomes(pipe, store: FeatureStore, rows: pd.DataFrame) -> list:
    """Decisions from the offline feature table (equal to the online features, tests/features/test_real_equivalence)
    for choosing populations fast. The attack itself always runs through the online path."""
    frame = store.frame.set_index("booking_id", drop=False)
    bks, fvs = [], []
    for r in rows.to_dict("records"):
        v = {k: r[k] for k in FEATURES}
        v.update(rule_flags(v))
        v["under_score"] = under_score(v)
        bks.append(FeatureStore.booking_from_row(frame.loc[r["booking_id"]]))
        fvs.append(FeatureVector(r["booking_id"], str(r["booked_at"]), v))
    scores = batch_gbm(pipe.gbm, fvs)
    return [(b, decide_pure(pipe, b, fv, s)) for b, fv, s in zip(bks, fvs, scores)]


# ---------- attack ----------
def first_hit(res, target: str) -> int | None:
    cur = ACTIONS.index(res.original.action)
    for i, t in enumerate(res.tried):
        a = ACTIONS.index(t.outcome.action)
        if (a == 0) if target == "allow" else (a < cur):
            return i + 1
    return None


def attack(wi: WhatIf, b, original, seed: int, typ: str) -> dict:
    res = search(wi, b, original, budget=BUDGET, max_fields=MAX_FIELDS, feedback="action", seed=seed)
    best = res.best.get("allow") or res.best.get("softer")
    ev = apply_changes(b, best.changes) if best else None
    return {"seed": seed, "typology": typ, "booking_id": b.booking_id, "original": original.action,
            "flip_allow": res.flipped("allow"), "flip_softer": res.flipped("softer"),
            "queries_to_allow": first_hit(res, "allow"), "queries_to_softer": first_hit(res, "softer"),
            "queries": res.evaluations,
            "evasion_action": best.outcome.action if best else None,
            "evasion_fields": [c.field for c in best.changes] if best else [],
            "cost_kept": round(ev.carrier_cost / b.carrier_cost, 4) if ev and b.carrier_cost else None,
            "latency_ms": res.latency_ms, "_res": res}


def run_seed(seed: int, scorer, cfg, per_seed: int, held_n: int, log) -> tuple[list[dict], dict]:
    t0 = time.perf_counter()
    store = seed_store(seed)
    pipe = make_pipeline(scorer, cfg)
    wi = WhatIf(pipe, store)
    f = pd.read_parquet(PROC / "features" / f"seed_{seed}.parquet")
    te = f[(f.split == "test") & f.is_fraud.astype(bool) & f.typology.isin(TRAINED + HELD_OUT)]
    outs = offline_outcomes(pipe, store, te)
    typ = dict(zip(te.booking_id, te.typology.astype(str)))
    stopped = [(b, o) for b, o in outs if o.action in STOP_ACTIONS]
    rng = random.Random(seed)
    pop = []
    for group, cap in ((TRAINED, per_seed), (HELD_OUT, held_n)):
        g = [x for x in stopped if typ[x[0].booking_id] in group]
        pop += rng.sample(g, min(cap, len(g)))
    info = {"seed": seed, "n_test_fraud": {t: int((te.typology == t).sum()) for t in TRAINED + HELD_OUT},
            "n_stopped": {t: sum(typ[b.booking_id] == t for b, _ in stopped) for t in TRAINED + HELD_OUT},
            "n_attacked": {t: sum(typ[b.booking_id] == t for b, _ in pop) for t in TRAINED + HELD_OUT},
            "offline_online_mismatch": 0}
    rows = []
    for b, off in pop:
        on = wi.evaluate([b], wi.view(b))[0]
        if on.action != off.action or abs(on.probability - off.probability) > 1e-6:
            info["offline_online_mismatch"] += 1
        if on.action not in STOP_ACTIONS:
            continue
        rows.append(attack(wi, b, on, seed, typ[b.booking_id]))
    info["seconds"] = round(time.perf_counter() - t0, 1)
    log(f"seed {seed}: attacked {len(rows)} in {info['seconds']} s, mismatches {info['offline_online_mismatch']}")
    return rows, info


# ---------- analyst counterfactuals ----------
def counterfactual_stats(scorer, cfg, n: int, log) -> dict:
    """Seed 0, test window: stopped decisions (fraud and legit), subsample n, the endpoint's search."""
    store = seed_store(0)
    pipe = make_pipeline(scorer, cfg)
    wi = WhatIf(pipe, store)
    f = pd.read_parquet(PROC / "features" / "seed_0.parquet")
    te = f[f.split == "test"]
    outs = offline_outcomes(pipe, store, te)
    stopped = [(b, o) for b, o in outs if o.action in STOP_ACTIONS]
    fraud = dict(zip(te.booking_id, te.is_fraud.astype(bool)))
    sample = random.Random(0).sample(stopped, min(n, len(stopped)))
    res = []
    for b, _ in sample:
        r = search(wi, b, None, feedback="probability")
        res.append({"fraud": fraud[b.booking_id], "action": r.original.action, "softer": r.flipped("softer"),
                    "allow": r.flipped("allow"), "evaluations": r.evaluations, "latency_ms": r.latency_ms,
                    "n_changes": len(r.best["softer"].changes) if r.best.get("softer") else None,
                    "fields": sorted({c.field for k in ("softer", "allow") if r.best.get(k)
                                      for c in r.best[k].changes})})
    lat = [x["latency_ms"] for x in res]
    out = {"split": "test", "seed": 0, "n_stopped_in_window": len(stopped), "n_sampled": len(res),
           "n_fraud_sampled": sum(x["fraud"] for x in res),
           "share_with_softer_cf": float(np.mean([x["softer"] for x in res])) if res else None,
           "share_with_allow_cf": float(np.mean([x["allow"] for x in res])) if res else None,
           "share_with_softer_cf_fraud": _mean([x["softer"] for x in res if x["fraud"]]),
           "share_with_softer_cf_legit": _mean([x["softer"] for x in res if not x["fraud"]]),
           "latency_ms_median": float(np.median(lat)) if lat else None,
           "latency_ms_p90": float(np.percentile(lat, 90)) if lat else None,
           "evaluations_median": float(np.median([x["evaluations"] for x in res])) if res else None,
           "fields_used": _count(f for x in res for f in x["fields"]), "rows": res}
    log(f"counterfactuals: {out['n_sampled']} sampled of {len(stopped)} stopped, softer {out['share_with_softer_cf']}")
    return out


def _mean(xs):
    return float(np.mean(xs)) if len(xs) else None


def _count(it) -> dict:
    c: dict[str, int] = {}
    for x in it:
        c[x] = c.get(x, 0) + 1
    return dict(sorted(c.items(), key=lambda kv: -kv[1]))


# ---------- hardening ----------
def harden(scorer, cfg, n: int, log) -> dict:
    from fraudshield.audit.log import AuditLog
    from fraudshield.learning.label_store import LabelStore, label_from_decision
    from fraudshield.learning.laya_export import LayaExport
    from fraudshield.learning.metrics import GateConfig
    from fraudshield.learning.registry import ModelRegistry
    from fraudshield.learning.service import LearningService
    from fraudshield.learning.wiring import DEFAULTS, load_new_pattern_pool, load_reference

    t0 = time.perf_counter()
    store = seed_store(0)
    pipe = make_pipeline(scorer, cfg)
    wi = WhatIf(pipe, store)
    f = pd.read_parquet(PROC / "features" / "seed_0.parquet")
    tr = f[(f.split == "train") & f.is_fraud.astype(bool) & f.typology.isin(TRAINED)]
    assert not tr.typology.isin(HELD_OUT).any()
    meta = tr.set_index("booking_id")[["typology", "campaign_id"]]
    stopped = [(b, o) for b, o in offline_outcomes(pipe, store, tr) if o.action in STOP_ACTIONS]
    sample = random.Random(1).sample(stopped, min(n, len(stopped)))
    labels, n_flipped = [], 0
    for b, _ in sample:
        on = wi.evaluate([b], wi.view(b))[0]
        if on.action not in STOP_ACTIONS:
            continue
        r = search(wi, b, on, feedback="action", seed=1)
        cur = ACTIONS.index(on.action)
        hits = [t for t in r.tried if ACTIONS.index(t.outcome.action) < cur][:3]
        n_flipped += bool(hits)
        view = wi.view(b)
        for i, t in enumerate(hits):
            eb = apply_changes(b, t.changes)
            fv = wi.featurize(eb, view)
            rec = {"decision_id": f"rt_{b.booking_id}_{i}", "booking_id": f"{b.booking_id}~rt{i}",
                   "account_id": b.account_id, "booked_at": eb.booked_at, "feature_values": dict(fv.values),
                   "action": t.outcome.action, "propensity": 1.0, "explored": False,
                   "model_versions": {"gbm": "gbm-B2-F-v1"},
                   "booking": {"carrier_cost": eb.carrier_cost, "account_id": b.account_id,
                               "meta": {"typology": str(meta.loc[b.booking_id, "typology"]),
                                        "campaign_id": meta.loc[b.booking_id, "campaign_id"]}}}
            labels.append(label_from_decision(rec, "fraud", "simulated_analyst", labelled_at=eb.booked_at,
                                              note="red-team evasion of a train-window fraud booking (synthetic)"))
    log(f"hardening: {len(sample)} train-window stopped fraud attacked, {n_flipped} evaded, {len(labels)} labels")
    scratch = Path(tempfile.mkdtemp(prefix="redteam_scratch_"))
    lc = {**DEFAULTS, **(cfg.get("learning") or {})}
    gate = GateConfig(min_new_labels=int(lc["min_new_labels"]), cost_tolerance_rel=float(lc["cost_tolerance_rel"]),
                      max_ece_increase=float(lc["max_ece_increase"]),
                      max_fpr_hn_increase=float(lc["max_fpr_hn_increase"]), mode=str(lc["gate_mode"]),
                      cost_margin_rel=float(lc["cost_margin_rel"]), pr_auc_margin=float(lc["pr_auc_margin"]))
    registry = ModelRegistry(scratch / "registry.json", ROOT / lc["models_dir"], lc["system"], lc["tier"],
                             versions_dir=scratch / "versions")
    store_l = LabelStore(scratch / "labels.jsonl")
    for lab in labels:
        store_l.append(lab)
    np_paths = sorted((ROOT / "data" / "processed" / "features").glob("seed_*.parquet"))
    svc = LearningService(pipeline=make_pipeline(scorer, cfg), audit=AuditLog(scratch / "audit.jsonl"),
                          registry=registry, label_store=store_l, laya_export=LayaExport(scratch / "laya.jsonl"),
                          reference=lambda: load_reference(ROOT / lc["features_path"]),
                          new_pattern=lambda: load_new_pattern_pool(np_paths), gate_cfg=gate,
                          max_train_rows=lc.get("max_train_rows"), B=int(lc["bootstrap_B"]),
                          calibration_dir=scratch / "calibration", wire_active=False,
                          train_options={"max_feedback_share": lc.get("max_feedback_share"),
                                         "ipw_clip": float(lc["ipw_clip"]), "lgb_params": lc.get("lgb_params") or None})
    out = svc.retrain(min_new_labels=int(lc["min_new_labels"]))
    cand = out["candidate"]["version"]
    log(f"hardening gate: passed={out['gate']['passed']} candidate={cand} ({time.perf_counter() - t0:.0f} s)")
    return {"scratch_dir": str(scratch), "n_train_fraud_stopped": len(stopped), "n_attacked": len(sample),
            "n_evaded": n_flipped, "n_labels": len(labels), "gate": out["gate"], "deployed_in_scratch": out["deployed"],
            "candidate_version": cand, "current": out["current"], "candidate": out["candidate"],
            "new_pattern_eval": out["new_pattern_eval"], "training_set": out["training_set"],
            "candidate_model": registry.load_model(cand)}


# ---------- report ----------
def rates(rows: list[dict], seeds: list[int], types) -> dict:
    out = {}
    for t in [*types, "all"]:
        sel = [r for r in rows if t == "all" and r["typology"] in types or r["typology"] == t]
        per = []
        for s in seeds:
            x = [r for r in sel if r["seed"] == s]
            if x:
                per.append((np.mean([r["flip_allow"] for r in x]), np.mean([r["flip_softer"] for r in x])))
        out[t] = {"n": len(sel), "n_seeds": len(per),
                  "flip_allow_pooled": _mean([r["flip_allow"] for r in sel]),
                  "flip_softer_pooled": _mean([r["flip_softer"] for r in sel]),
                  "flip_allow_mean": float(np.mean([p[0] for p in per])) if per else None,
                  "flip_allow_sd": float(np.std([p[0] for p in per], ddof=1)) if len(per) > 1 else None,
                  "flip_softer_mean": float(np.mean([p[1] for p in per])) if per else None,
                  "flip_softer_sd": float(np.std([p[1] for p in per], ddof=1)) if len(per) > 1 else None}
    return out


def _pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def _ms(m, s):
    return "n/a" if m is None else (f"{100 * m:.1f}%" if s is None else f"{100 * m:.1f} ± {100 * s:.1f}%")


def write(res: dict) -> None:
    a, cf = res["attack"], res["counterfactuals"]
    L = ["# Red team: we attack our own model", "",
         f"Real run of scripts/redteam.py on {res['run_at']}, {res['seconds']:.0f} s CPU wall time. Target: the "
         "deployed system as config/app.yaml runs it (B2_F LightGBM decides, Platt-calibrated, cost rule and rules "
         "floor; Laya not deciding), scored through the real online feature path. Search code: "
         "fraudshield/redteam/search.py (shared with GET /decisions/{id}/counterfactual).", "",
         f"Attacker: sees only the returned action; at most {BUDGET} queries and {MAX_FIELDS} changed fields per "
         "booking. Fields: declared value, declared weight (carrier cost follows the weight at the lane freight "
         "slope, never set freely), parcel size, service, a sender the account already used, booking time.", "",
         "## Flip rates, test window (2018-05-15..2018-08-31), evaluated once", "",
         f"Population: fraud bookings the system STOPS (owner_confirm, review, hold, block). Injection seeds "
         f"{a['seeds']} (as many as fit the time budget); per seed a seeded random subsample of at most "
         f"{a['per_seed']} trained-type and {a['held_n']} never-trained bookings. Mean ± sd over seeds; pooled n.", "",
         "| Type | n attacked | Flip to allow | Flip to any softer action |", "|---|---|---|---|"]
    for t, r in a["rates_trained"].items():
        L.append(f"| {t if t != 'all' else '**all trained**'} | {r['n']} | {_ms(r['flip_allow_mean'], r['flip_allow_sd'])} "
                 f"| {_ms(r['flip_softer_mean'], r['flip_softer_sd'])} |")
    st = {t: sum(i["n_stopped"][t] for i in a["per_seed_info"]) for t in TRAINED}
    tot = {t: sum(i["n_test_fraud"][t] for i in a["per_seed_info"]) for t in TRAINED}
    L += ["", "Stopped at booking over these seeds (before subsampling): "
          + ", ".join(f"{t} {st[t]} of {tot[t]}" for t in TRAINED)
          + ". T6 (under-declared weight) is mostly routed to a depot weight check (allow_scan_gated), not stopped, "
          "so few or no T6 bookings enter the population."]
    L += ["", "Never trained (T3, T5; reported only, never used for hardening):", "",
          "| Type | n attacked | Flip to allow | Flip to any softer action |", "|---|---|---|---|"]
    for t, r in a["rates_held_out"].items():
        L.append(f"| {t if t != 'all' else '**all never-trained**'} | {r['n']} | {_ms(r['flip_allow_mean'], r['flip_allow_sd'])} "
                 f"| {_ms(r['flip_softer_mean'], r['flip_softer_sd'])} |")
    L += ["", f"Fields in the evasions found (trained types): {a['fields_used']}.",
          f"Median queries to the first softer action, when found: {a['median_queries_softer']}; to allow: "
          f"{a['median_queries_allow']}. Median share of the carrier cost an evading booking keeps: "
          f"{_pct(a['median_cost_kept'])} (weight changes move the cost, so evading by under-declaring weight also "
          "shrinks the payoff).",
          f"Offline/online check: {a['mismatches']} of {a['n_checked']} attacked bookings scored differently from "
          "the offline feature table (0 expected).", "",
          "Softer includes allow_scan_gated: the parcel still gets a depot weight check, so a flip to it is a partial "
          "evasion at most.", "",
          "## Analyst counterfactuals (GET /decisions/{id}/counterfactual)", "",
          f"Test window, seed 0: {cf['n_sampled']} stopped decisions sampled at random from {cf['n_stopped_in_window']} "
          f"({cf['n_fraud_sampled']} fraud, the rest legit). The endpoint's search (model probability visible, "
          "bisection refinement), in process without HTTP.", "",
          "| Measure | Value |", "|---|---|",
          f"| Share with a counterfactual of at most 2 changes to a softer action | {_pct(cf['share_with_softer_cf'])} |",
          f"| ... on fraud / on legit stopped bookings | {_pct(cf['share_with_softer_cf_fraud'])} / "
          f"{_pct(cf['share_with_softer_cf_legit'])} |",
          f"| Share with one that reaches plain allow | {_pct(cf['share_with_allow_cf'])} |",
          f"| Latency median / p90 | {cf['latency_ms_median']:.0f} ms / {cf['latency_ms_p90']:.0f} ms |",
          f"| Evaluations used, median | {cf['evaluations_median']:.0f} of {BUDGET} |",
          f"| Fields in the counterfactuals | {cf['fields_used']} |", ""]
    h = res.get("hardening")
    L += ["## Hardening", ""]
    if h is None:
        L.append("Not run (cut for time).")
    else:
        g = h["gate"]
        L += [f"Evasions generated from TRAIN-window fraud only (seed 0, types {', '.join(TRAINED)}): {h['n_attacked']} "
              f"stopped train-window fraud bookings attacked, {h['n_evaded']} evaded, {h['n_labels']} evasive variants "
              "(at most 3 per booking) added as fraud labels (SIMULATED analyst labels: ground truth of synthetic "
              "evasions). They went through the existing LearningService retrain and the pre-registered gate "
              "(docs/LEARNING_GATE.md), code and thresholds unchanged, in a scratch registry (never "
              "artifacts/models/registry.json).", "",
              f"**Gate verdict: {'PASSED' if g['passed'] else 'REJECTED'}** (mode {g.get('mode')}).", "",
              "| Check | Passed | Detail |", "|---|---|---|"]
        for c in g.get("checks", []):
            L.append(f"| {c['name']} | {c['passed']} | {c.get('detail', '')} |")
        cr = h.get("candidate_rates")
        if cr:
            L += ["", f"Candidate {h['candidate_version']} ({'deployed in scratch' if h['deployed_in_scratch'] else 'NOT deployed'}),"
                  " test window seed 0, same attack, evaluated once. Each row attacks the test fraud that model stops "
                  "(seeded subsample), so the two populations differ:", "",
                  "| Model | n stopped fraud attacked | Flip to allow | Flip to softer |", "|---|---|---|---|",
                  f"| v1 (deployed) | {cr['baseline']['n']} | {_pct(cr['baseline']['flip_allow_pooled'])} | "
                  f"{_pct(cr['baseline']['flip_softer_pooled'])} |",
                  f"| candidate | {cr['candidate']['n']} | {_pct(cr['candidate']['flip_allow_pooled'])} | "
                  f"{_pct(cr['candidate']['flip_softer_pooled'])} |"]
    L += ["", "Sources: Fok et al., \"Foe for Fraud: Transferable Adversarial Attacks in Credit Card Fraud Detection\", "
          "arXiv 2508.14699 (2025); Khouna et al., \"Optimal Counterfactual Search in Tree Ensembles\", arXiv "
          "2605.06561 (2026). Fraud rows are synthetic injections on real Olist histories (DATA_CARD.md).", ""]
    OUT_MD.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-s", type=float, default=480.0, help="time budget for the seed loop")
    ap.add_argument("--per-seed", type=int, default=40)
    ap.add_argument("--held-n", type=int, default=10)
    ap.add_argument("--cf-n", type=int, default=60)
    ap.add_argument("--max-seeds", type=int, default=10)
    ap.add_argument("--harden", action="store_true")
    ap.add_argument("--harden-n", type=int, default=80)
    a = ap.parse_args()
    t0 = time.perf_counter()
    log = lambda m: print(f"[{time.perf_counter() - t0:6.0f}s] {m}", flush=True)  # noqa: E731
    cfg = app_config()
    model = GBMModel.load(ROOT / "artifacts" / "models", "B2", "F")
    scorer = gbm_scorer(model)

    cf = counterfactual_stats(scorer, cfg, a.cf_n, log)
    rows, infos, seeds = [], [], []
    per_seed_s = None
    for s in range(a.max_seeds):
        spent = time.perf_counter() - t0
        if per_seed_s is not None and spent + per_seed_s > a.budget_s:
            break
        ts = time.perf_counter()
        r, info = run_seed(s, scorer, cfg, a.per_seed, a.held_n, log)
        rows += r
        infos.append(info)
        seeds.append(s)
        per_seed_s = max(per_seed_s or 0.0, time.perf_counter() - ts)
    tr = [r for r in rows if r["typology"] in TRAINED]
    ok_s = [r["queries_to_softer"] for r in tr if r["queries_to_softer"]]
    ok_a = [r["queries_to_allow"] for r in tr if r["queries_to_allow"]]
    kept = [r["cost_kept"] for r in tr if r["cost_kept"] is not None]
    attack_out = {"split": "test", "seeds": seeds, "per_seed": a.per_seed, "held_n": a.held_n,
                  "rates_trained": rates(rows, seeds, TRAINED), "rates_held_out": rates(rows, seeds, HELD_OUT),
                  "fields_used": _count(f for r in tr for f in r["evasion_fields"]),
                  "median_queries_softer": statistics.median(ok_s) if ok_s else None,
                  "median_queries_allow": statistics.median(ok_a) if ok_a else None,
                  "median_cost_kept": statistics.median(kept) if kept else None,
                  "mismatches": sum(i["offline_online_mismatch"] for i in infos),
                  "n_checked": sum(sum(i["n_attacked"].values()) for i in infos), "per_seed_info": infos}
    hard = None
    if a.harden:
        hard = harden(scorer, cfg, a.harden_n, log)
        cand = hard.pop("candidate_model")
        base = [r for r in rows if r["seed"] == 0 and r["typology"] in TRAINED]
        crow, _ = run_seed(0, gbm_scorer(cand), cfg, a.per_seed, 0, log)
        hard["candidate_rates"] = {"baseline": rates(base, [0], TRAINED)["all"],
                                   "candidate": rates(crow, [0], TRAINED)["all"]}
    res = {"run_at": time.strftime("%Y-%m-%d %H:%M"), "seconds": time.perf_counter() - t0,
           "budget": BUDGET, "max_fields": MAX_FIELDS, "attack": attack_out,
           "counterfactuals": {k: v for k, v in cf.items() if k != "rows"}, "hardening": hard,
           "attack_rows": [{k: v for k, v in r.items() if k != "_res"} for r in rows]}
    OUT_JSON.write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    write(res)


if __name__ == "__main__":
    main()
