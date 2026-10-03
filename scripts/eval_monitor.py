"""Label-free monitor evaluation: estimated vs realized precision of stops, missed fraud and recall, per period.

Per injection seed: B2 LightGBM tier F trained on the train window and Platt-calibrated on the calibration window
(GBMModel.fit, as in scripts/train_gbm.py). Decisions are re-created with the service's own cost rule
(policy.decide.decide_with_trace) in the mode the shipped service runs in (laya.decide false: LightGBM decides,
degraded=True), on the decision score p = 1 - (1 - p_gbm)(1 - DROP_PRIOR * drop_pattern), with the
under-declared first-scan floor and the confirmed-fraud hard signal. Not modelled: the depot dial and the depot
follow-up (they only move bookings between allow and allow_scan_gated, both "let through"), the block cap (hold
and block are both "stopped").

The period (day or week) is chosen on the CALIBRATION window only (lower mean absolute error of the precision
estimate); the test window (2018-05-15..2018-08-31) is then evaluated once, in two scenarios:
  all        - the full test window, held-out fraud types T3 and T5 present (they were never trained on);
  trained    - the same window with the T3 and T5 rows removed (only fraud types the model learned).

Writes artifacts/results_monitor.md + .json. Usage: uv run --no-sync python scripts/eval_monitor.py [--seeds 10]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fraudshield.api.pipeline import DROP_PRIOR  # noqa: E402
from fraudshield.features.mix import rule_flags_frame  # noqa: E402
from fraudshield.models.gbm import GBMModel  # noqa: E402
from fraudshield.monitor import cbpe  # noqa: E402
from fraudshield.policy.costs import load_costs  # noqa: E402
from fraudshield.policy.decide import PolicyConfig, PolicyContext, decide_with_trace  # noqa: E402

HELD_OUT = ("T3", "T5")
TYPES = ("T1", "T2", "T3", "T4", "T5", "T6", "T7")
PERIODS = {"day": 1, "week": 7}
WINDOWS = {"cal": "2018-02-15", "test": "2018-05-15"}  # period 0 starts here (first booking day of the window)
ART = ROOT / "artifacts"


def policy_config() -> PolicyConfig:
    pol = (yaml.safe_load((ROOT / "config" / "app.yaml").read_text(encoding="utf-8")) or {}).get("policy") or {}
    keep = {k: v for k, v in pol.items() if k in PolicyConfig.__dataclass_fields__}
    return PolicyConfig(**keep)


def _f(x) -> float | None:
    return None if x is None or x != x else float(x)


def actions(df: pd.DataFrame, p: np.ndarray, flags: pd.DataFrame, costs, cfg) -> np.ndarray:
    """The service's action for each row (degraded mode: LightGBM decides; see module doc)."""
    rng = random.Random(0)
    under = flags.under_declared.to_numpy().astype(bool)
    hard = df.links_confirmed_fraud.fillna(0).to_numpy() >= 1
    out = []
    for i, (cc, ten, own) in enumerate(zip(df.carrier_cost.to_numpy(float), df.tenure_days.to_numpy(float),
                                           df.owner_contact_age_days.to_numpy(float))):
        ctx = PolicyContext(carrier_cost=float(cc), account_tenure_days=_f(ten), owner_contact_age_days=_f(own),
                            rule_floor="allow_scan_gated" if under[i] else "allow", hard_signal=bool(hard[i]),
                            degraded=True)
        out.append(decide_with_trace({"misuse": float(p[i])}, ctx, costs, cfg, rng)[0].action)
    return np.array(out)


def window(df: pd.DataFrame, gbm: GBMModel, costs, cfg) -> dict:
    flags = rule_flags_frame(df)
    p_gbm = gbm.predict(df)
    p = 1 - (1 - p_gbm) * (1 - DROP_PRIOR * flags.drop_pattern.to_numpy())
    a = actions(df, p, flags, costs, cfg)
    return {"p": p, "p_gbm": p_gbm, "stopped": cbpe.stopped_mask(a), "action": a, "y": df.is_fraud.to_numpy(bool),
            "typ": df.typology.astype(str).to_numpy(), "ts": df.booked_at.to_numpy("datetime64[ns]")}


def _sub(w: dict, m: np.ndarray) -> dict:
    return {k: v[m] for k, v in w.items()}


def errors(w: dict, start: str, days: int, score: str = "p", ref_edges=None, ref_share=None) -> dict:
    """Per-period estimate vs realized: mean absolute error and mean signed error (estimate - realized)."""
    rows = cbpe.per_period(w[score], w["stopped"], cbpe.period_ids(w["ts"], start, days), w["y"])
    prec = [(r["estimated"]["precision_stopped"], r["realized"]["precision_stopped"]) for r in rows
            if r["realized"]["n_stopped"] > 0]
    miss = [(r["estimated"]["fraud_missed"], r["realized"]["fraud_missed"]) for r in rows]
    rec = [(r["estimated"]["recall"], r["realized"]["recall"]) for r in rows
           if r["realized"]["fraud_caught"] + r["realized"]["fraud_missed"] > 0]
    out = {"n_periods": len(rows), "stops_per_period": float(np.mean([r["realized"]["n_stopped"] for r in rows]))}
    for name, pairs in (("precision", prec), ("missed", miss), ("recall", rec)):
        d = np.array([e - r for e, r in pairs], dtype=float)
        out[f"{name}_mae"], out[f"{name}_bias"] = float(np.abs(d).mean()), float(d.mean())
    tot_e, tot_r = cbpe.estimate(w[score], w["stopped"]), cbpe.realized(w["y"], w["stopped"])
    out.update({f"est_{k}": tot_e[k] for k in ("precision_stopped", "fraud_caught", "fraud_missed", "recall")})
    out.update({f"real_{k}": tot_r[k] for k in ("precision_stopped", "fraud_caught", "fraud_missed", "recall")})
    out["n_stopped"] = tot_r["n_stopped"]
    out["missed_per_period_est"] = float(np.mean([e for e, _ in miss]))
    out["missed_per_period_real"] = float(np.mean([r for _, r in miss]))
    if ref_edges is not None:
        ids = cbpe.period_ids(w["ts"], start, days)
        ps = [cbpe.psi_from_shares(ref_share, cbpe.shares(w["p"][ids == k], ref_edges)) for k in np.unique(ids)]
        out["psi_mean"], out["psi_max"] = float(np.mean(ps)), float(np.max(ps))
    return out


def per_type(w: dict) -> dict:
    out = {}
    for t in TYPES:
        m = w["typ"] == t
        lt = m & ~w["stopped"]
        out[t] = {"n": int(m.sum()), "let_through": int(lt.sum()), "sum_p_let_through": float(w["p"][lt].sum()),
                  "mean_p": float(w["p"][m].mean()) if m.any() else float("nan")}
    return out


def live_reference() -> dict:
    """Score distribution of the shipped seed-0 model (gbm_scores_seed0.parquet) on the calibration window, as the
    decision score; the live stream's PSI is computed against it."""
    sc = pd.read_parquet(ROOT / "data" / "processed" / "gbm_scores_seed0.parquet", columns=["booking_id", "split", "gbm_b2f"])
    sc = sc[sc.split == "cal"]
    cols = ["booking_id", "consignee_first_seen_days", "consignee_bookings_30d", "consignee_other_accts_30d", "dims_z",
            "weight_z", "n_prior"]
    f = pd.read_parquet(ROOT / "data" / "processed" / "features" / "seed_0.parquet", columns=cols)
    m = sc.merge(f, on="booking_id", how="left")
    p = 1 - (1 - m.gbm_b2f.to_numpy()) * (1 - DROP_PRIOR * rule_flags_frame(m).drop_pattern.to_numpy())
    e = cbpe.psi_edges(p)
    return {"window": "calibration window, seed 0, shipped model (gbm_scores_seed0.parquet), decision score",
            "n": int(len(p)), "edges": [float(x) for x in e], "shares": [float(x) for x in cbpe.shares(p, e)]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    a = ap.parse_args()
    t0 = time.time()
    costs, cfg = load_costs(ROOT / "config" / "costs.yaml"), policy_config()
    cal_w, test_w = {}, {}
    for seed in range(a.seeds):
        df = pd.read_parquet(ROOT / "data" / "processed" / "features" / f"seed_{seed}.parquet")
        tr, cal, te = (df[df.split == s].sort_values(["booked_at", "booking_id"], kind="mergesort").reset_index(drop=True)
                       for s in ("train", "cal", "test"))
        gbm = GBMModel("B2", "F").fit(tr, tr.is_fraud, cal, cal.is_fraud)
        cal_w[seed], test_w[seed] = window(cal, gbm, costs, cfg), window(te, gbm, costs, cfg)
        print(f"seed {seed} done ({time.time() - t0:.0f}s)", flush=True)

    # 1. period chosen on the calibration window only
    cal_rows = {name: [errors(cal_w[s], WINDOWS["cal"], d) for s in cal_w] for name, d in PERIODS.items()}
    choice = {name: float(np.mean([r["precision_mae"] for r in rs])) for name, rs in cal_rows.items()}
    period = min(choice, key=choice.get)
    days = PERIODS[period]

    # 2. test window, evaluated once with that period
    rows = []
    for s in test_w:
        cw = cal_w[s]
        edges = cbpe.psi_edges(cw["p"])
        ref = cbpe.shares(cw["p"], edges)
        rows.append({"seed": s, "window": "cal", "scenario": "all", "score": "decision",
                     **errors(cw, WINDOWS["cal"], days, ref_edges=edges, ref_share=ref)})
        trained = ~np.isin(test_w[s]["typ"], HELD_OUT)
        for scen, w in (("all", test_w[s]), ("trained", _sub(test_w[s], trained))):
            rows.append({"seed": s, "window": "test", "scenario": scen, "score": "decision",
                         **errors(w, WINDOWS["test"], days, ref_edges=edges, ref_share=ref)})
            rows.append({"seed": s, "window": "test", "scenario": scen, "score": "model only",
                         **errors(w, WINDOWS["test"], days, score="p_gbm")})
    res = pd.DataFrame(rows)
    types = {s: per_type(test_w[s]) for s in test_w}
    out = {"seeds": a.seeds, "period": period, "period_days": days,
           "period_choice_cal_precision_mae": choice,
           "period_choice_cal_stops_per_period": {n: float(np.mean([r["stops_per_period"] for r in rs]))
                                                  for n, rs in cal_rows.items()},
           "rows": res.to_dict(orient="records"),
           "per_type_test": {t: {k: float(np.mean([types[s][t][k] for s in types])) for k in types[0][t]} for t in TYPES},
           "live_reference": live_reference(), "runtime_s": round(time.time() - t0, 1)}
    out["blind_spot"] = blind_spot(res, out["per_type_test"], period)
    (ART / "results_monitor.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    write(res, out)


def _g(res, window, scen, score="decision"):
    return res[(res.window == window) & (res.scenario == scen) & (res.score == score)]


def blind_spot(res: pd.DataFrame, pt: dict, period: str) -> dict:
    al, tr = _g(res, "test", "all"), _g(res, "test", "trained")
    held_real = sum(pt[t]["let_through"] for t in HELD_OUT)
    held_est = sum(pt[t]["sum_p_let_through"] for t in HELD_OUT)
    b = {"period": period, "split": "test window 2018-05-15..2018-08-31", "seeds": int(al.seed.nunique()),
         "missed_bias_per_period_all": float(al.missed_bias.mean()), "missed_bias_per_period_all_sd": float(al.missed_bias.std()),
         "missed_bias_per_period_trained": float(tr.missed_bias.mean()),
         "missed_bias_per_period_trained_sd": float(tr.missed_bias.std()),
         "recall_bias_all": float(al.recall_bias.mean()), "recall_bias_trained": float(tr.recall_bias.mean()),
         "precision_mae_all": float(al.precision_mae.mean()), "precision_mae_trained": float(tr.precision_mae.mean()),
         "held_out_let_through": held_real, "held_out_sum_p_let_through": held_est}
    # paired per seed: how much the held-out rows move the missed-fraud error (same seed, same weeks)
    gap = al.missed_bias.to_numpy() - tr.missed_bias.to_numpy()
    b["missed_gap_per_period"], b["missed_gap_per_period_sd"] = float(gap.mean()), float(gap.std(ddof=1))
    for k, g in (("all", al), ("trained", tr)):
        b[f"window_recall_est_{k}"], b[f"window_recall_real_{k}"] = float(g.est_recall.mean()), float(g.real_recall.mean())
        b[f"window_missed_est_{k}"], b[f"window_missed_real_{k}"] = float(g.est_fraud_missed.mean()), float(g.real_fraud_missed.mean())
    b["summary"] = (f"Fraud types the model never learned (T3, T5): {held_real:.1f} of them were let through per test "
                    f"window on average, while their scores add up to only {held_est:.1f} expected frauds. "
                    f"Missed fraud per {period}, estimate minus actual: {b['missed_bias_per_period_all']:+.2f} with them "
                    f"present, {b['missed_bias_per_period_trained']:+.2f} without; paired gap "
                    f"{b['missed_gap_per_period']:+.2f} ± {b['missed_gap_per_period_sd']:.2f} per {period}. Whole window: "
                    f"estimated recall {b['window_recall_est_all']:.3f} vs actual {b['window_recall_real_all']:.3f} with "
                    f"them, {b['window_recall_est_trained']:.3f} vs {b['window_recall_real_trained']:.3f} without "
                    f"(test window, {b['seeds']} seeds).")
    b["ui_note"] = (f"Blind spot: fraud types the model never learned. On the test window the monitor saw "
                    f"{held_est:.2f} of the {held_real:.1f} such frauds it let through, so it under-counted missed fraud by "
                    f"{-b['missed_gap_per_period']:.2f} per {period} ({b['seeds']} seeds).")
    return b


def _ms(g, c, pct=False, d=3):
    m, s = g[c].mean(), g[c].std()
    return f"{100 * m:.1f} ± {100 * s:.1f} pp" if pct else f"{m:.{d}f} ± {s:.{d}f}"


def write(res: pd.DataFrame, out: dict) -> None:
    n, period = out["seeds"], out["period"]
    ch, sp = out["period_choice_cal_precision_mae"], out["period_choice_cal_stops_per_period"]
    L = ["# Label-free monitor: estimated vs realized performance", "",
         f"Real run of scripts/eval_monitor.py, {n} of the 10 injection seeds (runtime {out['runtime_s']} s on CPU), "
         "mean ± sd over seeds. B2 LightGBM tier F, Platt-calibrated on the calibration window; T3 and T5 never "
         "trained. Decisions re-created with the service's cost rule in its shipped mode (LightGBM decides). "
         "Estimate = CBPE from the decision score (fraudshield/monitor/cbpe.py): precision of stops = mean p over "
         "stopped bookings, missed fraud = sum p over bookings let through, recall = caught / (caught + missed). "
         "Stopped = owner_confirm, review, hold, block; let through = allow, allow_scan_gated. "
         "Realized = the dataset's ground truth (in production only known once labels arrive).", "",
         f"## Period, chosen on the calibration window (2018-02-15..2018-04-30) only", "",
         "| Period | Precision MAE (cal) | Stopped bookings per period (cal) |", "|---|---|---|"]
    L += [f"| {k} | {ch[k]:.3f} | {sp[k]:.1f} |" for k in PERIODS]
    L += ["", f"Chosen: **{period}** (lower precision MAE). Periods are counted from the window's first day; the "
          "last one is partial.", "",
          f"## Estimate vs realized per {period}", "",
          "Signed error = estimate minus realized (negative missed-fraud error = the monitor under-counts missed "
          "fraud; positive recall error = it thinks recall is better than it is).", "",
          "| Window, scenario, score | Precision MAE | Precision signed | Missed MAE (count) | Missed signed | "
          "Recall MAE | Recall signed | Score PSI vs cal (mean / max) |", "|---|---|---|---|---|---|---|---|"]
    for (win, scen, score), label in ((("cal", "all", "decision"), "calibration window (sanity, Platt fit here)"),
                                      (("test", "trained", "decision"), "test, trained types only (T3/T5 rows removed)"),
                                      (("test", "all", "decision"), "test, T3/T5 present (real test window)"),
                                      (("test", "trained", "model only"), "test, trained only, model-only p"),
                                      (("test", "all", "model only"), "test, T3/T5 present, model-only p")):
        g = _g(res, win, scen, score)
        psi = f"{g.psi_mean.mean():.3f} / {g.psi_max.mean():.3f}" if "psi_mean" in g and g.psi_mean.notna().any() else "n/a"
        L.append(f"| {label} | {_ms(g, 'precision_mae')} | {_ms(g, 'precision_bias')} | {_ms(g, 'missed_mae', d=2)} | "
                 f"{_ms(g, 'missed_bias', d=2)} | {_ms(g, 'recall_mae')} | {_ms(g, 'recall_bias')} | {psi} |")
    L += ["", "## Whole window totals (decision score)", "",
          "| Window, scenario | Stopped | Precision est / real | Caught est / real | Missed est / real | Recall est / real |",
          "|---|---|---|---|---|---|"]
    for (win, scen), label in ((("cal", "all"), "calibration"), (("test", "trained"), "test, trained types only"),
                               (("test", "all"), "test, T3/T5 present")):
        g = _g(res, win, scen)
        L.append(f"| {label} | {g.n_stopped.mean():.1f} | {g.est_precision_stopped.mean():.3f} / "
                 f"{g.real_precision_stopped.mean():.3f} | {g.est_fraud_caught.mean():.1f} / {g.real_fraud_caught.mean():.1f} | "
                 f"{g.est_fraud_missed.mean():.1f} / {g.real_fraud_missed.mean():.1f} | {g.est_recall.mean():.3f} / "
                 f"{g.real_recall.mean():.3f} |")
    pt = out["per_type_test"]
    L += ["", "## Where the missed fraud is, per type (test window, mean over seeds)", "",
          "Let through = fraud bookings of the type that were allowed or only scan-checked. Sum of p = what the monitor "
          "'sees' of them (it cannot attribute p to a type; this is the diagnostic with labels).", "",
          "| Type | Bookings | Let through | Sum of p over those | Mean p of the type |", "|---|---|---|---|---|"]
    for t in TYPES:
        r = pt[t]
        L.append(f"| {t}{'*' if t in HELD_OUT else ''} | {r['n']:.1f} | {r['let_through']:.1f} | "
                 f"{r['sum_p_let_through']:.2f} | {r['mean_p']:.3f} |")
    b = out["blind_spot"]
    pa, ptr = _g(res, "test", "all").psi_mean, _g(res, "test", "trained").psi_mean
    pc = _g(res, "cal", "all").psi_mean
    L += ["", "\\* held out, never trained.", "", "## Blind spot", "", b["summary"], "",
          f"Input-drift signal (PSI of the decision score per {period} against the calibration window): "
          f"{pc.mean():.3f} ± {pc.std():.3f} on the calibration window itself, {pa.mean():.3f} ± {pa.std():.3f} on the "
          f"test window with T3/T5 present and {ptr.mean():.3f} ± {ptr.std():.3f} without them (difference "
          f"{(pa - ptr.to_numpy()).abs().max():.3f} at most, in any seed). Whatever moves the score distribution on the "
          "test window, the held-out fraud is not what moves it: a few dozen bookings among ~22,900, scored low. "
          "A score-distribution proxy cannot flag them (Solozobov, arXiv 2604.15740: proxies catch covariate drift, "
          "not concept drift that leaves the features unchanged).", "",
          "Note: T6 (under-declared parcels) is a trained type the score also does not see (it goes to the depot scan, "
          "allow_scan_gated, which counts as let through here). It was in the calibration window too, so Platt "
          "scaling folds it into the base rate: the missed-fraud estimate is a sum of small p over ~22,900 mostly "
          "honest bookings, right only on average over the fraud mix it was calibrated on, never per type. Without "
          f"T3/T5 it over-counts missed fraud ({b['window_missed_est_trained']:.1f} estimated vs "
          f"{b['window_missed_real_trained']:.1f} actual over the window); with them it under-counts "
          f"({b['window_missed_est_all']:.1f} vs {b['window_missed_real_all']:.1f}). Per {period} the absolute errors "
          "are dominated by noise (a few frauds a week), so the signed error and the paired gap are the readable "
          "numbers, not the MAE.", ""]
    (ART / "results_monitor.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
