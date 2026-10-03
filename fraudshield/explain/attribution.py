"""Does an explanation cite the reasons the model actually leaned on?

LightGBM's pred_contrib gives each feature's contribution to one booking's score (TreeSHAP, in log-odds; the Platt
step after it is monotone, so the ranking is the same). Contributions are summed per reason code with FEATURE_CODE
(features that no reason code describes are left out, and their share is reported), and the model's top reasons are
the codes with the largest positive sums, at most 3.

Agreement for one explanation (well defined, kept simple):
  cited  = the first 3 distinct reason codes the explanation cites, in order
  model  = the model's top reason codes (at most 3)
  hits   = |cited & model|, of = len(model), hit@3 = hits / of, top1 = cited[0] == model[0]
  reachable = |decision reasons & model|: the most hits any explanation can get, since a validated explanation
              may only cite the decision's own reasons (codes whose rule fired).
Why measure it: LLM self-explanations disagree with SHAP, and LightGBM SHAP is the more reliable account of what
drives a score on financial tabular data (arXiv 2512.00163).
"""
from __future__ import annotations

from typing import Any

import numpy as np

# Model feature -> the reason code that describes it (the code's own rule feature plus the features measuring the
# same thing). Kept small and explicit; features absent here (distance, size in cm, category, channel, ...) are
# not described by any reason code.
FEATURE_CODE: dict[str, str] = {
    "cost_vs_median": "COST_FAR_ABOVE_ACCOUNT_NORM", "payoff_z": "COST_FAR_ABOVE_ACCOUNT_NORM",
    "new_senders_l10": "NEW_SENDERS_UNDER_PAYER", "base_senders_l10": "NEW_SENDERS_UNDER_PAYER",
    "distinct_senders_7d": "NEW_SENDERS_UNDER_PAYER", "base_senders_7d": "NEW_SENDERS_UNDER_PAYER",
    "sender_entropy_jump": "NEW_SENDERS_UNDER_PAYER", "sender_seen": "NEW_SENDERS_UNDER_PAYER",
    "payer_sender_pair_age_days": "NEW_SENDERS_UNDER_PAYER",
    "new_origins_l10": "NEW_ORIGINS", "base_origins_l10": "NEW_ORIGINS", "distinct_origins_7d": "NEW_ORIGINS",
    "base_origins_7d": "NEW_ORIGINS", "origin_entropy_jump": "NEW_ORIGINS", "origin_seen": "NEW_ORIGINS",
    "payer_origin_pair_age_days": "NEW_ORIGINS", "is_home_origin": "NEW_ORIGINS", "payer_origins_30d": "NEW_ORIGINS",
    "login_device_age_days": "NEW_LOGIN_DEVICE",
    "sender_differs": "SENDER_DIFFERS_FROM_ACCOUNT",
    "payer_senders_30d": "PAYER_MANY_SENDERS",
    "consignee_accts_30d": "CONSIGNEE_MANY_SENDERS",
    "consignee_first_seen_days": "DROP_ADDRESS_PATTERN", "consignee_other_accts_30d": "DROP_ADDRESS_PATTERN",
    "consignee_bookings_30d": "DROP_ADDRESS_PATTERN", "consignee_prior": "DROP_ADDRESS_PATTERN",
    "weight_z": "WEIGHT_UNUSUAL",
    "dims_z": "DIMS_UNUSUAL",
    "burst_ratio": "BURST_LAST_24H", "bookings_24h": "BURST_LAST_24H", "bookings_72h": "BURST_LAST_24H",
    "tenure_days": "NEW_ACCOUNT", "n_prior": "NEW_ACCOUNT",
    "hour_pct": "UNUSUAL_HOUR", "hour": "UNUSUAL_HOUR", "night": "UNUSUAL_HOUR",
    "links_confirmed_fraud": "LINK_TO_CONFIRMED_FRAUD", "prior_confirmed_fraud": "LINK_TO_CONFIRMED_FRAUD",
}
TOP_K = 3


def model_of(gbm: Any) -> Any:
    """The GBMModel behind the pipeline's scorer: a learning-deployed version (a gbm_scorer closure) or the base
    model of the real components. None if the scorer is not a LightGBM model (tests, fallback heuristic)."""
    for cell in getattr(gbm, "__closure__", None) or ():
        try:
            m = cell.cell_contents
        except ValueError:
            continue
        if hasattr(m, "booster") and hasattr(m, "features"):
            return m
    m = getattr(gbm, "__self__", None)
    if hasattr(m, "booster") and hasattr(m, "features"):
        return m
    if getattr(gbm, "__module__", "") == "fraudshield.api.real_components" and getattr(gbm, "__name__", "") == "gbm":
        from fraudshield.api import real_components  # noqa: PLC0415
        return real_components._get_gbm()
    return None


def _f(x: Any) -> float:
    try:
        return float("nan") if x is None else float(x)
    except (TypeError, ValueError):
        return float("nan")


def contributions(model: Any, values: dict[str, Any]) -> np.ndarray:
    """Per-feature contributions (log-odds) for one booking, aligned with model.features (bias column dropped)."""
    row = np.array([[_f(values.get(f)) for f in model.features]], dtype=float)
    return np.asarray(model.booster.predict(row, pred_contrib=True))[0][:-1]


def model_reasons(model: Any, values: dict[str, Any], k: int = TOP_K) -> dict[str, Any]:
    c = contributions(model, values)
    by_code: dict[str, float] = {}
    for f, v in zip(model.features, c):
        code = FEATURE_CODE.get(f)
        if code is not None:
            by_code[code] = by_code.get(code, 0.0) + float(v)
    codes = sorted(((v, code) for code, v in by_code.items() if v > 0), reverse=True)[:k]
    pos = c[c > 0].sum()
    mapped = sum(v for f, v in zip(model.features, c) if v > 0 and f in FEATURE_CODE)
    order = np.argsort(-c)[:5]
    return {
        "top_codes": [{"code": code, "contribution": round(v, 4)} for v, code in codes],
        "top_features": [{"name": model.features[i], "value": None if np.isnan(_f(values.get(model.features[i])))
                          else round(_f(values.get(model.features[i])), 4),
                          "contribution": round(float(c[i]), 4), "code": FEATURE_CODE.get(model.features[i])}
                         for i in order if c[i] > 0],
        "mapped_share": round(float(mapped / pos), 4) if pos > 0 else None,
    }


def agreement(cited_codes: list[str], model_codes: list[str], reasons: list[str] | None = None) -> dict[str, Any]:
    cited = list(dict.fromkeys(c for c in cited_codes if c))[:TOP_K]
    model = list(model_codes)[:TOP_K]
    out: dict[str, Any] = {"cited": cited, "model": model, "hits": None, "of": len(model), "hit_at_3": None,
                           "top1_match": None, "reachable": None}
    if not model:
        return out
    hits = len(set(cited) & set(model))
    out.update(hits=hits, hit_at_3=round(hits / len(model), 4), top1_match=bool(cited) and cited[0] == model[0])
    if reasons is not None:
        out["reachable"] = len(set(reasons) & set(model))
    return out


def attribution_view(record: dict[str, Any], gbm: Any, cited_codes: list[str]) -> dict[str, Any]:
    """Attribution and agreement for the console; 'available': False with a reason when it cannot be computed."""
    values = record.get("feature_values")
    if not values:
        return {"available": False, "why": "this decision has no stored feature values"}
    model = model_of(gbm)
    if model is None:
        return {"available": False, "why": "the scoring model is not a LightGBM model"}
    try:
        mr = model_reasons(model, values)
    except Exception as e:  # never break the decision page over the attribution
        return {"available": False, "why": f"{type(e).__name__}: {e}"}
    ag = agreement(cited_codes, [x["code"] for x in mr["top_codes"]], list(record.get("reasons") or []))
    version = (getattr(model, "metadata", None) or {}).get("active_version") or (record.get("model_versions") or {}).get("gbm")
    return {"available": True, "method": "LightGBM pred_contrib (TreeSHAP), summed per reason code",
            "model_version": version, **mr, "agreement": ag}
