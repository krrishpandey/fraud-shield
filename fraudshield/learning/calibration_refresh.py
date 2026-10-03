"""Refit Laya's misuse Platt map on recent labelled decisions that have raw Laya probabilities.

Rows: labels whose decision carried a raw (uncalibrated) Laya misuse probability. Needs >= 50 rows,
otherwise the refresh is reported as skipped (nothing is faked). Earliest 70% by booking time fit
Platt on logit(raw tempered with the current misuse temperature); the latest 30% compare ECE
(15 equal-mass bins) of the current vs the refitted map. Deployed only if ECE improves.
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from fraudshield.learning.metrics import ece_equal_mass
from fraudshield.learning.retrain import split_feedback_by_time
from fraudshield.models.calibration import apply_calibration, p_yes

MIN_ROWS = 50
EPS = 1e-6


def _logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _calibrated(raw: np.ndarray, cal: dict | None) -> np.ndarray:
    return np.array([p_yes(apply_calibration({"misuse": {"a": float(r), "b": 1 - float(r)}}, cal)[0]["misuse"])
                     for r in raw])


def refresh_misuse_calibration(labels: list[dict[str, Any]], current: dict | None, min_rows: int = MIN_ROWS,
                               min_improvement: float = 0.0) -> dict[str, Any]:
    rows = [x for x in labels if x.get("raw_laya_misuse") is not None]
    out: dict[str, Any] = {"status": "skipped", "n_rows": len(rows), "calibration": None,
                           "current_version": (current or {}).get("version", "uncalibrated")}
    if len(rows) < min_rows:
        out["reason"] = (f"only {len(rows)} labelled decisions have raw Laya probabilities; need at least "
                         f"{min_rows} (Laya was degraded or cached for the others)")
        return out
    fit, ev = split_feedback_by_time(rows, 0.7)
    T = float(((current or {}).get("temperatures") or {}).get("misuse", 1.0))

    def tempered(rs):
        return np.array([p_yes(apply_calibration({"misuse": {"a": r["raw_laya_misuse"],
                                                             "b": 1 - r["raw_laya_misuse"]}},
                                                 {"temperatures": {"misuse": T}})[0]["misuse"]) for r in rs])

    y_fit = np.array([x["label"] == "fraud" for x in fit])
    y_ev = np.array([x["label"] == "fraud" for x in ev])
    if y_fit.all() or not y_fit.any():
        out["reason"] = "fit rows contain only one class"
        return out
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(tempered(fit)).reshape(-1, 1), y_fit.astype(int))
    a, b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
    base = dict(current or {})
    new = {**base, "temperatures": {**(base.get("temperatures") or {}), "misuse": T},
           "platt": {**(base.get("platt") or {}), "misuse": {"a": a, "b": b}}}
    raw_ev = np.array([x["raw_laya_misuse"] for x in ev])
    ece_before = ece_equal_mass(y_ev, _calibrated(raw_ev, current))
    ece_after = ece_equal_mass(y_ev, _calibrated(raw_ev, new))
    out.update(n_fit=len(fit), n_eval=len(ev), ece_before=round(ece_before, 4), ece_after=round(ece_after, 4),
               platt={"a": round(a, 4), "b": round(b, 4)})
    if not (ece_after < ece_before - min_improvement) or math.isnan(ece_after):
        out["status"] = "rejected"
        out["reason"] = "refitted map did not lower ECE on the latest 30% of these rows"
        return out
    h = hashlib.sha256(json.dumps(new, sort_keys=True, default=str).encode()).hexdigest()[:8]
    new["version"] = f"cal-{datetime.now():%Y%m%d}-{h}"
    new["parent"] = base.get("version")
    new["source"] = "learning.calibration_refresh (analyst-labelled decisions)"
    out.update(status="updated", calibration=new, version=new["version"])
    return out
