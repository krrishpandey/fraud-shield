"""The label-free monitor over the live stream's decisions so far (GET /monitor/estimate).

The estimate uses only what production has at decision time: the decision score (probabilities.misuse) and the
action. The stream replays a labelled dataset, so the realized numbers are shown beside it, labelled as simulation
ground truth: in production they only exist once labels arrive. The blind-spot line comes from the offline test
(artifacts/results_monitor.json, scripts/eval_monitor.py).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from fraudshield.monitor import cbpe

REALIZED_LABEL = "simulation ground truth, not available in production until labels arrive"
ESTIMATE_LABEL = "from calibrated scores and the actions taken, no labels needed"
HELD_OUT = ("T3", "T5")
PSI_MIN_N = 500  # fewer bookings than this: the score histogram is too noisy for a drift reading


def load_results(path: str | Path | None) -> dict | None:
    if path is None or not Path(path).exists():
        return None
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def live_estimate(rows: list[dict], results: dict | None) -> dict[str, Any]:
    """rows: StreamMetrics rows (score, action, is_fraud, typology)."""
    p = np.array([float(r.get("score") or 0.0) for r in rows], dtype=float)
    s = cbpe.stopped_mask([r.get("action") for r in rows])
    y = np.array([bool(r.get("is_fraud")) for r in rows], dtype=bool)
    typ = [str(r.get("typology") or "none") for r in rows]
    real = cbpe.realized(y, s)
    missed_by_type: dict[str, int] = {}
    for t, fraud, stop in zip(typ, y, s):
        if fraud and not stop:
            missed_by_type[t] = missed_by_type.get(t, 0) + 1
    real.update({"label": REALIZED_LABEL, "missed_by_type": dict(sorted(missed_by_type.items())),
                 "missed_held_out": sum(v for k, v in missed_by_type.items() if k in HELD_OUT)})
    ref = (results or {}).get("live_reference")
    drift: dict[str, Any] = {"psi": None, "min_n": PSI_MIN_N, "reference": ref.get("window") if ref else None,
                             "note": "input drift of the score distribution; it does not see new fraud the model scores low"}
    if ref and len(p) >= PSI_MIN_N:
        drift["psi"] = cbpe.psi_from_shares(ref["shares"], cbpe.shares(p, np.asarray(ref["edges"], dtype=float)))
    bs = (results or {}).get("blind_spot") or {}
    return {
        "n": int(len(rows)),
        "estimated": {**cbpe.estimate(p, s), "label": ESTIMATE_LABEL},
        "realized": real,
        "drift": drift,
        "blind_spot": {k: bs.get(k) for k in ("ui_note", "missed_gap_per_period", "missed_gap_per_period_sd", "period",
                                              "seeds", "split", "held_out_let_through", "held_out_sum_p_let_through")}
        if bs else None,
        "evidence": "artifacts/results_monitor.md" if results else None,
    }
