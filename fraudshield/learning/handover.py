"""Model handover: which GBM scores new bookings from now on, in plain words (docs/API.md v1.2 "handover").

After every retrain or rollback the API states one verdict, the versions involved, since when the active
model has been serving, the gate checks that failed (translated from the gate's numbers into sentences)
and the version a rollback would go back to. Built from the structured report, never by parsing the
gate's detail strings.
"""
from __future__ import annotations

from typing import Any

from fraudshield.learning.metrics import GateConfig

NEW_IN_USE = "new_model_in_use"
KEPT = "previous_model_kept"
ROLLED_BACK = "rolled_back"


def _pct(x: float, k: int = 2) -> str:
    return f"{x * 100:.{k}f}%"


def plain_check(check: dict[str, Any], rep: dict[str, Any], cfg: GateConfig, n_new_labels: int) -> str:
    """One failed gate check as a sentence a non-specialist can read."""
    name, cur, cand = check["name"], rep.get("current") or {}, rep.get("candidate") or {}
    boot = rep.get("bootstrap") or {}
    if name in ("cost_noninferior", "cost_not_worse"):
        lo, hi = (boot.get("cost_improvement_ci95") or [float("nan")] * 2)[:2]
        rel = cfg.cost_margin_rel if name == "cost_noninferior" else cfg.cost_tolerance_rel
        tol = rel * float(cur["cost_per_1k_brl"])
        return (f"the cost per 1,000 bookings could get worse by up to R${-lo:.2f} (95% range of the saving: "
                f"R${lo:.2f} to R${hi:.2f}); the limit is R${tol:.2f}, {rel:.0%} of the current "
                f"R${cur['cost_per_1k_brl']:.2f}")
    if name == "pr_auc_noninferior":
        return (f"ranking quality (PR-AUC) would fall from {cur['pr_auc']:.3f} to {cand['pr_auc']:.3f}; "
                f"the limit is {cur['pr_auc'] - cfg.pr_auc_margin:.3f}")
    if name == "pr_auc_not_worse":
        lo, hi = (boot.get("pr_auc_diff_ci95") or [float("nan")] * 2)[:2]
        return (f"ranking quality (PR-AUC) would fall from {cur['pr_auc']:.3f} to {cand['pr_auc']:.3f}, and the "
                f"whole 95% range of the change ({lo:+.4f} to {hi:+.4f}) is below zero")
    if name in ("ece_noninferior", "ece_not_worse"):
        return (f"calibration error would rise from {cur['ece']:.3f} to {cand['ece']:.3f}; "
                f"the limit is +{cfg.max_ece_increase:.2f}")
    if name in ("fpr_hard_negative_noninferior", "fpr_hard_negative_not_worse"):
        return (f"honest hard-case false positives would rise from {_pct(cur['fpr_hard_negative'])} to "
                f"{_pct(cand['fpr_hard_negative'])}; the limit is +{cfg.max_fpr_hn_increase * 100:.1f} points")
    if name == "superiority":
        lo, hi = (boot.get("cost_improvement_ci95") or [float("nan")] * 2)[:2]
        npe = rep.get("new_pattern_eval")
        txt = (f"no proven improvement: the cost saving is not above zero with 95% confidence "
               f"(R${lo:.2f} to R${hi:.2f} per 1,000 bookings)")
        if npe and npe.get("recall_diff_ci95"):
            a, b = npe["recall_diff_ci95"]
            txt += f", and neither is the gain in new-pattern recall ({a * 100:+.1f} to {b * 100:+.1f} points)"
        else:
            txt += ", and no new-pattern eval set was available"
        return txt
    if name == "min_new_labels":
        return f"only {n_new_labels} new labels since the last retrain; at least {cfg.min_new_labels} are needed"
    return check.get("detail", name)


def new_pattern_summary(rep: dict[str, Any]) -> dict[str, Any] | None:
    """Recall on patterns the active model never trained on (eval set b if present, else the eval-set slice)."""
    npe = rep.get("new_pattern_eval")
    if npe:
        return {"current": npe["current"]["recall"], "candidate": npe["candidate"]["recall"],
                "ci95": npe.get("recall_diff_ci95"), "typologies": npe.get("typologies"), "source": "new-pattern eval set"}
    a, b = (rep.get("current") or {}).get("recall_new_pattern"), (rep.get("candidate") or {}).get("recall_new_pattern")
    if a is None or b is None:
        return None
    return {"current": a, "candidate": b, "ci95": None, "typologies": rep.get("new_pattern_typologies"),
            "source": "overall eval set"}


def retrain_handover(rep: dict[str, Any], cfg: GateConfig, n_new_labels: int, *, run_id: str, at: str,
                     previous: str, candidate: str, active: str, active_since: str | None,
                     rollback_target: str | None) -> dict[str, Any]:
    passed = bool(rep["gate"]["passed"])
    failed = [{"name": c["name"], "plain": plain_check(c, rep, cfg, n_new_labels), "detail": c["detail"]}
              for c in rep["gate"]["checks"] if not c["passed"]]
    learned = new_pattern_summary(rep)
    if passed:
        msg = (f"From now on, new bookings are scored by {active} (since {active_since}). "
               f"Previous model {previous} is kept for rollback.")
    else:
        gain = ""
        if learned and learned["candidate"] > learned["current"]:
            gain = (f" learned the new pattern (new-pattern recall {_pct(learned['current'], 1)} -> "
                    f"{_pct(learned['candidate'], 1)}) but")
        msg = (f"Still using {active} (since {active_since}). Candidate {candidate}{gain} was not deployed because: "
               + " ".join(f"({i}) {f['plain']}." for i, f in enumerate(failed, 1)))
    return {"event": "retrain", "run_id": run_id, "at": at, "verdict": NEW_IN_USE if passed else KEPT,
            "active_version": active, "previous_version": previous, "candidate_version": candidate,
            "active_since": active_since, "rollback_target": rollback_target, "failed_checks": failed,
            "new_pattern": learned, "message": msg}


def rollback_handover(*, at: str, previous: str, active: str, rollback_target: str | None) -> dict[str, Any]:
    msg = (f"Rolled back. From now on, new bookings are scored by {active} (since {at}). "
           f"{previous} is no longer used for new bookings.")
    return {"event": "rollback", "run_id": None, "at": at, "verdict": ROLLED_BACK, "active_version": active,
            "previous_version": previous, "candidate_version": None, "active_since": at,
            "rollback_target": rollback_target, "failed_checks": [], "new_pattern": None, "message": msg}
