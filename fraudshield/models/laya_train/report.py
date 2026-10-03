"""Rule-based, honest verdict for results_laya.md (DESIGN section 10 win conditions).
A side "wins" only when the 95% bootstrap CI of the paired difference excludes zero."""
from __future__ import annotations


def verdict_text(res: dict, fpr: float = 0.01) -> str:
    rows = [r for r in res["headline_a"] if r.get("fpr") == fpr]
    parts, a_met = [], []
    for r in rows:
        lo, hi = r["M1_minus_B2_ci"]
        if lo > 0:
            a_met.append(r["typology"])
            parts.append(f"on held-out {r['typology']} the fine-tuned Laya catches more than the GBM at {fpr:.0%} "
                         f"legit friction (recall {r['M1']:.3f} vs {r['B2']:.3f}, difference CI [{lo:.3f}, {hi:.3f}])")
        elif hi < 0:
            parts.append(f"on held-out {r['typology']} the GBM catches more than the fine-tuned Laya "
                         f"(recall {r['B2']:.3f} vs {r['M1']:.3f}, difference CI [{lo:.3f}, {hi:.3f}])")
        else:
            parts.append(f"on held-out {r['typology']} the two are not distinguishable (Laya {r['M1']:.3f}, "
                         f"GBM {r['B2']:.3f}, difference CI [{lo:.3f}, {hi:.3f}])")
    hb = res["headline_b"]
    lo, hi = hb["M1_minus_B3_auc_ci"]
    b_met = lo > 0
    if b_met:
        zs = (f"the zero-shot drop_consignee question on the fine-tuned model beats stock Laya on T3 "
              f"(ROC-AUC {hb['M1']['auc']:.3f} vs {hb['B3']['auc']:.3f}, difference CI [{lo:.3f}, {hi:.3f}])")
    elif hi < 0:
        zs = (f"fine-tuning made the zero-shot drop_consignee question worse on T3 "
              f"(ROC-AUC {hb['M1']['auc']:.3f} vs stock {hb['B3']['auc']:.3f}, CI [{lo:.3f}, {hi:.3f}])")
    else:
        zs = (f"the zero-shot drop_consignee question is not distinguishable between fine-tuned and stock Laya "
              f"(ROC-AUC {hb['M1']['auc']:.3f} vs {hb['B3']['auc']:.3f}, CI [{lo:.3f}, {hi:.3f}])")
    head = ("Win condition (a) " + (f"met for {', '.join(a_met)}" if a_met else "not met") + "; condition (b) "
            + ("met" if b_met else "not met") + ". ")
    tail = ("" if (a_met or b_met) else
            "So the GBM stays the ranker and Laya is presented as the calibrated, question-level evidence layer "
            "(why a booking looks like misuse, which mode it resembles), not as the detector. ")
    body = "; ".join(parts) + "; and " + zs
    return head + body[:1].upper() + body[1:] + ". " + tail + \
        "Held-out T3 has few campaigns, so its intervals are wide."
