"""Training objective, ported from the official notebook (train_ddp.py) with our action-head branch.

Probability heads (misuse, foreign_senders, payoff_max, risk_level): strictly proper reward
(log + spherical, RPS for the ordinal score) via laya.common.proper_reward, REINFORCE on Gaussian
logit noise with a group-mean baseline, plus soft cross-entropy (the notebook objective).
Action head: reward -(q . cost) / C (linear in q, NOT a proper scoring rule; its outputs are never
used as probabilities), CE weight 0. Per-item weights carry the per-question loss weights.
"""
from __future__ import annotations

import torch

SIGMA_START = 0.4
SIGMA_END = 0.1


def sigma_at(progress: float, start: float = SIGMA_START, end: float = SIGMA_END) -> float:
    p = min(max(float(progress), 0.0), 1.0)
    return start + (end - start) * p


def cost_reward(q: torch.Tensor, cost: torch.Tensor, scale: float) -> torch.Tensor:
    return -(q * cost).sum(-1) / float(scale)


def _norm_adv(r: torch.Tensor, sel: torch.Tensor) -> torch.Tensor:
    """Group-mean baseline per item, then scale by the std over the selected items."""
    adv = r - r.mean(0, keepdim=True)
    out = torch.zeros_like(adv)
    if sel.any():
        a = adv[:, sel]
        out[:, sel] = a / (a.std(unbiased=False) + 1e-6) if a.numel() > 1 else a * 0
    return out


def compute_loss(logits: torch.Tensor, batch: dict, sigma: float, group_size: int, cost_scale: float,
                 ce_weight: float = 1.0, w_sph: float = 0.75, w_rps: float = 1.0,
                 generator: torch.Generator | None = None):
    """logits [N, K] (masked with -1e4 outside options). Returns (loss, info dict of floats)."""
    from laya.common import proper_reward  # noqa: PLC0415

    dev = logits.device
    logits = logits.float()
    mask = batch["marker_mask"].to(dev)
    target = batch["target"].to(dev)
    qtype = batch["qtype"].to(dev)
    is_action = batch["is_action"].to(dev).bool()
    cost = batch["cost"].to(dev)
    w = batch["weight"].to(dev).float()
    k = mask.sum(-1, keepdim=True).float()

    noise = torch.randn((group_size,) + tuple(logits.shape), generator=generator,
                        device="cpu" if generator is not None else dev).to(dev)
    eps = noise * sigma * mask
    eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
    z = logits.detach().unsqueeze(0) + eps
    q = torch.softmax(z.masked_fill(~mask, -1e4), -1)
    with torch.no_grad():
        r_prob = proper_reward(q, target.unsqueeze(0), qtype, mask, w_sph=w_sph, w_rps=w_rps)
        r_cost = cost_reward(q, cost.unsqueeze(0), cost_scale)
        adv = _norm_adv(r_prob, ~is_action) + _norm_adv(r_cost, is_action)
    logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
    loss_rl = -(w.unsqueeze(0) * adv * logp).mean()
    ce_item = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1)
    ce_item = ce_item * (~is_action).float() * w
    loss_ce = ce_item.mean()
    loss = loss_rl + ce_weight * loss_ce
    info = {"rl": float(loss_rl.detach()), "ce": float(loss_ce.detach()),
            "reward_prob": float(r_prob[:, ~is_action].mean()) if (~is_action).any() else float("nan"),
            "reward_cost": float(r_cost[:, is_action].mean()) if is_action.any() else float("nan")}
    return loss, info
