import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("laya")

from fraudshield.models.laya_train import objective as O  # noqa: E402


def _batch(targets, qtypes, is_action, costs=None, weights=None):
    n = len(targets)
    k = max(len(t) for t in targets)
    mask = torch.zeros(n, k, dtype=torch.bool)
    tgt = torch.zeros(n, k)
    cost = torch.zeros(n, k)
    for i, t in enumerate(targets):
        mask[i, :len(t)] = True
        tgt[i, :len(t)] = torch.tensor(t)
        if costs and costs[i]:
            cost[i, :len(costs[i])] = torch.tensor(costs[i])
    return {"marker_mask": mask, "target": tgt, "qtype": torch.tensor(qtypes),
            "is_action": torch.tensor(is_action), "cost": cost,
            "weight": torch.tensor(weights or [1.0] * n)}


def test_sigma_schedule_linear():
    assert O.sigma_at(0.0) == pytest.approx(0.4)
    assert O.sigma_at(1.0) == pytest.approx(0.1)
    assert O.sigma_at(0.5) == pytest.approx(0.25)
    assert O.sigma_at(2.0) == pytest.approx(0.1)


def test_cost_reward_is_negative_expected_cost_over_scale():
    q = torch.tensor([[0.5, 0.5, 0.0]])
    cost = torch.tensor([[10.0, 0.0, 99.0]])
    assert O.cost_reward(q, cost, 10.0).item() == pytest.approx(-0.5)


def test_ce_is_zero_for_action_items_and_soft_ce_for_prob_items():
    b = _batch([[1.0, 0.0], [0, 0, 1, 0, 0, 0]], [0, 0], [False, True], costs=[None, [5, 4, 0, 1, 2, 3]])
    logits = torch.zeros(2, 6)
    logits = logits.masked_fill(~b["marker_mask"], -1e4)
    _, info = O.compute_loss(logits, b, sigma=0.4, group_size=4, cost_scale=10.0,
                             generator=torch.Generator().manual_seed(0))
    # only the prob item contributes CE: -log(0.5)
    assert info["ce"] == pytest.approx(math.log(2) / 2, rel=1e-4)


def test_item_weights_scale_ce():
    b1 = _batch([[1.0, 0.0]], [0], [False], weights=[1.0])
    b2 = _batch([[1.0, 0.0]], [0], [False], weights=[2.0])
    lg = torch.zeros(1, 2)
    _, i1 = O.compute_loss(lg, b1, 0.4, 4, 10.0, generator=torch.Generator().manual_seed(0))
    _, i2 = O.compute_loss(lg, b2, 0.4, 4, 10.0, generator=torch.Generator().manual_seed(0))
    assert i2["ce"] == pytest.approx(2 * i1["ce"])


def _train(b, steps=300, lr=0.5, k=None):
    k = k or b["marker_mask"].shape[1]
    logits = torch.zeros(b["marker_mask"].shape, requires_grad=True)
    opt = torch.optim.SGD([logits], lr=lr)
    g = torch.Generator().manual_seed(0)
    for _ in range(steps):
        opt.zero_grad()
        loss, _ = O.compute_loss(logits.masked_fill(~b["marker_mask"], -1e4), b, 0.4, 4, 10.0, generator=g)
        loss.backward()
        opt.step()
    return torch.softmax(logits.detach().masked_fill(~b["marker_mask"], -1e4), -1)


def test_action_head_learns_cheapest_action_from_cost_reward_alone():
    b = _batch([[0, 0, 1.0]], [0], [True], costs=[[10.0, 6.0, 0.0]])
    p = _train(b)
    assert p[0, 2] > 0.8


def test_prob_head_moves_toward_target():
    b = _batch([[1.0, 0.0], [0, 0, 0, 1.0, 0]], [0, 1], [False, False])
    p = _train(b, steps=100, lr=0.2)
    assert p[0, 0] > 0.9 and p[1, 3] > 0.8


def test_rl_term_alone_has_group_mean_baseline():
    # identical rewards across the group -> zero advantage -> zero RL gradient
    b = _batch([[0.5, 0.5]], [0], [False])
    logits = torch.zeros(1, 2, requires_grad=True)
    loss, info = O.compute_loss(logits, b, 0.4, 1, 10.0, ce_weight=0.0, generator=torch.Generator().manual_seed(0))
    loss.backward()
    assert torch.allclose(logits.grad, torch.zeros_like(logits.grad))
