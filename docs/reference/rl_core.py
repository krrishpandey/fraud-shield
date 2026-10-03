"""FraudShield RL core sketch: cost model, Bayes rule, logging policy, OPE, attacker.
All money in BRL. Cost parameters are ASSUMPTIONS except F (booking freight, from data)."""
import numpy as np

ACTIONS = ["allow", "allow_scan_gated", "owner_confirm", "review", "hold", "block"]
K = len(ACTIONS)

DEFAULT_PARAMS = dict(
    c_dispute=30.0,     # ops + dispute handling per fraud that gets through
    margin=0.30,        # carrier contribution margin on freight
    goodwill=15.0,      # churn/goodwill proxy when a legit booking is lost
    c_scan=0.50, q_scan=0.40, c_intercept=8.0, ab_scan=0.02,
    c_msg=0.20, fatigue=2.0, p_unreach=0.30, ab_wait=0.30, p_spoof=0.10,
    analyst_per_min=1.00, review_min=6.0, ab_review=0.05, analyst_miss=0.10,
    c_hold=3.0, ab_hold=0.25, c_block_support=5.0,
)


def cost_matrix(F, prm=DEFAULT_PARAMS):
    """(K,2) array C[a, y]; y=0 legit, y=1 fraud; booking freight F in BRL."""
    p = prm
    lost = p["margin"] * F + p["goodwill"]          # value of a lost legit booking
    leak = F + p["c_dispute"]                       # loss if fraud passes
    rev = p["analyst_per_min"] * p["review_min"]
    return np.array([
        [0.0, leak],
        [p["c_scan"] + p["ab_scan"] * lost,
         p["c_scan"] + p["q_scan"] * p["c_intercept"] + (1 - p["q_scan"]) * leak],
        [p["c_msg"] + p["fatigue"] + p["p_unreach"] * p["ab_wait"] * lost,
         p["c_msg"] + p["p_unreach"] * p["c_hold"] + (1 - p["p_unreach"]) * p["p_spoof"] * leak],
        [rev + p["ab_review"] * lost, rev + p["analyst_miss"] * leak],
        [p["c_hold"] + p["ab_hold"] * lost, p["c_hold"]],
        [lost + p["c_block_support"], 0.0],
    ])


def expected_costs(p_fraud, C, allowed=None):
    """dict action -> (1-p)C[a,0] + p C[a,1], restricted to allowed actions."""
    ec = (1 - p_fraud) * C[:, 0] + p_fraud * C[:, 1]
    return {a: float(ec[i]) for i, a in enumerate(ACTIONS) if allowed is None or a in allowed}


def bayes_action(p_fraud, F, prm=DEFAULT_PARAMS, allowed=None):
    ec = expected_costs(p_fraud, cost_matrix(F, prm), allowed)
    return min(ec, key=ec.get)


def logging_policy(p_fraud, F, rng, prm=DEFAULT_PARAMS, allowed=None, tau=2.0, eps=0.05,
                   no_explore_to_allow_above=0.30):
    """Softmax over -expected cost with epsilon floor. Returns (action, propensity, dist).
    Safety: 'allow' gets zero mass above a probability cap, so it is never explored there."""
    ec = expected_costs(p_fraud, cost_matrix(F, prm), allowed)
    acts = list(ec)
    if p_fraud > no_explore_to_allow_above and "allow" in acts and len(acts) > 1:
        acts.remove("allow")
    v = np.array([ec[a] for a in acts])
    s = np.exp(-(v - v.min()) / tau)
    s /= s.sum()
    dist_sub = (1 - eps) * s + eps / len(acts)
    dist = np.zeros(K)
    for a, q in zip(acts, dist_sub):
        dist[ACTIONS.index(a)] = q
    i = rng.choice(K, p=dist)
    return ACTIONS[i], float(dist[i]), dist


# ---------- OPE: rewards r = -realized cost; pi_e is (n,K) target-policy probabilities ----------
def _w(a, pscore, pi_e):
    return pi_e[np.arange(len(a)), a] / pscore


def ips(r, a, pscore, pi_e):
    return float(np.mean(_w(a, pscore, pi_e) * r))


def snips(r, a, pscore, pi_e):
    w = _w(a, pscore, pi_e)
    return float(np.sum(w * r) / np.sum(w))


def dr(r, a, pscore, pi_e, q_hat):
    """q_hat: (n,K) cross-fitted reward model predictions."""
    n = len(a)
    w = _w(a, pscore, pi_e)
    dm = np.sum(pi_e * q_hat, axis=1)
    return float(np.mean(dm + w * (r - q_hat[np.arange(n), a])))


def ess(w):
    return float(w.sum() ** 2 / np.sum(w ** 2))


def paired_bootstrap_diff(est, r, a, pscore, pi_c, pi_b, q_hat=None, B=2000, alpha=0.05, seed=0):
    """Percentile CI on V(candidate) - V(baseline); same resample for both (paired)."""
    rng = np.random.default_rng(seed)
    n = len(a)
    d = np.empty(B)
    for b in range(B):
        i = rng.integers(0, n, n)
        args = (r[i], a[i], pscore[i])
        if q_hat is None:
            d[b] = est(*args, pi_c[i]) - est(*args, pi_b[i])
        else:
            d[b] = est(*args, pi_c[i], q_hat[i]) - est(*args, pi_b[i], q_hat[i])
    return float(np.quantile(d, alpha / 2)), float(np.quantile(d, 1 - alpha / 2))


def dr_pseudo_rewards(r, a, pscore, q_hat):
    """Dudik et al. 2011 DR pseudo-reward for every action. Candidate policy =
    argmax over actions of a regressor fitted to these (cost-sensitive policy learning)."""
    n = len(a)
    G = q_hat.copy()
    G[np.arange(n), a] += (r - q_hat[np.arange(n), a]) / pscore
    return G


def deployment_gate(lo_dr_diff, ess_val, max_w, floor, min_ess=500):
    return lo_dr_diff > 0 and ess_val >= min_ess and max_w <= 1.0 / floor


# ---------- Adaptive attacker: CEM over mutation-operator parameters ----------
THETA_NAMES = ["rate_per_day", "decoy_frac", "weight_cap_q", "express_match", "warmup_days"]
LO = np.array([0.5, 0.0, 0.1, 0.0, 0.0])
HI = np.array([20.0, 0.9, 1.0, 1.0, 21.0])


def attacker_cem(run_campaign, n_iter=15, pop=48, elite_frac=0.2, episodes=8, seed=0):
    """run_campaign(theta, rng) -> (value_extracted_BRL, n_fraud_bookings, n_passed).
    Reward = value extracted before the account is burned (first block / hold /
    owner 'not me' / analyst fraud label) inside the takeover window."""
    rng = np.random.default_rng(seed)
    mu, sd = (LO + HI) / 2, (HI - LO) / 2
    n_el = max(2, int(pop * elite_frac))
    hist = []
    for it in range(n_iter):
        th = np.clip(rng.normal(mu, sd, size=(pop, len(mu))), LO, HI)
        res = np.array([np.mean([run_campaign(t, rng) for _ in range(episodes)], axis=0) for t in th])
        el = np.argsort(res[:, 0])[-n_el:]
        mu, sd = th[el].mean(0), th[el].std(0) + 1e-3 * (HI - LO)
        hist.append(dict(iter=it, best_value=float(res[:, 0].max()),
                         elite_value=float(res[el, 0].mean()),
                         elite_evasion=float(res[el, 2].sum() / max(res[el, 1].sum(), 1e-9)),
                         mu=mu.tolist()))
    return hist, mu
