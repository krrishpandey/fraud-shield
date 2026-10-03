import numpy as np
from rl_core import ACTIONS, bayes_action, cost_matrix

np.set_printoptions(suppress=True)
print(np.round(cost_matrix(45.0), 2))
for F in [15, 45, 120]:
    for name, allowed in [("owner", None), ("no-owner", [a for a in ACTIONS if a != "owner_confirm"])]:
        prev, out = None, []
        for p in np.linspace(0, 1, 4001):
            b = bayes_action(p, F, allowed=allowed)
            if b != prev:
                out.append((round(float(p), 3), b))
                prev = b
        print(F, name, out)
