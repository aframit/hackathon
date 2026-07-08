"""End-to-end demo: many small expert orderings -> regroup -> refit -> recover.

We invent a hidden "true" weighting, use it to order random subsets of scenarios
(simulating experts), feed those partial orderings through the workflow, re-fit
starting from the known WHC weights, and check we recover the global ranking.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from refit import RefitWorkflow, load_scenarios
from refit import scoring


def make_true_params(seed: int = 0, sigma: float = 0.8) -> dict:
    """A hidden "true" weighting: the known WHC constants, perturbed.

    Each effective weight is multiplied by a lognormal factor (so it stays
    positive), then mapped back to raw space. This is the expert model the
    re-fit must recover from orderings alone.
    """
    rng = np.random.default_rng(seed)
    true = {}
    for k, v in scoring.CONSTANTS.items():
        v = np.asarray(v, dtype=float)
        factor = np.exp(rng.normal(0.0, sigma, size=v.shape))
        true[k] = scoring._inv_sp(jnp.asarray(v * factor))
    return true


def held_out_agreement(true_s, est_s, idx, rng, n=4000) -> float:
    ts, es = true_s[idx], est_s[idx]
    a, b = rng.integers(0, len(idx), n), rng.integers(0, len(idx), n)
    m = a != b
    return float(np.mean((es[a[m]] > es[b[m]]) == (ts[a[m]] > ts[b[m]])))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> None:
    scenarios = load_scenarios()
    print(f"Loaded {len(scenarios)} scenarios across {len(scenarios.group_names)} groups")

    true_params = make_true_params()
    true_scores = np.asarray(scoring.score(true_params, scenarios.features))

    # in-memory workflow (no files written) starting from neutral weights
    wf = RefitWorkflow(scenarios, orderings_path=None, weights_path=None)

    # --- simulate the interface: many small orderings of random subsets --------
    rng = np.random.default_rng(1)
    groups = scenarios.group_names[:6]
    n_orderings, subset_size = 40, 8
    for _ in range(n_orderings):
        group = rng.choice(groups)
        batch = wf.serve_subset(group, n=subset_size, strategy="random", seed=int(rng.integers(1e9)))
        ids = [b["scenario_id"] for b in batch]
        rows = scenarios.index_of(ids)
        # expert orders this subset by the (hidden) true score, most critical first
        ordered = [ids[i] for i in np.argsort(-true_scores[rows])]
        wf.submit_ordering(ordered, group=group)

    pool = wf.store.regroup()
    print(f"\n{len(wf.store)} orderings -> {len(pool.net())} net pairwise constraints "
          f"(pooled across disjoint subsets)")

    # --- regroup + refit -------------------------------------------------------
    before_scores = wf.scores()
    result = wf.refit(epochs=600, lr=0.05)
    after_scores = wf.scores()

    print("\n--- REFIT ---")
    print(f"training pairs             : {result.n_pairs}")
    print(f"final loss                 : {result.loss:.4f}")
    print(f"train pair accuracy        : {result.pair_accuracy:.3f}")
    print(f"Spearman vs truth  (before): {spearman(before_scores, true_scores):+.3f}")
    print(f"Spearman vs truth  (after) : {spearman(after_scores, true_scores):+.3f}")

    # --- interpretable recovery: how the WHC coefficients moved --------------
    # Ordinal data pins down the ordering, not the absolute weight scale, so we
    # compare relative emphasis (each weight vector normalised to sum to 1).
    def rel(p, key):
        w = np.asarray(scoring.effective_weights(p)[key])
        return np.round(w / w.sum(), 2)

    print("\n--- H_Sev relative emphasis [crit_surf, steril, condition, proximity] ---")
    print(f"  known WHC : {rel(scoring.init_params(), 'hsw')}")
    print(f"  hidden    : {rel(true_params, 'hsw')}")
    print(f"  recovered : {rel(wf.params, 'hsw')}")

    # --- generalization: held-out pairs (mostly never co-ranked in a subset) ---
    g = groups[0]
    gi = scenarios.index_of(scenarios.in_group(g))
    print(f"\nheld-out pair agreement in '{g[:40]}...':")
    print(f"  before refit : {held_out_agreement(true_scores, before_scores, gi, rng):.3f}")
    print(f"  after  refit : {held_out_agreement(true_scores, after_scores, gi, rng):.3f}")


if __name__ == "__main__":
    main()
