"""Re-fit the calculation weights from pooled pairwise constraints.

We learn the weights of the known calculation (refit.scoring) so that
score(winner) > score(loser) holds for as many regrouped constraints as
possible. This is a weighted Bradley-Terry (logistic) ranking loss.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax

from . import scoring


@dataclass
class FitResult:
    params: dict
    loss: float
    pair_accuracy: float          # fraction of constraints satisfied (weighted)
    n_pairs: int
    history: list[float]


def _pairs_to_arrays(pool_net: dict[tuple[str, str], int], id_to_row: dict[str, int]):
    winners, losers, weights = [], [], []
    for (w, l), margin in pool_net.items():
        if w in id_to_row and l in id_to_row:
            winners.append(id_to_row[w])
            losers.append(id_to_row[l])
            weights.append(float(margin))
    return (
        jnp.array(winners, dtype=jnp.int32),
        jnp.array(losers, dtype=jnp.int32),
        jnp.array(weights, dtype=jnp.float32),
    )


def refit(
    features: np.ndarray,
    id_to_row: dict[str, int],
    pool_net: dict[tuple[str, str], int],
    *,
    init: dict | None = None,
    prior: dict | None = None,
    epochs: int = 300,
    lr: float = 0.05,
    l2: float = 1e-3,
) -> FitResult:
    """Fit calculation weights to satisfy the regrouped ``(winner, loser)`` pairs.

    features   : (N, len(FEATURES)) matrix for all referenced scenarios.
    id_to_row  : scenario_id -> row index into features.
    pool_net   : output of PairPool.net() (pair -> positive margin weight).
    prior      : weights to regularise toward (default: the known WHC constants).
                 Anchors the fit to the known calculation unless the orderings
                 give strong evidence to move away from it.
    """
    X = jnp.asarray(features, dtype=jnp.float32)
    win, lose, w = _pairs_to_arrays(pool_net, id_to_row)
    if win.shape[0] == 0:
        raise ValueError("No usable pairwise constraints to fit on.")
    w_norm = w / jnp.sum(w)
    params = init if init is not None else scoring.init_params()
    prior = prior if prior is not None else scoring.init_params()

    def loss_fn(params):
        s = scoring.score(params, X)
        logits = s[win] - s[lose]
        bt = -jnp.sum(w_norm * jax.nn.log_sigmoid(logits))
        reg = l2 * sum(jnp.sum((params[k] - prior[k]) ** 2) for k in params)
        return bt + reg

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    history = []
    for _ in range(epochs):
        params, opt_state, loss = step(params, opt_state)
        history.append(float(loss))

    # weighted fraction of constraints the fitted weights satisfy
    s = scoring.score(params, X)
    satisfied = (s[win] > s[lose]).astype(jnp.float32)
    acc = float(jnp.sum(w_norm * satisfied))

    return FitResult(
        params=params,
        loss=history[-1],
        pair_accuracy=acc,
        n_pairs=int(win.shape[0]),
        history=history,
    )
