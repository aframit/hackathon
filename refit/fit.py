"""Re-fit the calculation weights from pooled pairwise constraints.

We learn the parameters of the known calculation (refit.scoring) so that
score(winner) > score(loser) holds for as many regrouped constraints as
possible. This is a weighted Bradley-Terry (logistic) ranking loss.

Any subset of parameters can be fit at once via ``trainable`` (e.g. just one
label encoding); the rest stay frozen at their known values. Passing every key
fits everything simultaneously.
"""
from __future__ import annotations

from dataclasses import dataclass, field

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
    trainable: list[str]
    history: list[float]
    frustration: np.ndarray = field(default_factory=lambda: np.zeros(0))  # per row
    involvement: np.ndarray = field(default_factory=lambda: np.zeros(0))   # per row


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


def _frustration(scores, win, lose, w, n_rows):
    """Per-scenario average violation strength over the constraints it appears in.

    For a constraint winner>loser the violation is 1 - P(model ranks it right).
    A scenario scores high here when the fitted model keeps contradicting the
    ground-truth orderings that involve it.
    """
    p_correct = np.asarray(jax.nn.sigmoid(scores[win] - scores[lose]))
    viol = np.asarray(w) * (1.0 - p_correct)
    wnp, lnp, wgt = np.asarray(win), np.asarray(lose), np.asarray(w)
    frust = np.zeros(n_rows)
    invol = np.zeros(n_rows)
    np.add.at(frust, wnp, viol)
    np.add.at(frust, lnp, viol)
    np.add.at(invol, wnp, wgt)
    np.add.at(invol, lnp, wgt)
    return frust / np.maximum(invol, 1e-9), invol


def refit(
    features: np.ndarray,
    id_to_row: dict[str, int],
    pool_net: dict[tuple[str, str], int],
    *,
    enc_label_idx: dict[str, np.ndarray] | None = None,
    trainable: list[str] | None = None,
    init: dict | None = None,
    prior: dict | None = None,
    epochs: int = 300,
    lr: float = 0.05,
    l2: float = 1e-3,
) -> FitResult:
    """Fit calculation parameters to satisfy the regrouped ``(winner, loser)`` pairs.

    features       : (N, len(FEATURES)) matrix for all referenced scenarios.
    id_to_row      : scenario_id -> row index into features.
    pool_net       : output of PairPool.net() (pair -> positive margin weight).
    enc_label_idx  : param -> (N,) label index, for fittable encodings.
    trainable      : parameter keys to optimise (default: all combination weights;
                     encodings stay frozen). Pass e.g. ["enc:distance_to_object"]
                     to focus on one thing, or every key to fit all at once.
    prior          : values to regularise toward (default: the known WHC values).
    """
    X0 = jnp.asarray(features, dtype=jnp.float32)
    enc_label_idx = enc_label_idx or {}
    win, lose, w = _pairs_to_arrays(pool_net, id_to_row)
    if win.shape[0] == 0:
        raise ValueError("No usable pairwise constraints to fit on.")
    w_norm = w / jnp.sum(w)

    params = init if init is not None else scoring.init_params()
    prior = prior if prior is not None else scoring.init_params()
    trainable = list(trainable) if trainable is not None else list(scoring.WEIGHT_KEYS)

    frozen = {k: v for k, v in params.items() if k not in trainable}
    tparams = {k: params[k] for k in trainable}

    def loss_fn(tparams):
        merged = {**frozen, **tparams}
        X = scoring.assemble_features(X0, merged, enc_label_idx)
        s = scoring.score(merged, X)
        logits = s[win] - s[lose]
        bt = -jnp.sum(w_norm * jax.nn.log_sigmoid(logits))
        reg = l2 * sum(jnp.sum((tparams[k] - prior[k]) ** 2) for k in tparams)
        return bt + reg

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(tparams)

    @jax.jit
    def step(tparams, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(tparams)
        updates, opt_state = optimizer.update(grads, opt_state, tparams)
        return optax.apply_updates(tparams, updates), opt_state, loss

    history = []
    for _ in range(epochs):
        tparams, opt_state, loss = step(tparams, opt_state)
        history.append(float(loss))

    fitted = {**frozen, **tparams}
    X = scoring.assemble_features(X0, fitted, enc_label_idx)
    s = scoring.score(fitted, X)
    satisfied = (s[win] > s[lose]).astype(jnp.float32)
    acc = float(jnp.sum(w_norm * satisfied))
    frust, invol = _frustration(s, win, lose, w, X0.shape[0])

    return FitResult(
        params=fitted,
        loss=history[-1],
        pair_accuracy=acc,
        n_pairs=int(win.shape[0]),
        trainable=trainable,
        history=history,
        frustration=frust,
        involvement=invol,
    )
