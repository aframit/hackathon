"""The known WHC risk calculation, with its coefficients exposed as weights.

This is a differentiable version of WHC.py (verified to reproduce the stored WHC
column exactly). At init the weights equal the verified WHC constants, so score
reproduces the known calculation; re-fitting from orderings nudges them and is
regularised back toward these values (see refit.fit).

    WHC     = WH_Lik * WH_Sev
    WH_Lik  = h_lik_base + RM_Lik * H_Lik
    H_Lik   = hlw . [active, ergonomic_freedom, complexity]
    ergo    = ergw . [visibility, distance, tactility]
    RM_Lik  = rm_lik_base + rmlw . [movement_speed, execution_pace, frame_progress]
    WH_Sev  = h_sev_base + RM_Sev * H_Sev
    H_Sev   = hsw . [critical_surfaces, sterilization, product_condition, proximity]
    RM_Sev  = rm_sev_base + rmsw . [batch_recoverability, decontamination,
                                    barrier, gowning, interaction]
    rank_key = ln(WHC)   (the LN column experts rank by)

Weights are kept non-negative via softplus (each feature can only add risk, which
matches WHC and keeps WHC > 0 so ln is always defined). Parameters are stored in
pre-softplus space, and init_params returns the inverse-softplus of the constants
so the effective weights start exactly at the WHC values.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

# Canonical leaf-feature order.  The feature matrix (data.py) has one column per
# entry here, in this exact order.  Names match WHC.WHCInputs fields.
FEATURES: list[str] = [
    # likelihood leaves
    "number_of_active_objects",       # 0
    "degree_of_visibility",           # 1
    "distance_to_object",             # 2
    "degree_of_tactility",            # 3
    "complexity",                     # 4
    "allowed_movement_speed",         # 5
    "execution_pace",                 # 6
    "frame_progress_tracker",         # 7
    # severity leaves
    "number_of_critical_surfaces",    # 8
    "product_sterilization_status",   # 9
    "product_condition",              # 10
    "spatial_proximity_to_product",   # 11
    "batch_recoverability",           # 12
    "decontamination_status",         # 13
    "barrier_system",                 # 14
    "gowning",                        # 15
    "interaction_with_critical_surfaces",  # 16
]
N_FEATURES = len(FEATURES)

# The verified WHC coefficients (effective, post-softplus values). See WHC.py.
CONSTANTS: dict[str, list | float] = {
    "hlw": [1 / 3, 1 / 3, 1 / 3],          # H_Lik weights: active, ergo, complexity
    "ergw": [0.5, 0.25, 0.25],             # ergo: visibility, distance, tactility
    "rmlw": [1.0, 1.0, 1.0],               # RM_Lik: movement, exec_pace, frame_progress
    "rm_lik_base": 1.0,
    "h_lik_base": 0.4,
    "hsw": [0.3, 0.3, 0.2, 0.2],           # H_Sev: crit_surf, steril, condition, proximity
    "rmsw": [1.0, 1.0, 1.0, 1.0, 1.0],     # RM_Sev: batch, decontam, barrier, gowning, inter
    "rm_sev_base": 1.0,
    "h_sev_base": 0.5,
}


def _sp(x):
    return jax.nn.softplus(x)


def _inv_sp(y):
    # inverse softplus: raw such that softplus(raw) == y  (y > 0)
    return jnp.log(jnp.expm1(jnp.asarray(y, dtype=jnp.float32)))


def init_params() -> dict:
    """Raw (pre-softplus) params whose effective values are the WHC constants."""
    return {k: _inv_sp(jnp.asarray(v)) for k, v in CONSTANTS.items()}


def effective_weights(params: dict) -> dict:
    """The interpretable, post-softplus weights (what the calculation uses)."""
    return {k: _sp(v) for k, v in params.items()}


def wh_lik(params: dict, x: jnp.ndarray) -> jnp.ndarray:
    hlw, ergw, rmlw = _sp(params["hlw"]), _sp(params["ergw"]), _sp(params["rmlw"])
    ergo = x[..., 1:4] @ ergw
    h_lik = hlw[0] * x[..., 0] + hlw[1] * ergo + hlw[2] * x[..., 4]
    rm_lik = _sp(params["rm_lik_base"]) + x[..., 5:8] @ rmlw
    return _sp(params["h_lik_base"]) + rm_lik * h_lik


def wh_sev(params: dict, x: jnp.ndarray) -> jnp.ndarray:
    hsw, rmsw = _sp(params["hsw"]), _sp(params["rmsw"])
    h_sev = x[..., 8:12] @ hsw
    rm_sev = _sp(params["rm_sev_base"]) + x[..., 12:17] @ rmsw
    return _sp(params["h_sev_base"]) + rm_sev * h_sev


def whc(params: dict, x: jnp.ndarray) -> jnp.ndarray:
    """Weighted Hazard Criticality (== the stored WHC column at init)."""
    return wh_lik(params, x) * wh_sev(params, x)


def score(params: dict, x: jnp.ndarray) -> jnp.ndarray:
    """Ranking key experts sort by: ln(WHC) (== the LN column).

    Computed as ln(WH_Lik) + ln(WH_Sev); both branches are strictly positive by
    construction, so the log is always defined during fitting.
    """
    return jnp.log(wh_lik(params, x)) + jnp.log(wh_sev(params, x))
