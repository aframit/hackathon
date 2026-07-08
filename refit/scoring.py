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
import numpy as np

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
WEIGHT_KEYS = list(CONSTANTS)

# --- learnable label encodings ------------------------------------------------
# Every categorical WHC leaf parameter maps its labels to numeric sub-scores
# (WHC.SCORE_MAPS). Each is fittable via "enc:<param>" in the params dict, so the
# "all parameters" case can re-fit every leaf score at once. Encodings are free
# (can be negative), unlike the softplus-positive combination weights.
#
# Three of them (weight/size/handling_object) are averaged into `complexity`
# rather than mapping to their own feature column; they carry group="complexity".
# frame_progress_tracker is position-derived, so it is encoded over its 4 modifier
# levels. distance_to_object is treated as 10 individual integer bins on the 1-10
# scale (the near/far categories are dropped).
DISTANCE_BINS = 10
COMPLEXITY_PARTS = ("weight_object", "size_object", "handling_object")

# WHC.SCORE_MAPS (label -> score), verified against the Roche dataset.
_SCORE_MAPS: dict[str, dict[str, float]] = {
    "number_of_active_objects": {"1": 1.0, "2": 2.5, "3": 5.0, "4": 7.5},
    "degree_of_visibility": {"high": 1.0, "medium": 5.0, "low": 10.0},
    "degree_of_tactility": {"single gloved hands": 1.0, "double gloved hands": 5.0,
                            "isolator / rabs gloves": 10.0},
    "weight_object": {"200 g-1 kg": 1.0, "1-3 kg": 5.0, "less than 200 g": 10.0,
                      "more than 3 kg": 10.0},
    "size_object": {"5-30 cm": 1.0, "31-50 cm": 5.0, "more than 50 cm": 10.0,
                    "less than 5 cm": 10.0},
    "handling_object": {"easy": 1.0, "moderate": 5.0, "difficult": 10.0},
    "allowed_movement_speed": {"medium": 0.0, "slow": 0.2},
    "execution_pace": {"regular": 0.0, "slow": 0.2},
    "number_of_critical_surfaces": {"0": 1.0, "1": 1.5, "2": 2.5, "3": 5.0,
                                    "4": 7.5, "5+": 10.0},
    "product_sterilization_status": {"not applicable": 1.0,
                                     "post-terminal sterilization": 10.0},
    "product_condition": {"not applicable": 1.0, "open": 10.0},
    "spatial_proximity_to_product": {"far": 1.0, "moderate": 5.0, "close": 10.0},
    "batch_recoverability": {"no risk of losing product": 0.0},
    "decontamination_status": {"pre-decontamination": -0.1, "post-decontamination": 0.1},
    "barrier_system": {"isolator": -0.2, "isolator with doors open (c background)": 0.2,
                       "workbench": 0.3},
    "gowning": {"grade c gowning": 0.0},
    "interaction_with_critical_surfaces": {"usage of sterile tool": -0.1,
                                           "not applicable": 0.0,
                                           "using isolator/rabs gloves": 0.1,
                                           "using gloved hand": 0.2,
                                           "usage of non-sterile tool": 0.2},
}
# frame_progress_tracker modifier levels (from position; see WHC.fpt_score_from_position)
_FPT_LEVELS = [-0.2, 0.0, 0.2, 0.4]

_COMPLEXITY_COL = FEATURES.index("complexity")


def _build_encodable() -> dict[str, dict]:
    enc: dict[str, dict] = {
        # distance: 10 integer bins on the 1-10 scale (categories forgotten).
        "distance_to_object": {
            "col": FEATURES.index("distance_to_object"),
            "labels": [str(b) for b in range(1, DISTANCE_BINS + 1)],
            "init": [float(b) for b in range(1, DISTANCE_BINS + 1)],
            "group": None,
        },
    }
    for param, smap in _SCORE_MAPS.items():
        group = "complexity" if param in COMPLEXITY_PARTS else None
        col = _COMPLEXITY_COL if group == "complexity" else FEATURES.index(param)
        enc[param] = {"col": col, "labels": list(smap),
                      "init": list(smap.values()), "group": group}
    enc["frame_progress_tracker"] = {
        "col": FEATURES.index("frame_progress_tracker"),
        "labels": [f"{v:+.1f}" for v in _FPT_LEVELS],
        "init": list(_FPT_LEVELS),
        "group": None,
    }
    return enc


ENCODABLE: dict[str, dict] = _build_encodable()


def encoding_key(param: str) -> str:
    return f"enc:{param}"


ENCODING_KEYS = [encoding_key(p) for p in ENCODABLE]


def to_integer_bins(vec, lo: int = 1, hi: int = 10) -> np.ndarray:
    """Snap a fitted encoding to integer bin scores by rounding, clipped to [lo, hi].

    distance_to_object lives on a fixed 1-10 integer bin scale, so each fitted bin
    score is simply rounded to the nearest integer and kept within range.
    """
    return np.clip(np.rint(np.asarray(vec, dtype=float)), lo, hi)


def _sp(x):
    return jax.nn.softplus(x)


def _inv_sp(y):
    # inverse softplus: raw such that softplus(raw) == y  (y > 0)
    return jnp.log(jnp.expm1(jnp.asarray(y, dtype=jnp.float32)))


def init_params() -> dict:
    """Params whose effective values are the verified WHC constants.

    Weights are stored pre-softplus; encodings ("enc:<param>") are stored as-is.
    At init the encodings equal the precomputed feature scores, so ``score`` is
    unchanged until an encoding is chosen for fitting.
    """
    params = {k: _inv_sp(jnp.asarray(v)) for k, v in CONSTANTS.items()}
    for param, spec in ENCODABLE.items():
        params[encoding_key(param)] = jnp.asarray(spec["init"], dtype=jnp.float32)
    return params


def effective_weights(params: dict) -> dict:
    """The interpretable weights (softplus applied) and encodings (as-is)."""
    return {
        k: (v if k.startswith("enc:") else _sp(v)) for k, v in params.items()
    }


def _encoded_column(param: str, params: dict, enc_label_idx: dict) -> jnp.ndarray:
    """Per-scenario score for ``param`` from its learnable encoding."""
    vec = params[encoding_key(param)]
    idx = jnp.clip(jnp.asarray(enc_label_idx[param]), 0, vec.shape[0] - 1)
    return vec[idx]


def assemble_features(x0, params: dict, enc_label_idx: dict) -> jnp.ndarray:
    """Feature matrix with encoded columns replaced by their learnable scores.

    ``x0``            : (N, N_FEATURES) precomputed sub-scores.
    ``enc_label_idx`` : param -> (N,) label index into ENCODABLE[param]["labels"].
    Only parameters whose "enc:<param>" key is present in ``params`` are overridden
    (so freezing an encoding leaves its precomputed column untouched). The three
    complexity parts (weight/size/handling) are averaged into the complexity column.
    """
    x = jnp.asarray(x0, dtype=jnp.float32)
    for param, spec in ENCODABLE.items():
        if spec["group"] == "complexity":
            continue
        key = encoding_key(param)
        if key in params and param in enc_label_idx:
            x = x.at[:, spec["col"]].set(_encoded_column(param, params, enc_label_idx))

    parts = [p for p in COMPLEXITY_PARTS
             if encoding_key(p) in params and p in enc_label_idx]
    if len(parts) == len(COMPLEXITY_PARTS):
        total = sum(_encoded_column(p, params, enc_label_idx) for p in parts)
        x = x.at[:, _COMPLEXITY_COL].set(total / len(COMPLEXITY_PARTS))
    return x


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

    Computed as ln(WH_Lik) + ln(WH_Sev). Weights are positive by construction;
    encodings are free, so both branches are floored at a tiny positive value to
    keep the log defined while an encoding is being fit.
    """
    eps = 1e-6
    return jnp.log(jnp.clip(wh_lik(params, x), eps, None)) + jnp.log(
        jnp.clip(wh_sev(params, x), eps, None)
    )
