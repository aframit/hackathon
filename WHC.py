"""Weighted Hazard Criticality (WHC) score.

Minimal, self-contained, JAX implementation of the WHC risk score used in the
Roche IREM risk-profiling dataset.

    WHC = WH_Lik * WH_Sev

Likelihood side
    WH_Lik = H_lik_base + RM_lik * H_lik
    H_lik  = (1/3)*number_of_active_objects + (1/3)*ergonomic_freedom + (1/3)*complexity
    ergonomic_freedom = 0.5*degree_of_visibility + 0.25*distance_to_object + 0.25*degree_of_tactility
    RM_lik = 1 + allowed_movement_speed + execution_pace + frame_progress_tracker
    H_lik_base = 0.4

Severity side
    WH_Sev = H_sev_base + RM_sev * H_sev
    H_sev  = 0.3*number_of_critical_surfaces + 0.3*product_sterilization_status
             + 0.2*product_condition + 0.2*spatial_proximity_to_product
    RM_sev = 1 + batch_recoverability + decontamination_status + barrier_system
             + gowning + interaction_with_critical_surfaces
    H_sev_base = 0.5

All functions operate on numeric *scores* and accept scalars or JAX/NumPy
arrays (they broadcast, vmap and differentiate). Raw categorical labels can be
turned into scores with `encode_inputs` / `SCORE_MAPS`.

The equation and every score map below were verified to reproduce the stored
`WHC` column of the Roche dataset exactly (0 mismatches over 3940 rows).
"""

from __future__ import annotations

from typing import Mapping, NamedTuple

import jax.numpy as jnp

# --------------------------------------------------------------------------- #
# Coefficients
# --------------------------------------------------------------------------- #

# H_lik = l1*active_objects + l2*ergonomic_freedom + l3*complexity
L1 = L2 = L3 = 1.0 / 3.0

# ergonomic_freedom = alpha*visibility + beta*distance + gamma*tactility
ALPHA = 0.5
BETA = GAMMA = 0.25

# H_sev = s1*critical_surfaces + s2*sterilization + s3*product_condition + s4*proximity
S1 = S2 = 0.3
S3 = S4 = 0.2

H_LIK_BASE = 0.4
H_SEV_BASE = 0.5

# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #


class WHCInputs(NamedTuple):
    """Leaf scores that feed the WHC equation (scalars or arrays)."""

    # likelihood
    number_of_active_objects: jnp.ndarray
    degree_of_visibility: jnp.ndarray
    distance_to_object: jnp.ndarray
    degree_of_tactility: jnp.ndarray
    complexity: jnp.ndarray
    allowed_movement_speed: jnp.ndarray
    execution_pace: jnp.ndarray
    frame_progress_tracker: jnp.ndarray
    # severity
    number_of_critical_surfaces: jnp.ndarray
    product_sterilization_status: jnp.ndarray
    product_condition: jnp.ndarray
    spatial_proximity_to_product: jnp.ndarray
    batch_recoverability: jnp.ndarray
    decontamination_status: jnp.ndarray
    barrier_system: jnp.ndarray
    gowning: jnp.ndarray
    interaction_with_critical_surfaces: jnp.ndarray


# --------------------------------------------------------------------------- #
# Likelihood
# --------------------------------------------------------------------------- #


def ergonomic_freedom(degree_of_visibility, distance_to_object, degree_of_tactility):
    return ALPHA * degree_of_visibility + BETA * distance_to_object + GAMMA * degree_of_tactility


def complexity_from_object(weight_object, size_object, handling_object):
    """Complexity = mean of the weight / size / handling scores of the object."""
    return (weight_object + size_object + handling_object) / 3.0


def h_lik(number_of_active_objects, ergonomic_freedom_score, complexity):
    return L1 * number_of_active_objects + L2 * ergonomic_freedom_score + L3 * complexity


def rm_lik(allowed_movement_speed, execution_pace, frame_progress_tracker):
    return 1.0 + allowed_movement_speed + execution_pace + frame_progress_tracker


def wh_lik(h_lik_score, rm_lik_score):
    return H_LIK_BASE + rm_lik_score * h_lik_score


# --------------------------------------------------------------------------- #
# Severity
# --------------------------------------------------------------------------- #


def h_sev(
    number_of_critical_surfaces,
    product_sterilization_status,
    product_condition,
    spatial_proximity_to_product,
):
    return (
        S1 * number_of_critical_surfaces
        + S2 * product_sterilization_status
        + S3 * product_condition
        + S4 * spatial_proximity_to_product
    )


def rm_sev(
    batch_recoverability,
    decontamination_status,
    barrier_system,
    gowning,
    interaction_with_critical_surfaces,
):
    return (
        1.0
        + batch_recoverability
        + decontamination_status
        + barrier_system
        + gowning
        + interaction_with_critical_surfaces
    )


def wh_sev(h_sev_score, rm_sev_score):
    return H_SEV_BASE + rm_sev_score * h_sev_score


# --------------------------------------------------------------------------- #
# WHC
# --------------------------------------------------------------------------- #


def compute_whc(x: WHCInputs):
    """WHC = WH_Lik * WH_Sev from a `WHCInputs` of scores."""
    return whc_components(x)["WHC"]


def whc_components(x: WHCInputs) -> dict:
    """Return WHC plus every intermediate term (handy for validation)."""
    erg = ergonomic_freedom(x.degree_of_visibility, x.distance_to_object, x.degree_of_tactility)
    hl = h_lik(x.number_of_active_objects, erg, x.complexity)
    rl = rm_lik(x.allowed_movement_speed, x.execution_pace, x.frame_progress_tracker)
    whl = wh_lik(hl, rl)

    hs = h_sev(
        x.number_of_critical_surfaces,
        x.product_sterilization_status,
        x.product_condition,
        x.spatial_proximity_to_product,
    )
    rs = rm_sev(
        x.batch_recoverability,
        x.decontamination_status,
        x.barrier_system,
        x.gowning,
        x.interaction_with_critical_surfaces,
    )
    whs = wh_sev(hs, rs)

    return {
        "ergonomic_freedom": erg,
        "H_Lik": hl,
        "RM_Lik": rl,
        "WH_Lik": whl,
        "H_Sev": hs,
        "RM_Sev": rs,
        "WH_Sev": whs,
        "WHC": whl * whs,
    }


# --------------------------------------------------------------------------- #
# Frame progress tracker
# --------------------------------------------------------------------------- #
#
# `frame_progress_tracker` is not a per-row label: it is the 0-based position of
# the frame *within its process* (0 = first frame of the process), mapped to a
# modifier score. In the raw dataset this counter was buggy (built via factorize
# on the frame *name*, and not reset per process). The fix: within each process,
# rank each unique frame_groupid by first appearance -> 0-based position, then
# map position -> score below. See `docs/WHC_equation.md`.


def fpt_score_from_position(position):
    """0-based frame position within a process -> FPT modifier score.

        pos in [0, 48]   -> -0.2
        pos in [49, 98]  ->  0.0
        pos in [99, 198] ->  0.2
        pos >= 199       ->  0.4
    """
    position = jnp.asarray(position)
    return jnp.select(
        [position <= 48, position <= 98, position <= 198],
        [-0.2, 0.0, 0.2],
        default=0.4,
    )


# --------------------------------------------------------------------------- #
# Label -> score maps (verified against the Roche dataset)
# --------------------------------------------------------------------------- #
#
# Every categorical WHC input maps to a fixed numeric score. `frame_progress_tracker`
# is derived from frame position (see `fpt_score_from_position`), not from a label.

SCORE_MAPS: Mapping[str, Mapping[str, float]] = {
    # ---- likelihood ------------------------------------------------------- #
    "number_of_active_objects": {"1": 1.0, "2": 2.5, "3": 5.0, "4": 7.5},
    "degree_of_visibility": {"high": 1.0, "medium": 5.0, "low": 10.0},
    "distance_to_object": {
        "near - no obstacles": 1.0,
        "near - with obstacles": 5.0,
        "far - no obstacles": 5.0,
        "far - with obstacles": 10.0,
    },
    "degree_of_tactility": {
        "single gloved hands": 1.0,
        "double gloved hands": 5.0,
        "isolator / rabs gloves": 10.0,
    },
    "allowed_movement_speed": {"medium": 0.0, "slow": 0.2},
    "execution_pace": {"regular": 0.0, "slow": 0.2},
    # complexity = mean of the three object scores below (use complexity_from_object)
    "weight_object": {
        "200 g-1 kg": 1.0,
        "1-3 kg": 5.0,
        "less than 200 g": 10.0,
        "more than 3 kg": 10.0,
    },
    "size_object": {
        "5-30 cm": 1.0,
        "31-50 cm": 5.0,
        "more than 50 cm": 10.0,
        "less than 5 cm": 10.0,
    },
    "handling_object": {"easy": 1.0, "moderate": 5.0, "difficult": 10.0},
    # ---- severity --------------------------------------------------------- #
    "number_of_critical_surfaces": {
        "0": 1.0,
        "1": 1.5,
        "2": 2.5,
        "3": 5.0,
        "4": 7.5,
        "5+": 10.0,
    },
    "product_sterilization_status": {
        "not applicable": 1.0,
        "post-terminal sterilization": 10.0,
    },
    "product_condition": {"not applicable": 1.0, "open": 10.0},
    "spatial_proximity_to_product": {"far": 1.0, "moderate": 5.0, "close": 10.0},
    "batch_recoverability": {"no risk of losing product": 0.0},
    "decontamination_status": {"pre-decontamination": -0.1, "post-decontamination": 0.1},
    "barrier_system": {
        "isolator": -0.2,
        "isolator with doors open (c background)": 0.2,
        "workbench": 0.3,
    },
    "gowning": {"grade c gowning": 0.0},
    "interaction_with_critical_surfaces": {
        "usage of sterile tool": -0.1,
        "not applicable": 0.0,
        "using isolator/rabs gloves": 0.1,
        "using gloved hand": 0.2,
        "usage of non-sterile tool": 0.2,
    },
}


def encode(parameter: str, label: str) -> float:
    """Look up the numeric score of a single categorical `label`.

    Matching is case-insensitive / whitespace-trimmed.
    """
    table = SCORE_MAPS[parameter]
    key = str(label).strip().lower()
    for raw_label, score in table.items():
        if raw_label.lower() == key:
            return score
    raise KeyError(f"unknown label {label!r} for parameter {parameter!r}")


def encode_inputs(
    labels: Mapping[str, str],
    *,
    frame_position: int | None = None,
) -> WHCInputs:
    """Build `WHCInputs` from a dict of raw categorical labels.

    `labels` may provide `complexity` directly, or `weight_object` / `size_object`
    / `handling_object` (which are averaged into complexity).

    `frame_progress_tracker` is taken from `frame_position` (0-based position in
    the process) via `fpt_score_from_position`; alternatively pass a
    `frame_progress_tracker` score directly in `labels`.
    """
    labels = dict(labels)

    if "complexity" in labels:
        complexity = float(labels["complexity"])
    else:
        complexity = float(
            complexity_from_object(
                encode("weight_object", labels["weight_object"]),
                encode("size_object", labels["size_object"]),
                encode("handling_object", labels["handling_object"]),
            )
        )

    if frame_position is not None:
        fpt = float(fpt_score_from_position(frame_position))
    elif "frame_progress_tracker" in labels:
        fpt = float(labels["frame_progress_tracker"])
    else:
        raise ValueError("provide `frame_position` or a `frame_progress_tracker` score")

    return WHCInputs(
        number_of_active_objects=encode("number_of_active_objects", labels["number_of_active_objects"]),
        degree_of_visibility=encode("degree_of_visibility", labels["degree_of_visibility"]),
        distance_to_object=encode("distance_to_object", labels["distance_to_object"]),
        degree_of_tactility=encode("degree_of_tactility", labels["degree_of_tactility"]),
        complexity=complexity,
        allowed_movement_speed=encode("allowed_movement_speed", labels["allowed_movement_speed"]),
        execution_pace=encode("execution_pace", labels["execution_pace"]),
        frame_progress_tracker=fpt,
        number_of_critical_surfaces=encode("number_of_critical_surfaces", labels["number_of_critical_surfaces"]),
        product_sterilization_status=encode("product_sterilization_status", labels["product_sterilization_status"]),
        product_condition=encode("product_condition", labels["product_condition"]),
        spatial_proximity_to_product=encode("spatial_proximity_to_product", labels["spatial_proximity_to_product"]),
        batch_recoverability=encode("batch_recoverability", labels["batch_recoverability"]),
        decontamination_status=encode("decontamination_status", labels["decontamination_status"]),
        barrier_system=encode("barrier_system", labels["barrier_system"]),
        gowning=encode("gowning", labels["gowning"]),
        interaction_with_critical_surfaces=encode("interaction_with_critical_surfaces", labels["interaction_with_critical_surfaces"]),
    )


if __name__ == "__main__":
    demo = WHCInputs(
        number_of_active_objects=5.0,
        degree_of_visibility=5.0,
        distance_to_object=5.0,
        degree_of_tactility=1.0,
        complexity=4.0,
        allowed_movement_speed=0.0,
        execution_pace=0.0,
        frame_progress_tracker=fpt_score_from_position(0),  # first frame -> -0.2
        number_of_critical_surfaces=2.5,
        product_sterilization_status=1.0,
        product_condition=10.0,
        spatial_proximity_to_product=5.0,
        batch_recoverability=0.0,
        decontamination_status=0.1,
        barrier_system=0.3,
        gowning=0.0,
        interaction_with_critical_surfaces=0.2,
    )
    for name, value in whc_components(demo).items():
        print(f"{name:20s} {float(value):.4f}")
