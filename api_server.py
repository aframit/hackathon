"""Minimal FastAPI server that serves hazard scenarios and runs the re-fit."""
from __future__ import annotations

import random

import jax.numpy as jnp
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from refit import RefitWorkflow, load_scenarios, scoring

_DIST = "distance_to_object"
_INTERACTION = "interaction_with_critical_surfaces"

# The parameter groupings the "Done" button can re-fit. Each case maps to the set
# of parameters that move; everything else stays pinned at its known WHC value.
#   - interaction: re-bin the interaction_with_critical_surfaces encoding
#   - distance:    re-bin distance_to_object (integer bins, see below)
#   - all:         fit every combination weight and every encoding at once
CASES: dict[str, list[str]] = {
    _INTERACTION: [scoring.encoding_key(_INTERACTION)],
    _DIST: [scoring.encoding_key(_DIST)],
    # "all" re-fits the label/bin scores of every WHC leaf parameter at once.
    "all": list(scoring.ENCODING_KEYS),
}
DEFAULT_CASE = _INTERACTION

app = FastAPI(title="Hazard Scenarios API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:4173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Load once on startup
_table = None
_workflow = None


def get_table():
    global _table
    if _table is None:
        _table = load_scenarios()
    return _table


def get_workflow():
    # One persistent workflow: orderings accumulate and the fit keeps improving.
    global _workflow
    if _workflow is None:
        _workflow = RefitWorkflow(get_table())
    return _workflow


class RefitRequest(BaseModel):
    ordered_ids: list[list[str]] | list[str]  # batched (preferred) or legacy flat list
    case: str | None = None         # one of CASES; default: interaction study
    epochs: int = 500
    lr: float = 0.05


@app.get("/scenarios")
def get_scenarios(n: int = Query(default=10, ge=1, le=200)):
    """Return n randomly sampled hazard scenarios as a list of row objects."""
    table = get_table()
    indices = random.sample(range(len(table)), min(n, len(table)))

    label_cols = list(table.labels.columns)
    rows = [
        {col: str(table.labels.iloc[i][col]) for col in label_cols}
        for i in indices
    ]

    return {"rows": rows, "total": len(table), "columns": label_cols}


@app.post("/refit")
def refit(req: RefitRequest):
    """Store returned ordering batches, re-fit the selected case, return results.

    Body: {"ordered_ids": [[...], [...], ...], "case"?: ...}. Returns the
    re-ranked list (raw WHC), the frustration list, and the fitted encodings for
    the trained parameter(s). Each refit cold-starts from the known WHC values, so
    frozen parameters stay exact and free fits cannot drift/compound to zero.

    For the "all" case the label/bin scores of every WHC leaf parameter are re-fit
    simultaneously. distance_to_object bins are snapped to the integer 1-10 scale.
    """
    case = req.case or DEFAULT_CASE
    trainable = CASES.get(case)
    if trainable is None:
        raise HTTPException(400, f"Unknown case: {case!r}. Choose one of {list(CASES)}.")

    wf = get_workflow()
    ordered_batches = req.ordered_ids
    if not ordered_batches:
        raise HTTPException(400, "Need at least one batch of ordered ids.")

    # Backward compatibility: a flat list is treated as one batch.
    if isinstance(ordered_batches[0], str):
        ordered_batches = [ordered_batches]

    valid_batches: list[list[str]] = []
    for batch in ordered_batches:
        valid = [sid for sid in batch if sid in wf.id_to_row]
        if len(valid) >= 2:
            valid_batches.append(valid)

    if not valid_batches:
        raise HTTPException(400, "Need at least one valid batch with >= 2 known scenario ids.")

    for batch in valid_batches:
        wf.submit_ordering(batch)

    result = wf.refit(trainable=trainable, warm_start=False, epochs=req.epochs, lr=req.lr)

    # distance_to_object bins are integer-valued on the 1-10 scale: snap the fitted
    # encoding and persist it so the returned scores and bins are integers.
    dist_key = scoring.encoding_key(_DIST)
    if dist_key in trainable:
        ints = scoring.to_integer_bins(wf.params[dist_key])
        wf.params[dist_key] = jnp.asarray(ints, dtype=jnp.float32)
        wf._save_weights()

    whc = wf.whc_scores()
    labels = wf.scenarios.labels

    submitted_ids = [sid for batch in valid_batches for sid in batch]
    items = [
        {
            "scenario_id": sid,
            "hazard": labels.iloc[wf.id_to_row[sid]]["hazard name"],
            "whc": float(whc[wf.id_to_row[sid]]),
            "frustration": float(result.frustration[wf.id_to_row[sid]]),
        }
        for sid in submitted_ids
    ]

    encodings = {
        p: wf.effective_encoding(p)
        for p in scoring.ENCODABLE
        if scoring.encoding_key(p) in trainable
    }

    return {
        "case": case,
        "trainable": trainable,
        "n_orderings": len(wf.store),
        "n_pairs": result.n_pairs,
        "train_accuracy": result.pair_accuracy,
        "encodings": encodings,
        "fitted_list": sorted(items, key=lambda d: -d["whc"]),
        "frustration_list": sorted(items, key=lambda d: -d["frustration"]),
    }
