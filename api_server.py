"""Minimal FastAPI server that serves hazard scenarios and runs the re-fit."""
from __future__ import annotations

import random

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from refit import RefitWorkflow, load_scenarios, scoring

# Parameter whose label->score encoding "Done" re-fits. Default: the interaction
# study case. Any parameter in scoring.ENCODABLE works; pass "param" to override.
DEFAULT_PARAM = "interaction_with_critical_surfaces"

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
    ordered_ids: list[str]          # most critical first
    param: str | None = None
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
    """Store the returned ordering, re-fit the studied parameter, return results.

    Body: {"ordered_ids": [...most critical first...], "param"?: ...}. Returns the
    re-ranked list, the frustration list, and the fitted label->score encoding.
    """
    param = req.param or DEFAULT_PARAM
    if param not in scoring.ENCODABLE:
        raise HTTPException(400, f"Unknown/unsupported param: {param!r}")

    wf = get_workflow()
    valid = [i for i in req.ordered_ids if i in wf.id_to_row]
    if len(valid) < 2:
        raise HTTPException(400, "Need at least 2 known scenario ids to form an ordering.")

    wf.submit_ordering(valid)
    result = wf.refit(trainable=[scoring.encoding_key(param)], epochs=req.epochs, lr=req.lr)

    scores = wf.scores()
    labels = wf.scenarios.labels
    items = []
    for sid in valid:
        row = wf.id_to_row[sid]
        items.append({
            "scenario_id": sid,
            "hazard": labels.iloc[row]["hazard name"],
            "score": float(scores[row]),
            "frustration": float(result.frustration[row]),
        })

    return {
        "param": param,
        "n_orderings": len(wf.store),
        "n_pairs": result.n_pairs,
        "train_accuracy": result.pair_accuracy,
        "encoding": wf.effective_encoding(param),
        "fitted_list": sorted(items, key=lambda d: -d["score"]),
        "frustration_list": sorted(items, key=lambda d: -d["frustration"]),
    }
