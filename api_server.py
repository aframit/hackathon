"""Minimal FastAPI server that serves hazard scenarios from load_scenarios."""
from __future__ import annotations

import random

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from refit import load_scenarios

app = FastAPI(title="Hazard Scenarios API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:4173"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# Load once on startup
_table = None


def get_table():
    global _table
    if _table is None:
        _table = load_scenarios()
    return _table


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
