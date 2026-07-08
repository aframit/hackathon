"""Load hazard scenarios from the warehouse into a tidy, model-ready table.

A scenario is one hazard-scenario row. We keep:
  - scenario_id: a stable, unique id
  - group: the process it belongs to (orderings happen within a group)
  - features: the numeric sub-score vector in scoring.FEATURES order
  - labels: a few human-readable columns for the ordering interface
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .scoring import ENCODABLE, FEATURES

DEFAULT_DUMP = Path(__file__).resolve().parent.parent / "data_dump"

# display column (as shown in the reorder-app) -> raw label column.
DISPLAY_LABEL_COLUMNS: dict[str, str] = {
    "barrier": "BarrierSystem",
    "critical surfaces": "NumberOfCriticalSurfaces",
    "interaction": "InteractionWithCritSurf",
    "visibility": "DegreeOfVisibility",
    "distance to object": "DistToObj",
    "size": "SizeObj",
    "weight": "WeightObj",
    "handling": "HandlingOfObj",
}


def _project_from_source(source: pd.Series) -> pd.Series:
    """Short project label from the source filename (e.g. '..._ops4_...' -> 'ops4')."""
    return source.astype(str).str.extract(r"_rp_([a-z0-9]+)_", expand=False).fillna(source)

# canonical leaf-feature name -> column in raw_risk_profiling.parquet.
# These are exactly the leaf sub-scores WHC.py consumes (verified to reproduce
# the stored WHC column), in the same names as scoring.FEATURES.
RAW_SCORE_COLUMNS: dict[str, str] = {
    "number_of_active_objects": "NumbActObjScr",
    "degree_of_visibility": "DegreeOfVisibilityScr",
    "distance_to_object": "DistToObjScr",
    "degree_of_tactility": "DegreeOfTactilityScr",
    "complexity": "ComplexityScr",
    "allowed_movement_speed": "MovementSpeedScr",
    "execution_pace": "ExectionPaceScore",
    "frame_progress_tracker": "FrameProgressTrackerScr",
    "number_of_critical_surfaces": "NumberOfCriticalSurfacesScore",
    "product_sterilization_status": "ProductSterilizationStatusScr\n",
    "product_condition": "ProductConditionScr",
    "spatial_proximity_to_product": "SpatialProximityToProductScore",
    "batch_recoverability": "BatchRecoverabilityScr",
    "decontamination_status": "DecontaminationStatusScr",
    "barrier_system": "BarrierSystemScr",
    "gowning": "GowningStatusScr",
    "interaction_with_critical_surfaces": "InteractionWithCritSurfScr",
}

# score column per fittable parameter (scoring.ENCODABLE). Same as the feature
# score columns, plus the three object sub-scores that compose complexity.
ENC_SCORE_COLUMNS: dict[str, str] = {
    **RAW_SCORE_COLUMNS,
    "weight_object": "WeightObjScr",
    "size_object": "SizeObjScr",
    "handling_object": "HandlingOfObjScr",
}


# columns exported for the reorder-app (scenario_id is hidden from its display,
# but sent back on "Done" so the backend can map rows to scenarios exactly).
APP_CSV_COLUMNS = [
    "scenario_id", "project", "process", "hazard scenario", "barrier",
    "critical surfaces", "interaction", "visibility", "distance to object",
    "size", "weight", "handling",
]


@dataclass
class ScenarioTable:
    """Feature matrix + aligned metadata for a set of scenarios."""

    ids: np.ndarray            # (N,) str scenario ids
    features: np.ndarray       # (N, len(FEATURES)) float feature sub-scores
    groups: np.ndarray         # (N,) str group (process) per scenario
    labels: pd.DataFrame       # (N, ...) human-readable columns, aligned to ids
    # param -> (N,) label index into ENCODABLE[param]["labels"], for fittable encodings
    enc_label_idx: dict[str, np.ndarray] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.ids)

    def index_of(self, ids: list[str]) -> np.ndarray:
        """Row indices for the given scenario ids (preserving the given order)."""
        pos = {sid: i for i, sid in enumerate(self.ids)}
        return np.array([pos[s] for s in ids], dtype=int)

    def subset(self, ids: list[str]) -> "ScenarioTable":
        idx = self.index_of(ids)
        return ScenarioTable(
            ids=self.ids[idx],
            features=self.features[idx],
            groups=self.groups[idx],
            labels=self.labels.iloc[idx].reset_index(drop=True),
            enc_label_idx={k: v[idx] for k, v in self.enc_label_idx.items()},
        )

    def in_group(self, group: str) -> list[str]:
        return list(self.ids[self.groups == group])

    @property
    def group_names(self) -> list[str]:
        return sorted(set(self.groups.tolist()))


def load_scenarios(dump: Path | str = DEFAULT_DUMP) -> ScenarioTable:
    """Read raw_risk_profiling and build a ScenarioTable."""
    dump = Path(dump)
    raw = pd.read_parquet(dump / "raw_risk_profiling.parquet")

    # numeric feature matrix in canonical FEATURES order
    cols = [RAW_SCORE_COLUMNS[f] for f in FEATURES]
    feats = raw[cols].apply(pd.to_numeric, errors="coerce")

    # keep only fully-scored rows (a scenario needs every sub-score to be ranked)
    keep = feats.notna().all(axis=1)
    raw = raw[keep].reset_index(drop=True)
    feats = feats[keep].reset_index(drop=True)

    # stable, unique scenario id from the natural hazard-scenario key
    key_cols = ["ProcessName", "SubprocessName", "TaskName", "FrameID", "HazardScenario"]
    key_cols = [c for c in key_cols if c in raw.columns]
    scenario_id = (
        raw[key_cols].astype(str).agg(" | ".join, axis=1)
        + " #"
        + raw.groupby(key_cols).cumcount().astype(str)  # disambiguate exact dups
    )

    def _col(name: str) -> pd.Series:
        return raw.get(name, pd.Series([""] * len(raw))).astype(str)

    labels = pd.DataFrame(
        {
            "scenario_id": scenario_id.values,
            "project": _project_from_source(_col("_source")).values,
            "process": _col("ProcessName").values,
            "hazard scenario": _col("HazardScenario").values,
            "barrier": _col("BarrierSystem").values,
            "critical surfaces": _col("NumberOfCriticalSurfaces").values,
            "interaction": _col("InteractionWithCritSurf").values,
            "visibility": _col("DegreeOfVisibility").values,
            "distance to object": _col("DistToObj").values,
            "size": _col("SizeObj").values,
            "weight": _col("WeightObj").values,
            "handling": _col("HandlingOfObj").values,
        }
    )
    # alias used across the code for short display
    labels["hazard"] = labels["hazard scenario"]

    # Label/bin index per scenario for every fittable parameter. We map each
    # scenario to the encoding entry whose init score matches its precomputed
    # sub-score (argmin). This is uniform across parameters and reproduces the
    # WHC column exactly at init (scores come straight from the same score maps).
    enc_label_idx: dict[str, np.ndarray] = {}
    for param in ENCODABLE:
        init = np.asarray(ENCODABLE[param]["init"], dtype=float)
        score = pd.to_numeric(raw[ENC_SCORE_COLUMNS[param]], errors="coerce").to_numpy(dtype=float)
        enc_label_idx[param] = np.abs(score[:, None] - init[None, :]).argmin(axis=1)

    # Note: some RM modifier sub-scores are legitimately negative (e.g. barrier
    # 'isolator' = -0.2), so we keep raw values; WHC's structure stays positive.
    return ScenarioTable(
        ids=scenario_id.to_numpy(),
        features=feats.to_numpy(dtype=float),
        groups=raw["ProcessName"].astype(str).to_numpy(),
        labels=labels,
        enc_label_idx=enc_label_idx,
    )


def export_scenarios_csv(path: Path | str, dump: Path | str = DEFAULT_DUMP) -> int:
    """Write the reorder-app's scenario CSV (scenario_id + display columns)."""
    table = load_scenarios(dump)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.labels[APP_CSV_COLUMNS].to_csv(path, index=False)
    return len(table)
