"""Load hazard scenarios from the warehouse into a tidy, model-ready table.

A scenario is one hazard-scenario row. We keep:
  - scenario_id: a stable, unique id
  - group: the process it belongs to (orderings happen within a group)
  - features: the numeric sub-score vector in scoring.FEATURES order
  - labels: a few human-readable columns for the ordering interface
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .scoring import FEATURES

DEFAULT_DUMP = Path(__file__).resolve().parent.parent / "data_dump"

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


@dataclass
class ScenarioTable:
    """Feature matrix + aligned metadata for a set of scenarios."""

    ids: np.ndarray            # (N,) str scenario ids
    features: np.ndarray       # (N, len(FEATURES)) float feature sub-scores
    groups: np.ndarray         # (N,) str group (process) per scenario
    labels: pd.DataFrame       # (N, ...) human-readable columns, aligned to ids

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
            "project": _col("_source").values,
            "process": _col("ProcessName").values,
            "hazard name": _col("HazardScenario").values,
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

    # Note: some RM modifier sub-scores are legitimately negative (e.g. barrier
    # 'isolator' = -0.2), so we keep raw values; WHC's structure stays positive.
    return ScenarioTable(
        ids=scenario_id.to_numpy(),
        features=feats.to_numpy(dtype=float),
        groups=raw["ProcessName"].astype(str).to_numpy(),
        labels=labels,
    )
