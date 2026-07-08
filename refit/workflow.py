"""The interface contract: serve subsets, take orderings back, regroup, re-fit.

Typical loop an interface drives:

    wf = RefitWorkflow(load_scenarios())
    batch = wf.serve_subset(group="...", n=8)      # -> show to expert
    wf.submit_ordering([...ids best->worst...])     # <- expert sends it back
    wf.refit()                                       # regroup all + fit weights
    wf.current_ranking(group="...")                  # updated global ranking
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import scoring
from .data import ScenarioTable
from .fit import FitResult, refit
from .orderings import Ordering, OrderingStore


class RefitWorkflow:
    def __init__(
        self,
        scenarios: ScenarioTable,
        *,
        orderings_path: Path | str | None = "orderings.jsonl",
        weights_path: Path | str | None = "weights.json",
    ):
        self.scenarios = scenarios
        self.store = OrderingStore(orderings_path)
        self.weights_path = Path(weights_path) if weights_path else None
        self.id_to_row = {sid: i for i, sid in enumerate(scenarios.ids)}
        self.params = self._load_weights() or scoring.init_params()

    # --- scoring -------------------------------------------------------------
    def scores(self) -> np.ndarray:
        return np.asarray(scoring.score(self.params, self.scenarios.features))

    def current_ranking(self, group: str | None = None) -> list[dict]:
        """All scenarios (optionally within a group) sorted most -> least critical."""
        s = self.scores()
        idx = np.arange(len(self.scenarios))
        if group is not None:
            idx = idx[self.scenarios.groups == group]
        order = idx[np.argsort(-s[idx])]
        lab = self.scenarios.labels
        return [
            {"scenario_id": self.scenarios.ids[i], "score": float(s[i]),
             "hazard": lab.iloc[i]["hazard"]}
            for i in order
        ]

    # --- serve ---------------------------------------------------------------
    def serve_subset(
        self, group: str, n: int = 8, strategy: str = "random", seed: int | None = None
    ) -> list[dict]:
        """Pick ``n`` scenarios from ``group`` for an expert to order.

        strategy:
          - random:    uniform sample (good for coverage / cold start)
          - uncertain: scenarios currently closest in score (most ambiguous)
        """
        rng = np.random.default_rng(seed)
        ids = np.array(self.scenarios.in_group(group))
        if len(ids) == 0:
            raise ValueError(f"Unknown / empty group: {group!r}")
        n = min(n, len(ids))
        if strategy == "random":
            chosen = rng.choice(ids, size=n, replace=False)
        elif strategy == "uncertain":
            rows = self.scenarios.index_of(list(ids))
            s = self.scores()[rows]
            centre = np.argsort(s)[len(s) // 2]  # densest region of the score scale
            near = np.argsort(np.abs(s - s[centre]))[:n]
            chosen = ids[near]
        else:
            raise ValueError(f"Unknown strategy: {strategy!r}")
        sub = self.scenarios.subset(list(chosen))
        s = np.asarray(scoring.score(self.params, sub.features))
        return [
            {"scenario_id": sub.ids[i], "hazard": sub.labels.iloc[i]["hazard"],
             "current_score": float(s[i])}
            for i in range(len(sub))
        ]

    # --- take back -----------------------------------------------------------
    def submit_ordering(self, ordered_ids: list[str], group: str | None = None, **meta) -> None:
        """Store an expert's returned ordering (most critical first)."""
        unknown = [s for s in ordered_ids if s not in self.id_to_row]
        if unknown:
            raise ValueError(f"Unknown scenario ids: {unknown[:3]}...")
        self.store.add(Ordering(ordered_ids=list(ordered_ids), group=group, meta=meta))

    # --- regroup + refit -----------------------------------------------------
    def refit(self, *, adjacent_only: bool = False, warm_start: bool = True, **fit_kwargs) -> FitResult:
        """Regroup every stored ordering into one pool and re-fit the weights."""
        pool = self.store.regroup(adjacent_only=adjacent_only)
        result = refit(
            self.scenarios.features,
            self.id_to_row,
            pool.net(),
            init=self.params if warm_start else None,
            prior=scoring.init_params(),  # anchor to the known WHC constants
            **fit_kwargs,
        )
        self.params = result.params
        self._save_weights()
        return result

    # --- persistence ---------------------------------------------------------
    def _save_weights(self) -> None:
        if not self.weights_path:
            return
        self.weights_path.write_text(
            json.dumps({k: np.asarray(v).tolist() for k, v in self.params.items()}, indent=2)
        )

    def _load_weights(self) -> dict | None:
        if not (self.weights_path and self.weights_path.exists()):
            return None
        import jax.numpy as jnp

        raw = json.loads(self.weights_path.read_text())
        return {k: jnp.asarray(v) for k, v in raw.items()}
