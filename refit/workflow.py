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
        x = scoring.assemble_features(
            self.scenarios.features, self.params, self.scenarios.enc_label_idx
        )
        return np.asarray(scoring.score(self.params, x))

    def whc_scores(self) -> np.ndarray:
        """Raw WHC per scenario (the ranking score is ln(WHC); this is WHC itself)."""
        x = scoring.assemble_features(
            self.scenarios.features, self.params, self.scenarios.enc_label_idx
        )
        return np.asarray(scoring.whc(self.params, x))

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
        xs = scoring.assemble_features(sub.features, self.params, sub.enc_label_idx)
        s = np.asarray(scoring.score(self.params, xs))
        return [
            {"scenario_id": sub.ids[i], "hazard": sub.labels.iloc[i]["hazard"],
             "current_score": float(s[i])}
            for i in range(len(sub))
        ]

    def serve_contrast(
        self, param: str, n: int = 10, group: str | None = None,
        seed: int | None = None, pool_factor: int = 8,
    ) -> list[dict]:
        """Pick scenarios that isolate ``param``: similar in every other feature
        but spanning different labels of ``param``.

        Ordering such a subset makes the returned ranking informative about that
        one parameter (rather than being dominated by everything else), which is
        what you want when studying/re-binning a single parameter.
        """
        col = scoring.ENCODABLE[param]["col"]
        ids = np.array(self.scenarios.in_group(group)) if group else self.scenarios.ids
        rows = self.scenarios.index_of(list(ids))
        label_idx = self.scenarios.enc_label_idx[param][rows]

        # z-scored features excluding the target column -> "context" similarity
        other = np.delete(self.scenarios.features[rows], col, axis=1)
        std = other.std(0)
        std[std == 0] = 1.0
        other = (other - other.mean(0)) / std

        rng = np.random.default_rng(seed)
        anchor = rng.integers(len(rows))
        nearest = np.argsort(np.linalg.norm(other - other[anchor], axis=1))
        pool = nearest[: min(len(nearest), n * pool_factor)]

        # round-robin across labels (nearest first) so the subset spans labels
        from collections import defaultdict
        by_label: dict[int, list[int]] = defaultdict(list)
        for i in pool:
            by_label[int(label_idx[i])].append(i)
        chosen: list[int] = []
        while len(chosen) < n and any(by_label.values()):
            for lab in list(by_label):
                if by_label[lab]:
                    chosen.append(by_label[lab].pop(0))
                    if len(chosen) >= n:
                        break
        chosen_ids = list(ids[chosen])
        sub = self.scenarios.subset(chosen_ids)
        xs = scoring.assemble_features(sub.features, self.params, sub.enc_label_idx)
        s = np.asarray(scoring.score(self.params, xs))
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
    def refit(
        self,
        *,
        trainable: list[str] | None = None,
        adjacent_only: bool = False,
        warm_start: bool = True,
        **fit_kwargs,
    ) -> FitResult:
        """Regroup every stored ordering into one pool and re-fit the parameters.

        ``trainable`` selects which parameters to move (default: all combination
        weights). Focus on one thing with e.g. ["enc:distance_to_object"], or fit
        everything with scoring.WEIGHT_KEYS + scoring.ENCODING_KEYS.
        """
        pool = self.store.regroup(adjacent_only=adjacent_only)
        result = refit(
            self.scenarios.features,
            self.id_to_row,
            pool.net(),
            enc_label_idx=self.scenarios.enc_label_idx,
            trainable=trainable,
            init=self.params if warm_start else None,
            prior=scoring.init_params(),  # anchor to the known WHC values
            **fit_kwargs,
        )
        self.params = result.params
        self._save_weights()
        return result

    # --- readouts ------------------------------------------------------------
    def effective_encoding(self, param: str) -> dict:
        """Current label -> score map for a fittable parameter."""
        vec = np.asarray(self.params[scoring.encoding_key(param)])
        return dict(zip(scoring.ENCODABLE[param]["labels"], vec.tolist()))

    def worst_offenders(self, result: FitResult, n: int = 10) -> list[dict]:
        """Scenarios the fitted model most disagrees with the orderings on."""
        order = np.argsort(-result.frustration)
        out = []
        for i in order:
            if result.involvement[i] <= 0:
                continue
            out.append({
                "scenario_id": self.scenarios.ids[i],
                "hazard": self.scenarios.labels.iloc[i]["hazard"],
                "frustration": float(result.frustration[i]),
            })
            if len(out) >= n:
                break
        return out

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
