"""Expert orderings of scenario subsets, and how they regroup into one signal.

The interface serves a subset of scenarios; an expert returns them ordered from
most critical (rank 0) to least critical. Each ordering only constrains the items
within its subset, so we turn it into pairwise (winner, loser) constraints and
pool the constraints from every ordering into one global bag.

Because the model score is a function of shared weights (not a free per-scenario
value), constraints from disjoint subsets are all comparable through those shared
weights, so no overlapping/anchor scenarios are needed to merge them.
"""
from __future__ import annotations

import itertools
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Ordering:
    """One returned ranking of a subset (``ordered_ids[0]`` = most critical)."""

    ordered_ids: list[str]
    group: str | None = None
    meta: dict = field(default_factory=dict)  # e.g. who/when/source

    def pairs(self, adjacent_only: bool = False) -> list[tuple[str, str]]:
        """Pairwise (winner, loser) constraints implied by this ordering.

        By default every earlier item beats every later one; adjacent_only keeps
        only consecutive pairs.
        """
        if adjacent_only:
            return list(zip(self.ordered_ids, self.ordered_ids[1:]))
        return list(itertools.combinations(self.ordered_ids, 2))

    def to_json(self) -> dict:
        return {"ordered_ids": self.ordered_ids, "group": self.group, "meta": self.meta}

    @classmethod
    def from_json(cls, d: dict) -> "Ordering":
        return cls(ordered_ids=list(d["ordered_ids"]), group=d.get("group"), meta=d.get("meta", {}))


@dataclass
class PairPool:
    """Regrouped pairwise constraints across all orderings.

    weights[(w, l)] counts how many orderings put winner w over loser l. Training
    uses the net weight (support minus contradictions) so later corrections can
    outweigh earlier ones.
    """

    weights: Counter = field(default_factory=Counter)

    def add(self, pairs: list[tuple[str, str]]) -> None:
        for w, l in pairs:
            self.weights[(w, l)] += 1

    def net(self) -> dict[tuple[str, str], int]:
        """Collapse contradictory pairs: keep the majority direction with its margin."""
        seen: set[frozenset] = set()
        out: dict[tuple[str, str], int] = {}
        for (w, l) in list(self.weights):
            key = frozenset((w, l))
            if key in seen:
                continue
            seen.add(key)
            fwd = self.weights.get((w, l), 0)
            rev = self.weights.get((l, w), 0)
            margin = fwd - rev
            if margin > 0:
                out[(w, l)] = margin
            elif margin < 0:
                out[(l, w)] = -margin
            # margin == 0 -> perfectly contradictory, drop
        return out


class OrderingStore:
    """Append-only collection of orderings with optional JSONL persistence."""

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else None
        self.orderings: list[Ordering] = []
        if self.path and self.path.exists():
            self._load()

    def _load(self) -> None:
        with self.path.open() as f:
            self.orderings = [Ordering.from_json(json.loads(line)) for line in f if line.strip()]

    def add(self, ordering: Ordering) -> None:
        self.orderings.append(ordering)
        if self.path:
            with self.path.open("a") as f:
                f.write(json.dumps(ordering.to_json()) + "\n")

    def regroup(self, adjacent_only: bool = False) -> PairPool:
        """Merge every ordering's pairwise constraints into one PairPool."""
        pool = PairPool()
        for o in self.orderings:
            pool.add(o.pairs(adjacent_only=adjacent_only))
        return pool

    def __len__(self) -> int:
        return len(self.orderings)
