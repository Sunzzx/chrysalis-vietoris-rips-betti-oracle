"""Persistence diagrams and interval records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

import numpy as np


@dataclass(frozen=True, order=True)
class PersistenceInterval:
    """A single persistence bar in homology degree ``dim``."""

    dim: int
    birth: float
    death: float  # np.inf for essential classes

    @property
    def persistence(self) -> float:
        if np.isinf(self.death):
            return float("inf")
        return float(self.death - self.birth)

    @property
    def is_finite(self) -> bool:
        return not np.isinf(self.death)

    def contains_scale(self, eps: float) -> bool:
        """True if the class is alive at filtration value ``eps`` (half-open [b, d))."""
        if eps < self.birth:
            return False
        if np.isinf(self.death):
            return True
        return eps < self.death


@dataclass
class PersistenceDiagram:
    """Collection of persistence intervals, optionally filtered by dimension."""

    intervals: list[PersistenceInterval]

    def __iter__(self) -> Iterator[PersistenceInterval]:
        return iter(self.intervals)

    def __len__(self) -> int:
        return len(self.intervals)

    def in_dimension(self, dim: int) -> "PersistenceDiagram":
        return PersistenceDiagram([iv for iv in self.intervals if iv.dim == dim])

    def finite(self) -> "PersistenceDiagram":
        return PersistenceDiagram([iv for iv in self.intervals if iv.is_finite])

    def essential(self) -> "PersistenceDiagram":
        return PersistenceDiagram([iv for iv in self.intervals if not iv.is_finite])

    def significant(self, min_persistence: float) -> "PersistenceDiagram":
        """Drop short bars (topological noise) below a persistence threshold."""
        kept: list[PersistenceInterval] = []
        for iv in self.intervals:
            if not iv.is_finite or iv.persistence >= min_persistence:
                kept.append(iv)
        return PersistenceDiagram(kept)

    def as_array(self, dim: int | None = None) -> np.ndarray:
        """Return ``(birth, death)`` pairs; essential deaths encoded as ``-1``."""
        ivs = self.intervals if dim is None else [i for i in self.intervals if i.dim == dim]
        if not ivs:
            return np.zeros((0, 2), dtype=np.float64)
        rows = []
        for iv in ivs:
            d = -1.0 if np.isinf(iv.death) else float(iv.death)
            rows.append((iv.birth, d))
        return np.asarray(rows, dtype=np.float64)

    @staticmethod
    def merge(diagrams: Sequence["PersistenceDiagram"]) -> "PersistenceDiagram":
        ivs: list[PersistenceInterval] = []
        for d in diagrams:
            ivs.extend(d.intervals)
        ivs.sort()
        return PersistenceDiagram(ivs)

    def summary(self) -> str:
        lines = ["Persistence diagram summary:"]
        dims = sorted({iv.dim for iv in self.intervals})
        for d in dims:
            sub = self.in_dimension(d)
            n_ess = len(sub.essential())
            n_fin = len(sub.finite())
            lines.append(f"  H_{d}: {n_fin} finite + {n_ess} essential bars")
        return "\n".join(lines)
