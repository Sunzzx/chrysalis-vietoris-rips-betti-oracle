"""Vietoris–Rips filtration construction from finite metric point clouds."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Sequence

import numpy as np


def pairwise_distances(points: np.ndarray) -> np.ndarray:
    """Return the symmetric Euclidean distance matrix.

    Parameters
    ----------
    points:
        Array of shape ``(n, d)``.

    Returns
    -------
    ndarray
        Shape ``(n, n)`` with zeros on the diagonal.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2:
        raise ValueError("points must be a 2-D array of shape (n, d)")
    diff = pts[:, None, :] - pts[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1))


@dataclass(frozen=True, order=True)
class Simplex:
    """An abstract simplex with filtration value and sorted vertex tuple."""

    filtration: float
    vertices: tuple[int, ...]
    dimension: int

    @staticmethod
    def make(vertices: Iterable[int], filtration: float) -> "Simplex":
        verts = tuple(sorted(int(v) for v in vertices))
        if len(verts) != len(set(verts)):
            raise ValueError("simplex vertices must be unique")
        return Simplex(filtration=float(filtration), vertices=verts, dimension=len(verts) - 1)


class VietorisRipsFiltration:
    """Build a Vietoris–Rips filtration up to a maximum dimension and scale.

    The VR complex at scale ε contains a k-simplex on vertices
    ``{v0,...,vk}`` iff every pairwise distance among them is ≤ ε.
    The filtration value of a simplex is the diameter of its vertex set
    (maximum pairwise edge length).

    Complexity
    ----------
    - Distance matrix: ``O(n² d)``
    - Enumerating simplices up to dimension ``max_dim``:
      ``O(sum_{k=0}^{max_dim} C(n, k+1))`` which is exponential in
      ``max_dim``; practical for small n or low max_dim (≤ 2).
    """

    def __init__(
        self,
        points: np.ndarray,
        *,
        max_dim: int = 2,
        max_scale: float | None = None,
    ) -> None:
        if max_dim < 0:
            raise ValueError("max_dim must be >= 0")
        self.points = np.asarray(points, dtype=np.float64)
        if self.points.ndim != 2:
            raise ValueError("points must have shape (n, d)")
        self.n = self.points.shape[0]
        self.max_dim = int(max_dim)
        self.distances = pairwise_distances(self.points)
        finite = self.distances[np.triu_indices(self.n, k=1)]
        self.diameter = float(finite.max()) if finite.size else 0.0
        if max_scale is None:
            self.max_scale = self.diameter
        else:
            self.max_scale = float(max_scale)
        self._simplices: list[Simplex] | None = None

    def _edge_length(self, i: int, j: int) -> float:
        return float(self.distances[i, j])

    def _simplex_filtration(self, verts: Sequence[int]) -> float:
        if len(verts) == 1:
            return 0.0
        m = 0.0
        for a, b in combinations(verts, 2):
            m = max(m, self._edge_length(a, b))
        return m

    def build(self) -> list[Simplex]:
        """Enumerate all simplices with filtration ≤ max_scale, sorted."""
        if self._simplices is not None:
            return self._simplices

        simplices: list[Simplex] = []
        # 0-simplices (vertices)
        for i in range(self.n):
            simplices.append(Simplex.make((i,), 0.0))

        # Higher simplices
        for dim in range(1, self.max_dim + 1):
            for verts in combinations(range(self.n), dim + 1):
                filt = self._simplex_filtration(verts)
                if filt <= self.max_scale + 1e-15:
                    simplices.append(Simplex.make(verts, filt))

        simplices.sort()  # by filtration, then vertices
        self._simplices = simplices
        return simplices

    def simplices_by_dimension(self) -> dict[int, list[Simplex]]:
        """Group built simplices by dimension."""
        out: dict[int, list[Simplex]] = {}
        for s in self.build():
            out.setdefault(s.dimension, []).append(s)
        return out

    def critical_scales(self) -> np.ndarray:
        """Unique filtration values appearing in the complex."""
        vals = sorted({s.filtration for s in self.build()})
        return np.asarray(vals, dtype=np.float64)
