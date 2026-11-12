"""Boundary operators and ordered simplex indexing for chain complexes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from chrysalis.rips import Simplex, VietorisRipsFiltration


def boundary_faces(simplex: Simplex) -> list[tuple[int, ...]]:
    """Oriented faces of a simplex (omitting one vertex at a time).

    Face i omits vertex i and inherits the induced orientation sign
    ``(-1)^i`` which is tracked separately by callers when needed.
    """
    verts = simplex.vertices
    faces: list[tuple[int, ...]] = []
    for i in range(len(verts)):
        face = verts[:i] + verts[i + 1 :]
        faces.append(face)
    return faces


@dataclass
class IndexedComplex:
    """Simplices ordered by filtration with stable indices for matrix reduction.

    Ordering (standard for the clearing / standard algorithm):
    1. increasing filtration value
    2. increasing dimension
    3. lexicographic vertices
    """

    simplices: list[Simplex]
    index_of: dict[tuple[int, ...], int]
    filtration: np.ndarray
    dimension: np.ndarray

    @classmethod
    def from_filtration(cls, filt: VietorisRipsFiltration) -> "IndexedComplex":
        raw = list(filt.build())
        # Stable sort: filtration, dimension, vertices
        raw.sort(key=lambda s: (s.filtration, s.dimension, s.vertices))
        index_of = {s.vertices: i for i, s in enumerate(raw)}
        filtration = np.array([s.filtration for s in raw], dtype=np.float64)
        dimension = np.array([s.dimension for s in raw], dtype=np.int32)
        return cls(simplices=raw, index_of=index_of, filtration=filtration, dimension=dimension)

    def __len__(self) -> int:
        return len(self.simplices)

    def boundary_column(self, j: int) -> list[int]:
        """Return sorted row indices of ∂σ_j over GF(2) (no signs needed).

        Working over ``ℤ/2ℤ`` is standard for barcode computation of
        ``ℤ/2``-homology and simplifies reduction.
        """
        s = self.simplices[j]
        if s.dimension == 0:
            return []
        rows: list[int] = []
        for face in boundary_faces(s):
            rows.append(self.index_of[face])
        rows.sort()
        return rows


def reduce_boundary_matrix(
    complex_: IndexedComplex,
) -> tuple[dict[int, int], list[list[int]]]:
    """Standard left-to-right column reduction over GF(2).

    Returns
    -------
    pairs:
        Map ``death_column -> birth_column`` for finite bars. A column j
        that reduces to zero and is never used as a pivot is an essential
        (infinite) class born at j.
    reduced:
        Reduced columns as sorted lists of pivot-row indices (GF2 support).
    """
    m = len(complex_)
    # R[j] = list of row indices with 1 in column j (sorted)
    R: list[list[int]] = [complex_.boundary_column(j) for j in range(m)]
    # pivot_row -> column that currently owns that lowest-one
    pivot_of_row: dict[int, int] = {}
    pairs: dict[int, int] = {}  # death -> birth

    def lowest(col: list[int]) -> int | None:
        return col[-1] if col else None

    def gf2_add(a: list[int], b: list[int]) -> list[int]:
        """Symmetric difference of two sorted supports."""
        i = j = 0
        out: list[int] = []
        while i < len(a) and j < len(b):
            if a[i] == b[j]:
                i += 1
                j += 1
            elif a[i] < b[j]:
                out.append(a[i])
                i += 1
            else:
                out.append(b[j])
                j += 1
        if i < len(a):
            out.extend(a[i:])
        if j < len(b):
            out.extend(b[j:])
        return out

    for j in range(m):
        col = R[j]
        while True:
            low = lowest(col)
            if low is None:
                break
            owner = pivot_of_row.get(low)
            if owner is None:
                pivot_of_row[low] = j
                pairs[j] = low  # death j kills birth low
                break
            # Add owner column into j
            col = gf2_add(col, R[owner])
        R[j] = col

    return pairs, R
