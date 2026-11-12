"""Persistent homology computation via boundary-matrix reduction."""

from __future__ import annotations

import numpy as np

from chrysalis.boundary import IndexedComplex, reduce_boundary_matrix
from chrysalis.diagrams import PersistenceDiagram, PersistenceInterval
from chrysalis.rips import VietorisRipsFiltration


def compute_persistence(
    points: np.ndarray,
    *,
    max_dim: int = 1,
    max_scale: float | None = None,
    min_persistence: float = 0.0,
) -> PersistenceDiagram:
    """Compute the Vietoris–Rips persistence diagram of a point cloud.

    Parameters
    ----------
    points:
        ``(n, d)`` Euclidean point cloud.
    max_dim:
        Maximum homology dimension to track (requires simplices up to
        dimension ``max_dim + 1`` so that boundaries of (max_dim+1)-cells
        can kill max_dim cycles).
    max_scale:
        Optional hard cap on filtration diameter.
    min_persistence:
        Drop finite bars shorter than this threshold.

    Returns
    -------
    PersistenceDiagram
        Birth/death intervals for dimensions ``0 .. max_dim``.
    """
    # Need simplices one dimension above max_dim for death events
    filt = VietorisRipsFiltration(points, max_dim=max_dim + 1, max_scale=max_scale)
    complex_ = IndexedComplex.from_filtration(filt)
    pairs, reduced = reduce_boundary_matrix(complex_)

    death_cols = set(pairs.keys())
    birth_cols = set(pairs.values())

    intervals: list[PersistenceInterval] = []

    # Finite bars: pair (birth_row = low, death_col = j)
    for death_j, birth_i in pairs.items():
        dim = int(complex_.dimension[birth_i])
        if dim > max_dim:
            continue
        b = float(complex_.filtration[birth_i])
        d = float(complex_.filtration[death_j])
        if d - b < min_persistence:
            continue
        # Degenerate (birth==death) bars are topological noise from same-scale
        if d <= b + 1e-15 and min_persistence > 0:
            continue
        intervals.append(PersistenceInterval(dim=dim, birth=b, death=d))

    # Essential (infinite) classes: columns that reduced to zero and were
    # never paired as a birth pivot.
    for j in range(len(complex_)):
        if j in death_cols:
            continue
        if reduced[j]:  # non-zero reduced column => was a death pivot owner path
            # Actually non-zero means it has a pivot; those are death columns
            # already in pairs. Zero columns that are not births of something
            # still alive are essential births.
            continue
        # Zero column
        if j in birth_cols:
            # It was born and later killed — already recorded as finite bar
            continue
        dim = int(complex_.dimension[j])
        if dim > max_dim:
            continue
        b = float(complex_.filtration[j])
        intervals.append(PersistenceInterval(dim=dim, birth=b, death=float("inf")))

    intervals.sort()
    return PersistenceDiagram(intervals)


def compute_persistence_from_filtration(
    filt: VietorisRipsFiltration,
    *,
    max_dim: int | None = None,
    min_persistence: float = 0.0,
) -> PersistenceDiagram:
    """Like :func:`compute_persistence` but reuses an existing filtration."""
    max_d = filt.max_dim - 1 if max_dim is None else max_dim
    if max_d < 0:
        max_d = 0
    # Rebuild with +1 dim if needed
    if filt.max_dim < max_d + 1:
        filt = VietorisRipsFiltration(
            filt.points, max_dim=max_d + 1, max_scale=filt.max_scale
        )
    complex_ = IndexedComplex.from_filtration(filt)
    pairs, reduced = reduce_boundary_matrix(complex_)
    death_cols = set(pairs.keys())
    birth_cols = set(pairs.values())
    intervals: list[PersistenceInterval] = []

    for death_j, birth_i in pairs.items():
        dim = int(complex_.dimension[birth_i])
        if dim > max_d:
            continue
        b = float(complex_.filtration[birth_i])
        d = float(complex_.filtration[death_j])
        if d - b < min_persistence:
            continue
        intervals.append(PersistenceInterval(dim=dim, birth=b, death=d))

    for j in range(len(complex_)):
        if j in death_cols:
            continue
        if reduced[j]:
            continue
        if j in birth_cols:
            continue
        dim = int(complex_.dimension[j])
        if dim > max_d:
            continue
        b = float(complex_.filtration[j])
        intervals.append(PersistenceInterval(dim=dim, birth=b, death=float("inf")))

    intervals.sort()
    return PersistenceDiagram(intervals)
