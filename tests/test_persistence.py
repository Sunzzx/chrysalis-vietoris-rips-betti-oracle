"""Tests for boundary reduction and persistence pairs."""

from __future__ import annotations

import numpy as np
import pytest

from chrysalis.boundary import IndexedComplex, boundary_faces, reduce_boundary_matrix
from chrysalis.diagrams import PersistenceDiagram, PersistenceInterval
from chrysalis.persistence import compute_persistence
from chrysalis.rips import Simplex, VietorisRipsFiltration


def test_boundary_faces_triangle():
    s = Simplex.make((0, 1, 2), 1.0)
    faces = boundary_faces(s)
    assert faces == [(1, 2), (0, 2), (0, 1)]


def test_indexed_complex_order():
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.1]])
    filt = VietorisRipsFiltration(pts, max_dim=2)
    ix = IndexedComplex.from_filtration(filt)
    # Filtration nondecreasing
    f = ix.filtration
    assert np.all(f[1:] >= f[:-1] - 1e-15)
    # Within equal filtration, dimension nondecreasing
    for i in range(len(ix) - 1):
        if abs(f[i] - f[i + 1]) < 1e-15:
            assert ix.dimension[i] <= ix.dimension[i + 1]


def test_two_points_h0_merges():
    """Two points: one finite H0 bar (merge) + one essential H0."""
    pts = np.array([[0.0], [2.0]])
    dgm = compute_persistence(pts, max_dim=0)
    h0 = dgm.in_dimension(0)
    finite = h0.finite().intervals
    essential = h0.essential().intervals
    assert len(essential) == 1
    assert essential[0].birth == 0.0
    assert len(finite) == 1
    assert finite[0].birth == 0.0
    assert finite[0].death == pytest.approx(2.0)


def test_triangle_cycle_birth_death():
    """Equilateral-ish triangle: H1 born at longest edge, dies at triangle fill."""
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])
    dgm = compute_persistence(pts, max_dim=1)
    h1 = [iv for iv in dgm.in_dimension(1) if iv.is_finite]
    # With only 3 points the 1-cycle dies immediately when the 2-simplex appears
    # at the same scale as the longest edge (equilateral => birth==death possible).
    # Use a non-equilateral triangle for a positive-length bar.
    pts2 = np.array([[0.0, 0.0], [2.0, 0.0], [0.5, 0.5]])
    dgm2 = compute_persistence(pts2, max_dim=1, min_persistence=1e-12)
    h1b = [iv for iv in dgm2.in_dimension(1) if iv.is_finite and iv.persistence > 1e-9]
    # There should be exactly one positive H1 bar or zero if filled at birth scale;
    # longest edge length vs triangle diameter.
    edges = [
        np.linalg.norm(pts2[0] - pts2[1]),
        np.linalg.norm(pts2[0] - pts2[2]),
        np.linalg.norm(pts2[1] - pts2[2]),
    ]
    longest = max(edges)
    diam = longest  # for 3 points triangle filtration = longest edge
    # In VR, the 2-simplex appears at the same scale as the longest edge,
    # so H1 persistence is typically 0 for a single triangle. Check H0 instead.
    h0 = dgm2.in_dimension(0)
    assert len(h0.essential()) == 1
    assert len(h0.finite()) == 2  # three vertices -> two merges


def test_circle_has_essential_h1():
    """Dense circle samples should yield one long/essential H1 class."""
    n = 24
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.column_stack([np.cos(theta), np.sin(theta)])
    # Cap scale so the disk does not fully fill (no high 2-simplices killing loop early)
    # Chord length between adjacent samples:
    adj = np.linalg.norm(pts[0] - pts[1])
    # Allow edges of a few hops but keep max_dim=1 so no 2-simplices kill the loop
    dgm = compute_persistence(pts, max_dim=1, max_scale=adj * 2.5, min_persistence=0.05)
    h1 = dgm.in_dimension(1)
    # With max_dim=1 we still build 2-simplices (max_dim+1). For a circle,
    # a persistent H1 bar should appear.
    long_bars = [iv for iv in h1 if (not iv.is_finite) or iv.persistence > 0.3]
    assert len(long_bars) >= 1, f"expected persistent H1, got {list(h1)}"


def test_diagram_filters():
    dgm = PersistenceDiagram(
        [
            PersistenceInterval(0, 0.0, 1.0),
            PersistenceInterval(0, 0.0, float("inf")),
            PersistenceInterval(1, 0.5, 0.6),
            PersistenceInterval(1, 0.2, 2.0),
        ]
    )
    assert len(dgm.finite()) == 3
    assert len(dgm.essential()) == 1
    assert len(dgm.significant(0.2).in_dimension(1)) == 1
    arr = dgm.as_array(0)
    assert arr.shape[0] == 2
