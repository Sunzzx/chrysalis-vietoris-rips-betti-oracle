"""Unit tests for Vietoris–Rips construction."""

from __future__ import annotations

import numpy as np
import pytest

from chrysalis.rips import Simplex, VietorisRipsFiltration, pairwise_distances


def test_pairwise_distances_symmetry_and_zero_diag():
    pts = np.array([[0.0, 0.0], [3.0, 4.0], [1.0, 0.0]])
    d = pairwise_distances(pts)
    assert d.shape == (3, 3)
    np.testing.assert_allclose(d, d.T)
    np.testing.assert_allclose(np.diag(d), 0.0)
    np.testing.assert_allclose(d[0, 1], 5.0)


def test_simplex_ordering_and_validation():
    s = Simplex.make((2, 0, 1), 1.5)
    assert s.vertices == (0, 1, 2)
    assert s.dimension == 2
    assert s.filtration == 1.5
    with pytest.raises(ValueError):
        Simplex.make((0, 0), 1.0)


def test_vr_vertices_only():
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    filt = VietorisRipsFiltration(pts, max_dim=0, max_scale=10.0)
    simps = filt.build()
    assert len(simps) == 3
    assert all(s.dimension == 0 for s in simps)
    assert all(s.filtration == 0.0 for s in simps)


def test_vr_edge_filtration_is_diameter():
    pts = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
    filt = VietorisRipsFiltration(pts, max_dim=1, max_scale=10.0)
    by_dim = filt.simplices_by_dimension()
    edges = {s.vertices: s.filtration for s in by_dim[1]}
    assert edges[(0, 1)] == pytest.approx(2.0)
    assert edges[(0, 2)] == pytest.approx(2.0)
    assert edges[(1, 2)] == pytest.approx(np.sqrt(8.0))


def test_vr_triangle_filtration():
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    filt = VietorisRipsFiltration(pts, max_dim=2)
    tris = filt.simplices_by_dimension()[2]
    assert len(tris) == 1
    # diameter is the hypotenuse
    assert tris[0].filtration == pytest.approx(np.sqrt(2.0))


def test_max_scale_prunes_long_edges():
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0]])
    filt = VietorisRipsFiltration(pts, max_dim=1, max_scale=2.0)
    edges = filt.simplices_by_dimension().get(1, [])
    verts = {e.vertices for e in edges}
    assert (0, 1) in verts
    assert (0, 2) not in verts
    assert (1, 2) not in verts


def test_critical_scales_sorted_unique():
    pts = np.array([[0.0], [1.0], [3.0]])
    filt = VietorisRipsFiltration(pts, max_dim=1)
    scales = filt.critical_scales()
    assert scales[0] == 0.0
    assert list(scales) == sorted(set(scales.tolist()))
