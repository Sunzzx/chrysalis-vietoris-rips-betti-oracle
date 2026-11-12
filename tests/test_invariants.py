"""Numerical invariance and edge-case stress tests."""

from __future__ import annotations

import numpy as np
import pytest

from chrysalis.boundary import IndexedComplex, reduce_boundary_matrix
from chrysalis.datasets import circle_cloud
from chrysalis.persistence import compute_persistence
from chrysalis.rips import VietorisRipsFiltration


def test_empty_cloud_rejected_or_trivial():
    pts = np.zeros((0, 2))
    filt = VietorisRipsFiltration(pts, max_dim=1)
    assert filt.build() == []
    dgm = compute_persistence(pts, max_dim=0)
    assert len(dgm) == 0


def test_single_point():
    pts = np.array([[1.0, 2.0, 3.0]])
    dgm = compute_persistence(pts, max_dim=1)
    assert len(dgm.in_dimension(0).essential()) == 1
    assert len(dgm.in_dimension(0).finite()) == 0
    assert len(dgm.in_dimension(1)) == 0


def test_duplicate_points_merge_at_zero():
    pts = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
    dgm = compute_persistence(pts, max_dim=0)
    finite = dgm.in_dimension(0).finite().intervals
    # One merge at death 0 (duplicates) and one at death 1
    deaths = sorted(iv.death for iv in finite)
    assert any(d == pytest.approx(0.0) for d in deaths)
    assert any(d == pytest.approx(1.0) for d in deaths)


def test_reduction_idempotent_pivots_unique():
    pts = circle_cloud(16, seed=5)
    filt = VietorisRipsFiltration(pts, max_dim=2, max_scale=1.2)
    ix = IndexedComplex.from_filtration(filt)
    pairs, reduced = reduce_boundary_matrix(ix)
    # Each birth appears at most once
    births = list(pairs.values())
    assert len(births) == len(set(births))
    # Reduced nonzero columns have unique lowest ones
    lows = []
    for col in reduced:
        if col:
            lows.append(col[-1])
    assert len(lows) == len(set(lows))


def test_filtration_monotonic_inclusion():
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    f_small = VietorisRipsFiltration(pts, max_dim=1, max_scale=1.0)
    f_large = VietorisRipsFiltration(pts, max_dim=1, max_scale=2.0)
    s_small = {s.vertices for s in f_small.build()}
    s_large = {s.vertices for s in f_large.build()}
    assert s_small.issubset(s_large)


def test_h0_count_matches_union_find_intuition():
    """n points on a line: n-1 finite H0 bars + 1 essential."""
    pts = np.array([[float(i), 0.0] for i in range(7)])
    dgm = compute_persistence(pts, max_dim=0)
    h0 = dgm.in_dimension(0)
    assert len(h0.essential()) == 1
    assert len(h0.finite()) == 6
    deaths = sorted(iv.death for iv in h0.finite())
    # Consecutive merges at distance 1
    assert deaths == pytest.approx([1, 1, 1, 1, 1, 1])
