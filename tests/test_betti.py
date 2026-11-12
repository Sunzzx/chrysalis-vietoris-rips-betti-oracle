"""Betti number / curve invariants and topological oracle tests."""

from __future__ import annotations

import numpy as np
import pytest

from chrysalis.betti import betti_at_scale, betti_curve, euler_characteristic_at_scale
from chrysalis.datasets import circle_cloud, figure_eight_cloud, two_clusters
from chrysalis.diagrams import PersistenceDiagram, PersistenceInterval
from chrysalis.persistence import compute_persistence
from chrysalis.rips import pairwise_distances


def test_betti_at_scale_manual_diagram():
    dgm = PersistenceDiagram(
        [
            PersistenceInterval(0, 0.0, float("inf")),
            PersistenceInterval(0, 0.0, 1.0),
            PersistenceInterval(0, 0.0, 2.0),
            PersistenceInterval(1, 0.5, 1.5),
        ]
    )
    b0 = betti_at_scale(dgm, 0.25, max_dim=1)
    assert b0[0] == 3  # all three H0 alive
    assert b0[1] == 0
    b1 = betti_at_scale(dgm, 0.75, max_dim=1)
    # deaths are half-open [b,d): at 0.75, deaths at 1.0 and 2.0 still alive => β0=3
    assert b1[0] == 3
    assert b1[1] == 1
    b2 = betti_at_scale(dgm, 1.0, max_dim=1)
    assert b2[0] == 2  # [0,1) ended
    assert b2[1] == 1
    b3 = betti_at_scale(dgm, 1.5, max_dim=1)
    assert b3[1] == 0
    assert b3[0] == 2


def test_two_clusters_betti0_mid_scale():
    pts = two_clusters(10, separation=6.0, noise=0.05, seed=0)
    dgm = compute_persistence(pts, max_dim=0)
    # Immediately after birth all points are separate: β0 = n
    b_early = betti_at_scale(dgm, 0.0, max_dim=0)
    assert b_early[0] == len(pts)
    # Mid scale below separation: two components
    # Cluster internal scale ~ noise; separation = 6
    b_mid = betti_at_scale(dgm, 1.0, max_dim=0)
    assert b_mid[0] == 2
    # After full merge
    b_late = betti_at_scale(dgm, 10.0, max_dim=0)
    assert b_late[0] == 1


def test_circle_betti_oracle():
    pts = circle_cloud(30, radius=1.0, noise=0.0, seed=0)
    # Adjacent chord
    d = pairwise_distances(pts)
    adj = np.min(d + np.eye(len(pts)) * 1e9)
    dgm = compute_persistence(pts, max_dim=1, max_scale=adj * 3.0)
    scales, curves = betti_curve(dgm, max_dim=1, n_samples=50)
    # Somewhere in the middle β1 should hit 1 and β0 should be 1
    found = False
    for i, eps in enumerate(scales):
        if curves[0][i] == 1 and curves[1][i] >= 1:
            found = True
            break
    assert found, "circle should exhibit β0=1 and β1≥1 on some scale window"


def test_figure_eight_higher_beta1():
    pts = figure_eight_cloud(40, noise=0.0, seed=1)
    d = pairwise_distances(pts)
    # typical nearest neighbor
    nn = np.partition(d + np.eye(len(pts)) * 1e9, 1, axis=1)[:, 1]
    scale = float(np.median(nn) * 4.0)
    dgm = compute_persistence(pts, max_dim=1, max_scale=scale)
    scales, curves = betti_curve(dgm, max_dim=1, n_samples=60)
    max_b1 = int(curves[1].max())
    # Figure-eight should allow β1 up to 2
    assert max_b1 >= 2, f"expected max β1≥2 for figure-eight, got {max_b1}"


def test_euler_on_manual():
    dgm = PersistenceDiagram(
        [
            PersistenceInterval(0, 0.0, float("inf")),
            PersistenceInterval(1, 0.0, float("inf")),
        ]
    )
    # χ = β0 - β1 = 0
    assert euler_characteristic_at_scale(dgm, 1.0, max_dim=1) == 0


def test_distance_matrix_invariant_to_rigid_motion():
    rng = np.random.default_rng(4)
    pts = rng.normal(size=(12, 3))
    # Rotate + translate
    from math import cos, sin

    R = np.array([[cos(0.3), -sin(0.3), 0], [sin(0.3), cos(0.3), 0], [0, 0, 1]])
    pts2 = pts @ R.T + np.array([10.0, -3.0, 2.0])
    d1 = pairwise_distances(pts)
    d2 = pairwise_distances(pts2)
    np.testing.assert_allclose(np.sort(d1.ravel()), np.sort(d2.ravel()), atol=1e-9)


def test_persistence_stable_under_tiny_noise():
    pts = circle_cloud(20, noise=0.0, seed=0)
    dgm0 = compute_persistence(pts, max_dim=1, max_scale=1.5)
    rng = np.random.default_rng(0)
    pts_n = pts + rng.normal(0, 1e-4, size=pts.shape)
    dgm1 = compute_persistence(pts_n, max_dim=1, max_scale=1.5)
    # Same number of essential H0
    assert len(dgm0.in_dimension(0).essential()) == len(dgm1.in_dimension(0).essential()) == 1
