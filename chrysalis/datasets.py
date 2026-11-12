"""Synthetic point-cloud manifolds with known topological signatures."""

from __future__ import annotations

import numpy as np


def circle_cloud(
    n: int = 40,
    *,
    radius: float = 1.0,
    noise: float = 0.0,
    seed: int | None = 0,
) -> np.ndarray:
    """Uniform samples on S^1 ⊂ ℝ². Expected: β0=1, β1=1 for mid-range ε."""
    rng = np.random.default_rng(seed)
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    pts = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])
    if noise > 0:
        pts = pts + rng.normal(0.0, noise, size=pts.shape)
    return pts.astype(np.float64)


def figure_eight_cloud(
    n: int = 60,
    *,
    noise: float = 0.0,
    seed: int | None = 0,
) -> np.ndarray:
    """Two tangent circles (wedge of two S^1). Expected: β0=1, β1=2."""
    rng = np.random.default_rng(seed)
    n1 = n // 2
    n2 = n - n1
    t1 = np.linspace(0.0, 2.0 * np.pi, n1, endpoint=False)
    t2 = np.linspace(0.0, 2.0 * np.pi, n2, endpoint=False)
    c1 = np.column_stack([np.cos(t1) - 1.0, np.sin(t1)])
    c2 = np.column_stack([np.cos(t2) + 1.0, np.sin(t2)])
    pts = np.vstack([c1, c2])
    if noise > 0:
        pts = pts + rng.normal(0.0, noise, size=pts.shape)
    return pts.astype(np.float64)


def sphere_cloud(
    n: int = 80,
    *,
    radius: float = 1.0,
    noise: float = 0.0,
    seed: int | None = 0,
) -> np.ndarray:
    """Approximate samples on S^2 ⊂ ℝ³ via Gaussian normalization."""
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(n, 3))
    raw /= np.linalg.norm(raw, axis=1, keepdims=True)
    pts = radius * raw
    if noise > 0:
        pts = pts + rng.normal(0.0, noise, size=pts.shape)
    return pts.astype(np.float64)


def torus_cloud(
    n: int = 100,
    *,
    R: float = 2.0,
    r: float = 0.8,
    noise: float = 0.0,
    seed: int | None = 0,
) -> np.ndarray:
    """Uniform-ish samples on a torus of major R, minor r in ℝ³.

    Topologically T² has β0=1, β1=2, β2=1 (for dense enough samples).
    """
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 2.0 * np.pi, size=n)
    v = rng.uniform(0.0, 2.0 * np.pi, size=n)
    x = (R + r * np.cos(v)) * np.cos(u)
    y = (R + r * np.cos(v)) * np.sin(u)
    z = r * np.sin(v)
    pts = np.column_stack([x, y, z])
    if noise > 0:
        pts = pts + rng.normal(0.0, noise, size=pts.shape)
    return pts.astype(np.float64)


def two_clusters(
    n_per: int = 15,
    *,
    separation: float = 5.0,
    noise: float = 0.15,
    seed: int | None = 0,
) -> np.ndarray:
    """Two Gaussian blobs. Mid-scale: β0=2 until the bridge scale."""
    rng = np.random.default_rng(seed)
    a = rng.normal(0.0, noise, size=(n_per, 2))
    b = rng.normal(0.0, noise, size=(n_per, 2)) + np.array([separation, 0.0])
    return np.vstack([a, b]).astype(np.float64)
