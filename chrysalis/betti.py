"""Betti numbers and Betti curves from persistence diagrams."""

from __future__ import annotations

import numpy as np

from chrysalis.diagrams import PersistenceDiagram


def betti_at_scale(diagram: PersistenceDiagram, eps: float, max_dim: int | None = None) -> dict[int, int]:
    """Compute Betti numbers β_k(ε) = # classes of dim k alive at scale ε.

    A class with interval ``[birth, death)`` contributes to β_k at ε iff
    ``birth ≤ ε < death`` (essential classes have death = ∞).
    """
    dims = sorted({iv.dim for iv in diagram.intervals})
    if max_dim is not None:
        dims = [d for d in dims if d <= max_dim]
        for d in range(max_dim + 1):
            if d not in dims:
                dims.append(d)
        dims = sorted(set(dims))
    out: dict[int, int] = {d: 0 for d in dims}
    for iv in diagram.intervals:
        if max_dim is not None and iv.dim > max_dim:
            continue
        if iv.contains_scale(eps):
            out[iv.dim] = out.get(iv.dim, 0) + 1
    return out


def betti_curve(
    diagram: PersistenceDiagram,
    scales: np.ndarray | None = None,
    *,
    max_dim: int = 1,
    n_samples: int = 64,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Sample β_k along a grid of filtration scales.

    Returns
    -------
    scales:
        1-D array of ε values.
    curves:
        Mapping dimension → array of Betti numbers at each scale.
    """
    births = [iv.birth for iv in diagram.intervals]
    deaths = [iv.death for iv in diagram.intervals if np.isfinite(iv.death)]
    if scales is None:
        lo = 0.0
        hi = max(deaths) if deaths else (max(births) if births else 1.0)
        if hi <= lo:
            hi = lo + 1.0
        scales = np.linspace(lo, hi, n_samples)
    else:
        scales = np.asarray(scales, dtype=np.float64)

    curves: dict[int, list[int]] = {k: [] for k in range(max_dim + 1)}
    for eps in scales:
        b = betti_at_scale(diagram, float(eps), max_dim=max_dim)
        for k in range(max_dim + 1):
            curves[k].append(int(b.get(k, 0)))
    return scales, {k: np.asarray(v, dtype=np.int32) for k, v in curves.items()}


def euler_characteristic_at_scale(diagram: PersistenceDiagram, eps: float, max_dim: int = 2) -> int:
    """χ(ε) = Σ_k (-1)^k β_k(ε)."""
    b = betti_at_scale(diagram, eps, max_dim=max_dim)
    return int(sum(((-1) ** k) * b.get(k, 0) for k in range(max_dim + 1)))
