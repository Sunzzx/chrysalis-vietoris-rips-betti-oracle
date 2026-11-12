"""Chrysalis — Vietoris–Rips persistent homology & Betti barcode oracle.

Computational topology for biomedical and scientific point clouds:
filtrations, boundary-matrix reduction, persistence pairs, and Betti numbers.
"""

from __future__ import annotations

from chrysalis.betti import betti_at_scale, betti_curve
from chrysalis.datasets import circle_cloud, figure_eight_cloud, sphere_cloud, torus_cloud
from chrysalis.diagrams import PersistenceDiagram, PersistenceInterval
from chrysalis.persistence import compute_persistence
from chrysalis.rips import VietorisRipsFiltration, pairwise_distances

__all__ = [
    "VietorisRipsFiltration",
    "pairwise_distances",
    "compute_persistence",
    "PersistenceDiagram",
    "PersistenceInterval",
    "betti_at_scale",
    "betti_curve",
    "circle_cloud",
    "figure_eight_cloud",
    "sphere_cloud",
    "torus_cloud",
    "__version__",
]

__version__ = "0.1.0"
