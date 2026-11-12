"""Interactive terminal demo: barcodes, Betti curves, manifold oracles."""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from chrysalis import __version__
from chrysalis.betti import betti_at_scale, betti_curve
from chrysalis.datasets import circle_cloud, figure_eight_cloud, two_clusters
from chrysalis.diagrams import PersistenceDiagram
from chrysalis.persistence import compute_persistence


def _bar_row(birth: float, death: float, lo: float, hi: float, width: int = 48) -> str:
    span = max(hi - lo, 1e-12)
    if np.isinf(death):
        death_v = hi
        inf_mark = True
    else:
        death_v = death
        inf_mark = False
    b = int(round((birth - lo) / span * (width - 1)))
    d = int(round((death_v - lo) / span * (width - 1)))
    b = max(0, min(width - 1, b))
    d = max(b, min(width - 1, d))
    chars = ["·"] * width
    for i in range(b, d + 1):
        chars[i] = "█"
    if inf_mark:
        chars[-1] = "►"
    return "".join(chars)


def render_barcode(diagram: PersistenceDiagram, max_dim: int = 1, width: int = 48) -> str:
    finite_deaths = [iv.death for iv in diagram if iv.is_finite]
    births = [iv.birth for iv in diagram]
    lo = 0.0
    hi = max(finite_deaths) if finite_deaths else 1.0
    if births:
        hi = max(hi, max(births) * 1.05 + 1e-6)
    lines = [
        f"Persistence barcode  (scale {lo:.3f} → {hi:.3f})",
        "─" * (width + 18),
    ]
    for dim in range(max_dim + 1):
        ivs = sorted(diagram.in_dimension(dim).intervals, key=lambda x: (-x.persistence if x.is_finite else float("inf"), x.birth))
        lines.append(f"H_{dim}  ({len(ivs)} bars)")
        if not ivs:
            lines.append("  (empty)")
            continue
        # Show top bars by persistence (cap for readability)
        for iv in ivs[:12]:
            row = _bar_row(iv.birth, iv.death, lo, hi, width)
            dlabel = "∞" if not iv.is_finite else f"{iv.death:.3f}"
            lines.append(f"  [{iv.birth:6.3f}, {dlabel:>6}] |{row}|")
        if len(ivs) > 12:
            lines.append(f"  … +{len(ivs) - 12} more")
    return "\n".join(lines)


def render_betti_curve(diagram: PersistenceDiagram, max_dim: int = 1) -> str:
    scales, curves = betti_curve(diagram, max_dim=max_dim, n_samples=40)
    lines = ["Betti curves β_k(ε)", "─" * 56]
    max_b = max(int(curves[k].max()) if len(curves[k]) else 0 for k in curves) or 1
    height = min(8, max_b)
    for k in range(max_dim + 1):
        arr = curves[k]
        lines.append(f"β_{k}:")
        # sparkline
        blocks = " ▁▂▃▄▅▆▇█"
        spark = []
        for v in arr:
            idx = int(round(v / max_b * (len(blocks) - 1)))
            spark.append(blocks[idx])
        lines.append("  " + "".join(spark))
        # sample annotations
        mid = len(scales) // 2
        b_mid = betti_at_scale(diagram, float(scales[mid]), max_dim=max_dim)
        lines.append(
            f"  ε∈[{scales[0]:.3f},{scales[-1]:.3f}]  "
            f"β_{k}(mid={scales[mid]:.3f})={b_mid.get(k, 0)}"
        )
    return "\n".join(lines)


def run_suite() -> int:
    print(f"chrysalis v{__version__} — Vietoris–Rips Betti Oracle")
    print("=" * 60)

    cases = [
        ("circle S¹", circle_cloud(36, noise=0.02, seed=1), 1, "expect β0=1, β1=1"),
        ("figure-eight", figure_eight_cloud(48, noise=0.02, seed=2), 1, "expect β0=1, β1=2"),
        ("two clusters", two_clusters(12, separation=4.0, seed=3), 0, "expect β0=2 mid-scale"),
    ]

    t0 = time.perf_counter()
    for name, pts, max_dim, note in cases:
        print()
        print(f"▸ Dataset: {name}  (n={len(pts)}, d={pts.shape[1]})  [{note}]")
        t1 = time.perf_counter()
        diagram = compute_persistence(pts, max_dim=max_dim, min_persistence=1e-9)
        dt = time.perf_counter() - t1
        print(f"  computed in {dt*1000:.1f} ms")
        print(diagram.summary())
        # Pick a mid scale for Betti snapshot
        finite = [iv.death for iv in diagram if iv.is_finite]
        eps = float(np.median(finite)) if finite else 0.5
        beta = betti_at_scale(diagram, eps, max_dim=max_dim)
        beta_s = ", ".join(f"β{k}={beta.get(k, 0)}" for k in range(max_dim + 1))
        print(f"  Betti at ε≈{eps:.3f}: {beta_s}")
        print()
        print(render_barcode(diagram, max_dim=max_dim, width=40))
        print()
        print(render_betti_curve(diagram, max_dim=max_dim))
        print("─" * 60)

    print(f"\nTotal wall time: {time.perf_counter() - t0:.3f}s")
    print("Humanity impact: shape-aware biomarkers from point-cloud topology")
    print("  (protein conformations · tumor margins · neural manifolds)")
    return 0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="chrysalis",
        description="Vietoris–Rips persistent homology Betti barcode oracle",
    )
    parser.add_argument("--version", action="version", version=f"chrysalis {__version__}")
    parser.add_argument(
        "--dataset",
        choices=["circle", "eight", "clusters", "all"],
        default="all",
        help="Which synthetic manifold to analyse",
    )
    parser.add_argument("--n", type=int, default=36, help="Points per cloud")
    parser.add_argument("--max-dim", type=int, default=1, help="Max homology dimension")
    args = parser.parse_args(argv)

    if args.dataset == "all":
        raise SystemExit(run_suite())

    if args.dataset == "circle":
        pts = circle_cloud(args.n, noise=0.02, seed=7)
    elif args.dataset == "eight":
        pts = figure_eight_cloud(max(args.n, 40), noise=0.02, seed=7)
    else:
        pts = two_clusters(max(args.n // 2, 8), seed=7)

    diagram = compute_persistence(pts, max_dim=args.max_dim, min_persistence=1e-9)
    print(diagram.summary())
    print(render_barcode(diagram, max_dim=args.max_dim))
    print(render_betti_curve(diagram, max_dim=args.max_dim))


if __name__ == "__main__":
    main()
