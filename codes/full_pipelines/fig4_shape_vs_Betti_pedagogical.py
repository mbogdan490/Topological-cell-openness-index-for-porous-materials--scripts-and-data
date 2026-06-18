#!/usr/bin/env python3
"""
Regenerate the pedagogical figure ``fig4_shape_vs_Betti_pedagogical.png``.

The figure contrasts two idealised 2D porous geometries and their topology:

- TOP (closed cells): a 2x2 grid of isolated circular pores.
- BOTTOM (open cells): the same 2x2 grid connected by channels into a lattice.

For each case the script:

1. Procedurally generates the binary structure (1 = solid, 0 = pore).
2. Computes the signed distance transform (negative in pores, positive in solid).
3. Computes the B0 (components) and B1 (loops) Betti curves from the signed
   distance via a gudhi ``PeriodicCubicalComplex`` sublevel-set filtration.
4. Renders the structure panel with dimension arrows (a, h/2, d/2, r, w) and the
   Betti panels with filtration-level arrows marking the topological transitions.

The two cases are stacked into one composite that mirrors the manual Inkscape
original. This script is self-contained: the small generation / persistence
helpers are inlined (mirroring the proven gudhi-based toolchain in
``2D_problem/Scripts/experiments_on_2D_structure_vs_Betti``).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from gudhi import PeriodicCubicalComplex as PCC
from matplotlib.transforms import blended_transform_factory
from scipy.ndimage import binary_dilation, distance_transform_edt

B0_COLOR = "blue"
B1_COLOR = "green"


def _bresenham_line(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
    """Return the integer pixels on a Bresenham line between two endpoints."""
    points: list[tuple[int, int]] = []
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy
    return points


def _disk_structure(radius: float) -> np.ndarray:
    """Return a boolean disk structuring element of the given radius."""
    if radius <= 0:
        return np.ones((1, 1), dtype=bool)
    r = int(np.ceil(radius))
    y, x = np.ogrid[-r : r + 1, -r : r + 1]
    return x * x + y * y <= radius * radius


def generate_porous_2d(
    size: int,
    radius: float,
    spacing: float,
    width: float = 0.0,
    channel_prob: float = 0.0,
) -> np.ndarray:
    """
    Generate a binary 2D structure of circular pores on a regular grid.

    Pores are carved on a square lattice of period ``spacing``. With
    ``channel_prob = 1`` every pair of neighbouring pores (including those whose
    centres fall just outside the frame) is joined by a channel of the given
    ``width``, producing an open lattice; with ``channel_prob = 0`` the pores
    stay isolated (closed cells).

    Returns a ``uint8`` grid where 1 = solid and 0 = pore.
    """
    grid = np.ones((size, size), dtype=np.uint8)
    pores = np.zeros((size, size), dtype=bool)
    y, x = np.ogrid[:size, :size]

    half = spacing / 2.0
    centers_1d: list[float] = []
    c = -half
    while c < size + spacing:
        centers_1d.append(c)
        c += spacing
    n = len(centers_1d)
    grid_centers = [[(centers_1d[i], centers_1d[j]) for j in range(n)] for i in range(n)]

    for i in range(1, n - 1):
        for j in range(1, n - 1):
            xc, yc = grid_centers[i][j]
            pores[(x - xc) ** 2 + (y - yc) ** 2 <= radius**2] = True

    if channel_prob > 0.0 and width > 0.0:
        struct = _disk_structure(width / 2.0)
        for i in range(n):
            for j in range(n):
                xc, yc = grid_centers[i][j]
                for ni, nj in ((i, j + 1), (i + 1, j)):
                    if ni >= n or nj >= n:
                        continue
                    xc2, yc2 = grid_centers[ni][nj]
                    line = np.zeros_like(pores)
                    for px, py in _bresenham_line(
                        int(round(xc)), int(round(yc)), int(round(xc2)), int(round(yc2))
                    ):
                        if 0 <= px < size and 0 <= py < size:
                            line[py, px] = True
                    pores |= binary_dilation(line, structure=struct)

    grid[pores] = 0
    return grid


def compute_signed_distance_transform(grid: np.ndarray) -> np.ndarray:
    """Return the signed distance: positive in solid, negative inside pores."""
    solids = grid != 0
    pores = grid == 0
    dist_to_pore = distance_transform_edt(solids)
    dist_to_solid = distance_transform_edt(pores)
    signed = np.zeros_like(dist_to_pore, dtype=float)
    signed[solids] = dist_to_pore[solids]
    signed[pores] = -dist_to_solid[pores]
    return signed


def compute_persistence_2d(values: np.ndarray) -> list[tuple[int, tuple[float, float]]]:
    """Compute cubical persistence using the signed distance as filtration."""
    complex_ = PCC(
        top_dimensional_cells=values.astype(np.float64),
        periodic_dimensions=[False, False],
    )
    max_value = float(np.max(values))
    persistence: list[tuple[int, tuple[float, float]]] = []
    for dim, (birth, death) in complex_.persistence():
        finite_death = max_value if death == float("inf") else float(death)
        persistence.append((int(dim), (float(birth), finite_death)))
    return persistence


def betti_at_levels(
    persistence: list[tuple[int, tuple[float, float]]],
    dim: int,
    levels: np.ndarray,
) -> np.ndarray:
    """Evaluate the Betti curve of a given dimension at the requested levels."""
    intervals = [pair for d, pair in persistence if d == dim]
    return np.array(
        [sum(1 for birth, death in intervals if birth <= level < death) for level in levels],
        dtype=float,
    )


def _double_arrow(ax, p0, p1, *, lw: float = 1.6) -> None:
    """Draw a black double-headed arrow between two data-coordinate points."""
    ax.annotate(
        "",
        xy=p1,
        xytext=p0,
        arrowprops={"arrowstyle": "<->", "lw": lw, "color": "black"},
        annotation_clip=False,
    )


def _draw_structure(ax, grid: np.ndarray, size: int, *, font: int) -> None:
    """Render a binary structure panel (pores black, solid white) with axes."""
    ax.imshow(
        grid,
        origin="lower",
        cmap="gray",
        vmin=0,
        vmax=1,
        extent=[0, size, 0, size],
        interpolation="nearest",
    )
    ax.set_xlabel("x", fontsize=font)
    ax.set_ylabel("y", fontsize=font)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=font - 1)


def _annotate_closed_structure(ax, font: int) -> None:
    """Add the a, h/2, d/2 and r dimension arrows to the closed-cell panel."""
    # Centre-to-centre spacing between the two top pores.
    _double_arrow(ax, (20, 72), (60, 72))
    ax.text(40, 74, "a = 40", ha="center", va="bottom", fontsize=font)

    # Pore edge to lattice centre (half diagonal void scale).
    unit = (math.sqrt(0.5), -math.sqrt(0.5))
    start = (20 + 5 * unit[0], 60 + 5 * unit[1])
    _double_arrow(ax, start, (40, 40))
    ax.text(34, 51, "h/2 = 23.3", ha="center", va="bottom", fontsize=font - 1, rotation=-45)

    # Pore edge to mid-gap between the two bottom pores (half edge gap).
    _double_arrow(ax, (25, 20), (40, 20))
    ax.text(33, 15, "d/2 = 15", ha="center", va="top", fontsize=font - 1)

    # Pore radius on the bottom-right pore.
    _double_arrow(ax, (60, 13), (65, 13))
    ax.text(62.5, 9, "r = 5", ha="center", va="top", fontsize=font - 1)


def _annotate_open_structure(ax, font: int) -> None:
    """Add the w, r, h/2 and a dimension arrows to the open-cell panel."""
    xfrac = blended_transform_factory(ax.transData, ax.transAxes)

    ax.annotate(
        "",
        xy=(24, 1.04),
        xytext=(16, 1.04),
        xycoords=xfrac,
        arrowprops={"arrowstyle": "<->", "lw": 1.6, "color": "black"},
        annotation_clip=False,
    )
    ax.text(20, 1.07, "w = 8", ha="center", va="bottom", fontsize=font, transform=xfrac)

    ax.annotate(
        "",
        xy=(75, 1.04),
        xytext=(60, 1.04),
        xycoords=xfrac,
        arrowprops={"arrowstyle": "<->", "lw": 1.6, "color": "black"},
        annotation_clip=False,
    )
    ax.text(67.5, 1.07, "r = 15", ha="center", va="bottom", fontsize=font, transform=xfrac)

    # Pore edge to lattice centre (half diagonal void scale).
    unit = (math.sqrt(0.5), -math.sqrt(0.5))
    start = (20 + 15 * unit[0], 60 + 15 * unit[1])
    _double_arrow(ax, start, (40, 40))
    ax.text(44, 44, "h/2 = 13.3", ha="center", va="center", fontsize=font - 2, rotation=-45)

    ax.annotate(
        "",
        xy=(60, -0.19),
        xytext=(20, -0.19),
        xycoords=xfrac,
        arrowprops={"arrowstyle": "<->", "lw": 1.6, "color": "black"},
        annotation_clip=False,
    )
    ax.text(40, -0.25, "a = 40", ha="center", va="top", fontsize=font, transform=xfrac)


def _filtration_arrow(ax, level: float, label: str, font: int) -> None:
    """Draw an upward arrow below the axis marking a filtration transition."""
    ax.annotate(
        label,
        xy=(level, -0.30),
        xytext=(level, -0.62),
        xycoords=blended_transform_factory(ax.transData, ax.transAxes),
        textcoords=blended_transform_factory(ax.transData, ax.transAxes),
        ha="center",
        va="top",
        fontsize=font,
        arrowprops={"arrowstyle": "->", "lw": 1.6, "color": "black"},
        annotation_clip=False,
    )


def _draw_betti(
    ax0,
    ax1,
    levels: np.ndarray,
    b0: np.ndarray,
    b1: np.ndarray,
    *,
    xlim: tuple[float, float],
    b0_yticks: list[float],
    annotations: list[tuple[float, str]],
    font: int,
) -> None:
    """Render the stacked B0/B1 scatter panels with filtration annotations."""
    ax0.scatter(levels, b0, color=B0_COLOR, s=14)
    ax0.set_ylabel("Betti 0", fontsize=font)
    ax0.set_title("B0 (components)", fontsize=font + 1)
    ax0.set_yticks(b0_yticks)
    ax0.set_ylim(-0.4, max(b0_yticks) + 0.4)
    ax0.grid(True, axis="y", linestyle="--", alpha=0.4)

    ax1.scatter(levels, b1, color=B1_COLOR, s=14)
    ax1.set_ylabel("Betti 1", fontsize=font)
    ax1.set_title("B1 (loops)", fontsize=font + 1)
    ax1.set_yticks([0.0, 0.5, 1.0])
    ax1.set_ylim(-0.1, 1.15)
    ax1.set_xlabel("Filtration level", fontsize=font + 1)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.4)

    for ax in (ax0, ax1):
        ax.set_xlim(*xlim)
        ax.tick_params(labelsize=font - 1)

    for level, label in annotations:
        _filtration_arrow(ax1, level, label, font)


def build_figure(output_path: Path) -> None:
    """Generate both cases and assemble the composite pedagogical figure."""
    size = 80

    closed = generate_porous_2d(size=size, radius=5.0, spacing=40.0, channel_prob=0.0)
    open_ = generate_porous_2d(
        size=size, radius=15.0, spacing=40.0, width=8.0, channel_prob=1.0
    )

    closed_signed = compute_signed_distance_transform(closed)
    open_signed = compute_signed_distance_transform(open_)

    closed_levels = np.arange(-8, 27, dtype=float)
    open_levels = np.arange(-22, 23, dtype=float)

    closed_pers = compute_persistence_2d(closed_signed)
    open_pers = compute_persistence_2d(open_signed)

    closed_b0 = betti_at_levels(closed_pers, 0, closed_levels)
    closed_b1 = betti_at_levels(closed_pers, 1, closed_levels)
    open_b0 = betti_at_levels(open_pers, 0, open_levels)
    open_b1 = betti_at_levels(open_pers, 1, open_levels)

    fig = plt.figure(figsize=(13.0, 11.0))
    outer = fig.add_gridspec(
        2, 2, width_ratios=[1.0, 1.3], height_ratios=[1.0, 1.0], hspace=0.55, wspace=0.18
    )

    ax_closed = fig.add_subplot(outer[0, 0])
    _draw_structure(ax_closed, closed, size, font=13)
    _annotate_closed_structure(ax_closed, font=13)

    top_betti = outer[0, 1].subgridspec(2, 1, hspace=0.7)
    ax_c0 = fig.add_subplot(top_betti[0])
    ax_c1 = fig.add_subplot(top_betti[1], sharex=ax_c0)
    _draw_betti(
        ax_c0,
        ax_c1,
        closed_levels,
        closed_b0,
        closed_b1,
        xlim=(-8.5, 26.5),
        b0_yticks=[0, 2, 4],
        annotations=[(15.0, "d/2 = 15"), (23.3, "h/2 = 23.3")],
        font=12,
    )

    ax_open = fig.add_subplot(outer[1, 0])
    _draw_structure(ax_open, open_, size, font=13)
    _annotate_open_structure(ax_open, font=13)

    bot_betti = outer[1, 1].subgridspec(2, 1, hspace=0.7)
    ax_o0 = fig.add_subplot(bot_betti[0])
    ax_o1 = fig.add_subplot(bot_betti[1], sharex=ax_o0)
    _draw_betti(
        ax_o0,
        ax_o1,
        open_levels,
        open_b0,
        open_b1,
        xlim=(-21.5, 21.5),
        b0_yticks=[0, 1, 2, 3, 4],
        annotations=[(-15.0, "-r = -15"), (-4.0, "-w/2 = -4"), (13.3, "h/2 = 13.3")],
        font=12,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    official_repository = Path(__file__).resolve().parents[2]
    default_output = official_repository / "plots" / "fig4_shape_vs_Betti_pedagogical.png"

    parser = argparse.ArgumentParser(
        description="Regenerate the closed-vs-open pedagogical Betti figure."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help=f"Output PNG path (default: {default_output}).",
    )
    return parser.parse_args()


def main() -> None:
    """Build the composite figure and write it to disk."""
    args = parse_args()
    build_figure(args.output.resolve())
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
