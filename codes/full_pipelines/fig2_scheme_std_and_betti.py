#!/usr/bin/env python3
"""
Create a pedagogical 2D plot of four pores, their signed distance transform,
and Betti curves computed from that transform.

The generated structure uses the project convention:
    1 = solid, 0 = pore.
Signed distances are negative in pores and positive in the solid.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from gudhi import PeriodicCubicalComplex as PCC
from matplotlib.ticker import MaxNLocator
from scipy.ndimage import distance_transform_edt


SNAPSHOT_LEVELS = np.array([-10.0, -5.0, 2.0, 7.0, 10.0])
FONT_SIZE = 11


def make_pore_grid_structure(
    height: int = 50,
    width: int = 50,
    radius: float = 8.0,
    center_distance: float = 25.0,
) -> np.ndarray:
    """Return a binary 2D structure with a 2-by-2 grid of circular pores."""
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    if radius <= 0:
        raise ValueError("radius must be positive")
    if center_distance <= 0:
        raise ValueError("center_distance must be positive")
    if center_distance + 2.0 * radius > width:
        raise ValueError("pores do not fit horizontally in the requested width")
    if center_distance + 2.0 * radius > height:
        raise ValueError("pores do not fit vertically in the requested height")

    x_left = (width - 1 - center_distance) / 2.0
    x_right = x_left + center_distance
    y_bottom = (height - 1 - center_distance) / 2.0
    y_top = y_bottom + center_distance

    yy, xx = np.indices((height, width))
    pore_mask = np.zeros((height, width), dtype=bool)
    for x_center in (x_left, x_right):
        for y_center in (y_bottom, y_top):
            pore_mask |= (xx - x_center) ** 2 + (yy - y_center) ** 2 <= radius**2

    structure = np.ones((height, width), dtype=np.uint8)
    structure[pore_mask] = 0
    return structure


def compute_signed_distance_transform(structure: np.ndarray) -> np.ndarray:
    """Compute signed distance transform with positive solid and negative pores."""
    solids = structure != 0
    pores = structure == 0

    dist_to_pore = distance_transform_edt(solids)
    dist_to_solid = distance_transform_edt(pores)

    signed = np.zeros_like(dist_to_pore, dtype=float)
    signed[solids] = dist_to_pore[solids]
    signed[pores] = -dist_to_solid[pores]
    return signed


def compute_persistence_2d(values: np.ndarray) -> list[tuple[int, tuple[float, float]]]:
    """Compute 2D cubical persistence using the signed distance as filtration."""
    if values.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {values.shape}")

    complex_ = PCC(
        top_dimensional_cells=values.astype(np.float64),
        periodic_dimensions=[False, False],
    )
    max_value = float(np.max(values))

    persistence = []
    for dim, (birth, death) in complex_.persistence():
        finite_death = max_value if death == float("inf") else float(death)
        persistence.append((int(dim), (float(birth), finite_death)))
    return persistence


def betti_at_levels(
    persistence: list[tuple[int, tuple[float, float]]],
    dim: int,
    levels: np.ndarray,
) -> np.ndarray:
    """Evaluate a Betti curve at the requested filtration levels."""
    intervals = [pair for interval_dim, pair in persistence if interval_dim == dim]
    return np.array(
        [
            sum(1 for birth, death in intervals if birth <= level < death)
            for level in levels
        ],
        dtype=float,
    )


def make_plot(
    structure: np.ndarray,
    signed_distance: np.ndarray,
    levels: np.ndarray,
    betti0: np.ndarray,
    betti1: np.ndarray,
    snapshot_levels: np.ndarray,
    output_path: Path,
) -> None:
    """Save the pedagogical figure with structure, filtration, and Betti curves."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(
        figsize=(10.0, 7.4),
        constrained_layout=True,
    )
    layout = fig.add_gridspec(2, 1, height_ratios=[1.15, 0.95])
    top_row = layout[0].subgridspec(1, 2, width_ratios=[1.0, 1.25])

    ax_structure = fig.add_subplot(top_row[0, 0])
    ax_distance = fig.add_subplot(top_row[0, 1])
    ax_betti = fig.add_subplot(layout[1, 0])

    ax_structure.imshow(
        structure,
        origin="lower",
        cmap="gray",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    ax_structure.set_title("porous structure (pores in black)", fontsize=FONT_SIZE)
    ax_structure.set_ylabel("y [pixels]")
    ax_structure.set_aspect("equal")

    max_abs = float(max(abs(np.min(signed_distance)), abs(np.max(signed_distance))))
    image = ax_distance.imshow(
        signed_distance,
        origin="lower",
        cmap="RdBu_r",
        vmin=-max_abs,
        vmax=max_abs,
        interpolation="nearest",
    )
    ax_distance.set_title("Signed distance transform", fontsize=FONT_SIZE)
    ax_distance.set_ylabel("y [pixels]")
    ax_distance.set_aspect("equal")
    colorbar = fig.colorbar(image, ax=ax_distance, label="Signed distance [pixels]")
    colorbar.ax.set_ylabel("Signed distance [pixels]", fontsize=FONT_SIZE)
    colorbar.ax.tick_params(labelsize=FONT_SIZE)
    colorbar.set_ticks(np.arange(np.ceil(-max_abs), np.floor(max_abs) + 1.0, 2.0))

    ax_betti.plot(levels, betti0, label=r"$\beta_0$ components", color="tab:blue", linewidth=2.0)
    ax_betti.plot(levels, betti1, label=r"$\beta_1$ loops", color="tab:green", linewidth=2.0)
    ax_betti.axvline(-1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    ax_betti.text(
        -5.6,
        0.35,
        r"$\tau=\beta_1(t=-1)/(\beta_0(t=-1)+\beta_1(t=-1))$",
        fontsize=FONT_SIZE,
        ha="left",
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.5},
    )
    ax_betti.set_title("Betti curves", fontsize=FONT_SIZE)
    ax_betti.set_xlabel("t (filtration level)")
    ax_betti.set_ylabel("Betti number")
    ax_betti.grid(True, linestyle="--", alpha=0.35)
    ax_betti.legend(loc="best", fontsize=FONT_SIZE)

    ax_betti.set_xlim(float(np.min(snapshot_levels) - 1.5), float(np.max(snapshot_levels) + 1.5))
    ax_betti.set_ylim(-0.15, max(float(np.max(betti0)), float(np.max(betti1)), 4.0) + 0.15)

    inset_width = 2.8
    inset_height = 1.28
    inset_center_y = 2.0
    for level in snapshot_levels:
        inset = ax_betti.inset_axes(
            [
                level - inset_width / 2.0,
                inset_center_y - inset_height / 2.0,
                inset_width,
                inset_height,
            ],
            transform=ax_betti.transData,
        )
        caught = signed_distance <= level
        inset.imshow(
            np.where(caught, 0, 1),
            origin="lower",
            cmap="gray",
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        inset.set_title(f"t = {level:g}", fontsize=FONT_SIZE, pad=1)
        inset.set_xticks([])
        inset.set_yticks([])
        inset.set_aspect("equal")
        for spine in inset.spines.values():
            spine.set_linewidth(0.7)

    for ax in (ax_structure, ax_distance):
        ax.set_xlabel("x [pixels]")

    for ax in (ax_structure, ax_distance, ax_betti):
        ax.xaxis.label.set_size(FONT_SIZE)
        ax.yaxis.label.set_size(FONT_SIZE)
        ax.tick_params(axis="both", labelsize=FONT_SIZE)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax_betti.set_xticks([-9, -6, -3, -1, 0, 3, 6, 9])

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the pedagogical plot script."""
    official_repository = Path(__file__).resolve().parents[2]
    default_output = (
        official_repository
        / "plots"
        / "fig2_scheme_std_and_betti.png"
    )

    parser = argparse.ArgumentParser(
        description="Generate a three-row pedagogical signed-DT and Betti plot."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help=f"Final plot path (default: {default_output}).",
    )
    parser.add_argument("--height", type=int, default=50, help="Structure height in pixels.")
    parser.add_argument("--width", type=int, default=50, help="Structure width in pixels.")
    parser.add_argument("--radius", type=float, default=8.0, help="Pore radius in pixels.")
    parser.add_argument(
        "--center-distance",
        type=float,
        default=25.0,
        help="Horizontal and vertical distance between neighboring pore centers in pixels.",
    )
    parser.add_argument(
        "--level-step",
        type=float,
        default=0.25,
        help="Filtration-level spacing for plotted Betti curves.",
    )
    return parser.parse_args()


def main() -> None:
    """Generate the structure, compute topology, and save the final plot."""
    args = parse_args()
    if args.level_step <= 0:
        raise ValueError("--level-step must be positive")

    structure = make_pore_grid_structure(
        height=args.height,
        width=args.width,
        radius=args.radius,
        center_distance=args.center_distance,
    )
    signed_distance = compute_signed_distance_transform(structure)

    min_level = float(min(np.floor(np.min(signed_distance)), np.min(SNAPSHOT_LEVELS)))
    max_level = float(max(np.ceil(np.max(signed_distance)), np.max(SNAPSHOT_LEVELS)))
    levels = np.arange(min_level, max_level + args.level_step, args.level_step)

    persistence = compute_persistence_2d(signed_distance)
    betti0 = betti_at_levels(persistence, dim=0, levels=levels)
    betti1 = betti_at_levels(persistence, dim=1, levels=levels)

    make_plot(
        structure=structure,
        signed_distance=signed_distance,
        levels=levels,
        betti0=betti0,
        betti1=betti1,
        snapshot_levels=SNAPSHOT_LEVELS,
        output_path=args.output.resolve(),
    )
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
