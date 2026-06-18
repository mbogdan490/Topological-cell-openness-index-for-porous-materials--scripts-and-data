#!/usr/bin/env python3
"""
Draw a schematic of the 2D porous-structure generation scheme.

The plot mirrors the generator used by the 2D sponginess experiment:
centers are placed on an extended grid starting at -a/2, pores are carved
only from the interior rows and columns, and channels may connect neighboring
centers including the extra outside rows and columns.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Patch, Rectangle
from matplotlib.ticker import MaxNLocator


# All text sizes are doubled relative to the original schematic so the figure
# remains legible at paper scale. The base constant drives every explicit
# fontsize/labelsize in this script; rcParams below catch any unspecified text.
FONT_SIZE = 22

# Double matplotlib's default font sizes as a fallback for any text that is not
# given an explicit size (defaults are 10-12pt, so 2x lands at 20-24pt).
plt.rcParams.update(
    {
        "font.size": 20,
        "axes.titlesize": 24,
        "axes.labelsize": 24,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 20,
        "figure.titlesize": 24,
    }
)


def _sample_positive(
    rng: np.random.Generator,
    mean: float,
    std: float,
    max_attempts: int = 100,
) -> float:
    """Sample a positive normal perturbation, matching the structure generator."""
    if std <= 0:
        return mean
    for _ in range(max_attempts):
        value = float(rng.normal(mean, std))
        if value > 0:
            return value
    return mean


def generate_base_coordinates(size: float, spacing: float) -> np.ndarray:
    """Return the 1D extended lattice coordinates used by the generator."""
    centers_1d = []
    coord = -spacing / 2.0
    while coord < size + spacing:
        centers_1d.append(coord)
        coord += spacing
    return np.array(centers_1d, dtype=float)


def simulate_generation_grid(
    size: float,
    spacing: float,
    radius: float,
    radius_std: float,
    position_std: float,
    channel_prob: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[tuple[tuple[int, int], tuple[int, int]]]]:
    """Simulate the point, radius, and connection choices used before carving pores."""
    rng = np.random.default_rng(seed)
    base_coords = generate_base_coordinates(size=size, spacing=spacing)

    n_rows = len(base_coords)
    n_cols = len(base_coords)
    base_centers = np.zeros((n_rows, n_cols, 2), dtype=float)
    actual_centers = np.zeros_like(base_centers)
    radii = np.zeros((n_rows, n_cols), dtype=float)

    for i, base_cx in enumerate(base_coords):
        for j, base_cy in enumerate(base_coords):
            base_centers[i, j] = (base_cx, base_cy)
            actual_centers[i, j] = (
                base_cx + (float(rng.normal(0.0, position_std)) if position_std > 0 else 0.0),
                base_cy + (float(rng.normal(0.0, position_std)) if position_std > 0 else 0.0),
            )
            radii[i, j] = _sample_positive(rng, radius, radius_std)

    inside = (
        (0.0 < base_centers[:, :, 0])
        & (base_centers[:, :, 0] < size)
        & (0.0 < base_centers[:, :, 1])
        & (base_centers[:, :, 1] < size)
    )

    connections = []
    for i in range(n_rows):
        for j in range(n_cols):
            if j + 1 < n_cols and rng.random() < channel_prob:
                connections.append(((i, j), (i, j + 1)))
            if i + 1 < n_rows and rng.random() < channel_prob:
                connections.append(((i, j), (i + 1, j)))

    return base_centers, actual_centers, radii, inside, connections


def draw_double_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    label: str,
    label_offset: tuple[float, float] = (0.0, 0.0),
) -> None:
    """Draw a double-headed distance arrow with a centered label."""
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="<->",
        mutation_scale=12,
        linewidth=1.2,
        color="black",
    )
    ax.add_patch(arrow)
    label_x = (start[0] + end[0]) / 2.0 + label_offset[0]
    label_y = (start[1] + end[1]) / 2.0 + label_offset[1]
    ax.text(
        label_x,
        label_y,
        label,
        ha="center",
        va="center",
        fontsize=FONT_SIZE,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
    )


def _distance_to_segment(
    xx: np.ndarray,
    yy: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> np.ndarray:
    """Return pixelwise Euclidean distance to a finite line segment."""
    segment = end - start
    segment_len_sq = float(np.dot(segment, segment))
    if segment_len_sq == 0.0:
        return np.hypot(xx - start[0], yy - start[1])

    projection = ((xx - start[0]) * segment[0] + (yy - start[1]) * segment[1]) / segment_len_sq
    projection = np.clip(projection, 0.0, 1.0)
    closest_x = start[0] + projection * segment[0]
    closest_y = start[1] + projection * segment[1]
    return np.hypot(xx - closest_x, yy - closest_y)


def build_binary_structure(
    size: float,
    actual_centers: np.ndarray,
    radii: np.ndarray,
    inside: np.ndarray,
    connections: list[tuple[tuple[int, int], tuple[int, int]]],
    channel_width: float,
) -> np.ndarray:
    """Rasterize the generated pores and throats as a binary image."""
    n_pixels = int(round(size))
    yy, xx = np.indices((n_pixels, n_pixels))
    pores = np.zeros((n_pixels, n_pixels), dtype=bool)

    for i, j in np.argwhere(inside):
        x_center, y_center = actual_centers[i, j]
        pores |= (xx - x_center) ** 2 + (yy - y_center) ** 2 <= radii[i, j] ** 2

    for (i0, j0), (i1, j1) in connections:
        start = actual_centers[i0, j0]
        end = actual_centers[i1, j1]
        pores |= _distance_to_segment(xx, yy, start, end) <= channel_width / 2.0

    structure = np.ones((n_pixels, n_pixels), dtype=np.uint8)
    structure[pores] = 0
    return structure


def plot_generation_scheme(
    output_path: Path,
    size: float = 100.0,
    spacing: float = 25.0,
    radius: float = 8.0,
    radius_std: float = 1.8,
    position_std: float = 3.0,
    channel_width: float = 4.0,
    channel_prob: float = 0.55,
    seed: int = 24,
) -> None:
    """Save a schematic of one simulated extended grid and its parameters."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    base_centers, actual_centers, radii, inside, connections = simulate_generation_grid(
        size=size,
        spacing=spacing,
        radius=radius,
        radius_std=radius_std,
        position_std=position_std,
        channel_prob=channel_prob,
        seed=seed,
    )
    base_flat = base_centers.reshape(-1, 2)
    actual_flat = actual_centers.reshape(-1, 2)
    inside_flat = inside.ravel()

    pad = spacing * 0.95
    extended_span = size + 2.0 * pad
    intermediate_span = size + pad
    fig = plt.figure(figsize=(32.0, 20.4), constrained_layout=True)
    layout = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.95])
    # A single top row keeps all three panels in cells with a common top edge;
    # the 2nd/3rd columns are widened (1.25x) so the secondary panels are a bit
    # larger than before. Anchoring every panel to the north edge makes the
    # aspect-equal squares hug the top of their cells, so their titles line up.
    secondary_span = intermediate_span * 1.25
    top_row = layout[0].subgridspec(
        1,
        3,
        width_ratios=[extended_span, secondary_span, secondary_span],
    )
    bottom_row = layout[1].subgridspec(1, 3)
    ax = fig.add_subplot(top_row[0, 0])
    ax_middle = fig.add_subplot(top_row[0, 1])
    ax_final = fig.add_subplot(top_row[0, 2])
    for top_ax in (ax, ax_middle, ax_final):
        top_ax.set_anchor("N")
    example_axes = [fig.add_subplot(bottom_row[0, idx]) for idx in range(3)]

    ax.set_xlim(-pad, size + pad)
    ax.set_ylim(-pad, size + pad)
    ax.set_aspect("equal")

    # Light shading identifies the rows/columns of helper centers outside the domain.
    ax.add_patch(Rectangle((-pad, -pad), pad, size + 2 * pad, color="0.92", zorder=0))
    ax.add_patch(Rectangle((size, -pad), pad, size + 2 * pad, color="0.92", zorder=0))
    ax.add_patch(Rectangle((-pad, -pad), size + 2 * pad, pad, color="0.92", zorder=0))
    ax.add_patch(Rectangle((-pad, size), size + 2 * pad, pad, color="0.92", zorder=0))

    for x_base, y_base in base_flat:
        ax.add_patch(
            Circle(
                (x_base, y_base),
                1.8,
                facecolor="white",
                edgecolor="0.45",
                linestyle="--",
                linewidth=1.0,
                zorder=2,
            )
        )
    ax.scatter(
        [],
        [],
        s=35,
        facecolor="white",
        edgecolor="0.45",
        linestyle="--",
        linewidth=1.0,
        label="unperturbed grid centers",
    )

    for (i0, j0), (i1, j1) in connections:
        x0, y0 = actual_centers[i0, j0]
        x1, y1 = actual_centers[i1, j1]
        ax.plot(
            [x0, x1],
            [y0, y1],
            color="tab:orange",
            linewidth=channel_width,
            solid_capstyle="round",
            alpha=0.75,
            zorder=1,
        )

    ax.scatter(
        actual_flat[~inside_flat, 0],
        actual_flat[~inside_flat, 1],
        s=35,
        facecolor="white",
        edgecolor="0.35",
        linewidth=1.0,
        zorder=3,
        label="off-grid pore centers",
    )
    ax.scatter(
        actual_flat[inside_flat, 0],
        actual_flat[inside_flat, 1],
        s=38,
        facecolor="black",
        edgecolor="black",
        zorder=4,
        label="pore centers",
    )

    inside_indices = np.argwhere(inside)
    for i, j in inside_indices:
        x_center, y_center = actual_centers[i, j]
        ax.add_patch(
            Circle(
                (x_center, y_center),
                radii[i, j],
                facecolor="none",
                edgecolor="tab:blue",
                linewidth=1.1,
                alpha=0.75,
                zorder=2,
            )
        )

    boundary = Rectangle(
        (0.0, 0.0),
        size,
        size,
        fill=False,
        linestyle="--",
        linewidth=1.5,
        edgecolor="black",
        zorder=5,
    )
    ax.add_patch(boundary)

    draw_double_arrow(
        ax,
        (spacing / 2.0, -0.42 * spacing),
        (1.5 * spacing, -0.42 * spacing),
        r"$a$",
        label_offset=(0.0, 0.08 * spacing),
    )
    radius_i, radius_j = inside_indices[0]
    radius_center = actual_centers[radius_i, radius_j]
    draw_double_arrow(
        ax,
        (float(radius_center[0]), float(radius_center[1])),
        (float(radius_center[0] + radii[radius_i, radius_j]), float(radius_center[1])),
        r"$r$",
        label_offset=(0.0, 0.18 * spacing),
    )

    inside_connection = next(
        (
            connection
            for connection in connections
            if inside[connection[0]] and inside[connection[1]]
        ),
        None,
    )
    if inside_connection is not None:
        (i0, j0), (i1, j1) = inside_connection
        x0, y0 = actual_centers[i0, j0]
        x1, y1 = actual_centers[i1, j1]
        mid = np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0])
        direction = np.array([x1 - x0, y1 - y0], dtype=float)
        normal = np.array([-direction[1], direction[0]], dtype=float)
        normal /= np.linalg.norm(normal)
        draw_double_arrow(
            ax,
            tuple(mid - normal * channel_width / 2.0),
            tuple(mid + normal * channel_width / 2.0),
            r"$w$",
            label_offset=tuple(normal * 0.17 * spacing),
        )

    ax.text(
        size / 2.0,
        size + 0.22 * spacing,
        "physical domain",
        ha="center",
        va="bottom",
        fontsize=FONT_SIZE,
    )
    ax.text(
        size + 0.42 * spacing,
        size / 2.0,
        "extra\ncolumns",
        ha="center",
        va="center",
        fontsize=FONT_SIZE,
    )
    ax.text(
        size / 2.0,
        -0.75 * spacing,
        "extra rows",
        ha="center",
        va="center",
        fontsize=FONT_SIZE,
    )
    ax.text(
        size / 2.0,
        -0.36 * spacing,
        "connections between pore centers",
        ha="center",
        va="center",
        fontsize=FONT_SIZE,
        color="tab:orange",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
    )

    ax.set_title("1. Generate extended grid", fontsize=FONT_SIZE)
    ax.set_xlabel("x [pixels]", fontsize=FONT_SIZE)
    ax.set_ylabel("y [pixels]", fontsize=FONT_SIZE)
    ax.tick_params(axis="both", labelsize=FONT_SIZE)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    handles, labels = ax.get_legend_handles_labels()
    label_order = [
        "unperturbed grid centers",
        "pore centers",
        "off-grid pore centers",
    ]
    ordered_handles = [handles[labels.index(label)] for label in label_order]
    ax.legend(
        ordered_handles,
        label_order,
        loc="upper left",
        bbox_to_anchor=(0.0, 0.82),
        fontsize=FONT_SIZE - 1,
        framealpha=0.95,
    )

    for (i0, j0), (i1, j1) in connections:
        x0, y0 = actual_centers[i0, j0]
        x1, y1 = actual_centers[i1, j1]
        ax_middle.plot(
            [x0, x1],
            [y0, y1],
            color="tab:orange",
            linewidth=channel_width,
            solid_capstyle="round",
            alpha=0.75,
            zorder=1,
        )

    for i, j in inside_indices:
        x_center, y_center = actual_centers[i, j]
        ax_middle.add_patch(
            Circle(
                (x_center, y_center),
                radii[i, j],
                facecolor="none",
                edgecolor="tab:blue",
                linewidth=1.2,
                alpha=0.85,
                zorder=2,
            )
        )

    ax_middle.scatter(
        actual_flat[inside_flat, 0],
        actual_flat[inside_flat, 1],
        s=28,
        facecolor="black",
        edgecolor="black",
        zorder=3,
    )
    ax_middle.add_patch(
        Rectangle(
            (0.0, 0.0),
            size,
            size,
            fill=False,
            linestyle="--",
            linewidth=1.2,
            edgecolor="black",
            zorder=4,
        )
    )
    ax_middle.set_xlim(0.0, size)
    ax_middle.set_ylim(0.0, size)
    ax_middle.set_aspect("equal")
    ax_middle.set_title("2. Crop to physical domain", fontsize=FONT_SIZE)
    ax_middle.set_xlabel("x [pixels]", fontsize=FONT_SIZE)
    ax_middle.set_ylabel("y [pixels]", fontsize=FONT_SIZE)
    ax_middle.tick_params(axis="both", labelsize=FONT_SIZE)
    ax_middle.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_middle.yaxis.set_major_locator(MaxNLocator(integer=True))

    structure = build_binary_structure(
        size=size,
        actual_centers=actual_centers,
        radii=radii,
        inside=inside,
        connections=connections,
        channel_width=channel_width,
    )
    ax_final.imshow(
        structure,
        origin="lower",
        cmap="gray",
        vmin=0,
        vmax=1,
        extent=(0.0, size, 0.0, size),
        interpolation="nearest",
    )
    ax_final.set_aspect("equal")
    ax_final.set_title("3. Final binary structure", fontsize=FONT_SIZE)
    ax_final.set_xlabel("x [pixels]", fontsize=FONT_SIZE)
    ax_final.set_ylabel("y [pixels]", fontsize=FONT_SIZE)
    ax_final.tick_params(axis="both", labelsize=FONT_SIZE)
    ax_final.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_final.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax_final.legend(
        handles=[
            Patch(facecolor="black", edgecolor="black", label="pore"),
            Patch(facecolor="white", edgecolor="black", label="structure"),
        ],
        loc="upper right",
        fontsize=FONT_SIZE - 1,
        framealpha=0.95,
    )

    examples = [
        {"p": 0.1, "a": 31.0, "r": 7.0, "w": 3.0, "seed": seed + 101},
        {"p": 0.5, "a": 20.0, "r": 5.0, "w": 2.0, "seed": seed + 202},
        {"p": 0.9, "a": 40.0, "r": 15.0, "w": 10.0, "seed": seed + 303},
    ]
    for example_ax, params in zip(example_axes, examples, strict=True):
        _, example_centers, example_radii, example_inside, example_connections = simulate_generation_grid(
            size=size,
            spacing=params["a"],
            radius=params["r"],
            radius_std=0.15 * params["r"],
            position_std=0.12 * params["a"],
            channel_prob=params["p"],
            seed=params["seed"],
        )
        example_structure = build_binary_structure(
            size=size,
            actual_centers=example_centers,
            radii=example_radii,
            inside=example_inside,
            connections=example_connections,
            channel_width=params["w"],
        )
        example_ax.imshow(
            example_structure,
            origin="lower",
            cmap="gray",
            vmin=0,
            vmax=1,
            extent=(0.0, size, 0.0, size),
            interpolation="nearest",
        )
        example_ax.set_aspect("equal")
        example_ax.set_title("Example final structure", fontsize=FONT_SIZE)
        example_ax.set_xlabel("x [pixels]", fontsize=FONT_SIZE)
        example_ax.set_ylabel("y [pixels]", fontsize=FONT_SIZE)
        example_ax.tick_params(axis="both", labelsize=FONT_SIZE)
        example_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        example_ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        example_ax.text(
            0.97,
            0.97,
            (
                f"p = {params['p']:.1f}\n"
                f"a = {params['a']:.0f}\n"
                f"r = {params['r']:.0f}\n"
                f"w = {params['w']:.0f}"
            ),
            transform=example_ax.transAxes,
            ha="right",
            va="top",
            fontsize=FONT_SIZE - 1,
            bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.9, "pad": 2.0},
        )

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the generation-scheme plot."""
    official_repository = Path(__file__).resolve().parents[2]
    default_output = (
        official_repository
        / "plots"
        / "fig1_dataset_generation_scheme.png"
    )

    parser = argparse.ArgumentParser(
        description="Plot the 2D porous-structure generation scheme."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help=f"Final plot path (default: {default_output}).",
    )
    parser.add_argument("--size", type=float, default=100.0)
    parser.add_argument("--spacing", type=float, default=25.0)
    parser.add_argument("--radius", type=float, default=8.0)
    parser.add_argument("--radius-std", type=float, default=1.8)
    parser.add_argument("--position-std", type=float, default=3.0)
    parser.add_argument("--channel-width", type=float, default=4.0)
    parser.add_argument("--channel-prob", type=float, default=0.55)
    parser.add_argument("--seed", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    """Create and save the generation-scheme plot."""
    args = parse_args()
    plot_generation_scheme(
        output_path=args.output.resolve(),
        size=args.size,
        spacing=args.spacing,
        radius=args.radius,
        radius_std=args.radius_std,
        position_std=args.position_std,
        channel_width=args.channel_width,
        channel_prob=args.channel_prob,
        seed=args.seed,
    )
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
