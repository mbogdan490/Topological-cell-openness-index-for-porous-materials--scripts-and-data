#!/usr/bin/env python3
"""Plot phi_0 vs tau^(1/k) scatter rows from precomputed 2D and 3D CSV files.

A default run produces a 2-row x 2-column figure:
- upper row: 2D scatter (x = phi_0, y = tau^(1/3)) | a representative 2D structure
- lower row: 3D scatter (x = phi_0, y = tau^(1/4)) | a representative 3D structure

For each dimensionality a single "representative" structure is chosen among the
rows that are simultaneously very open (phi_0 > 0.98) and clearly spongy
(tau^(1/k) < 0.9). Its array is loaded, visualized in the right column, and the
matching scatter point is highlighted with an enlarged star marker.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch

# Script lives at official_repository/codes/regenerating_plots_from_datasets/fig_3/,
# so official_repository == Path(__file__).resolve().parents[3].
_OFFICIAL_REPO = Path(__file__).resolve().parents[3]
_DEFAULT_2D = _OFFICIAL_REPO / "datasets" / "fig_3" / "tau_phi0_metrics_2d.csv"
_DEFAULT_3D = _OFFICIAL_REPO / "datasets" / "fig_3" / "tau_phi0_metrics_3d.csv"
_DEFAULT_OUTPUT = _OFFICIAL_REPO / "plots" / "fig3_compare_tau_with_phi.png"

# Folders holding the binary structure arrays (convention 1=solid, 0=pore).
_STRUCT_DIR_2D = _OFFICIAL_REPO / "datasets" / "fig_3" / "2D_for_correlating_tau_and_phi_0"
_STRUCT_DIR_3D = _OFFICIAL_REPO / "datasets" / "fig_3" / "3D_for_correlating_tau_and_phi_0"

# Fixed roots used for each case (x = phi_0, y = tau^(1/k)).
_K_2D = 3
_K_3D = 4

# Selection criteria for the representative structure of each case.
_PHI0_MIN = 0.98
_TAU_ROOT_MAX = 0.9

# Edge length (voxels) of the centered sub-cube rendered for the 3D panel.
_CROP_SIZE_3D = 100


@dataclass
class StructureChoice:
    """A representative structure selected from a metrics CSV."""

    structure: str
    phi0: float
    tau: float
    k: int
    qualified: bool
    n_qualified: int

    @property
    def tau_root(self) -> float:
        """tau raised to the 1/k power used on the scatter y-axis."""
        return float(np.power(max(self.tau, 0.0), 1.0 / self.k))


def read_metrics(path: Path) -> list[dict[str, float | str]]:
    """Read all rows of a structure,phi_0,tau CSV as dicts (typed numerics)."""
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    parsed: list[dict[str, float | str]] = []
    for row in rows:
        parsed.append(
            {
                "structure": str(row["structure"]),
                "phi_0": float(row["phi_0"]),
                "tau": float(row["tau"]),
            }
        )
    return parsed


def read_phi0_tau(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read phi_0 and tau columns from a structure,phi_0,tau CSV file."""
    rows = read_metrics(path)
    phi0 = np.array([float(row["phi_0"]) for row in rows], dtype=float)
    tau = np.array([float(row["tau"]) for row in rows], dtype=float)
    return phi0, tau


def choose_representative(rows: list[dict[str, float | str]], k: int) -> StructureChoice:
    """Pick one representative structure (highest phi_0 among qualifiers).

    A row qualifies when phi_0 > _PHI0_MIN and tau^(1/k) < _TAU_ROOT_MAX. When
    several qualify the one with the largest phi_0 wins. When none qualify we
    fall back to the row closest (Euclidean, in the phi_0 / tau^(1/k) plane) to
    the corner (phi_0=1, tau^(1/k)=_TAU_ROOT_MAX) so the figure still renders.
    """
    tau_max = _TAU_ROOT_MAX**k
    qualified = [
        row for row in rows if float(row["phi_0"]) > _PHI0_MIN and float(row["tau"]) < tau_max
    ]
    if qualified:
        best = max(qualified, key=lambda row: float(row["phi_0"]))
        return StructureChoice(
            structure=str(best["structure"]),
            phi0=float(best["phi_0"]),
            tau=float(best["tau"]),
            k=k,
            qualified=True,
            n_qualified=len(qualified),
        )

    def distance(row: dict[str, float | str]) -> float:
        phi0 = float(row["phi_0"])
        tau_root = float(np.power(max(float(row["tau"]), 0.0), 1.0 / k))
        return (phi0 - 1.0) ** 2 + (tau_root - _TAU_ROOT_MAX) ** 2

    closest = min(rows, key=distance)
    return StructureChoice(
        structure=str(closest["structure"]),
        phi0=float(closest["phi_0"]),
        tau=float(closest["tau"]),
        k=k,
        qualified=False,
        n_qualified=0,
    )


def load_structure(struct_dir: Path, structure_name: str) -> np.ndarray:
    """Load a binary structure .npy by name, tolerating a missing extension."""
    candidate = struct_dir / structure_name
    if not candidate.exists() and not structure_name.endswith(".npy"):
        candidate = struct_dir / f"{structure_name}.npy"
    if not candidate.exists():
        raise FileNotFoundError(f"Structure {structure_name!r} not found in {struct_dir}")
    return np.load(candidate)


def _central_crop_3d(volume: np.ndarray, size: int) -> np.ndarray:
    """Return the centered size^3 sub-cube of a 3D volume at full resolution.

    The crop is centered on the array center along every axis; if an axis is
    shorter than `size` the whole axis is kept.
    """
    slices = []
    for length in volume.shape:
        edge = min(size, length)
        start = (length - edge) // 2
        slices.append(slice(start, start + edge))
    return volume[tuple(slices)]


def _render_structure_2d(ax: plt.Axes, structure: np.ndarray, choice: StructureChoice) -> None:
    """imshow a 2D binary structure with pores black and solid white.

    Uses the "gray" colormap so value 0 (pore) maps to black and value 1
    (solid) maps to white.
    """
    ax.imshow(structure, cmap="gray", interpolation="nearest", vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        rf"$\phi_0={choice.phi0:.3f}$, $\tau={choice.tau:.3f}$ "
        rf"($\tau^{{1/{choice.k}}}={choice.tau_root:.3f}$)",
        fontsize=11,
    )


def _render_structure_3d(ax: plt.Axes, structure: np.ndarray, choice: StructureChoice) -> tuple[int, ...]:
    """Voxel-render the pore phase of the central crop; return the crop shape.

    These structures are mostly solid with thin pore channels, so the open
    pore phase (the interesting, "spongy" part) is rendered rather than the
    near-solid block. Instead of downsampling the whole volume, the centered
    `_CROP_SIZE_3D`^3 sub-cube is rendered at full resolution for clarity.
    Pore voxels are drawn black; the solid phase is left as empty white space.
    """
    crop = _central_crop_3d(structure, _CROP_SIZE_3D)
    pores = crop == 0
    ax.voxels(
        pores,
        facecolors="black",
        edgecolors="#00000033",
        linewidth=0.12,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=22, azim=-58)
    ax.set_title(
        f"central {'x'.join(str(d) for d in crop.shape)} crop\n"
        rf"$\phi_0={choice.phi0:.3f}$, $\tau={choice.tau:.3f}$ "
        rf"($\tau^{{1/{choice.k}}}={choice.tau_root:.3f}$)",
        fontsize=11,
    )
    return crop.shape


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation coefficient (0.0 if either input is constant)."""
    if x.size == 0 or y.size == 0:
        return 0.0
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _plot_scatter_row(
    ax: plt.Axes,
    phi0: np.ndarray,
    tau: np.ndarray,
    k: int,
    label: str,
    choice: StructureChoice,
) -> float:
    """Draw one phi_0 vs tau^(1/k) scatter panel and highlight the chosen point."""
    y = np.power(np.clip(tau, 0.0, None), 1.0 / k)
    r = pearson_r(phi0, y)
    ax.scatter(phi0, y, c="steelblue", s=28, alpha=0.85, edgecolors="k", linewidths=0.35)
    ax.scatter(
        [choice.phi0],
        [choice.tau_root],
        marker="o",
        s=240,
        c="red",
        edgecolors="k",
        linewidths=1.4,
        zorder=6,
        label="visualized structure",
    )
    ax.set_xlabel(r"x: $\phi_0$", fontsize=11)
    ax.set_ylabel(rf"y: $\tau^{{1/{k}}}$", fontsize=11)
    ax.set_title(rf"{label}: $\phi_0$ vs $\tau^{{1/{k}}}$", fontsize=12)
    ax.text(
        0.02,
        0.98,
        rf"Pearson $r={r:.3f}$",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
    )
    # Pearson annotation sits at (0.02, 0.98) anchored top; place the legend just
    # below it (small gap) so the two stack neatly in the upper-left corner.
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.90), fontsize=9, framealpha=0.85)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle="--", alpha=0.35)
    return r


def _add_leader_lines(
    fig: plt.Figure,
    ax_scatter: plt.Axes,
    ax_struct: plt.Axes,
    choice: StructureChoice,
) -> None:
    """Connect the highlighted scatter point to the structure panel's left edge.

    Two subtle red leader lines run from the point (scatter data coordinates)
    to the upper-left (0, 1) and lower-left (0, 0) corners of the adjacent
    structure axes' bounding box, visually linking the point to its image.
    """
    point = (choice.phi0, choice.tau_root)
    for corner in ((0.0, 1.0), (0.0, 0.0)):
        connection = ConnectionPatch(
            xyA=point,
            coordsA=ax_scatter.transData,
            xyB=corner,
            coordsB=ax_struct.transAxes,
            color="red",
            alpha=0.45,
            linewidth=0.9,
            linestyle="--",
            zorder=1,
        )
        fig.add_artist(connection)


def build_two_row_figure(
    phi0_2d: np.ndarray,
    tau_2d: np.ndarray,
    phi0_3d: np.ndarray,
    tau_3d: np.ndarray,
    choice_2d: StructureChoice,
    struct_2d: np.ndarray,
    choice_3d: StructureChoice,
    struct_3d: np.ndarray,
    out_path: Path,
    *,
    dpi: int = 220,
) -> tuple[float, float, tuple[int, ...]]:
    """Build the 2x2 scatter + structure figure; return (r_2d, r_3d, crop_shape_3d)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(13.0, 10.0))
    ax_scatter_2d = fig.add_subplot(2, 2, 1)
    ax_struct_2d = fig.add_subplot(2, 2, 2)
    ax_scatter_3d = fig.add_subplot(2, 2, 3)
    ax_struct_3d = fig.add_subplot(2, 2, 4, projection="3d")

    r2d = _plot_scatter_row(ax_scatter_2d, phi0_2d, tau_2d, _K_2D, "2D", choice_2d)
    r3d = _plot_scatter_row(ax_scatter_3d, phi0_3d, tau_3d, _K_3D, "3D", choice_3d)
    _render_structure_2d(ax_struct_2d, struct_2d, choice_2d)
    crop_shape_3d = _render_structure_3d(ax_struct_3d, struct_3d, choice_3d)

    fig.suptitle(
        r"Correlation of openness ($\phi_0$) and sponginess ($\tau^{1/k}$)",
        y=0.99,
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    # Leader lines added after layout so the structure-panel corners are final.
    _add_leader_lines(fig, ax_scatter_2d, ax_struct_2d, choice_2d)
    _add_leader_lines(fig, ax_scatter_3d, ax_struct_3d, choice_3d)

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return r2d, r3d, crop_shape_3d


def _report_choice(label: str, choice: StructureChoice) -> None:
    """Print the selection outcome for one case."""
    status = (
        f"{choice.n_qualified} qualified"
        if choice.qualified
        else "NONE qualified -> closest fallback"
    )
    print(
        f"[{label}] {status}; chosen: {choice.structure}\n"
        f"        phi_0={choice.phi0:.6f}, tau={choice.tau:.6f}, "
        f"tau^(1/{choice.k})={choice.tau_root:.6f}"
    )


def main() -> None:
    """Create the 2x2 phi_0 vs tau^(1/k) + structure figure from existing CSVs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-2d", type=Path, default=_DEFAULT_2D)
    parser.add_argument("--metrics-3d", type=Path, default=_DEFAULT_3D)
    parser.add_argument("--struct-dir-2d", type=Path, default=_STRUCT_DIR_2D)
    parser.add_argument("--struct-dir-3d", type=Path, default=_STRUCT_DIR_3D)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()

    rows_2d = read_metrics(args.metrics_2d)
    rows_3d = read_metrics(args.metrics_3d)
    phi0_2d, tau_2d = read_phi0_tau(args.metrics_2d)
    phi0_3d, tau_3d = read_phi0_tau(args.metrics_3d)

    choice_2d = choose_representative(rows_2d, _K_2D)
    choice_3d = choose_representative(rows_3d, _K_3D)
    _report_choice("2D", choice_2d)
    _report_choice("3D", choice_3d)

    struct_2d = load_structure(args.struct_dir_2d, choice_2d.structure)
    struct_3d = load_structure(args.struct_dir_3d, choice_3d.structure)
    print(f"Loaded 2D array shape={struct_2d.shape}, 3D array shape={struct_3d.shape}")

    solid_frac_3d = float(struct_3d.mean())
    print(
        f"3D volume fractions (1=solid, 0=pore): solid={solid_frac_3d:.6f}, "
        f"pore={1.0 - solid_frac_3d:.6f}"
    )

    r2d, r3d, crop_shape_3d = build_two_row_figure(
        phi0_2d,
        tau_2d,
        phi0_3d,
        tau_3d,
        choice_2d,
        struct_2d,
        choice_3d,
        struct_3d,
        args.output,
        dpi=args.dpi,
    )
    print(f"Read rows: 2D={phi0_2d.size}, 3D={phi0_3d.size}")
    print(f"3D render central crop shape: {crop_shape_3d}")
    print(f"Pearson r: 2D (phi_0 vs tau^(1/{_K_2D}))={r2d:.6f}, 3D (phi_0 vs tau^(1/{_K_3D}))={r3d:.6f}")
    print(f"Figure: {args.output}")


if __name__ == "__main__":
    main()
