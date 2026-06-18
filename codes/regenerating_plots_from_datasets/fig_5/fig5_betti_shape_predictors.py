#!/usr/bin/env python3
"""Combined 3×3 figure: closed cells, open cells (iteration 6), open cells (σ = U(0, param/10)).

Reads precomputed summary CSVs (``h_half`` from ``compute_h_half()`` in
``paper_betti_predictors_core``). Scatter panels and through-origin fits use
``csv_rows_for_scatter_and_fit`` (structures with no strictly negative ∇β₁
excluded); inset example structures are chosen from the full CSV rows.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Script lives at official_repository/codes/regenerating_plots_from_datasets/fig_5/,
# so official_repository == Path(__file__).resolve().parents[3].
_OFFICIAL_REPO = Path(__file__).resolve().parents[3]

# Import the co-located core helper regardless of the current working directory.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from paper_betti_predictors_core import (  # noqa: E402
    B0_GRAD_MAX_LEVEL_LABEL,
    B0_GRAD_MAX_VS_NEG_R_TITLE,
    B1_GRAD_MIN_LEVEL_LABEL,
    B1_GRAD_MIN_VS_H_HALF_TITLE,
    D_HALF_AXIS_LABEL,
    EXAMPLE_SEED_DEFAULT,
    H_HALF_AXIS_LABEL,
    MIDPOINT_B0MIN_B1MAX_LEVEL_LABEL,
    MIDPOINT_VS_D_HALF_TITLE,
    MIDPOINT_VS_NEG_W_HALF_TITLE,
    NEG_R_AXIS_LABEL,
    NEG_W_HALF_AXIS_LABEL,
    csv_rows_for_scatter_and_fit,
    scatter_panel_from_csv,
)

_DATASET_DIR = _OFFICIAL_REPO / "datasets" / "fig_5"
_DEFAULT_CLOSED_CSV = _DATASET_DIR / "betti_predictors_closed_summary.csv"
_DEFAULT_OPEN_CSV = _DATASET_DIR / "betti_predictors_open_summary.csv"
_DEFAULT_OPEN_R10_CSV = (
    _DATASET_DIR
    / "2D_for_Betti_predictors_in_open_cells_reduced_noise/betti_predictors_open_summary.csv"
)
_DEFAULT_CLOSED_STRUCTURES = (
    _DATASET_DIR / "2D_for_Betti_predictors_in_closed_cells"
)
_DEFAULT_OPEN_STRUCTURES = (
    _DATASET_DIR / "2D_for_Betti_predictors_in_open_cells"
)
_DEFAULT_OPEN_R10_STRUCTURES = (
    _DATASET_DIR / "2D_for_Betti_predictors_in_open_cells_reduced_noise/structures"
)
_DEFAULT_OUTPUT = (
    _OFFICIAL_REPO / "plots" / "fig5_betti_shape_predictors.png"
)

_INSET_BOUNDS = [0.02, 0.55, 0.38, 0.40]
_ROW_LABELS = ["a", "b", "c"]


def read_summary_csv(path: Path) -> list[dict[str, str]]:
    """Read a precomputed Betti-predictor summary CSV."""
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def _row_by_seed(rows: list[dict[str, str]], seed: int) -> dict[str, str] | None:
    for row in rows:
        if int(row["seed"]) == seed:
            return row
    return None


def pick_closed_example_row(
    rows: list[dict[str, str]],
    *,
    example_seed: int = EXAMPLE_SEED_DEFAULT,
) -> dict[str, str]:
    """Closed inset structure: seed match or median index fallback."""
    return _row_by_seed(rows, example_seed) or rows[len(rows) // 2]


def pick_open_example_row(
    rows: list[dict[str, str]],
    *,
    example_seed: int = EXAMPLE_SEED_DEFAULT,
) -> dict[str, str]:
    """Open inset structure: seed match or largest relative std fallback."""
    if match := _row_by_seed(rows, example_seed):
        return match

    def heterogeneity(row: dict[str, str]) -> float:
        r = float(row["r_mean_param"])
        a = float(row["a_param"])
        w = float(row["w_param"])
        return (
            float(row["r_std_param"]) / max(r, 1e-12)
            + float(row["a_std_param"]) / max(a, 1e-12)
            + float(row["w_std_param"]) / max(w, 1e-12)
        )

    return max(rows, key=heterogeneity)


def add_structure_inset(ax: plt.Axes, structure: np.ndarray) -> None:
    """Upper-left inset showing a binary 2D structure (pipeline styling)."""
    inset = ax.inset_axes(_INSET_BOUNDS)
    inset.imshow(structure, origin="lower", cmap="gray", vmin=0, vmax=1)
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_linewidth(0.7)


def _add_row_labels(fig: plt.Figure, axes: np.ndarray, labels: list[str]) -> None:
    """Place row labels (a/b/c) at the upper-left corner of each row."""
    fig.canvas.draw()
    for row_idx, label in enumerate(labels):
        row_axes = axes[row_idx]
        bboxes = [ax.get_position() for ax in row_axes]
        x_left = min(bbox.x0 for bbox in bboxes)
        y_top = max(bbox.y1 for bbox in bboxes)
        fig.text(
            x_left - 0.015,
            y_top,
            label,
            ha="right",
            va="top",
            fontsize=28,
            fontweight="bold",
            transform=fig.transFigure,
        )


def _plot_closed_row(
    axes: np.ndarray,
    rows: list[dict[str, str]],
    *,
    structures_dir: Path,
    inset_rows: list[dict[str, str]],
    example_seed: int,
) -> str:
    """Top row: closed-cell panels with structure inset on column 0."""
    example = pick_closed_example_row(inset_rows, example_seed=example_seed)
    structure_name = example["structure"]
    structure_grid = np.load(structures_dir / structure_name)

    scatter_panel_from_csv(
        axes[0],
        rows,
        x_key="neg_r",
        y_key="b0_grad_max_level",
        x_label=NEG_R_AXIS_LABEL,
        y_label=B0_GRAD_MAX_LEVEL_LABEL,
        title=B0_GRAD_MAX_VS_NEG_R_TITLE,
    )
    add_structure_inset(axes[0], structure_grid)

    scatter_panel_from_csv(
        axes[1],
        rows,
        x_key="d_half",
        y_key="midpoint_b0min_b1max_level",
        x_label=D_HALF_AXIS_LABEL,
        y_label=MIDPOINT_B0MIN_B1MAX_LEVEL_LABEL,
        title=MIDPOINT_VS_D_HALF_TITLE,
    )
    scatter_panel_from_csv(
        axes[2],
        rows,
        x_key="h_half",
        y_key="b1_grad_min_level",
        # Closed cells (w = 0): h/2 reduces to the wall term a/2 - w/2.
        x_label=r"$h/2 = \sqrt{2}\,a/2 - r$",
        y_label=B1_GRAD_MIN_LEVEL_LABEL,
        title=B1_GRAD_MIN_VS_H_HALF_TITLE,
    )
    return structure_name


def _plot_open_row(
    axes: np.ndarray,
    rows: list[dict[str, str]],
    *,
    structures_dir: Path,
    inset_rows: list[dict[str, str]],
    example_seed: int,
) -> str:
    """Open-cell panels with structure inset on column 0."""
    example = pick_open_example_row(inset_rows, example_seed=example_seed)
    structure_name = example["structure"]
    structure_grid = np.load(structures_dir / structure_name)

    scatter_panel_from_csv(
        axes[0],
        rows,
        x_key="neg_r",
        y_key="b0_grad_max_level",
        x_label=NEG_R_AXIS_LABEL,
        y_label=B0_GRAD_MAX_LEVEL_LABEL,
        title=B0_GRAD_MAX_VS_NEG_R_TITLE,
    )
    add_structure_inset(axes[0], structure_grid)

    scatter_panel_from_csv(
        axes[1],
        rows,
        x_key="neg_w_half",
        y_key="midpoint_b0min_b1max_level",
        x_label=NEG_W_HALF_AXIS_LABEL,
        y_label=MIDPOINT_B0MIN_B1MAX_LEVEL_LABEL,
        title=MIDPOINT_VS_NEG_W_HALF_TITLE,
    )
    scatter_panel_from_csv(
        axes[2],
        rows,
        x_key="h_half",
        y_key="b1_grad_min_level",
        x_label=H_HALF_AXIS_LABEL,
        y_label=B1_GRAD_MIN_LEVEL_LABEL,
        title=B1_GRAD_MIN_VS_H_HALF_TITLE,
    )
    return structure_name


def save_combined_plot(
    closed_rows: list[dict[str, str]],
    open_rows: list[dict[str, str]],
    open_r10_rows: list[dict[str, str]],
    *,
    closed_structures_dir: Path,
    open_structures_dir: Path,
    open_r10_structures_dir: Path,
    output: Path,
    dpi: int,
    example_seed: int = EXAMPLE_SEED_DEFAULT,
    closed_inset_rows: list[dict[str, str]] | None = None,
    open_inset_rows: list[dict[str, str]] | None = None,
    open_r10_inset_rows: list[dict[str, str]] | None = None,
) -> tuple[str, str, str]:
    """Build the 3×3 combined figure; return inset structure filenames."""
    closed_inset_rows = closed_rows if closed_inset_rows is None else closed_inset_rows
    open_inset_rows = open_rows if open_inset_rows is None else open_inset_rows
    open_r10_inset_rows = open_r10_rows if open_r10_inset_rows is None else open_r10_inset_rows

    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 3, figsize=(18.0, 14.5), constrained_layout=True)

    closed_name = _plot_closed_row(
        axes[0],
        closed_rows,
        structures_dir=closed_structures_dir,
        inset_rows=closed_inset_rows,
        example_seed=example_seed,
    )
    open_name = _plot_open_row(
        axes[1],
        open_rows,
        structures_dir=open_structures_dir,
        inset_rows=open_inset_rows,
        example_seed=example_seed,
    )
    open_r10_name = _plot_open_row(
        axes[2],
        open_r10_rows,
        structures_dir=open_r10_structures_dir,
        inset_rows=open_r10_inset_rows,
        example_seed=example_seed,
    )

    _add_row_labels(fig, axes, _ROW_LABELS)

    fig.suptitle(
        "Betti-gradient predictors",
        fontsize=26,
        y=1.05,
    )
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return closed_name, open_name, open_r10_name


def main() -> None:
    """Create the combined closed/open/open-r10 Betti predictor figure from CSVs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--closed-csv", type=Path, default=_DEFAULT_CLOSED_CSV)
    parser.add_argument("--open-csv", type=Path, default=_DEFAULT_OPEN_CSV)
    parser.add_argument("--open-r10-csv", type=Path, default=_DEFAULT_OPEN_R10_CSV)
    parser.add_argument(
        "--closed-structures-dir",
        type=Path,
        default=_DEFAULT_CLOSED_STRUCTURES,
    )
    parser.add_argument(
        "--open-structures-dir",
        type=Path,
        default=_DEFAULT_OPEN_STRUCTURES,
    )
    parser.add_argument(
        "--open-r10-structures-dir",
        type=Path,
        default=_DEFAULT_OPEN_R10_STRUCTURES,
    )
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--example-seed",
        type=int,
        default=EXAMPLE_SEED_DEFAULT,
        help="Seed for panel-1 structure inset when present in the CSV.",
    )
    args = parser.parse_args()

    closed_rows = read_summary_csv(args.closed_csv)
    open_rows = read_summary_csv(args.open_csv)
    open_r10_rows = read_summary_csv(args.open_r10_csv)

    closed_plot_rows = csv_rows_for_scatter_and_fit(closed_rows)
    open_plot_rows = csv_rows_for_scatter_and_fit(open_rows)
    open_r10_plot_rows = csv_rows_for_scatter_and_fit(open_r10_rows)

    closed_name, open_name, open_r10_name = save_combined_plot(
        closed_plot_rows,
        open_plot_rows,
        open_r10_plot_rows,
        closed_structures_dir=args.closed_structures_dir,
        open_structures_dir=args.open_structures_dir,
        open_r10_structures_dir=args.open_r10_structures_dir,
        output=args.output,
        dpi=args.dpi,
        example_seed=args.example_seed,
        closed_inset_rows=closed_rows,
        open_inset_rows=open_rows,
        open_r10_inset_rows=open_r10_rows,
    )

    for label, total, plot in (
        ("Closed", closed_rows, closed_plot_rows),
        ("Open (iteration 6)", open_rows, open_plot_rows),
        ("Open r10 (σ = param/10)", open_r10_rows, open_r10_plot_rows),
    ):
        excluded = len(total) - len(plot)
        print(
            f"{label}: {len(plot)}/{len(total)} scatter/fit "
            f"(excluded {excluded} with no negative ∇β₁)"
        )

    print(f"Closed inset structure: {closed_name}")
    print(f"Open inset structure: {open_name}")
    print(f"Open r10 inset structure: {open_r10_name}")
    print(f"Figure: {args.output}")


if __name__ == "__main__":
    main()
