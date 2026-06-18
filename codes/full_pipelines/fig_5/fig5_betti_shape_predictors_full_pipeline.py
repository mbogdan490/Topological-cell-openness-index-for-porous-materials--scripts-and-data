#!/usr/bin/env python3
"""
Full pipeline for fig. 5 (Betti shape predictors): closed + open + open-r10 cells.

End-to-end orchestrator that REGENERATES the summary datasets from the raw
structure ``.npy`` files and then builds the combined 3x3 paper figure:

  1. Process closed-cell structures   -> closed summary CSV + 4-panel figure
  2. Process open-cell structures      -> open summary CSV + 4-panel figure
  3. Process open-r10 cell structures  -> open-r10 summary CSV + 4-panel figure
     (same open-cell recipe; sigma ranges r/10, w/10, a/10 already baked into
      the stored structures)
  4. Build the combined 3x3 figure from the three summary CSVs by invoking the
     co-repository plotting script
     (codes/regenerating_plots_from_datasets/fig_5/fig5_betti_shape_predictors.py).

Steps 1-3 run gudhi persistent homology over every structure (signed distance
transform -> 2D periodic cubical complex persistence -> Betti curves -> gradient
features). This is the computationally heavy part of the pipeline.

Default inputs (official_repository/datasets/fig_5/):
  2D_for_Betti_predictors_in_closed_cells/
  2D_for_Betti_predictors_in_open_cells/
  2D_for_Betti_predictors_in_open_cells_reduced_noise/structures/

Default outputs:
  datasets/fig_5/betti_predictors_closed_summary.csv (+ 4-panel PNG)
  datasets/fig_5/betti_predictors_open_summary.csv   (+ 4-panel PNG)
  datasets/fig_5/2D_for_Betti_predictors_in_open_cells_reduced_noise/betti_predictors_open_summary.csv (+ PNG)
  plots/fig5_betti_shape_predictors.png

Run from anywhere:

  python3 codes/full_pipelines/fig_5/fig5_betti_shape_predictors_full_pipeline.py

Use ``--skip-processing`` (or the per-dataset ``--skip-closed`` / ``--skip-open``
/ ``--skip-r10`` flags) to reuse existing CSVs, and ``--skip-plot`` to stop
before the combined figure.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from paper_betti_predictors_core import (  # noqa: E402
    EXAMPLE_SEED_DEFAULT,
    run_closed_pipeline,
    run_open_pipeline,
)

# Script lives at official_repository/codes/full_pipelines/fig_5/,
# so official_repository == _SCRIPT_DIR.parents[2].
_OFFICIAL_REPO = _SCRIPT_DIR.parents[2]
_DATASET_DIR = _OFFICIAL_REPO / "datasets" / "fig_5"

_DEFAULT_CLOSED_STRUCTURES = _DATASET_DIR / "2D_for_Betti_predictors_in_closed_cells"
_DEFAULT_OPEN_STRUCTURES = _DATASET_DIR / "2D_for_Betti_predictors_in_open_cells"
_DEFAULT_OPEN_R10_DIR = (
    _DATASET_DIR / "2D_for_Betti_predictors_in_open_cells_reduced_noise"
)
_DEFAULT_OPEN_R10_STRUCTURES = _DEFAULT_OPEN_R10_DIR / "structures"

_DEFAULT_COMBINED_OUTPUT = _OFFICIAL_REPO / "plots" / "fig5_betti_shape_predictors.png"
_COMBINED_PLOT_SCRIPT = (
    _OFFICIAL_REPO
    / "codes"
    / "regenerating_plots_from_datasets"
    / "fig_5"
    / "fig5_betti_shape_predictors.py"
)


def _run_combined_plot(
    *,
    output: Path,
    dpi: int,
    example_seed: int,
    closed_csv: Path,
    open_csv: Path,
    open_r10_csv: Path,
    closed_structures_dir: Path,
    open_structures_dir: Path,
    open_r10_structures_dir: Path,
) -> None:
    """Invoke the co-repository combined 3x3 figure script via subprocess."""
    if not _COMBINED_PLOT_SCRIPT.is_file():
        raise FileNotFoundError(f"Missing combined plot script: {_COMBINED_PLOT_SCRIPT}")

    cmd = [
        sys.executable,
        str(_COMBINED_PLOT_SCRIPT),
        "--closed-csv",
        str(closed_csv),
        "--open-csv",
        str(open_csv),
        "--open-r10-csv",
        str(open_r10_csv),
        "--closed-structures-dir",
        str(closed_structures_dir),
        "--open-structures-dir",
        str(open_structures_dir),
        "--open-r10-structures-dir",
        str(open_r10_structures_dir),
        "--output",
        str(output),
        "--dpi",
        str(dpi),
        "--example-seed",
        str(example_seed),
    ]
    print(f"[pipeline] combined plot: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    """Run closed, open, r10 Betti processing and the combined 3x3 figure."""
    parser = argparse.ArgumentParser(
        description="Combined closed/open/r10 Betti predictors full pipeline for fig. 5."
    )
    parser.add_argument(
        "--closed-structures-dir",
        type=Path,
        default=_DEFAULT_CLOSED_STRUCTURES,
        help="Closed-cell structure .npy directory.",
    )
    parser.add_argument(
        "--open-structures-dir",
        type=Path,
        default=_DEFAULT_OPEN_STRUCTURES,
        help="Open-cell structure .npy directory.",
    )
    parser.add_argument(
        "--open-r10-structures-dir",
        type=Path,
        default=_DEFAULT_OPEN_R10_STRUCTURES,
        help="Open-cell r10 structure .npy directory (sigma ranges r/10, w/10, a/10).",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=_DATASET_DIR,
        help="Output directory for the closed/open summary CSVs and 4-panel figures.",
    )
    parser.add_argument(
        "--open-r10-output-dir",
        type=Path,
        default=_DEFAULT_OPEN_R10_DIR,
        help="Output directory for the open-r10 summary CSV and 4-panel figure.",
    )
    parser.add_argument(
        "--combined-output",
        type=Path,
        default=_DEFAULT_COMBINED_OUTPUT,
        help="Path for the combined 3x3 figure PNG.",
    )
    parser.add_argument("--dpi", type=int, default=200, help="Figure DPI.")
    parser.add_argument(
        "--example-seed",
        type=int,
        default=EXAMPLE_SEED_DEFAULT,
        help="Seed for structure insets when present in the CSV.",
    )
    parser.add_argument(
        "--skip-closed",
        action="store_true",
        help="Skip closed-cell Betti processing.",
    )
    parser.add_argument(
        "--skip-open",
        action="store_true",
        help="Skip open-cell Betti processing.",
    )
    parser.add_argument(
        "--skip-r10",
        action="store_true",
        help="Skip open-r10 Betti processing.",
    )
    parser.add_argument(
        "--skip-processing",
        action="store_true",
        help="Skip all Betti CSV/figure processing (plot only, if not skipped).",
    )
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="Skip the combined 3x3 figure.",
    )
    args = parser.parse_args()

    closed_csv = args.dataset_dir / "betti_predictors_closed_summary.csv"
    open_csv = args.dataset_dir / "betti_predictors_open_summary.csv"
    open_r10_csv = args.open_r10_output_dir / "betti_predictors_open_summary.csv"

    if not args.skip_processing:
        if not args.skip_closed:
            print(f"[pipeline] Closed Betti processing -> {args.dataset_dir}", flush=True)
            run_closed_pipeline(
                args.closed_structures_dir,
                args.dataset_dir,
                args.dataset_dir,
                dpi=args.dpi,
                example_seed=args.example_seed,
            )
        if not args.skip_open:
            print(f"[pipeline] Open Betti processing -> {args.dataset_dir}", flush=True)
            run_open_pipeline(
                args.open_structures_dir,
                args.dataset_dir,
                args.dataset_dir,
                dpi=args.dpi,
                example_seed=args.example_seed,
            )
        if not args.skip_r10:
            print(
                f"[pipeline] Open r10 Betti processing -> {args.open_r10_output_dir}",
                flush=True,
            )
            run_open_pipeline(
                args.open_r10_structures_dir,
                args.open_r10_output_dir,
                args.open_r10_output_dir,
                dpi=args.dpi,
                example_seed=args.example_seed,
            )

    if not args.skip_plot:
        for label, path in (
            ("closed summary CSV", closed_csv),
            ("open summary CSV", open_csv),
            ("open r10 summary CSV", open_r10_csv),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"Missing {label}: {path}")
        if not args.open_r10_structures_dir.is_dir():
            raise FileNotFoundError(
                f"Missing open r10 structures directory: {args.open_r10_structures_dir}"
            )

        _run_combined_plot(
            output=args.combined_output,
            dpi=args.dpi,
            example_seed=args.example_seed,
            closed_csv=closed_csv,
            open_csv=open_csv,
            open_r10_csv=open_r10_csv,
            closed_structures_dir=args.closed_structures_dir,
            open_structures_dir=args.open_structures_dir,
            open_r10_structures_dir=args.open_r10_structures_dir,
        )

    print(f"[pipeline] Done. Combined figure: {args.combined_output}", flush=True)


if __name__ == "__main__":
    main()
