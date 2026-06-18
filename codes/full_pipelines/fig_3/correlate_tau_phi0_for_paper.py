#!/usr/bin/env python3
"""
Paper pipeline: correlate tau and phi_0 for 2D and 3D porous structures.

Loads existing binary structures (no regeneration), computes signed distance
transforms, Betti-based tau at filtration level -1, numerical picnometry phi_0,
Pearson correlations for k=1..6, and writes one combined figure (2D scatter,
3D scatter, correlation table).

Per-structure phi_0 and tau are appended to CSV files in the plot output folder
as each structure finishes. Large arrays are released before moving on.

Default inputs:
  official_repository/datasets/fig_3/2D_for_correlating_tau_and_phi_0
  official_repository/datasets/fig_3/3D_for_correlating_tau_and_phi_0

Default outputs (official_repository/plots/):
  tau_phi0_correlation_2d_3d.png
  tau_phi0_metrics_2d.csv
  tau_phi0_metrics_3d.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from paper_tau_phi0_core import (  # noqa: E402
    build_combined_figure,
    process_structures_folder,
)

# Script lives at official_repository/codes/full_pipelines/fig_3/
_OFFICIAL_REPO = _SCRIPT_DIR.parents[2]
_DEFAULT_2D = _OFFICIAL_REPO / "datasets" / "fig_3" / "2D_for_correlating_tau_and_phi_0"
_DEFAULT_3D = _OFFICIAL_REPO / "datasets" / "fig_3" / "3D_for_correlating_tau_and_phi_0"
_DEFAULT_OUT = _OFFICIAL_REPO / "plots" / "tau_phi0_correlation_2d_3d.png"


def main() -> None:
    """Run 2D and 3D analysis and build the combined paper figure."""
    parser = argparse.ArgumentParser(
        description="Correlate tau and phi_0 for 2D and 3D structures; write one combined plot.",
    )
    parser.add_argument(
        "--structures-2d",
        type=Path,
        default=_DEFAULT_2D,
        help="Directory with 2D structure .npy files.",
    )
    parser.add_argument(
        "--structures-3d",
        type=Path,
        default=_DEFAULT_3D,
        help="Directory with 3D structure .npy files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUT,
        help="Output PNG path.",
    )
    parser.add_argument("--dpi", type=int, default=220, help="Figure DPI.")
    parser.add_argument(
        "--workers-3d",
        type=int,
        default=2,
        metavar="N",
        help="Parallel worker processes for the 3D phase (default: 2). Use 1 for sequential.",
    )
    args = parser.parse_args()
    if args.workers_3d < 1:
        parser.error("--workers-3d must be at least 1.")

    out_dir = args.output.parent
    csv_2d = out_dir / "tau_phi0_metrics_2d.csv"
    csv_3d = out_dir / "tau_phi0_metrics_3d.csv"

    print("=== 2D dataset ===", flush=True)
    metrics_2d = process_structures_folder(
        args.structures_2d,
        ndim=2,
        label="2D",
        metrics_csv=csv_2d,
    )
    print(f"[2D] Collected metrics for n={metrics_2d.phi0.size} structures.", flush=True)

    print("=== 3D dataset ===", flush=True)
    metrics_3d = process_structures_folder(
        args.structures_3d,
        ndim=3,
        label="3D",
        max_workers=args.workers_3d,
        metrics_csv=csv_3d,
    )
    print(f"[3D] Collected metrics for n={metrics_3d.phi0.size} structures.", flush=True)

    print("Building combined figure ...", flush=True)
    k2d, k3d = build_combined_figure(metrics_2d, metrics_3d, args.output, dpi=args.dpi)
    print(f"Optimal k: 2D={k2d}, 3D={k3d}", flush=True)
    print(f"Saved {args.output}", flush=True)
    print(f"Saved {csv_2d}", flush=True)
    print(f"Saved {csv_3d}", flush=True)


if __name__ == "__main__":
    main()
