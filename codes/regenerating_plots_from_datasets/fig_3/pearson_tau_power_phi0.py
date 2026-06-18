"""Compute Pearson correlations between ``phi_0`` and ``tau ** (1/k)``.

Purpose
-------
For both the 2D and 3D datasets behind Figure 3, this script measures how
strongly the openness proxy ``phi_0`` correlates with fractional powers of the
persistence metric ``tau``. For each exponent ``1/k`` (k = 1..6) it computes the
Pearson correlation coefficient ``r`` and the associated two-sided p-value, for
both dimensionalities.

Output layout
-------------
The results are written as a tabular CSV with one row per ``k`` and the columns::

    k, exponent, pearson_r_2d, p_value_2d, pearson_r_3d, p_value_3d

This "one row per k" layout was chosen because the natural independent variable
of the experiment is the exponent ``1/k``; keeping 2D and 3D side by side in the
same row makes the dimensional comparison easy to read at a glance.

Edge cases
----------
``tau`` is expected to lie in ``[0, 1]`` so ``tau ** (1/k)`` is always defined
(``0 ** (1/k) == 0``). If any negative ``tau`` values are present, the affected
rows are dropped (pairwise) before computing the correlation for the powered
column, and a note is printed. Observed data ranges are reported to stdout.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# official_repository = parents[3] of this file:
# fig_3 -> regenerating_plots_from_datasets -> codes -> official_repository
REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET_DIR = REPO_ROOT / "datasets" / "fig_3"

DEFAULT_CSV_2D = DATASET_DIR / "tau_phi0_metrics_2d.csv"
DEFAULT_CSV_3D = DATASET_DIR / "tau_phi0_metrics_3d.csv"
DEFAULT_OUTPUT = DATASET_DIR / "pearson_tau_power_phi0.csv"

K_VALUES = (1, 2, 3, 4, 5, 6)


def load_metrics(csv_path: Path) -> pd.DataFrame:
    """Load a ``structure,phi_0,tau`` metrics CSV.

    Column names are matched case-insensitively and stripped of surrounding
    whitespace so minor header variations are handled robustly.
    """
    df = pd.read_csv(csv_path)
    normalized = {col.strip().lower(): col for col in df.columns}
    for required in ("phi_0", "tau"):
        if required not in normalized:
            raise KeyError(
                f"Column '{required}' not found in {csv_path} "
                f"(found columns: {list(df.columns)})"
            )
    out = pd.DataFrame(
        {
            "phi_0": pd.to_numeric(df[normalized["phi_0"]], errors="coerce"),
            "tau": pd.to_numeric(df[normalized["tau"]], errors="coerce"),
        }
    )
    return out


def pearson_for_power(phi_0: np.ndarray, tau: np.ndarray, k: int) -> tuple[float, float]:
    """Return ``(r, p_value)`` between ``phi_0`` and ``tau ** (1/k)``.

    Pairs containing NaN or negative ``tau`` (for which the fractional power is
    undefined) are dropped before the correlation is computed.
    """
    exponent = 1.0 / k
    valid = np.isfinite(phi_0) & np.isfinite(tau) & (tau >= 0)
    phi_valid = phi_0[valid]
    tau_valid = tau[valid]
    powered = np.power(tau_valid, exponent)
    r, p_value = pearsonr(phi_valid, powered)
    return float(r), float(p_value)


def build_correlation_table(
    df_2d: pd.DataFrame,
    df_3d: pd.DataFrame,
    k_values: tuple[int, ...] = K_VALUES,
    decimals: int = 6,
) -> pd.DataFrame:
    """Build the per-``k`` Pearson correlation table for both dimensions."""
    phi_2d, tau_2d = df_2d["phi_0"].to_numpy(), df_2d["tau"].to_numpy()
    phi_3d, tau_3d = df_3d["phi_0"].to_numpy(), df_3d["tau"].to_numpy()

    rows = []
    for k in k_values:
        r2, p2 = pearson_for_power(phi_2d, tau_2d, k)
        r3, p3 = pearson_for_power(phi_3d, tau_3d, k)
        rows.append(
            {
                "k": k,
                "exponent": round(1.0 / k, decimals),
                "pearson_r_2d": round(r2, decimals),
                "p_value_2d": round(p2, decimals),
                "pearson_r_3d": round(r3, decimals),
                "p_value_3d": round(p3, decimals),
            }
        )
    return pd.DataFrame(rows)


def report_data_range(label: str, df: pd.DataFrame) -> None:
    """Print the observed ``tau`` range and warn about negative values."""
    tau = df["tau"].to_numpy()
    n_neg = int(np.sum(tau < 0))
    print(
        f"[{label}] n={len(df)}  tau range: "
        f"min={np.nanmin(tau):.6f}, max={np.nanmax(tau):.6f}"
    )
    if n_neg:
        print(
            f"[{label}] WARNING: {n_neg} negative tau value(s) found; "
            "those rows were dropped before computing tau**(1/k)."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-2d", type=Path, default=DEFAULT_CSV_2D)
    parser.add_argument("--csv-3d", type=Path, default=DEFAULT_CSV_3D)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--decimals", type=int, default=6)
    args = parser.parse_args()

    df_2d = load_metrics(args.csv_2d)
    df_3d = load_metrics(args.csv_3d)

    report_data_range("2D", df_2d)
    report_data_range("3D", df_3d)

    table = build_correlation_table(df_2d, df_3d, decimals=args.decimals)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)

    print("\nPearson r between phi_0 and tau**(1/k):\n")
    print(table.to_string(index=False))
    print(f"\nSaved table to: {args.output}")


if __name__ == "__main__":
    main()
