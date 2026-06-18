#!/usr/bin/env python3
"""Regenerate manuscript Table 3 from the closed-cells dataset.

Table 3 (``\\label{tab:exp3_regressions}``) reports through-origin linear
regressions (intercept fixed to 0) between Betti-curve-derived predictors and
geometric targets on the closed-cells dataset. This script recomputes the five
regression rows directly from ``betti_predictors_closed_summary.csv`` and emits
the table as LaTeX matching the manuscript's structure.

The regression and predictor extraction reuse the co-located
``paper_betti_predictors_core`` helpers (``csv_rows_for_scatter_and_fit``,
``csv_column``, ``linear_regression_through_origin``) so the numbers match
exactly what ``fig5_betti_shape_predictors.py`` and the full pipeline produce.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

# Script lives at official_repository/codes/regenerating_plots_from_datasets/fig_5/,
# so official_repository == Path(__file__).resolve().parents[3].
_OFFICIAL_REPO = Path(__file__).resolve().parents[3]
assert _OFFICIAL_REPO.name == "official_repository", _OFFICIAL_REPO

# Import the co-located core helper regardless of the current working directory.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from paper_betti_predictors_core import (  # noqa: E402
    csv_column,
    csv_rows_for_scatter_and_fit,
    linear_regression_through_origin,
)

_DEFAULT_CSV = (
    _OFFICIAL_REPO / "datasets" / "fig_5" / "betti_predictors_closed_summary.csv"
)
_DEFAULT_OUTPUT = _OFFICIAL_REPO / "tables" / "table3.tex"
_DEFAULT_CSV_OUTPUT = _OFFICIAL_REPO / "tables" / "table3.csv"


@dataclass(frozen=True)
class RowSpec:
    """One Table 3 regression row: a (target, predictor) pair and its CSV columns.

    The through-origin fit is ``predictor_level = slope * target`` (the same
    orientation used in the figure), where ``x_key`` is the target column and
    ``y_key`` is the predictor-level column in the summary CSV.
    """

    target_tex: str
    predictor_tex: str
    predictor_plain: str
    summary: str
    x_key: str
    y_key: str


# Order, labels, and math notation match Table 3 in the manuscript.
ROW_SPECS: tuple[RowSpec, ...] = (
    RowSpec(
        target_tex="$r$",
        predictor_tex=r"$-\arg\max_t \nabla \beta_0(t)$",
        predictor_plain="-argmax_t grad_beta0",
        summary="r   ~ -argmax dB0",
        x_key="neg_r",
        y_key="b0_grad_max_level",
    ),
    RowSpec(
        target_tex="$d/2$",
        predictor_tex=r"$\arg\min_t \nabla \beta_0(t)$",
        predictor_plain="argmin_t grad_beta0",
        summary="d/2 ~ argmin dB0",
        x_key="d_half",
        y_key="b0_grad_min_level",
    ),
    RowSpec(
        target_tex="$d/2$",
        predictor_tex=r"$\arg\max_t \nabla \beta_1(t)$",
        predictor_plain="argmax_t grad_beta1",
        summary="d/2 ~ argmax dB1",
        x_key="d_half",
        y_key="b1_grad_max_level",
    ),
    RowSpec(
        target_tex="$d/2$",
        predictor_tex=(
            r"$\frac{1}{2}\!\left(\arg\min_t \nabla \beta_0(t)"
            r"+\arg\max_t \nabla \beta_1(t)\right)$"
        ),
        predictor_plain="0.5*(argmin_t grad_beta0 + argmax_t grad_beta1)",
        summary="d/2 ~ midpoint(argmin dB0, argmax dB1)",
        x_key="d_half",
        y_key="midpoint_b0min_b1max_level",
    ),
    RowSpec(
        target_tex="$h/2$",
        predictor_tex=r"$\arg\min_t \nabla \beta_1(t)$",
        predictor_plain="argmin_t grad_beta1",
        summary="h/2 ~ argmin dB1",
        x_key="h_half",
        y_key="b1_grad_min_level",
    ),
)


@dataclass(frozen=True)
class RegressionResult:
    """Computed regression statistics for one Table 3 row."""

    spec: RowSpec
    slope: float
    r2: float
    n_samples: int


def load_fit_rows(csv_path: Path) -> list[dict[str, str]]:
    """Load the closed-cells summary CSV and keep only scatter/fit-eligible rows."""
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")
    return csv_rows_for_scatter_and_fit(rows)


def compute_regression(rows: list[dict[str, str]], spec: RowSpec) -> RegressionResult:
    """Through-origin regression (slope, R², n) for one target/predictor pair."""
    x = csv_column(rows, spec.x_key)
    y = csv_column(rows, spec.y_key)
    slope, r2 = linear_regression_through_origin(x, y)
    return RegressionResult(spec=spec, slope=slope, r2=r2, n_samples=len(x))


def build_all_rows(rows: list[dict[str, str]]) -> list[RegressionResult]:
    """Compute all five Table 3 regression rows."""
    return [compute_regression(rows, spec) for spec in ROW_SPECS]


def check_midpoint_consistency(results: list[RegressionResult]) -> bool:
    """Verify row 4 (midpoint) slope equals the mean of rows 2 and 3.

    Holds exactly because the midpoint predictor is the per-structure average of
    rows 2 and 3 and all three share the same target column, so the through-origin
    slope is linear in the predictor.
    """
    expected = 0.5 * (results[1].slope + results[2].slope)
    return abs(results[3].slope - expected) < 1e-9


def render_latex(results: list[RegressionResult]) -> str:
    """Render the results as a LaTeX table matching the manuscript's Table 3."""
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Linear regressions (intercept fixed to $0$) between "
        r"Betti-curve-derived predictors and geometric targets on the "
        r"closed-cells dataset.}",
        r"\label{tab:exp3_regressions}",
        r"\begin{tabular}{llllr}",
        r"\hline",
        r"\textbf{target} & \textbf{predictor} & \textbf{slope} & "
        r"\textbf{$R^2$} & \textbf{$n_{\mathrm{samples}}$} \\",
        r"\hline",
    ]
    for res in results:
        lines.append(
            f"{res.spec.target_tex} & {res.spec.predictor_tex} & "
            f"${res.slope:.3f}$ & ${res.r2:.3f}$ & {res.n_samples} \\\\"
        )
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def write_csv(results: list[RegressionResult], path: Path) -> None:
    """Write the regenerated table rows to a plain-text CSV (same computed values)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["target", "predictor", "slope", "R2", "n_samples"])
        for res in results:
            target_plain = res.spec.target_tex.strip("$")
            writer.writerow(
                [
                    target_plain,
                    res.spec.predictor_plain,
                    f"{res.slope:.3f}",
                    f"{res.r2:.3f}",
                    res.n_samples,
                ]
            )


def format_summary(results: list[RegressionResult]) -> str:
    """Human-readable one-line-per-row summary of the regenerated table."""
    header = f"{'target/predictor':40s} {'slope':>8s} {'R^2':>8s} {'n':>5s}"
    out = [header, "-" * len(header)]
    for res in results:
        out.append(
            f"{res.spec.summary:40s} {res.slope:8.3f} {res.r2:8.3f} "
            f"{res.n_samples:5d}"
        )
    return "\n".join(out)


def main() -> None:
    """Recompute Table 3 from the closed-cells CSV and write it as LaTeX."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=_DEFAULT_CSV)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--csv-output", type=Path, default=_DEFAULT_CSV_OUTPUT)
    args = parser.parse_args()

    rows = load_fit_rows(args.csv)
    results = build_all_rows(rows)

    print(f"Closed-cells CSV: {args.csv}")
    print(f"Samples used in fits: {len(rows)}\n")
    print(format_summary(results))

    consistent = check_midpoint_consistency(results)
    expected_mid = 0.5 * (results[1].slope + results[2].slope)
    print(
        f"\nSanity check row4 == 1/2(row2 + row3): {consistent} "
        f"(row4={results[3].slope:.3f}, 1/2(row2+row3)={expected_mid:.3f})"
    )

    latex = render_latex(results)
    print("\n" + latex)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(latex)
    print(f"Wrote {args.output}")

    write_csv(results, args.csv_output)
    print(f"Wrote {args.csv_output}")


if __name__ == "__main__":
    main()
