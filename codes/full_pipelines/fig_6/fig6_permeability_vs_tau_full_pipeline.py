#!/usr/bin/env python3
"""Scatter: tau Betti observables vs permeability (log x-axis) with Pearson r, p, R².

R² is the squared Pearson correlation (coefficient of determination for a
simple linear fit of the observable vs the x variable).

Observables are discrete sums over integer filtration levels t = -50, …, 0 from
``data/Betti_curves/{Sample}_betti.csv``, with per-level 3D tau:

    tau(t) = (beta_1 + beta_2) / (beta_0 + beta_1 + beta_2)  if denom > 0 else 0

Two-panel figure (2×1 vertical stack):

  a. Σ tau(t) (``integral_tau_m50_to_0``)
  b. −Σ t·tau(t) = Σ (−t)·tau(t) (``sum_negative_k_times_tau_m50_to_0``)

Permeability
------------
Uses **permeability** (mD) from ``tau_vs_permeability_metrics.csv`` or
``data/permeability.csv``. The plot x-axis is log10(permeability) with a log
scale spanning the sample range (~9–386 mD). Subplot annotations show Pearson
statistics vs log10(permeability); the correlations text file reports both
linear permeability and log10(permeability) associations.

Writes ``betti_observables_vs_permeability.png`` and
``betti_observables_vs_permeability_correlations.txt``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import pearsonr  # noqa: E402

# Script lives at official_repository/codes/full_pipelines/fig_6/, so
# official_repository == _SCRIPT_DIR.parents[2].
_SCRIPT_DIR = Path(__file__).resolve().parent
_OFFICIAL_REPO = _SCRIPT_DIR.parents[2]
assert _OFFICIAL_REPO.name == "official_repository", _OFFICIAL_REPO
_DATA_DIR = _OFFICIAL_REPO / "datasets" / "fig_6"
_BETTI_DIR = _DATA_DIR / "Betti_curves"
_METRICS_CSV = _DATA_DIR / "tau_vs_permeability_metrics.csv"
_PERMEABILITY_CSV = _DATA_DIR / "permeability.csv"
_PLOT_PATH = _OFFICIAL_REPO / "plots" / "fig6_permeability_vs_tau.png"
_CORRELATIONS_TXT = _OFFICIAL_REPO / "plots" / "fig6_permeability_vs_tau_correlations.txt"
_INTEGRALS_CSV = _DATA_DIR / "betti_integrals_vs_permeability.csv"

LEVEL_MIN = -50
LEVEL_MAX = 0

BETTI_FILE_TO_SAMPLE: dict[str, str] = {
    "Bandera_Brown": "Bandera Brown",
    "Bandera_Gray": "Bandera Gray",
    "Buff_Berea": "Buff Berea",
    "Bentheimer": "Bentheimer",
    "Berea": "Berea",
    "BSG": "BSG",
    "BUG": "BUG",
    "Castle_Gate": "Castle Gate",
    "Kirby": "Kirby",
    "Leopard": "Leopard",
    "Parker": "Parker",
}

# (column name, short LaTeX label for subplot y-axis)
OBSERVABLES: tuple[tuple[str, str], ...] = (
    ("integral_tau_m50_to_0", r"$\sum \tau(t)$"),
    ("sum_negative_k_times_tau_m50_to_0", r"$\sum (-t)\,\tau(t)$"),
)

# Human-readable names for correlation text output (plot/display notation uses t).
OBSERVABLE_DISPLAY_NAMES: dict[str, str] = {
    "integral_tau_m50_to_0": r"Σ τ(t) over t = -50…0",
    "sum_negative_k_times_tau_m50_to_0": r"Σ (−t)·τ(t) over t = -50…0",
}

PANEL_LABELS = ("a", "b")

INTEGRALS_CSV_COLUMNS = (
    "integral_b1_times_k2_m50_to_0",
    "integral_b1_squared_times_k_m50_to_0",
    "integral_b1_times_k_times_tau_m50_to_0",
    "tau_at_filtration_minus_1",
    "integral_k2_times_tau_m50_to_0",
    "integral_k2_times_b0_times_tau_m50_to_0",
    "integral_tau_m50_to_0",
    "sum_negative_k_times_tau_m50_to_0",
    "sum_k_times_tau_m50_to_0",
)


def tau_from_betti(b0: float, b1: float, b2: float) -> float:
    """3D sponginess index tau at one filtration level (from Betti counts)."""
    denom = b0 + b1 + b2
    return float((b1 + b2) / denom) if denom > 0 else 0.0


def load_metrics() -> pd.DataFrame:
    """Load metrics keyed by human-readable sample name."""
    if _METRICS_CSV.is_file():
        return pd.read_csv(_METRICS_CSV).set_index("sample")
    return pd.read_csv(_PERMEABILITY_CSV).set_index("sample")


def compute_observables(betti_path: Path) -> dict[str, float]:
    """Discrete sums over filtration level t in [LEVEL_MIN, LEVEL_MAX]."""
    df = pd.read_csv(betti_path)
    mask = (df["filtration_level"] >= LEVEL_MIN) & (df["filtration_level"] <= LEVEL_MAX)
    sub = df.loc[mask].copy()
    sub["filtration_level"] = sub["filtration_level"].astype(int)
    expected = LEVEL_MAX - LEVEL_MIN + 1
    if len(sub) != expected:
        missing = set(range(LEVEL_MIN, LEVEL_MAX + 1)) - set(sub["filtration_level"])
        raise ValueError(f"{betti_path.name}: missing levels {sorted(missing)}")

    k = sub["filtration_level"].to_numpy(dtype=float)
    b0 = sub["beta_0"].to_numpy(dtype=float)
    b1 = sub["beta_1"].to_numpy(dtype=float)
    b2 = sub["beta_2"].to_numpy(dtype=float)
    tau = np.array([tau_from_betti(x0, x1, x2) for x0, x1, x2 in zip(b0, b1, b2)])

    idx_m1 = int(np.where(k == -1)[0][0])
    tau_m1 = float(tau[idx_m1])

    sum_k_tau = float(np.sum(k * tau))
    return {
        "sum_k_times_tau_m50_to_0": sum_k_tau,
        "sum_negative_k_times_tau_m50_to_0": float(np.sum(-k * tau)),
        "integral_b1_times_level_m50_to_0": float(np.sum(b1 * k)),
        "integral_b0_times_level_m50_to_0": float(np.sum(b0 * k)),
        "integral_b0_times_level_times_tau_m50_to_0": float(np.sum(b0 * k * tau)),
        "integral_b1_times_k2_m50_to_0": float(np.sum(b1 * k**2)),
        "integral_b1_squared_times_k_m50_to_0": float(np.sum(b1**2 * k)),
        "integral_b1_times_k_times_tau_m50_to_0": float(np.sum(b1 * k * tau)),
        "tau_at_filtration_minus_1": tau_m1,
        "integral_k2_times_tau_m50_to_0": float(np.sum(k**2 * tau)),
        "integral_k2_times_b0_times_tau_m50_to_0": float(np.sum(k**2 * b0 * tau)),
        "integral_tau_m50_to_0": float(np.sum(tau)),
    }


def build_table() -> pd.DataFrame:
    """One row per sample: observables, permeability, log10(permeability)."""
    metrics = load_metrics()
    rows: list[dict[str, object]] = []

    for stem, sample in sorted(BETTI_FILE_TO_SAMPLE.items(), key=lambda x: x[1]):
        betti_path = _BETTI_DIR / f"{stem}_betti.csv"
        if not betti_path.is_file():
            raise FileNotFoundError(f"Missing Betti CSV: {betti_path}")
        if sample not in metrics.index:
            raise KeyError(f"Sample {sample!r} not in metrics CSV")

        obs = compute_observables(betti_path)
        perm = float(metrics.loc[sample, "permeability"])
        reported_porosity = float(metrics.loc[sample, "reported_porosity"])
        rows.append(
            {
                "sample": sample,
                "permeability_mD": perm,
                "log10_permeability": float(np.log10(perm)),
                "reported_porosity": reported_porosity,
                **obs,
            }
        )

    return pd.DataFrame(rows)


def pearson_stats(y: np.ndarray, x: np.ndarray) -> tuple[float, float, float]:
    """Return (r, p, R²) with R² = r² for simple linear Pearson association."""
    r, p = pearsonr(y, x)
    r = float(r)
    p = float(p)
    return r, p, r * r


def compute_all_stats(
    df: pd.DataFrame,
) -> tuple[
    list[tuple[str, float, float, float]],
    list[tuple[str, float, float, float]],
]:
    """Pearson r, p, and R² for each observable vs permeability and log10(permeability)."""
    perm = df["permeability_mD"].astype(float).to_numpy()
    log_perm = df["log10_permeability"].astype(float).to_numpy()

    stats_perm: list[tuple[str, float, float, float]] = []
    stats_log_perm: list[tuple[str, float, float, float]] = []
    for col, _ in OBSERVABLES:
        y = df[col].astype(float).to_numpy()
        stats_perm.append((col, *pearson_stats(y, perm)))
        stats_log_perm.append((col, *pearson_stats(y, log_perm)))
    return stats_perm, stats_log_perm


def correlation_lines(
    stats_perm: list[tuple[str, float, float, float]],
    stats_log_perm: list[tuple[str, float, float, float]],
    n: int,
) -> list[str]:
    """Format Pearson r, p, R² for each observable vs permeability and log10(permeability)."""
    lines = [
        "Pearson correlations: tau Betti observables vs permeability",
        f"n = {n} DRP-317 samples",
        "Permeability in mD (Neumann Table 1 / tau_vs_permeability_metrics.csv)",
        "R² = r² (coefficient of determination for simple linear Pearson fit)",
        f"Filtration levels: integer t = {LEVEL_MIN}, …, {LEVEL_MAX} (discrete sum)",
        "Plot x-axis: log10(permeability) with log scale; annotations use log10(permeability)",
        "",
        "Observables",
        "",
        "vs permeability (mD, linear)",
        f"{'observable':<45} {'r':>9} {'p':>10} {'R²':>10}",
        "-" * 76,
    ]
    for col, r, p, r2 in stats_perm:
        label = OBSERVABLE_DISPLAY_NAMES.get(col, col)
        lines.append(f"{label:<45} {r:+9.4f} {p:10.4g} {r2:10.4g}")
        lines.append("")

    lines.extend(
        [
            "vs log10(permeability)",
            f"{'observable':<45} {'r':>9} {'p':>10} {'R²':>10}",
            "-" * 76,
        ]
    )
    for col, r, p, r2 in stats_log_perm:
        label = OBSERVABLE_DISPLAY_NAMES.get(col, col)
        lines.append(f"{label:<45} {r:+9.4f} {p:10.4g} {r2:10.4g}")
        lines.append("")

    return lines


def update_integrals_csv(df: pd.DataFrame) -> None:
    """Add new integral columns to betti_integrals_vs_permeability.csv if present."""
    if not _INTEGRALS_CSV.is_file():
        return

    existing = pd.read_csv(_INTEGRALS_CSV)
    updates = df[["sample", *INTEGRALS_CSV_COLUMNS]].copy()
    merged = existing.drop(columns=list(INTEGRALS_CSV_COLUMNS), errors="ignore").merge(
        updates,
        on="sample",
        how="left",
    )
    merged.to_csv(_INTEGRALS_CSV, index=False, float_format="%.10g")
    print(f"Updated {_INTEGRALS_CSV} with columns: {', '.join(INTEGRALS_CSV_COLUMNS)}")


# ~1 cm in figure fraction (fig height 10 in) and panel-label offsets (fig coords).
_STATS_BOX_Y_AXES = 0.90  # was 0.95; shifted down ~1 cm
_PANEL_LABEL_X_OFFSET = 0.035  # left of axes edge (was 0.015)
_PANEL_LABEL_Y_OFFSET = 0.018  # above axes top (was 0)


def _add_panel_labels(fig: plt.Figure, axes: np.ndarray, labels: tuple[str, ...]) -> None:
    """Place panel labels (a/b) above-left of each panel, outside the axes."""
    fig.canvas.draw()
    for ax, label in zip(axes, labels):
        bbox = ax.get_position()
        fig.text(
            bbox.x0 - _PANEL_LABEL_X_OFFSET,
            bbox.y1 + _PANEL_LABEL_Y_OFFSET,
            label,
            ha="right",
            va="top",
            fontsize=14,
            fontweight="bold",
            transform=fig.transFigure,
        )


def plot_scatter(
    df: pd.DataFrame,
    stats_log_perm: list[tuple[str, float, float, float]],
) -> None:
    """2×1 figure: Σ τ(t) and −Σ t·τ(t) vs log10(permeability)."""
    perm = df["permeability_mD"].astype(float)
    perm_min = float(perm.min())
    perm_max = float(perm.max())

    fig, axes = plt.subplots(2, 1, figsize=(8, 10))
    axes = np.atleast_1d(axes)

    colors = ("C1", "C0")
    for ax, (col, ylabel), (_, r, p, r2), color in zip(
        axes, OBSERVABLES, stats_log_perm, colors
    ):
        y = df[col].astype(float)
        ax.scatter(perm, y, s=70, color=color, edgecolors="k", linewidths=0.4)
        for _, row in df.iterrows():
            ax.annotate(
                row["sample"],
                (row["permeability_mD"], row[col]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=7,
                alpha=0.85,
            )
        ax.set_xscale("log")
        ax.set_xlim(perm_min * 0.85, perm_max * 1.15)
        ax.set_xlabel(r"Permeability (mD), $\log_{10}$ scale")
        ax.set_ylabel(ylabel)
        ax.text(
            0.05,
            _STATS_BOX_Y_AXES,
            (
                f"Pearson r = {r:+.3f} vs log10(perm)\n"
                f"p = {p:.3g}\n"
                f"$R^2$ = {r2:.3f} (n={len(df)})"
            ),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
        )
        ax.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    _add_panel_labels(fig, axes, PANEL_LABELS)
    fig.savefig(_PLOT_PATH, dpi=150)
    plt.close(fig)


def print_stats_table(
    title: str,
    stats: list[tuple[str, float, float, float]],
) -> None:
    """Print a formatted correlation table to stdout."""
    print(title)
    print(f"{'observable':<45} {'r':>9} {'p':>10} {'R²':>10}")
    print("-" * 76)
    for col, r, p, r2 in stats:
        label = OBSERVABLE_DISPLAY_NAMES.get(col, col)
        print(f"{label:<45} {r:+9.4f} {p:10.4g} {r2:10.4g}")
    print()


def main() -> None:
    """Build the permeability-vs-tau figure from the full-pipeline dataset CSVs."""
    _PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = build_table()
    stats_perm, stats_log_perm = compute_all_stats(df)
    lines = correlation_lines(stats_perm, stats_log_perm, len(df))
    _CORRELATIONS_TXT.write_text("\n".join(lines))
    update_integrals_csv(df)
    plot_scatter(df, stats_log_perm)

    print(f"Wrote {_PLOT_PATH}")
    print(f"Wrote {_CORRELATIONS_TXT}")
    print()
    print_stats_table("vs permeability (mD, linear)", stats_perm)
    print_stats_table("vs log10(permeability)", stats_log_perm)
    print()
    print("Sample values:")
    print(
        f"{'sample':<16} {'integral_tau_m50_to_0':>24} "
        f"{'sum_negative_k_times_tau_m50_to_0':>32} {'log10_perm':>12}"
    )
    print("-" * 88)
    for _, row in df.sort_values("sample").iterrows():
        print(
            f"{row['sample']:<16} {row['integral_tau_m50_to_0']:24.6f} "
            f"{row['sum_negative_k_times_tau_m50_to_0']:32.6f} {row['log10_permeability']:12.4f}"
        )


if __name__ == "__main__":
    main()
