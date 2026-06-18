"""Core pipeline for 2D Betti-gradient vs structure-scale paper figures."""

from __future__ import annotations

import csv
import gc
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from gudhi import PeriodicCubicalComplex as PCC
from scipy.ndimage import distance_transform_edt

FILTRATION_LEVELS = np.arange(-30, 51, dtype=float)
PERSISTENCE_MIN_OPEN = 1.5
EXAMPLE_SEED_DEFAULT = 2105

# Geometrical x-axis labels (scatter panels: x = geometry, y = gradient observable).
NEG_R_AXIS_LABEL = r"$-r$"
NEG_W_HALF_AXIS_LABEL = r"$-w/2$"
D_HALF_AXIS_LABEL = r"$d/2 = a/2 - r$"
H_HALF_AXIS_LABEL = r"$h/2 = \min(\sqrt{2}\,a/2 - r,\; a/2 - w/2)$"

# Gradient-related y-axis labels.
B0_GRAD_MAX_LEVEL_LABEL = r"$\mathrm{level}(\max \nabla\beta_0)$"
B1_GRAD_MIN_LEVEL_LABEL = r"$\mathrm{level}(\min \nabla\beta_1)$"
MIDPOINT_B0MIN_B1MAX_LEVEL_LABEL = (
    r"$\frac{1}{2}\left(\mathrm{level}(\min \nabla\beta_0)"
    r"+\mathrm{level}(\max \nabla\beta_1)\right)$"
)

# Panel titles (y vs x: gradient observable vs geometrical scale).
B0_GRAD_MAX_VS_NEG_R_TITLE = rf"{B0_GRAD_MAX_LEVEL_LABEL} vs $-r$"
MIDPOINT_VS_NEG_W_HALF_TITLE = rf"{MIDPOINT_B0MIN_B1MAX_LEVEL_LABEL} vs $-w/2$"
MIDPOINT_VS_D_HALF_TITLE = rf"{MIDPOINT_B0MIN_B1MAX_LEVEL_LABEL} vs $d/2$"
B1_GRAD_MIN_VS_H_HALF_TITLE = rf"{B1_GRAD_MIN_LEVEL_LABEL} vs $h/2$"

_CLOSED_RE = re.compile(
    r"exp3_closed_size(?P<size>\d+)_r(?P<r>[\d.]+)_rstd(?P<rstd>[\d.]+)_"
    r"a(?P<a>[\d.]+)_astd(?P<astd>[\d.]+)_p(?P<p>[\d.]+)_seed(?P<seed>\d+)\.npy$"
)
_OPEN_RE = re.compile(
    r"exp4_open_size(?P<size>\d+)_r(?P<r>[\d.]+)_rstd(?P<rstd>[\d.]+)_"
    r"a(?P<a>[\d.]+)_astd(?P<astd>[\d.]+)_w(?P<w>[\d.]+)_wstd(?P<wstd>[\d.]+)_"
    r"p(?P<p>[\d.]+)_seed(?P<seed>\d+)\.npy$"
)


@dataclass
class ClosedCellRecord:
    """Per-structure summary for closed-cell Betti predictors."""

    structure: str
    sample_id: int
    seed: int
    size: int
    p: float
    r_mean_param: float
    r_std_param: float
    neg_r: float
    a_param: float
    a_std_param: float
    w_param: float
    w_std_param: float
    d_half: float
    h_half: float
    b0_grad_max_level: float
    b0_grad_min_level: float
    b1_grad_max_level: float
    b1_grad_min_level: float
    midpoint_b0min_b1max_level: float
    b1_gradient_ever_negative: bool
    exclude_from_fit: bool


@dataclass
class OpenCellRecord:
    """Per-structure summary for open-cell Betti predictors."""

    structure: str
    sample_id: int
    seed: int
    size: int
    p: float
    r_mean_param: float
    r_std_param: float
    neg_r: float
    a_param: float
    a_std_param: float
    w_param: float
    w_std_param: float
    r_half: float
    w_half: float
    h_half: float
    neg_w_half: float
    b0_grad_max_level: float
    b0_grad_min_level: float
    b1_grad_max_level: float
    b1_grad_min_level: float
    midpoint_b0min_b1max_level: float
    b1_gradient_ever_negative: bool
    exclude_from_fit: bool


def compute_betti_gradient(levels: np.ndarray, betti: np.ndarray) -> np.ndarray:
    """Discrete gradient of a Betti curve over filtration levels (same as extrema helper)."""
    grad = np.zeros_like(betti, dtype=float)
    grad[0] = (betti[1] - betti[0]) / (levels[1] - levels[0])
    for i in range(1, len(betti) - 1):
        grad[i] = (betti[i + 1] - betti[i - 1]) / (levels[i + 1] - levels[i - 1])
    grad[-1] = (betti[-1] - betti[-2]) / (levels[-1] - levels[-2])
    return grad


def b1_gradient_ever_negative(levels: np.ndarray, b1: np.ndarray) -> bool:
    """True if ∇β₁(t) is strictly negative at any filtration level."""
    return bool(np.any(compute_betti_gradient(levels, b1) < 0))


def records_for_scatter_and_fit(
    records: list[ClosedCellRecord] | list[OpenCellRecord],
) -> list[ClosedCellRecord] | list[OpenCellRecord]:
    """Structures eligible for scatter panels and through-origin fits."""
    return [r for r in records if r.b1_gradient_ever_negative]


def csv_rows_for_scatter_and_fit(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Filter precomputed CSV rows using ``b1_gradient_ever_negative``."""
    if not rows:
        return rows
    if "b1_gradient_ever_negative" not in rows[0]:
        raise ValueError(
            "Summary CSV missing column b1_gradient_ever_negative; re-run the full pipeline."
        )
    return [row for row in rows if row["b1_gradient_ever_negative"].lower() in ("true", "1", "yes")]


def csv_column(
    rows: list[dict[str, str]],
    key: str,
) -> np.ndarray:
    """Read one numeric CSV column; derive ``neg_r`` from ``r_mean_param`` when absent."""
    if not rows:
        return np.array([], dtype=float)
    if key == "neg_r" and "neg_r" not in rows[0]:
        return np.array([-float(row["r_mean_param"]) for row in rows], dtype=float)
    return np.array([float(row[key]) for row in rows], dtype=float)


def compute_signed_distance_transform(array: np.ndarray) -> np.ndarray:
    """Signed distance transform: solid positive, pore negative."""
    solids = array != 0
    pores = array == 0
    dist_to_pore = distance_transform_edt(solids)
    dist_to_solid = distance_transform_edt(pores)
    signed = np.zeros_like(dist_to_pore, dtype=float)
    signed[solids] = dist_to_pore[solids]
    signed[pores] = -dist_to_solid[pores]
    return signed


def compute_persistence_2d(signed: np.ndarray) -> list[tuple[int, tuple[float, float]]]:
    """Persistence intervals on a 2D signed distance field (non-periodic)."""
    pcc = PCC(
        top_dimensional_cells=signed.astype(np.float64),
        periodic_dimensions=[False, False],
    )
    max_val = float(np.max(signed))
    out: list[tuple[int, tuple[float, float]]] = []
    for dim, (birth, death) in pcc.persistence():
        if death == float("inf"):
            out.append((dim, (float(birth), max_val)))
        else:
            out.append((dim, (float(birth), float(death))))
    return out


def filter_persistence(
    persistence: Iterable[tuple[int, tuple[float, float]]],
    *,
    min_persistence: float,
) -> list[tuple[int, tuple[float, float]]]:
    """Keep intervals with death - birth >= min_persistence."""
    return [
        (dim, (birth, death))
        for dim, (birth, death) in persistence
        if (death - birth) >= min_persistence
    ]


def betti_at_levels(
    persistence: list[tuple[int, tuple[float, float]]],
    dim: int,
    levels: np.ndarray = FILTRATION_LEVELS,
) -> np.ndarray:
    """Betti numbers for one homology dimension across filtration levels."""
    intervals = [pair for d, pair in persistence if d == dim]
    vals = []
    for lev in levels:
        vals.append(sum(1 for birth, death in intervals if birth <= lev < death))
    return np.array(vals, dtype=float)


def betti_gradient_extrema(levels: np.ndarray, betti: np.ndarray) -> tuple[float, float]:
    """Filtration levels where the discrete gradient of a Betti curve is max/min."""
    grad = compute_betti_gradient(levels, betti)
    max_idx = int(np.argmax(grad))
    min_idx = int(np.argmin(grad))
    return float(levels[max_idx]), float(levels[min_idx])


def linear_regression_with_intercept(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Fit y ~ slope*x + intercept; return (slope, intercept, r2)."""
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(slope), float(intercept), float(r2)


def compute_h_half(*, a: float, r: float, w: float = 0.0) -> tuple[float, float, float]:
    """Return (h_geom, h_wall, h_half) with h_half = min(h_geom, h_wall)."""
    h_geom = float(np.sqrt(2.0) * a / 2.0 - r)
    h_wall = a / 2.0 - w / 2.0
    return h_geom, h_wall, float(min(h_geom, h_wall))


def record_by_seed(
    records: list[ClosedCellRecord] | list[OpenCellRecord],
    seed: int,
) -> ClosedCellRecord | OpenCellRecord | None:
    """Find one pipeline record by generation seed."""
    for record in records:
        if record.seed == seed:
            return record
    return None


def linear_regression_through_origin(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Fit y ~ slope*x (intercept 0); return (slope, r2)."""
    denom = float(np.sum(x * x))
    slope = float(np.sum(x * y) / denom) if denom > 0 else 0.0
    y_pred = slope * x
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return slope, float(r2)


_FIT_STATS_BBOX = dict(boxstyle="round", facecolor="white", alpha=0.8)


def add_fit_stats_box(
    ax: plt.Axes,
    *,
    slope: float,
    r2: float,
    through_origin: bool = True,
    intercept: float | None = None,
) -> None:
    """Lower-right annotation box with fit line, R², and intercept."""
    if through_origin:
        lines = [
            f"$y = {slope:.3f}x$",
            f"$R^2 = {r2:.3f}$",
            "intercept $= 0$",
        ]
    else:
        intercept_val = 0.0 if intercept is None else intercept
        lines = [
            f"$y = {slope:.3f}x + {intercept_val:.3f}$",
            f"$R^2 = {r2:.3f}$",
            f"intercept $= {intercept_val:.3f}$",
        ]
    ax.text(
        0.98,
        0.02,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox=_FIT_STATS_BBOX,
    )


def _parse_closed_name(name: str) -> dict[str, float | int]:
    match = _CLOSED_RE.match(name)
    if not match:
        raise ValueError(f"Cannot parse closed structure name: {name}")
    return {k: (int(v) if k in {"size", "seed"} else float(v)) for k, v in match.groupdict().items()}


def _parse_open_name(name: str) -> dict[str, float | int]:
    match = _OPEN_RE.match(name)
    if not match:
        raise ValueError(f"Cannot parse open structure name: {name}")
    return {k: (int(v) if k in {"size", "seed"} else float(v)) for k, v in match.groupdict().items()}


def process_closed_structure(path: Path, sample_id: int) -> ClosedCellRecord:
    """SDT, Betti curves, and gradient features for one closed structure."""
    params = _parse_closed_name(path.name)
    r = float(params["r"])
    a = float(params["a"])
    w = 0.0
    d_half = a / 2.0 - r
    _, _, h_half = compute_h_half(a=a, r=r, w=w)

    structure = np.load(path)
    signed = compute_signed_distance_transform(structure)
    persistence = compute_persistence_2d(signed)
    b0 = betti_at_levels(persistence, 0)
    b1 = betti_at_levels(persistence, 1)

    b0_grad_max, b0_grad_min = betti_gradient_extrema(FILTRATION_LEVELS, b0)
    b1_grad_max, b1_grad_min = betti_gradient_extrema(FILTRATION_LEVELS, b1)
    midpoint = 0.5 * (b0_grad_min + b1_grad_max)
    b1_neg = b1_gradient_ever_negative(FILTRATION_LEVELS, b1)

    del structure, signed, persistence, b0, b1
    gc.collect()

    return ClosedCellRecord(
        structure=path.name,
        sample_id=sample_id,
        seed=int(params["seed"]),
        size=int(params["size"]),
        p=float(params["p"]),
        r_mean_param=r,
        r_std_param=float(params["rstd"]),
        neg_r=-r,
        a_param=a,
        a_std_param=float(params["astd"]),
        w_param=0.0,
        w_std_param=0.0,
        d_half=float(d_half),
        h_half=h_half,
        b0_grad_max_level=b0_grad_max,
        b0_grad_min_level=b0_grad_min,
        b1_grad_max_level=b1_grad_max,
        b1_grad_min_level=b1_grad_min,
        midpoint_b0min_b1max_level=float(midpoint),
        b1_gradient_ever_negative=b1_neg,
        exclude_from_fit=not b1_neg,
    )


def process_open_structure(path: Path, sample_id: int) -> OpenCellRecord:
    """SDT, filtered Betti curves, and gradient features for one open structure."""
    params = _parse_open_name(path.name)
    r = float(params["r"])
    a = float(params["a"])
    w = float(params["w"])
    r_half = r / 2.0
    w_half = w / 2.0
    _, _, h_half = compute_h_half(a=a, r=r, w=w)
    neg_w_half = -w_half

    structure = np.load(path)
    signed = compute_signed_distance_transform(structure)
    persistence = filter_persistence(
        compute_persistence_2d(signed),
        min_persistence=PERSISTENCE_MIN_OPEN,
    )
    b0 = betti_at_levels(persistence, 0)
    b1 = betti_at_levels(persistence, 1)

    b0_grad_max, b0_grad_min = betti_gradient_extrema(FILTRATION_LEVELS, b0)
    b1_grad_max, b1_grad_min = betti_gradient_extrema(FILTRATION_LEVELS, b1)
    midpoint = 0.5 * (b0_grad_min + b1_grad_max)
    b1_neg = b1_gradient_ever_negative(FILTRATION_LEVELS, b1)

    del structure, signed, persistence, b0, b1
    gc.collect()

    return OpenCellRecord(
        structure=path.name,
        sample_id=sample_id,
        seed=int(params["seed"]),
        size=int(params["size"]),
        p=float(params["p"]),
        r_mean_param=r,
        r_std_param=float(params["rstd"]),
        neg_r=-r,
        a_param=a,
        a_std_param=float(params["astd"]),
        w_param=w,
        w_std_param=float(params["wstd"]),
        r_half=float(r_half),
        w_half=float(w_half),
        h_half=h_half,
        neg_w_half=float(neg_w_half),
        b0_grad_max_level=b0_grad_max,
        b0_grad_min_level=b0_grad_min,
        b1_grad_max_level=b1_grad_max,
        b1_grad_min_level=b1_grad_min,
        midpoint_b0min_b1max_level=float(midpoint),
        b1_gradient_ever_negative=b1_neg,
        exclude_from_fit=not b1_neg,
    )


def write_summary_csv(rows: list[ClosedCellRecord] | list[OpenCellRecord], path: Path) -> None:
    """Write per-structure parameter table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def scatter_panel_from_csv(
    ax: plt.Axes,
    rows: list[dict[str, str]],
    *,
    x_key: str,
    y_key: str,
    x_label: str,
    y_label: str,
    title: str,
) -> None:
    """Scatter panel with through-origin fit from precomputed CSV rows."""
    x = csv_column(rows, x_key)
    y = csv_column(rows, y_key)
    _scatter_through_origin(
        ax,
        x,
        y,
        x_label=x_label,
        y_label=y_label,
        title=title,
        extend_xlim=False,
    )


def _scatter_with_intercept(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    x_label: str,
    y_label: str,
    title: str,
) -> None:
    """Scatter plus OLS line (with intercept)."""
    slope, intercept, r2 = linear_regression_with_intercept(x, y)
    ax.scatter(x, y, c="steelblue", s=22, alpha=0.85, edgecolors="k", linewidths=0.35)
    x_line = np.linspace(float(np.min(x)), float(np.max(x)), 200)
    ax.plot(x_line, slope * x_line + intercept, color="red", linewidth=2)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    add_fit_stats_box(
        ax,
        slope=slope,
        r2=r2,
        through_origin=False,
        intercept=intercept,
    )
    ax.grid(True, linestyle="--", alpha=0.35)


def _scatter_through_origin(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    x_label: str,
    y_label: str,
    title: str,
    extend_xlim: bool = True,
) -> None:
    """Scatter plus through-origin OLS line."""
    slope, r2 = linear_regression_through_origin(x, y)
    ax.scatter(x, y, c="steelblue", s=22, alpha=0.85, edgecolors="k", linewidths=0.35)
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if extend_xlim:
        x_start = min(-10.0, x_min)
        x_end = max(x_max * 1.05, x_max + 1.0)
        ax.set_xlim(x_start, x_end)
    else:
        x_start, x_end = x_min, x_max
    x_line = np.linspace(x_start, x_end, 200)
    ax.plot(x_line, slope * x_line, color="red", linewidth=2)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    add_fit_stats_box(ax, slope=slope, r2=r2, through_origin=True)
    ax.grid(True, linestyle="--", alpha=0.35)


def build_closed_figure(
    records: list[ClosedCellRecord],
    structures_dir: Path,
    out_path: Path,
    *,
    dpi: int = 200,
    example_seed: int = EXAMPLE_SEED_DEFAULT,
) -> None:
    """Four-panel figure for closed cells (experiment 3 analog)."""
    example = record_by_seed(records, example_seed) or records[len(records) // 2]
    grid = np.load(structures_dir / example.structure)
    plot_records = records_for_scatter_and_fit(records)

    x_negr = np.array([r.neg_r for r in plot_records], dtype=float)
    y_b0max = np.array([r.b0_grad_max_level for r in plot_records], dtype=float)
    x_dhalf = np.array([r.d_half for r in plot_records], dtype=float)
    y_mid = np.array([r.midpoint_b0min_b1max_level for r in plot_records], dtype=float)
    x_hhalf = np.array([r.h_half for r in plot_records], dtype=float)
    y_b1min = np.array([r.b1_grad_min_level for r in plot_records], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    ax0, ax1, ax2, ax3 = axes.ravel()

    ax0.imshow(grid, origin="lower", cmap="gray", vmin=0, vmax=1)
    ax0.set_title("Example structure")
    ax0.set_xlabel("x")
    ax0.set_ylabel("y")

    _scatter_through_origin(
        ax1,
        x_negr,
        y_b0max,
        x_label=NEG_R_AXIS_LABEL,
        y_label=B0_GRAD_MAX_LEVEL_LABEL,
        title=B0_GRAD_MAX_VS_NEG_R_TITLE,
    )
    _scatter_through_origin(
        ax2,
        x_dhalf,
        y_mid,
        x_label=D_HALF_AXIS_LABEL,
        y_label=MIDPOINT_B0MIN_B1MAX_LEVEL_LABEL,
        title=MIDPOINT_VS_D_HALF_TITLE,
    )
    _scatter_through_origin(
        ax3,
        x_hhalf,
        y_b1min,
        x_label=H_HALF_AXIS_LABEL,
        y_label=B1_GRAD_MIN_LEVEL_LABEL,
        title=B1_GRAD_MIN_VS_H_HALF_TITLE,
    )

    fig.suptitle("Closed cells: Betti-gradient predictors", y=1.02, fontsize=13)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def build_open_figure(
    records: list[OpenCellRecord],
    structures_dir: Path,
    out_path: Path,
    *,
    dpi: int = 200,
    example_seed: int = EXAMPLE_SEED_DEFAULT,
) -> None:
    """Four-panel figure for open cells (experiment 4 analog)."""
    example = record_by_seed(records, example_seed)
    if example is None:
        example = max(
            records,
            key=lambda row: (
                row.r_std_param / max(row.r_mean_param, 1e-12)
                + row.a_std_param / max(row.a_param, 1e-12)
                + row.w_std_param / max(row.w_param, 1e-12)
            ),
        )
    grid = np.load(structures_dir / example.structure)
    plot_records = records_for_scatter_and_fit(records)

    x_negr = np.array([r.neg_r for r in plot_records], dtype=float)
    y_b0max = np.array([r.b0_grad_max_level for r in plot_records], dtype=float)
    x_negw = np.array([r.neg_w_half for r in plot_records], dtype=float)
    y_mid = np.array([r.midpoint_b0min_b1max_level for r in plot_records], dtype=float)
    x_hhalf = np.array([r.h_half for r in plot_records], dtype=float)
    y_b1min = np.array([r.b1_grad_min_level for r in plot_records], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    ax0, ax1, ax2, ax3 = axes.ravel()

    ax0.imshow(grid, origin="lower", cmap="gray", vmin=0, vmax=1)
    ax0.set_title("Example structure")
    ax0.set_xlabel("x")
    ax0.set_ylabel("y")

    _scatter_through_origin(
        ax1,
        x_negr,
        y_b0max,
        x_label=NEG_R_AXIS_LABEL,
        y_label=B0_GRAD_MAX_LEVEL_LABEL,
        title=B0_GRAD_MAX_VS_NEG_R_TITLE,
    )
    _scatter_through_origin(
        ax2,
        x_negw,
        y_mid,
        x_label=NEG_W_HALF_AXIS_LABEL,
        y_label=MIDPOINT_B0MIN_B1MAX_LEVEL_LABEL,
        title=MIDPOINT_VS_NEG_W_HALF_TITLE,
    )
    _scatter_through_origin(
        ax3,
        x_hhalf,
        y_b1min,
        x_label=H_HALF_AXIS_LABEL,
        y_label=B1_GRAD_MIN_LEVEL_LABEL,
        title=B1_GRAD_MIN_VS_H_HALF_TITLE,
    )

    fig.suptitle(
        f"Open cells: Betti-gradient predictors (persistence $\\geq$ {PERSISTENCE_MIN_OPEN})",
        y=1.02,
        fontsize=13,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def run_closed_pipeline(
    structures_dir: Path,
    data_dir: Path,
    figure_dir: Path,
    *,
    dpi: int = 200,
    example_seed: int = EXAMPLE_SEED_DEFAULT,
) -> Path:
    """Process all closed structures and write CSV + figure."""
    paths = sorted(structures_dir.glob("*.npy"))
    if not paths:
        raise FileNotFoundError(f"No .npy files in {structures_dir}")

    summary_csv = data_dir / "betti_predictors_closed_summary.csv"
    figure_png = figure_dir / "betti_predictors_2D_closed_cells.png"

    records: list[ClosedCellRecord] = []
    n_total = len(paths)
    print(f"[closed] Found {n_total} structures in {structures_dir}", flush=True)

    for index, path in enumerate(paths):
        print(f"[closed {index + 1}/{n_total}] {path.name} ...", flush=True)
        records.append(process_closed_structure(path, index))
        print(f"[closed {index + 1}/{n_total}] done.", flush=True)

    write_summary_csv(records, summary_csv)
    n_excluded = sum(1 for r in records if r.exclude_from_fit)
    plot_n = len(records) - n_excluded
    print(
        f"[closed] Scatter/fit: {plot_n}/{len(records)} structures "
        f"(excluded {n_excluded} with no negative ∇β₁)",
        flush=True,
    )
    build_closed_figure(records, structures_dir, figure_png, dpi=dpi, example_seed=example_seed)
    print(f"[closed] Wrote {summary_csv}", flush=True)
    print(f"[closed] Wrote {figure_png}", flush=True)
    return figure_png


def run_open_pipeline(
    structures_dir: Path,
    data_dir: Path,
    figure_dir: Path,
    *,
    dpi: int = 200,
    example_seed: int = EXAMPLE_SEED_DEFAULT,
) -> Path:
    """Process all open structures and write CSV + figure."""
    paths = sorted(structures_dir.glob("*.npy"))
    if not paths:
        raise FileNotFoundError(f"No .npy files in {structures_dir}")

    summary_csv = data_dir / "betti_predictors_open_summary.csv"
    figure_png = figure_dir / "betti_predictors_2D_open_cells.png"

    records: list[OpenCellRecord] = []
    n_total = len(paths)
    print(f"[open] Found {n_total} structures in {structures_dir}", flush=True)

    for index, path in enumerate(paths):
        print(f"[open {index + 1}/{n_total}] {path.name} ...", flush=True)
        records.append(process_open_structure(path, index))
        print(f"[open {index + 1}/{n_total}] done.", flush=True)

    write_summary_csv(records, summary_csv)
    n_excluded = sum(1 for r in records if r.exclude_from_fit)
    plot_n = len(records) - n_excluded
    print(
        f"[open] Scatter/fit: {plot_n}/{len(records)} structures "
        f"(excluded {n_excluded} with no negative ∇β₁)",
        flush=True,
    )
    build_open_figure(records, structures_dir, figure_png, dpi=dpi, example_seed=example_seed)
    print(f"[open] Wrote {summary_csv}", flush=True)
    print(f"[open] Wrote {figure_png}", flush=True)
    return figure_png
