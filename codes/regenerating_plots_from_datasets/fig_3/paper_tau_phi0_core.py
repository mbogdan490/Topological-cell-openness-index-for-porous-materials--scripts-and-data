"""Shared in-memory pipeline for paper tau–phi_0 correlation figures."""

from __future__ import annotations

import csv
import gc
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from gudhi import PeriodicCubicalComplex as PCC
from matplotlib.gridspec import GridSpec
from scipy.ndimage import binary_propagation, distance_transform_edt, generate_binary_structure

TAU_FILTRATION_LEVEL = -1.0
PERSISTENCE_MIN_3D = 1.5
METRICS_CSV_HEADER = ("structure", "phi_0", "tau")


@dataclass
class DatasetMetrics:
    """Per-structure metrics computed in memory."""

    phi0: np.ndarray
    tau: np.ndarray
    porosity: np.ndarray


@dataclass
class RootCorrelation:
    """Pearson correlation and Fisher-z confidence interval."""

    k: int
    pearson_r: float
    fisher_z: float
    z_ci95_low: float
    z_ci95_high: float
    r_ci95_low: float
    r_ci95_high: float


def compute_signed_distance_2d(structure: np.ndarray) -> np.ndarray:
    """Signed distance transform for a 2D binary structure (solid +, pore -)."""
    solids = structure != 0
    pores = structure == 0
    dist_to_pore = distance_transform_edt(solids)
    dist_to_solid = distance_transform_edt(pores)
    signed = np.zeros_like(dist_to_pore, dtype=float)
    signed[solids] = dist_to_pore[solids]
    signed[pores] = -dist_to_solid[pores]
    return signed


def compute_signed_distance_3d(structure: np.ndarray) -> np.ndarray:
    """Signed distance transform for a 3D binary structure (solid +, pore -)."""
    solids = structure != 0
    pores = structure == 0
    dist_to_pore = distance_transform_edt(solids)
    dist_to_solid = distance_transform_edt(pores)
    signed = np.zeros_like(dist_to_pore, dtype=float)
    signed[solids] = dist_to_pore[solids]
    signed[pores] = -dist_to_solid[pores]
    return signed


def _persistence_from_signed(signed: np.ndarray, ndim: int) -> list[tuple[int, tuple[float, float]]]:
    """Compute persistence intervals from a signed distance field."""
    periodic = [False] * ndim
    pcc = PCC(top_dimensional_cells=signed.astype(np.float64), periodic_dimensions=periodic)
    max_val = float(np.max(signed))
    out: list[tuple[int, tuple[float, float]]] = []
    for dim, (birth, death) in pcc.persistence():
        if death == float("inf"):
            out.append((dim, (float(birth), max_val)))
        else:
            out.append((dim, (float(birth), float(death))))
    return out


def _filter_persistence(
    persistence: list[tuple[int, tuple[float, float]]],
    *,
    min_persistence: float,
) -> list[tuple[int, tuple[float, float]]]:
    """Keep intervals with sufficient persistence length."""
    return [
        (dim, (birth, death))
        for dim, (birth, death) in persistence
        if (death - birth) >= min_persistence
    ]


def _betti_at_level(
    persistence: list[tuple[int, tuple[float, float]]],
    dim: int,
    level: float = TAU_FILTRATION_LEVEL,
) -> float:
    """Betti number for one homology dimension at a single filtration level."""
    intervals = [pair for d, pair in persistence if d == dim]
    return float(sum(1 for birth, death in intervals if birth <= level < death))


def _tau_at_filtration_level(
    persistence: list[tuple[int, tuple[float, float]]],
    *,
    ndim: int,
    level: float = TAU_FILTRATION_LEVEL,
) -> float:
    """Sponginess index tau at a single filtration level (2D or 3D formula)."""
    b0 = _betti_at_level(persistence, 0, level)
    b1 = _betti_at_level(persistence, 1, level)
    if ndim == 2:
        denom = b0 + b1
        return float(b1 / denom) if denom > 0 else 0.0
    b2 = _betti_at_level(persistence, 2, level)
    denom = b0 + b1 + b2
    return float((b1 + b2) / denom) if denom > 0 else 0.0


def _open_pore_fraction(structure: np.ndarray, ndim: int) -> float:
    """Fraction of pore voxels connected to the domain boundary."""
    pores = structure == 0
    total = int(np.sum(pores))
    if total == 0:
        return 0.0

    boundary = np.zeros_like(pores, dtype=bool)
    for axis in range(ndim):
        sl0 = [slice(None)] * ndim
        sl1 = [slice(None)] * ndim
        sl0[axis] = 0
        sl1[axis] = -1
        boundary[tuple(sl0)] = True
        boundary[tuple(sl1)] = True

    seeds = pores & boundary
    conn = generate_binary_structure(rank=ndim, connectivity=ndim)
    reachable = binary_propagation(seeds, structure=conn, mask=pores)
    return float(np.sum(reachable) / total)


def _init_metrics_csv(csv_path: Path) -> None:
    """Create (or overwrite) a metrics CSV with header row."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        csv.writer(handle).writerow(METRICS_CSV_HEADER)


def _append_metrics_csv(csv_path: Path, structure_name: str, phi0: float, tau: float) -> None:
    """Append one structure's metrics to the CSV."""
    with csv_path.open("a", newline="") as handle:
        csv.writer(handle).writerow([structure_name, phi0, tau])


def _compute_structure_metrics(path_str: str, ndim: int) -> tuple[float, float, float]:
    """Compute (phi_0, tau, porosity) for one structure file (worker-safe)."""
    path = Path(path_str)
    structure = np.load(path)
    if structure.ndim != ndim:
        raise ValueError(f"Expected {ndim}D structure in {path.name}, got shape {structure.shape}")

    signed = compute_signed_distance_2d(structure) if ndim == 2 else compute_signed_distance_3d(structure)
    persistence = _persistence_from_signed(signed, ndim)
    if ndim == 3:
        persistence = _filter_persistence(persistence, min_persistence=PERSISTENCE_MIN_3D)

    tau = _tau_at_filtration_level(persistence, ndim=ndim)
    phi0 = _open_pore_fraction(structure, ndim)
    porosity = float(1.0 - structure.mean())

    del structure, signed, persistence
    gc.collect()
    return phi0, tau, porosity


def _record_structure_metrics(
    csv_path: Path,
    structure_name: str,
    phi0: float,
    tau: float,
    porosity: float,
) -> tuple[float, float, float]:
    """Append metrics to CSV; return scalars kept for downstream aggregation."""
    _append_metrics_csv(csv_path, structure_name, phi0, tau)
    return phi0, tau, porosity


def _process_one_structure_logged(
    index: int,
    path_str: str,
    ndim: int,
    dim_label: str,
    n_total: int,
    csv_path: Path,
) -> tuple[int, float, float, float]:
    """Compute metrics for one structure and print progress (sequential path)."""
    path = Path(path_str)
    prefix = f"[{dim_label} {index}/{n_total}] {path.name}"
    print(f"{prefix}: computing signed distance transform ...", flush=True)
    phi0, tau, porosity = _compute_structure_metrics(path_str, ndim)
    _record_structure_metrics(csv_path, path.name, phi0, tau, porosity)
    print(
        f"{prefix}: signed distance transform done; "
        f"tau (persistence + Betti at level -1) done; "
        f"tau={tau:.6f}, phi_0={phi0:.6f}; saved to {csv_path.name}.",
        flush=True,
    )
    return index, phi0, tau, porosity


def _executor_pool(max_workers: int) -> ProcessPoolExecutor:
    """Process pool; recycle workers after each task on 3.11+ to free memory."""
    if sys.version_info >= (3, 11):
        return ProcessPoolExecutor(max_workers=max_workers, max_tasks_per_child=1)
    return ProcessPoolExecutor(max_workers=max_workers)


def _process_structures_parallel(
    paths: list[Path],
    *,
    ndim: int,
    dim_label: str,
    max_workers: int,
    csv_path: Path,
) -> DatasetMetrics:
    """Process structures with a process pool (intended for heavy 3D runs)."""
    n_total = len(paths)
    print(
        f"[{dim_label}] Using {max_workers} parallel worker process(es) "
        f"({n_total} structures). Metrics CSV: {csv_path}",
        flush=True,
    )

    phi0_vals: list[float] = []
    tau_vals: list[float] = []
    porosity_vals: list[float] = []
    order_index: list[int] = []
    completed = 0

    with _executor_pool(max_workers) as executor:
        futures = {
            executor.submit(_compute_structure_metrics, str(path), ndim): (index, path)
            for index, path in enumerate(paths, start=1)
        }
        for future in as_completed(futures):
            index, path = futures[future]
            phi0, tau, porosity = future.result()
            _record_structure_metrics(csv_path, path.name, phi0, tau, porosity)
            order_index.append(index)
            phi0_vals.append(phi0)
            tau_vals.append(tau)
            porosity_vals.append(porosity)
            completed += 1
            print(
                f"[{dim_label} {completed}/{n_total} finished] {path.name}: "
                f"tau={tau:.6f}, phi_0={phi0:.6f}; saved to {csv_path.name}.",
                flush=True,
            )
            del future

    sorted_rows = sorted(zip(order_index, phi0_vals, tau_vals, porosity_vals, strict=True), key=lambda r: r[0])
    print(f"[{dim_label}] Finished all {n_total} structures.", flush=True)
    return DatasetMetrics(
        phi0=np.array([row[1] for row in sorted_rows], dtype=float),
        tau=np.array([row[2] for row in sorted_rows], dtype=float),
        porosity=np.array([row[3] for row in sorted_rows], dtype=float),
    )


def process_structures_folder(
    structures_dir: Path,
    *,
    ndim: int,
    label: str | None = None,
    max_workers: int = 1,
    metrics_csv: Path,
) -> DatasetMetrics:
    """Run SDT, persistence, tau, and phi_0 for all structures in a folder."""
    paths = sorted(structures_dir.glob("*.npy"))
    if not paths:
        raise FileNotFoundError(f"No .npy structures found in {structures_dir}")

    dim_label = label if label is not None else f"{ndim}D"
    n_total = len(paths)
    _init_metrics_csv(metrics_csv)
    print(
        f"[{dim_label}] Found {n_total} structures in {structures_dir}; "
        f"writing metrics to {metrics_csv}",
        flush=True,
    )

    if max_workers > 1:
        return _process_structures_parallel(
            paths,
            ndim=ndim,
            dim_label=dim_label,
            max_workers=max_workers,
            csv_path=metrics_csv,
        )

    phi0_vals: list[float] = []
    tau_vals: list[float] = []
    porosity_vals: list[float] = []
    for index, path in enumerate(paths, start=1):
        _, phi0, tau, porosity = _process_one_structure_logged(
            index, str(path), ndim, dim_label, n_total, metrics_csv
        )
        phi0_vals.append(phi0)
        tau_vals.append(tau)
        porosity_vals.append(porosity)

    print(f"[{dim_label}] Finished all {n_total} structures.", flush=True)
    return DatasetMetrics(
        phi0=np.array(phi0_vals, dtype=float),
        tau=np.array(tau_vals, dtype=float),
        porosity=np.array(porosity_vals, dtype=float),
    )


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation coefficient."""
    if x.size == 0 or y.size == 0:
        return 0.0
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _fisher_ci(r: float, n: int) -> tuple[float, float, float, float, float]:
    """Return (z, z_lo, z_hi, r_lo, r_hi) for Pearson r with n samples."""
    if n <= 3:
        return 0.0, 0.0, 0.0, r, r
    r_clip = max(min(float(r), 0.999999999999), -0.999999999999)
    z = math.atanh(r_clip)
    se = 1.0 / math.sqrt(n - 3)
    z_lo = z - 1.96 * se
    z_hi = z + 1.96 * se
    return z, z_lo, z_hi, math.tanh(z_lo), math.tanh(z_hi)


def root_correlations(phi0: np.ndarray, tau: np.ndarray, k_max: int = 6) -> list[RootCorrelation]:
    """Pearson r(phi0, tau^(1/k)) for k=1..k_max with Fisher-z 95% CI."""
    n = int(phi0.size)
    rows: list[RootCorrelation] = []
    for k in range(1, k_max + 1):
        y = np.power(np.clip(tau, 0.0, None), 1.0 / k)
        r = _pearson(phi0, y)
        z, z_lo, z_hi, r_lo, r_hi = _fisher_ci(r, n)
        rows.append(
            RootCorrelation(
                k=k,
                pearson_r=r,
                fisher_z=z,
                z_ci95_low=z_lo,
                z_ci95_high=z_hi,
                r_ci95_low=r_lo,
                r_ci95_high=r_hi,
            )
        )
    return rows


def best_k(rows: list[RootCorrelation]) -> int:
    """Return k in 1..6 that maximizes Pearson correlation."""
    return max(rows, key=lambda row: row.pearson_r).k


def build_combined_figure(
    metrics_2d: DatasetMetrics,
    metrics_3d: DatasetMetrics,
    out_path: Path,
    *,
    dpi: int = 220,
) -> tuple[int, int]:
    """Create 2-row paper figure: 2D scatter, 3D scatter, correlation table."""
    rows_2d = root_correlations(metrics_2d.phi0, metrics_2d.tau)
    rows_3d = root_correlations(metrics_3d.phi0, metrics_3d.tau)
    k2d = best_k(rows_2d)
    k3d = best_k(rows_3d)

    y2d = np.power(np.clip(metrics_2d.tau, 0.0, None), 1.0 / k2d)
    y3d = np.power(np.clip(metrics_3d.tau, 0.0, None), 1.0 / k3d)
    r2d = _pearson(metrics_2d.phi0, y2d)
    r3d = _pearson(metrics_3d.phi0, y3d)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(11.5, 8.5))
    gs = GridSpec(
        2,
        2,
        figure=fig,
        height_ratios=[1.0, 1.0],
        width_ratios=[1.35, 1.0],
        hspace=0.34,
        wspace=0.28,
    )

    ax_2d = fig.add_subplot(gs[0, :])
    ax_3d = fig.add_subplot(gs[1, 0])
    ax_tbl = fig.add_subplot(gs[1, 1])

    for ax, phi0, y, r_val, k_opt, label in [
        (ax_2d, metrics_2d.phi0, y2d, r2d, k2d, "2D"),
        (ax_3d, metrics_3d.phi0, y3d, r3d, k3d, "3D"),
    ]:
        ax.scatter(phi0, y, c="steelblue", s=24, alpha=0.85, edgecolors="k", linewidths=0.35)
        ax.set_xlabel(r"$\phi_0$")
        ax.set_ylabel(rf"$\tau^{{1/{k_opt}}}$")
        ax.set_title(rf"{label}: $\phi_0$ vs $\tau^{{1/{k_opt}}}$")
        ax.text(
            0.02,
            0.98,
            rf"Pearson $r={r_val:.3f}$",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
        )
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, linestyle="--", alpha=0.35)

    ax_tbl.axis("off")
    header = ["k", "r (2D)", "95% CI (2D)", "r (3D)", "95% CI (3D)"]
    table_rows = [header]
    for row2d, row3d in zip(rows_2d, rows_3d, strict=True):
        table_rows.append(
            [
                str(row2d.k),
                f"{row2d.pearson_r:.3f}",
                f"[{row2d.r_ci95_low:.3f}, {row2d.r_ci95_high:.3f}]",
                f"{row3d.pearson_r:.3f}",
                f"[{row3d.r_ci95_low:.3f}, {row3d.r_ci95_high:.3f}]",
            ]
        )
    table = ax_tbl.table(
        cellText=table_rows,
        loc="center",
        cellLoc="center",
        colWidths=[0.08, 0.18, 0.28, 0.18, 0.28],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.35)
    for (row_idx, col_idx), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_text_props(weight="bold")
        if col_idx == 0:
            cell.get_text().set_ha("center")

    ax_tbl.set_title(
        r"Pearson correlation between $\phi_0$ and $\tau^{1/k}$",
        pad=12,
    )

    fig.suptitle(
        r"Correlation of openness ($\phi_0$) and sponginess ($\tau^{1/k}$)",
        y=0.98,
        fontsize=12,
    )
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return k2d, k3d
