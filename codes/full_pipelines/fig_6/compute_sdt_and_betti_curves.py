#!/usr/bin/env python3
"""Compute 3D signed distance transforms and Betti curves for cropped DRP-317 cubes.

Loads existing ``*_central400.npy`` binaries from ``data/``, writes float64 SDTs to
``data/signed_distance_transforms/``, and saves sublevel-filtration Betti curves
(β₀, β₁, β₂) at integer levels −50 … 100 to ``data/Betti_curves/``.

Persistence is built from the SDT via Gudhi (same convention as
``paper_tau_phi0_core``) with the 3D minimum-persistence filter (1.5).
Betti counts at each level follow ``paper_betti_predictors_core.betti_at_levels``.

Idempotent: skips existing SDT files and complete Betti CSVs (151 data rows)
unless ``--force`` is passed.
"""

from __future__ import annotations

import argparse
import csv
import gc
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Script lives at official_repository/codes/full_pipelines/fig_6/, so
# official_repository == _SCRIPT_DIR.parents[2]. Core helper modules are
# co-located in this folder and imported via the script's own directory.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
_OFFICIAL_REPO = _SCRIPT_DIR.parents[2]
assert _OFFICIAL_REPO.name == "official_repository", _OFFICIAL_REPO
_DATA_DIR = _OFFICIAL_REPO / "datasets" / "fig_6"
_SDT_DIR = _DATA_DIR / "signed_distance_transforms"
_BETTI_DIR = _DATA_DIR / "Betti_curves"

from paper_betti_predictors_core import betti_at_levels  # noqa: E402
from paper_tau_phi0_core import (  # noqa: E402
    PERSISTENCE_MIN_3D,
    TAU_FILTRATION_LEVEL,
    _filter_persistence,
    _open_pore_fraction,
    _persistence_from_signed,
    _tau_at_filtration_level,
    compute_signed_distance_3d,
)

FILTRATION_LEVELS = np.arange(-50, 101, dtype=float)
N_FILTRATION_LEVELS = int(FILTRATION_LEVELS.size)
BETTI_CSV_HEADER = ("filtration_level", "beta_0", "beta_1", "beta_2")
SKIPPED_HEADER = ("sample", "npy_file", "reason")
METRICS_HEADER = ("sample", "npy_file", "phi_0", "tau", "wall_time_s")

# npy stem (without .npy) → human-readable sample label
NPY_STEM_TO_SAMPLE: dict[str, str] = {
    "BanderaBrown_2d25um_binary_central400": "Bandera Brown",
    "BanderaGray_2d25um_binary_central400": "Bandera Gray",
    "BB_2d25um_binary_central400": "Buff Berea",
    "Bentheimer_2d25um_binary_central400": "Bentheimer",
    "Berea_2d25um_binary_central400": "Berea",
    "BSG_2d25um_binary_central400": "BSG",
    "BUG_2d25um_binary_central400": "BUG",
    "CastleGate_2d25um_binary_central400": "Castle Gate",
    "Kirby_2d25um_binary_central400": "Kirby",
    "Leopard_2d25um_binary_central400": "Leopard",
    "Parker_2d25um_binary_central400": "Parker",
}


@dataclass
class SampleResult:
    """Outputs and timing for one processed sample."""

    sample: str
    npy_file: str
    sdt_path: Path
    betti_path: Path
    sdt_bytes: int
    betti_rows: int
    phi_0: float
    tau: float
    wall_seconds: float
    skipped: bool = False


def _betti_csv_slug(sample: str) -> str:
    """Filesystem-safe basename for a Betti CSV from a sample label."""
    return sample.replace(" ", "_")


def _sdt_path_for_npy(npy_path: Path) -> Path:
    """Return the SDT output path for a cropped structure ``.npy`` file."""
    return _SDT_DIR / f"{npy_path.stem}_sdt.npy"


def _betti_path_for_sample(sample: str) -> Path:
    """Return the Betti-curve CSV path for a sample label."""
    return _BETTI_DIR / f"{_betti_csv_slug(sample)}_betti.csv"


def _count_betti_rows(path: Path) -> int:
    """Return the number of data rows in a Betti CSV (0 if missing or empty)."""
    if not path.exists() or path.stat().st_size == 0:
        return 0
    with path.open("r", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)  # header
        return sum(1 for _ in reader)


def _append_skipped(sample: str, npy_file: str, reason: str) -> None:
    """Append one failure row to ``data/skipped.csv``."""
    path = _DATA_DIR / "skipped.csv"
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(SKIPPED_HEADER)
        writer.writerow([sample, npy_file, reason])


def _write_betti_csv(path: Path, levels: np.ndarray, b0: np.ndarray, b1: np.ndarray, b2: np.ndarray) -> None:
    """Write filtration levels and Betti numbers to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(BETTI_CSV_HEADER)
        for lev, v0, v1, v2 in zip(levels, b0, b1, b2, strict=True):
            writer.writerow([int(lev), int(v0), int(v1), int(v2)])


def _append_metrics_row(path: Path, sample: str, npy_file: str, phi_0: float, tau: float, wall_s: float) -> None:
    """Append one row to the optional per-sample metrics CSV."""
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(METRICS_HEADER)
        writer.writerow([sample, npy_file, f"{phi_0:.10f}", f"{tau:.10f}", f"{wall_s:.2f}"])


def _compute_or_load_sdt(
    structure: np.ndarray,
    sdt_path: Path,
    *,
    force: bool,
) -> np.ndarray:
    """Return signed distance field; compute from ``structure`` or load cached SDT."""
    if not force and sdt_path.exists():
        print(f"    [skip SDT] loading {sdt_path.name} (mmap)", flush=True)
        return np.array(np.load(sdt_path, mmap_mode="r"), dtype=np.float64)

    print("    [SDT] computing ...", flush=True)
    t0 = time.monotonic()
    signed = compute_signed_distance_3d(structure)
    print(f"    [SDT] done in {time.monotonic() - t0:.1f} s", flush=True)

    sdt_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(sdt_path, signed.astype(np.float64))
    return signed


def _persistence_and_betti(signed: np.ndarray) -> tuple[list, np.ndarray, np.ndarray, np.ndarray, float]:
    """Filtered persistence, Betti curves, and tau at filtration −1."""
    print("    [persistence] computing ...", flush=True)
    t0 = time.monotonic()
    persistence = _persistence_from_signed(signed, ndim=3)
    persistence = _filter_persistence(persistence, min_persistence=PERSISTENCE_MIN_3D)
    print(
        f"    [persistence] done in {time.monotonic() - t0:.1f} s; "
        f"{len(persistence)} intervals (min pers {PERSISTENCE_MIN_3D})",
        flush=True,
    )

    print(f"    [Betti] evaluating {N_FILTRATION_LEVELS} filtration levels ...", flush=True)
    t0 = time.monotonic()
    b0 = betti_at_levels(persistence, 0, FILTRATION_LEVELS)
    b1 = betti_at_levels(persistence, 1, FILTRATION_LEVELS)
    b2 = betti_at_levels(persistence, 2, FILTRATION_LEVELS)
    tau = _tau_at_filtration_level(persistence, ndim=3, level=TAU_FILTRATION_LEVEL)
    print(f"    [Betti] done in {time.monotonic() - t0:.1f} s; tau={tau:.6f}", flush=True)
    return persistence, b0, b1, b2, tau


def process_sample(
    npy_path: Path,
    *,
    force: bool,
    metrics_csv: Path,
    index: int,
    n_total: int,
) -> SampleResult:
    """Process one cropped structure: SDT, Betti curves, tau, and phi_0."""
    stem = npy_path.stem
    sample = NPY_STEM_TO_SAMPLE.get(stem)
    if sample is None:
        raise KeyError(f"No sample label mapping for {npy_path.name}")

    prefix = f"[{index}/{n_total}] {sample} ({npy_path.name})"
    print(f"{prefix}: start", flush=True)
    t0 = time.monotonic()

    sdt_path = _sdt_path_for_npy(npy_path)
    betti_path = _betti_path_for_sample(sample)
    betti_rows = _count_betti_rows(betti_path)

    if not force and betti_rows == N_FILTRATION_LEVELS and sdt_path.exists():
        print(
            f"{prefix}: complete Betti CSV ({betti_rows} rows) and SDT exist; skipping.",
            flush=True,
        )
        return SampleResult(
            sample=sample,
            npy_file=npy_path.name,
            sdt_path=sdt_path,
            betti_path=betti_path,
            sdt_bytes=sdt_path.stat().st_size,
            betti_rows=betti_rows,
            phi_0=float("nan"),
            tau=float("nan"),
            wall_seconds=time.monotonic() - t0,
        )

    try:
        structure = np.array(np.load(npy_path, mmap_mode="r"), dtype=np.uint8)
        signed = _compute_or_load_sdt(structure, sdt_path, force=force)

        need_betti = force or betti_rows != N_FILTRATION_LEVELS
        if need_betti:
            _, b0, b1, b2, tau = _persistence_and_betti(signed)
            _write_betti_csv(betti_path, FILTRATION_LEVELS, b0, b1, b2)
            betti_rows = N_FILTRATION_LEVELS
            print(f"    [saved] {betti_path}", flush=True)
            del b0, b1, b2
        else:
            print(f"    [skip Betti] {betti_path.name} already has {betti_rows} rows", flush=True)
            _, _, _, _, tau = _persistence_and_betti(signed)

        phi_0 = _open_pore_fraction(structure, ndim=3)
        del structure, signed
        gc.collect()

        wall = time.monotonic() - t0
        _append_metrics_row(metrics_csv, sample, npy_path.name, phi_0, tau, wall)
        print(
            f"{prefix}: done in {wall:.1f} s ({wall / 60:.2f} min); "
            f"tau={tau:.6f}, phi_0={phi_0:.6f}",
            flush=True,
        )
        gc.collect()
        return SampleResult(
            sample=sample,
            npy_file=npy_path.name,
            sdt_path=sdt_path,
            betti_path=betti_path,
            sdt_bytes=sdt_path.stat().st_size,
            betti_rows=betti_rows,
            phi_0=phi_0,
            tau=tau,
            wall_seconds=wall,
        )
    except Exception as exc:  # noqa: BLE001
        reason = f"{type(exc).__name__}: {exc}"
        _append_skipped(sample, npy_path.name, reason)
        print(f"{prefix}: FAILED — {reason}", flush=True)
        gc.collect()
        return SampleResult(
            sample=sample,
            npy_file=npy_path.name,
            sdt_path=sdt_path,
            betti_path=betti_path,
            sdt_bytes=sdt_path.stat().st_size if sdt_path.exists() else 0,
            betti_rows=_count_betti_rows(betti_path),
            phi_0=float("nan"),
            tau=float("nan"),
            wall_seconds=time.monotonic() - t0,
            skipped=True,
        )


def _discover_npy_files() -> list[Path]:
    """Return sorted cropped ``*_central400.npy`` files present in ``data/``."""
    paths = sorted(_DATA_DIR.glob("*_central400.npy"))
    if not paths:
        raise FileNotFoundError(f"No *_central400.npy files in {_DATA_DIR}")
    return paths


def main() -> None:
    """Run SDT + Betti-curve pipeline for all cropped structures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute SDTs and Betti CSVs even when outputs already exist.",
    )
    args = parser.parse_args()

    _SDT_DIR.mkdir(parents=True, exist_ok=True)
    _BETTI_DIR.mkdir(parents=True, exist_ok=True)
    metrics_csv = _DATA_DIR / "sdt_betti_metrics.csv"
    if args.force and metrics_csv.exists():
        metrics_csv.unlink()

    paths = _discover_npy_files()
    n_total = len(paths)
    print(
        f"Found {n_total} cropped structures in {_DATA_DIR}; "
        f"filtration levels {int(FILTRATION_LEVELS[0])}..{int(FILTRATION_LEVELS[-1])} "
        f"({N_FILTRATION_LEVELS} levels)",
        flush=True,
    )

    results: list[SampleResult] = []
    for index, npy_path in enumerate(paths, start=1):
        results.append(
            process_sample(npy_path, force=args.force, metrics_csv=metrics_csv, index=index, n_total=n_total)
        )

    print("\n=== Summary ===", flush=True)
    for row in results:
        status = "FAILED" if row.skipped else "ok"
        print(
            f"  [{status}] {row.sample}: "
            f"SDT {row.sdt_bytes / (1024**2):.1f} MiB, "
            f"Betti {row.betti_rows} rows, "
            f"wall {row.wall_seconds:.1f} s ({row.wall_seconds / 60:.2f} min)",
            flush=True,
        )

    failures = [r for r in results if r.skipped]
    if failures:
        print(f"\n{len(failures)} failure(s) logged to {_DATA_DIR / 'skipped.csv'}", flush=True)
    print(f"Metrics appended to {metrics_csv}", flush=True)


if __name__ == "__main__":
    main()
