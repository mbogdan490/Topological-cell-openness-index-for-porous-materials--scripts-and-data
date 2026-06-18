#!/usr/bin/env python3
"""End-to-end pipeline: tau vs lab permeability for DRP-317 sandstones.

Phases (all idempotent — completed steps are skipped on re-run):

  A. Verify the 11 DRP-317 download URLs with HEAD requests and write
     ``data/manifest.csv``.
  B. Download each ``*_2d25um_binary.raw`` (1000^3 ``uint8``, ~1 GB each) into
     ``data/`` with curl resume + retry; failures are logged to
     ``data/skipped.csv``.
  C. Optional per-sample lab permeabilities (mD) are joined from
     ``data/permeability.csv`` (columns: ``sample,permeability``) when that
     file exists. If it is absent, the ``permeability`` column in the metrics
     CSV is left blank for every sample and the plot is rendered as a
     "permeability pending" placeholder. The audit trail for the lab values
     lives in ``data/permeability_source.md``.
  D. For each downloaded cube: map bytes to ``{0=pore (black), 1=solid (white)}``
     using ``(raw != 0).astype(uint8)`` (robust to the documented ``{0, 255}``
     convention and the actual DRP-317 ``{0, 1}`` encoding), apply the uniform
     central 400^3 crop documented in ``data/computation_notes.md`` (a full
     1000^3 SDT + persistence run is infeasible on the target machine, and
     400^3 matches the scale used by the in-tree 3D paper datasets), save the
     cropped ``.npy``, then compute the solid volume fraction, sponginess
     index tau, and open-pore fraction phi_0 via the shared core routines in
     the co-located ``paper_tau_phi0_core.py`` (imported from this script's
     own directory via ``sys.path.insert``).
  E. Write ``tau_vs_permeability_metrics.csv`` into the dataset directory and
     ``tau_vs_permeability.png`` (log-scaled permeability scatter with sample
     labels) into ``official_repository/plots/``.

Disk cleanup policy (introduced after the first run; see also
``data/computation_notes.md``): once a sample's cropped ``.npy`` is saved
AND its row has been appended to ``_partial_results.csv`` (the per-sample
crash-safe log written at the project root), the corresponding source
``.raw`` is deleted from ``data/`` to free disk. The sole exception is
``Berea_2d25um_binary.raw``, kept as the validation sample for the raw ->
numpy mapping. The pipeline is resumable: rerunning skips any sample
already present in ``_partial_results.csv``, so deleted ``.raw`` files do
not need to be re-downloaded.
"""

from __future__ import annotations

import csv
import gc
import shutil
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Script lives at official_repository/codes/full_pipelines/fig_6/, so
# official_repository == _SCRIPT_DIR.parents[2]. Core helper modules are
# co-located in this folder and imported via the script's own directory.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
_OFFICIAL_REPO = _SCRIPT_DIR.parents[2]
assert _OFFICIAL_REPO.name == "official_repository", _OFFICIAL_REPO
_DATA_DIR = _OFFICIAL_REPO / "datasets" / "fig_6"
_PLOTS_DIR = _OFFICIAL_REPO / "plots"

from paper_tau_phi0_core import (  # noqa: E402
    PERSISTENCE_MIN_3D,
    _filter_persistence,
    _open_pore_fraction,
    _persistence_from_signed,
    _tau_at_filtration_level,
    compute_signed_distance_3d,
)

NX = NY = NZ = 1000
CROP_SIZE = 400  # uniform central crop applied to every sample
DTYPE = "uint8"
BYTES_PER_VOXEL = 1
EXPECTED_BYTES = NX * NY * NZ * BYTES_PER_VOXEL
MIN_FREE_GB = 15

BASE_URL = "https://web.corral.tacc.utexas.edu/digitalporousmedia/DRP-317"

MANIFEST_HEADER = (
    "sample", "filename", "source_url",
    "nx", "ny", "nz", "dtype", "bytes_per_voxel",
    "expected_size_bytes", "actual_size_bytes",
)
METRICS_HEADER = (
    "sample",
    "reported_porosity",
    "porosity_in_cropped_structure",
    "permeability",
    "tau",
    "volume_fraction",
    "phi_0",
)
SKIPPED_HEADER = ("sample", "filename", "reason")
PARTIAL_HEADER = (
    "sample",
    "reported_porosity",
    "porosity_in_cropped_structure",
    "permeability",
    "filename",
    "tau",
    "volume_fraction",
    "phi_0",
    "shape",
    "wall_time_s",
)

# Always retained as the validation sample for the raw -> numpy mapping.
RAW_TO_ALWAYS_KEEP = "Berea_2d25um_binary.raw"

# Sample manifest with DRP-317 folder/filename layout. Lab permeabilities are
# joined separately from ``data/permeability.csv`` (see Phase C in the
# module docstring); this list intentionally carries no permeability column.
SAMPLES: list[dict[str, str]] = [
    {"sample": "Bandera Gray",  "folder": "Bandera%20Gray",
     "filename": "BanderaGray_2d25um_binary.raw"},
    {"sample": "Parker",        "folder": "Parker",
     "filename": "Parker_2d25um_binary.raw"},
    {"sample": "Kirby",         "folder": "Kirby",
     "filename": "Kirby_2d25um_binary.raw"},
    {"sample": "Bandera Brown", "folder": "Bandera%20Brown",
     "filename": "BanderaBrown_2d25um_binary.raw"},
    {"sample": "BSG",           "folder": "Berea%20Sister%20Gray",
     "filename": "BSG_2d25um_binary.raw"},
    {"sample": "BUG",           "folder": "Berea%20Upper%20Gray",
     "filename": "BUG_2d25um_binary.raw"},
    {"sample": "Berea",         "folder": "Berea",
     "filename": "Berea_2d25um_binary.raw"},
    {"sample": "Castle Gate",   "folder": "CastleGate",
     "filename": "CastleGate_2d25um_binary.raw"},
    {"sample": "Buff Berea",    "folder": "Buff%20Berea",
     "filename": "BB_2d25um_binary.raw"},
    {"sample": "Leopard",       "folder": "Leopard",
     "filename": "Leopard_2d25um_binary.raw"},
    {"sample": "Bentheimer",    "folder": "Bentheimer",
     "filename": "Bentheimer_2d25um_binary.raw"},
]


@dataclass
class ManifestRow:
    """One row of the per-sample manifest."""

    sample: str
    filename: str
    source_url: str
    actual_size_bytes: int = 0
    head_status: int = -1
    head_size: int = 0


@dataclass
class SampleMetrics:
    """Computed per-sample metrics for the scatter plot.

    ``permeability_mD`` defaults to NaN and is only filled in when
    ``data/permeability.csv`` is present (see ``_read_permeability_table``).
    """

    sample: str
    tau: float
    volume_fraction: float
    phi0: float
    permeability_mD: float = float("nan")
    reported_porosity: str = ""
    shape: tuple[int, int, int] = field(default_factory=lambda: (CROP_SIZE,) * 3)
    wall_seconds: float = 0.0


def _free_gb(path: Path) -> float:
    """Return free disk space (GiB) on the filesystem hosting ``path``."""
    return shutil.disk_usage(path).free / (1024**3)


def _head_size(url: str) -> tuple[int, int]:
    """Return (status_code, content_length) for a HEAD request to ``url``."""
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            length = int(resp.headers.get("Content-Length", "0"))
            return resp.status, length
    except Exception as exc:  # noqa: BLE001
        print(f"  HEAD failed for {url}: {exc}", flush=True)
        return -1, 0


def _append_skipped(path: Path, sample: str, filename: str, reason: str) -> None:
    """Append one skipped-sample row, creating the file with a header if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(SKIPPED_HEADER)
        writer.writerow([sample, filename, reason])


def _read_partial(path: Path) -> dict[str, dict[str, str]]:
    """Return ``{sample: row_dict}`` from ``_partial_results.csv`` (empty if absent)."""
    if not path.exists() or path.stat().st_size == 0:
        return {}
    out: dict[str, dict[str, str]] = {}
    with path.open("r", newline="") as handle:
        for row in csv.DictReader(handle):
            sample = (row.get("sample") or "").strip()
            if sample:
                out[sample] = row
    return out


def _read_permeability_table(path: Path) -> tuple[dict[str, float], dict[str, str]]:
    """Return ``({sample: permeability}, {sample: reported_porosity})``.

    Empty dicts if the file is missing or has no parseable rows. Expected
    columns: ``sample,permeability`` (and optionally ``reported_porosity``).
    Permeability values that are blank, non-numeric, or non-positive are
    skipped.
    """
    if not path.exists() or path.stat().st_size == 0:
        return {}, {}
    permeability: dict[str, float] = {}
    reported_porosity: dict[str, str] = {}
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sample = ""
            value = ""
            reported = ""
            for k, v in row.items():
                if k and k.strip().lower() == "sample":
                    sample = (v or "").strip()
                elif k and k.strip().lower() == "permeability":
                    value = (v or "").strip()
                elif k and k.strip().lower() == "reported_porosity":
                    reported = (v or "").strip()
            if not sample or not value:
                continue
            try:
                parsed = float(value)
            except ValueError:
                continue
            if parsed > 0:
                permeability[sample] = parsed
                if reported:
                    reported_porosity[sample] = reported
    return permeability, reported_porosity


def _append_partial(
    path: Path,
    metrics: "SampleMetrics",
    filename: str,
    *,
    permeability_table: dict[str, float] | None = None,
    reported_porosity_table: dict[str, str] | None = None,
) -> None:
    """Append one per-sample row to ``_partial_results.csv`` (creates header)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    permeability_table = permeability_table or {}
    reported_porosity_table = reported_porosity_table or {}
    perm = permeability_table.get(metrics.sample)
    perm_cell = f"{perm:g}" if perm is not None and perm > 0 else ""
    reported = reported_porosity_table.get(metrics.sample, "")
    porosity = 1.0 - metrics.volume_fraction
    with path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(PARTIAL_HEADER)
        writer.writerow([
            metrics.sample,
            reported,
            f"{porosity:.10f}",
            perm_cell,
            filename,
            f"{metrics.tau:.10f}",
            f"{metrics.volume_fraction:.10f}",
            f"{metrics.phi0:.10f}",
            "x".join(str(s) for s in metrics.shape),
            f"{metrics.wall_seconds:.2f}",
        ])


def _delete_raw_after_success(raw_path: Path) -> None:
    """Delete a processed ``.raw`` from ``data/``, except the Berea validator."""
    if raw_path.name == RAW_TO_ALWAYS_KEEP:
        print(
            f"    [keep] {raw_path.name}: retained as raw -> numpy validator.",
            flush=True,
        )
        return
    try:
        raw_path.unlink()
        print(f"    [rm  ] {raw_path.name}: removed after successful processing.",
              flush=True)
    except FileNotFoundError:
        pass


def _write_manifest(rows: list[ManifestRow], path: Path) -> None:
    """Write the per-sample manifest CSV (overwrites any existing file)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(MANIFEST_HEADER)
        for row in rows:
            writer.writerow([
                row.sample, row.filename, row.source_url,
                NX, NY, NZ, DTYPE, BYTES_PER_VOXEL,
                EXPECTED_BYTES,
                row.actual_size_bytes if row.actual_size_bytes else "",
            ])


def phase_a_verify_urls() -> list[ManifestRow]:
    """Phase A: HEAD-check every DRP-317 URL and build manifest rows."""
    print("[Phase A] Verifying DRP-317 download URLs ...", flush=True)
    rows: list[ManifestRow] = []
    for entry in SAMPLES:
        url = f"{BASE_URL}/{entry['folder']}/{entry['filename']}/{entry['filename']}"
        status, size = _head_size(url)
        print(
            f"  {str(entry['sample']):<14} HTTP {status} size={size} url={url}",
            flush=True,
        )
        rows.append(ManifestRow(
            sample=str(entry["sample"]),
            filename=str(entry["filename"]),
            source_url=url,
            head_status=status,
            head_size=size,
        ))
    return rows


def phase_b_download(
    rows: list[ManifestRow],
    skipped_csv: Path,
    completed_samples: set[str],
) -> None:
    """Phase B: Download each verified raw via curl with resume + retry.

    Samples whose metrics already live in ``_partial_results.csv`` are
    skipped: their ``.raw`` has been intentionally deleted by the disk
    cleanup policy and does not need to come back.
    """
    print("[Phase B] Downloading raw cubes (resume-capable) ...", flush=True)
    free = _free_gb(_DATA_DIR)
    if free < MIN_FREE_GB:
        raise RuntimeError(
            f"Insufficient free disk space ({free:.1f} GiB < {MIN_FREE_GB} GiB)."
        )

    for row in rows:
        out_path = _DATA_DIR / row.filename
        if row.sample in completed_samples and not out_path.exists():
            print(
                f"  [skip] {row.filename}: metrics already computed; "
                f"not re-downloading (disk cleanup policy).",
                flush=True,
            )
            continue
        if out_path.exists() and out_path.stat().st_size == EXPECTED_BYTES:
            row.actual_size_bytes = EXPECTED_BYTES
            print(f"  [skip] {row.filename}: already complete.", flush=True)
            continue

        if row.head_status != 200:
            _append_skipped(skipped_csv, row.sample, row.filename,
                            f"HEAD returned {row.head_status}")
            print(f"  [fail] {row.filename}: HEAD {row.head_status}", flush=True)
            continue

        print(f"  [get ] {row.filename}", flush=True)
        cmd = ["curl", "-sSL", "--retry", "3", "-C", "-",
               "-o", str(out_path), row.source_url]
        completed = subprocess.run(cmd, capture_output=True, text=True)
        if completed.returncode != 0:
            _append_skipped(skipped_csv, row.sample, row.filename,
                            f"curl rc={completed.returncode}: "
                            f"{completed.stderr.strip()}")
            print(f"    -> FAILED: {completed.stderr.strip()}", flush=True)
            continue

        actual = out_path.stat().st_size if out_path.exists() else 0
        row.actual_size_bytes = actual
        if actual != EXPECTED_BYTES:
            _append_skipped(skipped_csv, row.sample, row.filename,
                            f"size mismatch: got {actual}, expected {EXPECTED_BYTES}")
            print(f"    -> size mismatch ({actual}).", flush=True)


def _central_crop(volume: np.ndarray, size: int) -> np.ndarray:
    """Return the central ``size^3`` block of a cubic volume."""
    if any(d < size for d in volume.shape):
        raise ValueError(f"Cannot crop {volume.shape} to {size}^3")
    start = [(d - size) // 2 for d in volume.shape]
    sl = tuple(slice(s, s + size) for s in start)
    return volume[sl]


def _load_and_map(raw_path: Path) -> np.ndarray:
    """Load a 1000^3 ``uint8`` cube and map ``black -> 0 (pore), white -> 1 (solid)``.

    DRP-317 binary cubes ship in two possible byte conventions: ``{0, 1}`` or
    ``{0, 255}``. The mandatory final mapping is ``0 = pore (black)``,
    ``1 = solid (white)``. Using ``raw != 0`` collapses any nonzero "white"
    byte to ``1`` while preserving pore voxels at ``0``; this is equivalent to
    the documented ``(raw > 128).astype(uint8)`` for ``{0, 255}`` files and
    the correct identity-style cast for the actual DRP-317 ``{0, 1}`` files
    (where ``> 128`` would incorrectly zero the entire array).
    """
    raw = np.fromfile(raw_path, dtype=np.uint8)
    if raw.size != EXPECTED_BYTES:
        raise ValueError(
            f"Unexpected byte count for {raw_path.name}: "
            f"{raw.size} != {EXPECTED_BYTES}"
        )
    return (raw.reshape((NX, NY, NZ)) != 0).astype(np.uint8)


def _compute_metrics(volume: np.ndarray) -> tuple[float, float, float]:
    """Return (tau, phi_0, volume_fraction_solid) for one binary cube."""
    volume_fraction = float(volume.mean())
    signed = compute_signed_distance_3d(volume)
    persistence = _persistence_from_signed(signed, ndim=3)
    persistence = _filter_persistence(persistence, min_persistence=PERSISTENCE_MIN_3D)
    tau = _tau_at_filtration_level(persistence, ndim=3)
    phi_0 = _open_pore_fraction(volume, ndim=3)
    del signed, persistence
    gc.collect()
    return tau, phi_0, volume_fraction


def phase_d_metrics(
    rows: list[ManifestRow],
    skipped_csv: Path,
    partial_csv: Path,
    *,
    permeability_table: dict[str, float] | None = None,
    reported_porosity_table: dict[str, str] | None = None,
) -> list[SampleMetrics]:
    """Phase D: Run the full SDT + persistence + phi_0 pipeline per sample.

    Resumable. For each sample already present in ``_partial_results.csv``,
    the cached metrics are reused and no .raw is required. For every
    freshly-computed sample, the resulting ``.npy`` is saved, a row is
    appended to ``_partial_results.csv``, and the source ``.raw`` is
    deleted (unless it is the Berea validator).
    """
    print(
        f"[Phase D] Computing metrics on central {CROP_SIZE}^3 crops "
        f"(uniform across all samples).",
        flush=True,
    )

    cached = _read_partial(partial_csv)
    if cached:
        print(
            f"  Resume: {len(cached)} sample(s) already in "
            f"{partial_csv.name}: {sorted(cached)}",
            flush=True,
        )

    results: list[SampleMetrics] = []
    for row in rows:
        cache = cached.get(row.sample)
        if cache is not None:
            try:
                shape_parts = tuple(int(p) for p in cache["shape"].split("x"))
            except Exception:  # noqa: BLE001
                shape_parts = (CROP_SIZE,) * 3
            results.append(SampleMetrics(
                sample=row.sample,
                tau=float(cache["tau"]),
                volume_fraction=float(cache["volume_fraction"]),
                phi0=float(cache["phi_0"]),
                reported_porosity=(cache.get("reported_porosity") or "").strip(),
                shape=shape_parts,
                wall_seconds=float(cache.get("wall_time_s", "0") or 0.0),
            ))
            print(f"  [done] {row.sample}: cached in {partial_csv.name}.", flush=True)
            continue

        raw_path = _DATA_DIR / row.filename
        if not raw_path.exists() or raw_path.stat().st_size != EXPECTED_BYTES:
            _append_skipped(skipped_csv, row.sample, row.filename,
                            "raw missing or wrong size at metrics phase")
            print(f"  [skip] {row.sample}: raw not available.", flush=True)
            continue

        print(f"  [proc] {row.sample}", flush=True)
        t0 = time.monotonic()
        full = _load_and_map(raw_path)
        structure = _central_crop(full, CROP_SIZE).copy()
        del full
        gc.collect()

        npy_path = _DATA_DIR / f"{raw_path.stem}_central{CROP_SIZE}.npy"
        np.save(npy_path, structure)

        try:
            tau, phi_0, vf = _compute_metrics(structure)
        except Exception as exc:  # noqa: BLE001
            _append_skipped(skipped_csv, row.sample, row.filename,
                            f"metrics error: {exc}")
            print(f"    -> ERROR: {exc}", flush=True)
            del structure
            gc.collect()
            continue
        wall = time.monotonic() - t0
        print(
            f"    -> vol_frac(solid)={vf:.4f}  tau={tau:.4f}  "
            f"phi_0={phi_0:.4f}  ({wall/60:.1f} min)",
            flush=True,
        )
        metric = SampleMetrics(
            sample=row.sample,
            tau=tau,
            volume_fraction=vf,
            phi0=phi_0,
            shape=tuple(int(s) for s in structure.shape),
            wall_seconds=wall,
        )
        results.append(metric)
        _append_partial(
            partial_csv,
            metric,
            row.filename,
            permeability_table=permeability_table,
            reported_porosity_table=reported_porosity_table,
        )
        del structure
        gc.collect()
        _delete_raw_after_success(raw_path)
    return results


def _save_placeholder_plot(png_path: Path) -> None:
    """Write a 'permeability pending' placeholder PNG so the file exists."""
    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    ax.text(
        0.5, 0.5,
        "permeability pending\n"
        "Populate data/permeability.csv with columns\n"
        "(sample, permeability) and rerun this script\n"
        "to regenerate the scatter plot.",
        transform=ax.transAxes,
        ha="center", va="center",
        fontsize=12, color="dimgray",
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "lightgray"},
    )
    ax.set_xlabel("Lab permeability [mD]")
    ax.set_ylabel(r"Sponginess index $\tau$")
    ax.set_title(
        rf"DRP-317 sandstones: $\tau$ vs lab permeability "
        rf"(central {CROP_SIZE}$^3$ crop)"
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def phase_e_outputs(
    metrics: list[SampleMetrics],
    csv_path: Path,
    png_path: Path,
) -> None:
    """Phase E: Write the metrics CSV and the tau-vs-permeability scatter plot.

    The ``permeability`` cell is left blank for any sample without a finite,
    positive value in ``data/permeability.csv``. When no samples have a
    plottable permeability, a "permeability pending" placeholder PNG is
    saved so the file always exists.
    """
    print("[Phase E] Writing CSV and figure ...", flush=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(METRICS_HEADER)
        for row in metrics:
            perm_cell = (
                f"{row.permeability_mD:g}"
                if np.isfinite(row.permeability_mD) and row.permeability_mD > 0
                else ""
            )
            porosity = 1.0 - row.volume_fraction
            writer.writerow([
                row.sample,
                row.reported_porosity,
                f"{porosity:.10f}",
                perm_cell,
                f"{row.tau:.10f}",
                f"{row.volume_fraction:.10f}",
                f"{row.phi0:.10f}",
            ])

    plottable = [
        m for m in metrics
        if np.isfinite(m.permeability_mD) and m.permeability_mD > 0
    ]
    if not plottable:
        print(
            "  No permeability values found (data/permeability.csv missing or "
            "empty); writing placeholder plot.",
            flush=True,
        )
        _save_placeholder_plot(png_path)
        return

    xs = np.array([m.permeability_mD for m in plottable], dtype=float)
    ys = np.array([m.tau for m in plottable], dtype=float)
    labels = [m.sample for m in plottable]

    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    ax.scatter(xs, ys, c="steelblue", s=64, alpha=0.9,
               edgecolors="k", linewidths=0.6)
    for x, y, name in zip(xs, ys, labels, strict=True):
        ax.annotate(name, xy=(x, y), xytext=(6, 4),
                    textcoords="offset points", fontsize=9)
    if xs.max() / max(xs.min(), 1e-12) > 10:
        ax.set_xscale("log")
        ax.set_xlabel("Lab permeability [mD] (log scale)")
    else:
        ax.set_xlabel("Lab permeability [mD]")
    ax.set_ylabel(r"Sponginess index $\tau$")
    ax.set_title(
        rf"DRP-317 sandstones: $\tau$ vs lab permeability "
        rf"(central {CROP_SIZE}$^3$ crop)"
    )
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Run phases A through E end-to-end for the DRP-317 figure."""
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    _PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_csv = _DATA_DIR / "manifest.csv"
    skipped_csv = _DATA_DIR / "skipped.csv"
    partial_csv = _DATA_DIR / "_partial_results.csv"
    permeability_csv = _DATA_DIR / "permeability.csv"
    metrics_csv = _DATA_DIR / "tau_vs_permeability_metrics.csv"
    figure_png = _PLOTS_DIR / "tau_vs_permeability.png"

    completed_samples = set(_read_partial(partial_csv).keys())

    rows = phase_a_verify_urls()
    _write_manifest(rows, manifest_csv)

    phase_b_download(rows, skipped_csv, completed_samples)
    for row in rows:
        path = _DATA_DIR / row.filename
        row.actual_size_bytes = path.stat().st_size if path.exists() else 0
    _write_manifest(rows, manifest_csv)

    permeability_table, reported_porosity_table = _read_permeability_table(
        permeability_csv
    )
    metrics = phase_d_metrics(
        rows,
        skipped_csv,
        partial_csv,
        permeability_table=permeability_table,
        reported_porosity_table=reported_porosity_table,
    )

    if permeability_table:
        print(
            f"[Phase C] Joined permeability for {len(permeability_table)} "
            f"sample(s) from {permeability_csv.name}.",
            flush=True,
        )
    else:
        print(
            f"[Phase C] {permeability_csv.name} not found or empty; "
            f"permeability column will be blank.",
            flush=True,
        )
    for metric in metrics:
        metric.permeability_mD = permeability_table.get(
            metric.sample, float("nan")
        )
        if not metric.reported_porosity:
            metric.reported_porosity = reported_porosity_table.get(
                metric.sample, ""
            )

    phase_e_outputs(metrics, metrics_csv, figure_png)

    print(
        f"Done. metrics_csv={metrics_csv}  figure={figure_png}",
        flush=True,
    )


if __name__ == "__main__":
    main()
