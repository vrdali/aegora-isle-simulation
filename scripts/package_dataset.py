"""Packages the confirmatory dataset for distribution (Zenodo / Drive).

Produces three things under ``dist/``:

1. ``aegora_isle_dataset_v1.tar.gz`` -- the complete 360-replication dataset,
   every CSV with its JSON sidecar, plus the batch-level summary tables.
   This is the archival artifact; it is what the DOI should point at.
2. ``dataset_manifest.csv`` -- one row per file: path, bytes, row count,
   SHA-256, and the run's governance/scenario/replication/seed. Ships inside
   the archive *and* alongside it, so a downloader can verify integrity
   without unpacking 460 MB.
3. ``sample/`` -- a downsampled subset small enough to commit to the repo,
   so the dashboard has something to show on a fresh clone (see
   ``--sample-stride`` and the note in the setup guide about what this
   costs in fidelity).

Nothing here reads or writes anything inside ``data/runs/`` except to read
it: the dataset is immutable once collected.

Usage::

    python scripts/package_dataset.py --data_dir data/ --out dist/
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import tarfile
from pathlib import Path

import pandas as pd

# Batch-level artifacts that belong in the archive alongside the runs.
# Absent files are skipped rather than erroring -- not every batch writes
# every diagnostic.
SUMMARY_FILES: tuple[str, ...] = (
    "attempts_summary.csv",
    "cell_attempt_summary.csv",
    "convergence_failures.csv",
    "value_loss_diagnostics.csv",
    "README.md",
)

CHUNK = 1 << 20  # 1 MiB, so hashing a 1.4 MB CSV stays O(1) in memory


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def count_rows(path: Path) -> int:
    """Data rows, excluding the header."""
    with path.open("rb") as fh:
        return max(0, sum(1 for _ in fh) - 1)


def build_manifest(data_dir: Path) -> list[dict[str, object]]:
    """One row per run CSV, with its sidecar's provenance fields folded in."""
    rows: list[dict[str, object]] = []
    for csv_path in sorted((data_dir / "runs").glob("*/*.csv")):
        json_path = csv_path.with_suffix(".json")
        if not json_path.exists():
            # Mirrors the dashboard's rule: a CSV without its sidecar is a
            # half-written run and is not part of the dataset.
            continue
        meta = json.loads(json_path.read_text())
        rows.append({
            "run_id": meta.get("run_id", csv_path.stem),
            "governance_condition": meta.get("governance_condition", ""),
            "scenario": meta.get("scenario", ""),
            "random_seed": meta.get("random_seed", ""),
            "n_agents": meta.get("n_agents", ""),
            "collapse_occurred": meta.get("collapse_occurred", ""),
            "collapse_timestep": meta.get("collapse_timestep", ""),
            "csv_path": csv_path.relative_to(data_dir).as_posix(),
            "csv_bytes": csv_path.stat().st_size,
            "csv_rows": count_rows(csv_path),
            "csv_sha256": sha256_of(csv_path),
            "json_path": json_path.relative_to(data_dir).as_posix(),
            "json_sha256": sha256_of(json_path),
        })
    return rows


def write_manifest(rows: list[dict[str, object]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_sample(data_dir: Path, sample_dir: Path, stride: int) -> tuple[int, int]:
    """Downsampled copy of every run, one row per ``stride``-timestep bin.

    **Bin-averaged, not decimated.** Taking every Nth row aliases: the KPIs
    are spiky at full resolution, and sampling them at 1/25 rate misrepresents
    the ones that spike hardest -- GDP-equivalent showed ~9% mean error that
    way, against ~0.2% for Gini. Averaging within each bin is the correct
    anti-aliased reduction and it is also what the dashboard's own 50-step
    rolling mean is approximating, so the two compose cleanly.

    Column handling follows each column's meaning, not its dtype:
    numeric KPIs average, ``collapse_event`` takes ``any`` (a collapse
    anywhere in the bin is a collapse in that bin), and the run-identity
    columns take the first value, being constant within a run anyway.

    The sidecars are copied verbatim -- they are tiny, and they carry
    ``collapse_timestep``, which the dashboard's collapse marker reads
    directly. So the marker stays exact regardless of the binning.

    Returns (files_written, total_bytes).
    """
    if sample_dir.exists():
        shutil.rmtree(sample_dir)
    written = 0
    total = 0

    identity = ("governance_condition", "scenario", "run_id", "random_seed", "n_agents")

    for csv_path in sorted((data_dir / "runs").glob("*/*.csv")):
        json_path = csv_path.with_suffix(".json")
        if not json_path.exists():
            continue
        dest_dir = sample_dir / "runs" / csv_path.parent.name
        dest_dir.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(csv_path)
        bins = (df["timestep"] - 1) // stride

        agg: dict[str, object] = {"timestep": "last"}
        for col in df.columns:
            if col == "timestep":
                continue
            if col == "collapse_event":
                agg[col] = "any"
            elif col in identity:
                agg[col] = "first"
            else:
                agg[col] = "mean"

        out = df.groupby(bins, sort=True).agg(agg).reset_index(drop=True)
        out = out[df.columns]                      # preserve the locked column order
        dest_csv = dest_dir / csv_path.name
        out.to_csv(dest_csv, index=False)
        shutil.copy2(json_path, dest_dir / json_path.name)

        written += 2
        total += dest_csv.stat().st_size + json_path.stat().st_size

    for name in SUMMARY_FILES:
        src = data_dir / name
        if src.exists():
            shutil.copy2(src, sample_dir / name)
            written += 1
            total += src.stat().st_size
    return written, total


def build_archive(data_dir: Path, manifest_path: Path, out_path: Path) -> None:
    """tar.gz of runs/ + summaries + the manifest, under one top-level dir."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    root = "aegora_isle_dataset"
    with tarfile.open(out_path, "w:gz") as tar:
        tar.add(data_dir / "runs", arcname=f"{root}/runs")
        for name in SUMMARY_FILES:
            src = data_dir / name
            if src.exists():
                tar.add(src, arcname=f"{root}/{name}")
        tar.add(manifest_path, arcname=f"{root}/dataset_manifest.csv")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_dir", default="data/")
    ap.add_argument("--out", default="dist/")
    ap.add_argument("--sample-stride", type=int, default=25,
                    help="Keep every Nth timestep in the committed sample.")
    ap.add_argument("--skip-archive", action="store_true",
                    help="Manifest and sample only (the tar.gz is the slow part).")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out)

    print(f"Scanning {data_dir}/runs ...")
    rows = build_manifest(data_dir)
    if not rows:
        raise SystemExit(f"No run CSV/JSON pairs found under {data_dir}/runs")
    manifest_path = out_dir / "dataset_manifest.csv"
    write_manifest(rows, manifest_path)
    raw_bytes = sum(int(r["csv_bytes"]) for r in rows)
    print(f"  {len(rows)} replications, {raw_bytes / 1048576:.1f} MB of CSV")
    print(f"  manifest -> {manifest_path}")

    print(f"Writing sample (every {args.sample_stride}th timestep) ...")
    n_files, n_bytes = write_sample(data_dir, out_dir / "sample", args.sample_stride)
    print(f"  {n_files} files, {n_bytes / 1048576:.1f} MB -> {out_dir}/sample")

    if not args.skip_archive:
        archive = out_dir / "aegora_isle_dataset_v1.tar.gz"
        print("Building archive (this takes a minute) ...")
        build_archive(data_dir, manifest_path, archive)
        print(f"  {archive.stat().st_size / 1048576:.1f} MB -> {archive}")
        print(f"  sha256 {sha256_of(archive)}")


if __name__ == "__main__":
    main()
