"""Verifies a downloaded dataset against its manifest.

A truncated or partially-synced download is the most common cause of
charts that look subtly wrong, and it is invisible until someone tries to
explain an anomaly that is really a missing chunk of CSV. This checks every
file's SHA-256 against ``dataset_manifest.csv`` and reports what does not
match.

Usage::

    python scripts/verify_dataset.py --data_dir data/
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

CHUNK = 1 << 20


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def check_seed_consistency(data_dir: Path) -> list[str]:
    """Replications whose sidecar seed disagrees with ``attempts_summary.csv``.

    The retry protocol replaces a non-converged run in place at ``seed+1000``,
    keeping the same ``run_id``. If the replacement does not land, the file
    left on disk is the *excluded* attempt while the summary reports the
    replacement's seed -- a silent mismatch that no checksum can detect,
    because the stale file is internally consistent with itself.

    This happened once in the confirmatory batch (G1_C_001), so it is checked
    rather than assumed.
    """
    summary_path = data_dir / "attempts_summary.csv"
    if not summary_path.exists():
        return []

    seeds: dict[str, int] = {}
    for row in csv.DictReader(summary_path.open()):
        try:
            seeds[row["run_id"]] = int(row["seed"])
        except (KeyError, ValueError):
            continue

    bad: list[str] = []
    missing: list[str] = []
    for json_path in sorted((data_dir / "runs").glob("*/*.json")):
        try:
            meta = json.loads(json_path.read_text())
        except (OSError, ValueError):
            continue
        run_id = meta.get("run_id", json_path.stem)
        if run_id not in seeds:
            # A run with no entry in the attempt log cannot have its
            # provenance checked at all. Silently skipping these once let a
            # summary truncated to a single governance condition pass clean
            # while two thirds of the dataset went unverified.
            missing.append(run_id)
        elif int(meta.get("random_seed", -1)) != seeds[run_id]:
            bad.append(
                f"{run_id}: file has seed {meta.get('random_seed')}, "
                f"attempts_summary records {seeds[run_id]}"
            )

    if missing:
        shown = ", ".join(missing[:5])
        more = f" (and {len(missing) - 5} more)" if len(missing) > 5 else ""
        bad.append(
            f"{len(missing)} run(s) absent from attempts_summary.csv, so their "
            f"provenance could not be checked: {shown}{more}"
        )
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_dir", default="data/")
    ap.add_argument("--manifest", default=None,
                    help="Defaults to <data_dir>/dataset_manifest.csv")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    manifest = Path(args.manifest) if args.manifest else data_dir / "dataset_manifest.csv"

    if not manifest.exists():
        print(f"No manifest at {manifest}.")
        print("It ships inside the dataset archive -- unpack that first.")
        return 2

    rows = list(csv.DictReader(manifest.open()))
    missing: list[str] = []
    corrupt: list[str] = []

    for i, row in enumerate(rows, 1):
        if i % 50 == 0 or i == len(rows):
            print(f"  checked {i}/{len(rows)}", end="\r", flush=True)
        for path_key, hash_key in (("csv_path", "csv_sha256"), ("json_path", "json_sha256")):
            path = data_dir / row[path_key]
            if not path.exists():
                missing.append(row[path_key])
            elif sha256_of(path) != row[hash_key]:
                corrupt.append(row[path_key])

    print(" " * 40, end="\r")
    print(f"Manifest lists {len(rows)} replications ({len(rows) * 2} files).")

    seed_problems = check_seed_consistency(data_dir)

    if not missing and not corrupt and not seed_problems:
        print("All files present, every checksum matches, "
              "and every seed agrees with attempts_summary.csv.")
        return 0

    if seed_problems:
        print(f"\nSEED MISMATCH ({len(seed_problems)}):")
        for p in seed_problems:
            print(f"  {p}")
        print("  A replacement run did not land; the excluded attempt is still "
              "on disk. Re-run that replication at the recorded seed.")

    if missing:
        print(f"\nMISSING ({len(missing)}):")
        for p in missing[:20]:
            print(f"  {p}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")

    if corrupt:
        print(f"\nCHECKSUM MISMATCH ({len(corrupt)}):")
        for p in corrupt[:20]:
            print(f"  {p}")
        if len(corrupt) > 20:
            print(f"  ... and {len(corrupt) - 20} more")

    if missing or corrupt:
        # A seed mismatch is a provenance problem in the source dataset, not a
        # transfer problem -- re-downloading would faithfully reproduce it.
        print("\nRe-download the archive; a partial transfer is the usual cause.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
