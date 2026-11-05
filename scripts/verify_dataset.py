"""Verifies a dataset against its manifest and its provenance records.

Every check ends in one of three states: PASS, FAIL, or NOT RUN (the records
it needs are missing, empty or malformed). The overall result is success
only when every required check ran and passed. A check that could not be
completed is never reported as a pass (exit code 3).

Exit codes: 0 all checks passed; 1 a check failed; 2 no manifest;
3 no failures found, but at least one required check could not run.


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


PASS, FAIL, NOT_RUN = "PASS", "FAIL", "NOT RUN"

REQUIRED_RECORDS = {
    "attempts_summary.csv": ["run_id", "seed"],
    "cell_attempt_summary.csv": ["governance", "scenario", "total_replications",
                                 "total_replacement_attempts_across_cell"],
}
MANIFEST_COLUMNS = ["run_id", "random_seed", "csv_path", "csv_sha256",
                    "json_path", "json_sha256"]


class Result:
    def __init__(self, name: str, state: str, details: list[str] | None = None):
        self.name, self.state, self.details = name, state, details or []


def read_csv(path: Path, required_cols: list[str]):
    """Return (rows, problem). problem is None when the file is usable."""
    if not path.exists():
        return None, f"{path.name} is missing"
    try:
        text = path.read_text()
    except (OSError, UnicodeDecodeError) as exc:
        return None, f"{path.name} cannot be read ({exc})"
    if not text.strip():
        return None, f"{path.name} is empty"
    try:
        reader = csv.DictReader(text.splitlines())
        rows = list(reader)
        cols = reader.fieldnames or []
    except csv.Error as exc:
        return None, f"{path.name} is malformed ({exc})"
    absent = [c for c in required_cols if c not in cols]
    if absent:
        return None, f"{path.name} lacks required column(s): {', '.join(absent)}"
    if not rows:
        return None, f"{path.name} has a header but no rows"
    return rows, None


def check_checksums(data_dir: Path, manifest_rows) -> Result:
    missing, corrupt = [], []
    for i, row in enumerate(manifest_rows, 1):
        if i % 50 == 0 or i == len(manifest_rows):
            print(f"  checked {i}/{len(manifest_rows)}", end="\r", flush=True)
        for path_key, hash_key in (("csv_path", "csv_sha256"), ("json_path", "json_sha256")):
            path = data_dir / row[path_key]
            if not path.exists():
                missing.append(row[path_key])
            elif sha256_of(path) != row[hash_key]:
                corrupt.append(row[path_key])
    print(" " * 40, end="\r")
    d = [f"missing file: {p}" for p in missing] + [f"checksum mismatch: {p}" for p in corrupt]
    return Result("Files present and checksums match manifest", FAIL if d else PASS, d)


def check_provenance(data_dir: Path, manifest_rows) -> list[Result]:
    out: list[Result] = []
    summary, problem = read_csv(data_dir / "attempts_summary.csv",
                                REQUIRED_RECORDS["attempts_summary.csv"])
    if summary is None:
        for name in ("Run identifiers unique", "Seeds unique across runs",
                     "Sidecar seeds agree with attempts_summary.csv",
                     "Run coverage matches manifest"):
            out.append(Result(name, NOT_RUN, [problem]))
        return out

    # Malformed rows (non-integer seed) are failures, not silently skipped.
    seeds: dict[str, int] = {}
    ids: list[str] = []
    bad_rows: list[str] = []
    for n, row in enumerate(summary, 2):
        rid = (row.get("run_id") or "").strip()
        try:
            seed = int(row["seed"])
        except (KeyError, TypeError, ValueError):
            bad_rows.append(f"row {n}: run_id={rid or '?'} has a non-integer seed")
            continue
        if not rid:
            bad_rows.append(f"row {n}: empty run_id")
            continue
        ids.append(rid)
        seeds[rid] = seed

    dup = sorted({r for r in ids if ids.count(r) > 1})
    out.append(Result("Run identifiers unique",
                      FAIL if dup or bad_rows else PASS,
                      bad_rows + [f"duplicate run_id: {r}" for r in dup]))

    by_seed: dict[int, list[str]] = {}
    for rid, sd in seeds.items():
        by_seed.setdefault(sd, []).append(rid)
    coll = {sd: r for sd, r in by_seed.items() if len(r) > 1}
    out.append(Result("Seeds unique across runs", FAIL if coll else PASS,
                      [f"seed {sd} used by {', '.join(sorted(r))}" for sd, r in sorted(coll.items())]))

    # Sidecar seeds vs the summary (wrong replacement seeds, stale files).
    problems: list[str] = []
    seen_json = 0
    for json_path in sorted((data_dir / "runs").glob("*/*.json")):
        seen_json += 1
        try:
            meta = json.loads(json_path.read_text())
        except (OSError, ValueError):
            problems.append(f"{json_path.name}: sidecar unreadable or malformed")
            continue
        rid = meta.get("run_id", json_path.stem)
        if rid not in seeds:
            problems.append(f"{rid}: absent from attempts_summary.csv (provenance unchecked)")
        elif int(meta.get("random_seed", -1)) != seeds[rid]:
            problems.append(f"{rid}: file has seed {meta.get('random_seed')}, "
                            f"attempts_summary records {seeds[rid]}")
    if seen_json == 0:
        out.append(Result("Sidecar seeds agree with attempts_summary.csv", NOT_RUN,
                          ["no run sidecars found under runs/*/*.json"]))
    else:
        out.append(Result("Sidecar seeds agree with attempts_summary.csv",
                          FAIL if problems else PASS, problems))

    # Coverage: manifest vs summary, in both directions.
    m_ids = {r["run_id"] for r in manifest_rows}
    s_ids = set(ids)
    cov = [f"in manifest, not in attempts_summary: {r}" for r in sorted(m_ids - s_ids)] + \
          [f"in attempts_summary, not in manifest: {r}" for r in sorted(s_ids - m_ids)]
    out.append(Result("Run coverage matches manifest", FAIL if cov else PASS, cov))

    # Replacement accounting: per-cell roll-up vs per-run attempts.
    cells, problem2 = read_csv(data_dir / "cell_attempt_summary.csv",
                               REQUIRED_RECORDS["cell_attempt_summary.csv"])
    name = "Cell roll-up consistent with per-run attempts"
    if cells is None:
        out.append(Result(name, NOT_RUN, [problem2]))
    else:
        try:
            per_cell: dict[tuple, int] = {}
            for row in summary:
                k = (row["governance"], row["scenario"])
                per_cell[k] = per_cell.get(k, 0) + max(int(row["attempts_used"]) - 1, 0)
            d = []
            for row in cells:
                k = (row["governance"], row["scenario"])
                if per_cell.get(k, 0) != int(row["total_replacement_attempts_across_cell"]):
                    d.append(f"{k[0]}/{k[1]}: roll-up says "
                             f"{row['total_replacement_attempts_across_cell']} replacements, "
                             f"per-run records imply {per_cell.get(k, 0)}")
            out.append(Result(name, FAIL if d else PASS, d))
        except (KeyError, ValueError) as exc:
            out.append(Result(name, NOT_RUN, [f"attempt columns unusable ({exc})"]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_dir", default="data/")
    ap.add_argument("--manifest", default=None,
                    help="Defaults to <data_dir>/dataset_manifest.csv")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    manifest = Path(args.manifest) if args.manifest else data_dir / "dataset_manifest.csv"
    rows, problem = read_csv(manifest, MANIFEST_COLUMNS)
    if rows is None:
        print(f"Cannot verify: {problem}.")
        print("The manifest ships inside the dataset archive; unpack that first.")
        return 2

    print(f"Manifest lists {len(rows)} replications ({len(rows) * 2} files).")
    results = [check_checksums(data_dir, rows)] + check_provenance(data_dir, rows)

    print()
    for r in results:
        print(f"[{r.state:7}] {r.name}")
        for d in r.details[:20]:
            print(f"           {d}")
        if len(r.details) > 20:
            print(f"           ... and {len(r.details) - 20} more")
    failed = [r for r in results if r.state == FAIL]
    not_run = [r for r in results if r.state == NOT_RUN]
    print()
    if failed:
        print(f"VERIFICATION FAILED: {len(failed)} check(s) failed"
              + (f", {len(not_run)} could not run." if not_run else "."))
        return 1
    if not_run:
        print(f"VERIFICATION INCOMPLETE: {len(not_run)} required check(s) could not run. "
              "This is not a pass.")
        return 3
    print("All required checks ran and passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
