"""Fault-injection evaluation of the provenance verifier.

Builds a small valid dataset, injects one fault at a time, and records
whether ``verify_dataset.py`` detects it and which check caught it. Each
fault is applied to a fresh copy, so results are independent.

Usage::

    python scripts/fault_injection.py --out fault_matrix.csv

Dataset-level faults only. Code-level faults (global RNG call, unrecorded
configuration change) are covered by ``scripts/fault_injection_code.py``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import shutil
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("verify_dataset", HERE / "verify_dataset.py")
vd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vd)

CELLS = [("G1", "A"), ("G1", "B"), ("G2", "A"), ("G2", "B")]
REPS = 3


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_csv(path, header, rows):
    with Path(path).open("w", newline="") as fh:
        w = csv.writer(fh); w.writerow(header); w.writerows(rows)


def read_rows(path):
    return list(csv.DictReader(Path(path).open()))


def build(root: Path):
    man, att = [], []
    seed = 0
    for gi, (g, s) in enumerate(CELLS):
        for rep in range(1, REPS + 1):
            seed = gi * 1000 + rep
            rid = f"{g}_{s}_{rep:03d}"
            d = root / "runs" / g; d.mkdir(parents=True, exist_ok=True)
            c, j = d / f"{rid}.csv", d / f"{rid}.json"
            c.write_text("timestep,gini_raw,run_id,random_seed\n" +
                         "".join(f"{t},{0.5 + seed / 1e5 + t / 1e4},{rid},{seed}\n" for t in range(1, 6)))
            j.write_text(json.dumps({"run_id": rid, "random_seed": seed,
                                     "scenario_params": {"regeneration_rate": 0.05}}))
            man.append([rid, g, s, seed, 50, False, "", f"runs/{g}/{rid}.csv", c.stat().st_size, 5,
                        sha(c), f"runs/{g}/{rid}.json", sha(j)])
            att.append([rid, g, s, rep, seed, 1, "CONVERGED"])
    write_csv(root / "dataset_manifest.csv",
              ["run_id", "governance_condition", "scenario", "random_seed", "n_agents",
               "collapse_occurred", "collapse_timestep", "csv_path", "csv_bytes", "csv_rows",
               "csv_sha256", "json_path", "json_sha256"], man)
    write_csv(root / "attempts_summary.csv",
              ["run_id", "governance", "scenario", "replication", "seed", "attempts_used", "final_status"], att)
    write_csv(root / "cell_attempt_summary.csv",
              ["governance", "scenario", "total_replications", "total_attempts_across_cell",
               "total_replacement_attempts_across_cell"],
              [[g, s, REPS, REPS, 0] for g, s in CELLS])


def refresh_manifest(root: Path, rid: str):
    rows = read_rows(root / "dataset_manifest.csv")
    for r in rows:
        if r["run_id"] == rid:
            r["csv_sha256"] = sha(root / r["csv_path"]); r["json_sha256"] = sha(root / r["json_path"])
            r["csv_bytes"] = (root / r["csv_path"]).stat().st_size
    write_rows(root / "dataset_manifest.csv", rows)


def write_rows(path, rows):
    with Path(path).open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


# ---------------------------------------------------------------- faults

def f_wrong_run_consistent(root):
    """Replace run G1_A_002 with a copy of G1_A_001 and make manifest hashes consistent."""
    src, dst = "G1_A_001", "G1_A_002"
    for ext in ("csv", "json"):
        shutil.copy(root / f"runs/G1/{src}.{ext}", root / f"runs/G1/{dst}.{ext}")
    refresh_manifest(root, dst)

def f_truncate_summary(root):
    p = root / "attempts_summary.csv"
    p.write_text("\n".join(p.read_text().splitlines()[:5]) + "\n")

def f_drop_run_everywhere(root):
    """Remove one run from files, manifest and attempts summary (roll-up untouched)."""
    rid = "G2_B_003"
    for ext in ("csv", "json"):
        (root / f"runs/G2/{rid}.{ext}").unlink()
    for name in ("dataset_manifest.csv", "attempts_summary.csv"):
        rows = [r for r in read_rows(root / name) if r["run_id"] != rid]
        write_rows(root / name, rows)

def f_drop_run_and_rollup(root):
    """As above, and also edit the roll-up to match (a careful silent drop)."""
    f_drop_run_everywhere(root)
    rows = read_rows(root / "cell_attempt_summary.csv")
    for r in rows:
        if (r["governance"], r["scenario"]) == ("G2", "B"):
            r["total_replications"] = str(REPS - 1); r["total_attempts_across_cell"] = str(REPS - 1)
    write_rows(root / "cell_attempt_summary.csv", rows)

def f_edit_csv_value(root):
    p = root / "runs/G1/G1_A_001.csv"
    p.write_text(p.read_text().replace("0.5", "0.6", 1))

def f_edit_csv_and_manifest(root):
    f_edit_csv_value(root); refresh_manifest(root, "G1_A_001")

def f_config_changed_unrecorded(root):
    """Sidecar claims one regeneration rate; the run actually used another.
    Simulated by editing the sidecar after the fact and refreshing the manifest."""
    p = root / "runs/G1/G1_A_001.json"
    m = json.loads(p.read_text()); m["scenario_params"]["regeneration_rate"] = 0.08
    p.write_text(json.dumps(m)); refresh_manifest(root, "G1_A_001")

def f_wrong_replacement_seed(root):
    p = root / "runs/G1/G1_A_001.json"
    m = json.loads(p.read_text()); m["random_seed"] = 1001
    p.write_text(json.dumps(m)); refresh_manifest(root, "G1_A_001")

def f_seed_collision(root):
    rows = read_rows(root / "attempts_summary.csv"); rows[1]["seed"] = rows[0]["seed"]
    write_rows(root / "attempts_summary.csv", rows)

def f_delete_summary(root):
    (root / "attempts_summary.csv").unlink()

def f_sidecar_runid_swapped(root):
    """Sidecar of G1_A_003 claims to be G1_A_001 (copy-paste error)."""
    p = root / "runs/G1/G1_A_003.json"
    m = json.loads(p.read_text()); m["run_id"] = "G1_A_001"; m["random_seed"] = 1
    p.write_text(json.dumps(m)); refresh_manifest(root, "G1_A_003")

FAULTS = [
    ("Run file internally consistent but wrong (copied from another run, manifest refreshed)", f_wrong_run_consistent),
    ("Batch summary truncated", f_truncate_summary),
    ("Run dropped silently (files, manifest, summary; roll-up untouched)", f_drop_run_everywhere),
    ("Run dropped silently (roll-up edited to match)", f_drop_run_and_rollup),
    ("CSV value edited, manifest not refreshed", f_edit_csv_value),
    ("CSV value edited, manifest refreshed (only detectable with published manifest hash)", f_edit_csv_and_manifest),
    ("Configuration value changed without recording (sidecar edited, manifest refreshed)", f_config_changed_unrecorded),
    ("Replacement seed in sidecar differs from the summary", f_wrong_replacement_seed),
    ("Two runs share a seed", f_seed_collision),
    ("attempts_summary.csv deleted", f_delete_summary),
    ("Sidecar run_id and seed copied from another run", f_sidecar_runid_swapped),
]


def run_verifier(root: Path, anchor=None):
    import sys
    argv = sys.argv; sys.argv = ["verify", "--data_dir", str(root), "--expected_per_cell", str(REPS)] + (["--manifest_sha256", anchor] if anchor else [])
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            code = vd.main()
    finally:
        sys.argv = argv
    out = buf.getvalue()
    caught = [ln.split("]", 1)[1].strip() for ln in out.splitlines()
              if ln.startswith("[FAIL") or ln.startswith("[NOT RUN")]
    return code, caught


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="fault_matrix.csv")
    args = ap.parse_args()
    results = []
    with tempfile.TemporaryDirectory() as t:
        base = Path(t) / "base"; build(base)
        anchor = sha(base / "dataset_manifest.csv")
        code, _ = run_verifier(base, anchor)
        assert code == 0, "baseline dataset must verify cleanly"
        for name, fn in FAULTS:
            w = Path(t) / "work"
            if w.exists(): shutil.rmtree(w)
            shutil.copytree(base, w)
            fn(w)
            code, caught = run_verifier(w, anchor if fn.__name__ == "f_edit_csv_and_manifest" else None)
            results.append({"fault": name, "detected": "yes" if code != 0 else "NO",
                            "exit_code": code, "caught_by": "; ".join(caught) or "-"})
    write_rows(Path(args.out), results)
    for r in results:
        print(f"{r['detected']:3}  {r['fault']}\n       -> {r['caught_by']}")
    missed = [r for r in results if r["detected"] == "NO"]
    print(f"\n{len(results) - len(missed)}/{len(results)} detected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
