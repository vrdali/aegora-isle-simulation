"""Regression tests for scripts/verify_dataset.py.

Each test builds a tiny, valid dataset, breaks exactly one thing, and checks
that the verifier does not report success. The central case is the one found
in review: deleting attempts_summary.csv used to return success.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "verify_dataset", Path(__file__).resolve().parents[1] / "scripts" / "verify_dataset.py")
vd = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(vd)

RUNS = [("G1", "A", 1, 1), ("G1", "A", 2, 2), ("G2", "A", 1, 31), ("G2", "A", 2, 32)]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _write_csv(path: Path, header: list[str], rows: list[list]):
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


@pytest.fixture
def ds(tmp_path: Path) -> Path:
    man = []
    for gov, sc, rep, seed in RUNS:
        rid = f"{gov}_{sc}_{rep:03d}"
        d = tmp_path / "runs" / gov
        d.mkdir(parents=True, exist_ok=True)
        c, j = d / f"{rid}.csv", d / f"{rid}.json"
        c.write_text("timestep,gini\n25,0.5\n")
        j.write_text(json.dumps({"run_id": rid, "random_seed": seed}))
        man.append([rid, gov, sc, seed, 50, False, "", f"runs/{gov}/{rid}.csv", c.stat().st_size,
                    1, _sha(c), f"runs/{gov}/{rid}.json", _sha(j)])
    _write_csv(tmp_path / "dataset_manifest.csv",
               ["run_id", "governance_condition", "scenario", "random_seed", "n_agents",
                "collapse_occurred", "collapse_timestep", "csv_path", "csv_bytes", "csv_rows",
                "csv_sha256", "json_path", "json_sha256"], man)
    _write_csv(tmp_path / "attempts_summary.csv",
               ["run_id", "governance", "scenario", "replication", "seed", "attempts_used", "final_status"],
               [[f"{g}_{s}_{r:03d}", g, s, r, sd, 1, "CONVERGED"] for g, s, r, sd in RUNS])
    _write_csv(tmp_path / "cell_attempt_summary.csv",
               ["governance", "scenario", "total_replications", "total_attempts_across_cell",
                "total_replacement_attempts_across_cell"],
               [["G1", "A", 2, 2, 0], ["G2", "A", 2, 2, 0]])
    return tmp_path


def run(ds: Path, capsys, monkeypatch) -> tuple[int, str]:
    monkeypatch.setattr("sys.argv", ["verify", "--data_dir", str(ds)])
    code = vd.main()
    return code, capsys.readouterr().out


def test_valid_dataset_passes(ds, capsys, monkeypatch):
    assert run(ds, capsys, monkeypatch)[0] == 0


def test_missing_attempts_summary_is_not_success(ds, capsys, monkeypatch):
    (ds / "attempts_summary.csv").unlink()
    code, out = run(ds, capsys, monkeypatch)
    assert code == 3 and "NOT RUN" in out and "attempts_summary.csv is missing" in out
    assert "every seed agrees" not in out


def test_empty_attempts_summary_is_not_success(ds, capsys, monkeypatch):
    (ds / "attempts_summary.csv").write_text("")
    assert run(ds, capsys, monkeypatch)[0] == 3


def test_header_only_summary_is_not_success(ds, capsys, monkeypatch):
    p = ds / "attempts_summary.csv"
    p.write_text(p.read_text().splitlines()[0] + "\n")
    assert run(ds, capsys, monkeypatch)[0] == 3


def test_summary_missing_required_column(ds, capsys, monkeypatch):
    p = ds / "attempts_summary.csv"
    p.write_text(p.read_text().replace("seed,", "sd,", 1))
    code, out = run(ds, capsys, monkeypatch)
    assert code == 3 and "lacks required column" in out


def test_malformed_seed_value_fails(ds, capsys, monkeypatch):
    p = ds / "attempts_summary.csv"
    p.write_text(p.read_text().replace(",1,1,CONVERGED", ",x,1,CONVERGED", 1))
    assert run(ds, capsys, monkeypatch)[0] == 1


def test_missing_cell_rollup_is_not_success(ds, capsys, monkeypatch):
    (ds / "cell_attempt_summary.csv").unlink()
    assert run(ds, capsys, monkeypatch)[0] == 3


def test_duplicate_run_identifier_fails(ds, capsys, monkeypatch):
    p = ds / "attempts_summary.csv"
    lines = p.read_text().splitlines()
    p.write_text("\n".join(lines + [lines[1]]) + "\n")
    code, out = run(ds, capsys, monkeypatch)
    assert code == 1 and "duplicate run_id" in out


def test_wrong_replacement_seed_fails(ds, capsys, monkeypatch):
    j = ds / "runs" / "G1" / "G1_A_001.json"
    j.write_text(json.dumps({"run_id": "G1_A_001", "random_seed": 1001}))
    man = ds / "dataset_manifest.csv"
    man.write_text(man.read_text().replace(vd.sha256_of(ds / "runs/G1/G1_A_001.json"), "0" * 64))
    code, out = run(ds, capsys, monkeypatch)
    assert code == 1 and "attempts_summary records 1" in out


def test_seed_collision_fails(ds, capsys, monkeypatch):
    p = ds / "attempts_summary.csv"
    p.write_text(p.read_text().replace(",32,1,", ",1,1,"))
    code, out = run(ds, capsys, monkeypatch)
    assert code == 1 and "seed 1 used by" in out


def test_incomplete_coverage_fails(ds, capsys, monkeypatch):
    p = ds / "attempts_summary.csv"
    p.write_text("\n".join(p.read_text().splitlines()[:-1]) + "\n")
    code, out = run(ds, capsys, monkeypatch)
    assert code == 1 and "in manifest, not in attempts_summary" in out


def test_rollup_inconsistent_with_attempts_fails(ds, capsys, monkeypatch):
    p = ds / "attempts_summary.csv"
    p.write_text(p.read_text().replace("G1,A,1,1,1,CONVERGED", "G1,A,1,1,2,CONVERGED"))
    code, out = run(ds, capsys, monkeypatch)
    assert code == 1 and "roll-up says 0 replacements" in out


def test_missing_data_file_fails(ds, capsys, monkeypatch):
    (ds / "runs" / "G2" / "G2_A_002.csv").unlink()
    assert run(ds, capsys, monkeypatch)[0] == 1


def test_corrupt_data_file_fails(ds, capsys, monkeypatch):
    (ds / "runs" / "G2" / "G2_A_002.csv").write_text("tampered\n")
    assert run(ds, capsys, monkeypatch)[0] == 1


def test_missing_manifest_exit_2(tmp_path, capsys, monkeypatch):
    assert run(tmp_path, capsys, monkeypatch)[0] == 2
