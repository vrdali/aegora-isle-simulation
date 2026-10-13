"""Tests for the distribution packaging script.

This script produces the artifact that gets a DOI, so the properties that
matter are: the manifest can actually detect corruption, and the downsampled
sample preserves the schema and the KPI means rather than aliasing them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from package_dataset import build_manifest, sha256_of, write_sample  # noqa: E402

CSV_COLUMNS = [
    "timestep", "gini_coefficient", "hdi_proxy", "gdp_equivalent",
    "depletion_rate", "resource_stock", "collapse_event",
    "governance_condition", "scenario", "run_id", "random_seed", "n_agents",
    "gini_raw", "gdp_raw",
]


def _write_run(root: Path, run_id: str, gov: str, scen: str, n: int = 500,
               *, collapse_at: int | None = None) -> None:
    d = root / "runs" / gov
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    collapse = np.zeros(n, dtype=bool)
    if collapse_at is not None:
        collapse[collapse_at - 1] = True
    pd.DataFrame({
        "timestep": np.arange(1, n + 1),
        "gini_coefficient": np.linspace(0.4, 0.6, n),
        "hdi_proxy": np.linspace(0.7, 0.8, n),
        # Deliberately spiky: this is the column that aliased under decimation.
        "gdp_equivalent": rng.random(n) * 10.0,
        "depletion_rate": np.zeros(n),
        "resource_stock": np.linspace(9000, 8000, n),
        "collapse_event": collapse,
        "governance_condition": gov, "scenario": scen, "run_id": run_id,
        "random_seed": 1, "n_agents": 50,
        "gini_raw": np.linspace(0.4, 0.6, n),
        "gdp_raw": rng.random(n) * 10.0,
    })[CSV_COLUMNS].to_csv(d / f"{run_id}.csv", index=False)
    (d / f"{run_id}.json").write_text(json.dumps({
        "run_id": run_id, "random_seed": 1, "governance_condition": gov,
        "scenario": scen, "n_agents": 50,
        "collapse_occurred": collapse_at is not None,
        "collapse_timestep": collapse_at,
    }))


# ------------------------------------------------------------------ manifest


def test_manifest_has_a_row_per_replication(tmp_path) -> None:
    _write_run(tmp_path, "G1_A_001", "G1", "A")
    _write_run(tmp_path, "G2_B_003", "G2", "B", collapse_at=120)
    rows = build_manifest(tmp_path)
    assert {r["run_id"] for r in rows} == {"G1_A_001", "G2_B_003"}
    assert all(len(r["csv_sha256"]) == 64 for r in rows)


def test_manifest_row_count_excludes_header(tmp_path) -> None:
    _write_run(tmp_path, "G1_A_001", "G1", "A", n=500)
    assert build_manifest(tmp_path)[0]["csv_rows"] == 500


def test_manifest_checksum_changes_when_a_byte_changes(tmp_path) -> None:
    """The whole point of shipping the manifest is detecting a bad download."""
    _write_run(tmp_path, "G1_A_001", "G1", "A")
    csv_path = tmp_path / "runs" / "G1" / "G1_A_001.csv"
    before = sha256_of(csv_path)
    with csv_path.open("a") as fh:
        fh.write("truncated-or-appended\n")
    assert sha256_of(csv_path) != before


def test_manifest_skips_csv_without_sidecar(tmp_path) -> None:
    _write_run(tmp_path, "G1_A_001", "G1", "A")
    _write_run(tmp_path, "G1_A_002", "G1", "A")
    (tmp_path / "runs" / "G1" / "G1_A_002.json").unlink()
    assert [r["run_id"] for r in build_manifest(tmp_path)] == ["G1_A_001"]


# -------------------------------------------------------------------- sample


def test_sample_preserves_the_locked_column_order(tmp_path) -> None:
    """CLAUDE.md forbids renaming or reordering the CSV schema."""
    _write_run(tmp_path, "G1_A_001", "G1", "A")
    write_sample(tmp_path, tmp_path / "sample", stride=25)
    out = pd.read_csv(tmp_path / "sample" / "runs" / "G1" / "G1_A_001.csv")
    assert list(out.columns) == CSV_COLUMNS


def test_sample_row_count_matches_stride(tmp_path) -> None:
    _write_run(tmp_path, "G1_A_001", "G1", "A", n=500)
    write_sample(tmp_path, tmp_path / "sample", stride=25)
    out = pd.read_csv(tmp_path / "sample" / "runs" / "G1" / "G1_A_001.csv")
    assert len(out) == 20


def test_sample_averages_rather_than_decimates(tmp_path) -> None:
    """Decimation aliased GDP by ~9% on real data. Each sample row must be
    the mean of its bin, not the value of one sampled timestep."""
    _write_run(tmp_path, "G1_A_001", "G1", "A", n=500)
    full = pd.read_csv(tmp_path / "runs" / "G1" / "G1_A_001.csv")
    write_sample(tmp_path, tmp_path / "sample", stride=25)
    out = pd.read_csv(tmp_path / "sample" / "runs" / "G1" / "G1_A_001.csv")

    expected_first_bin = full["gdp_equivalent"].iloc[:25].mean()
    assert out["gdp_equivalent"].iloc[0] == pytest.approx(expected_first_bin)
    # And the overall mean survives, which decimation does not guarantee.
    assert out["gdp_equivalent"].mean() == pytest.approx(
        full["gdp_equivalent"].mean(), rel=1e-9)


def test_sample_keeps_a_collapse_flag_inside_its_bin(tmp_path) -> None:
    """A single True in 25 rows must not be averaged away -- the collapse
    flag is the rarest and most consequential signal in the file."""
    _write_run(tmp_path, "G1_A_001", "G1", "A", n=500, collapse_at=120)
    write_sample(tmp_path, tmp_path / "sample", stride=25)
    out = pd.read_csv(tmp_path / "sample" / "runs" / "G1" / "G1_A_001.csv")
    assert bool(out["collapse_event"].any())


def test_sample_copies_sidecars_verbatim(tmp_path) -> None:
    """The dashboard reads collapse_timestep from the sidecar, so the marker
    stays exact no matter how the series is binned."""
    _write_run(tmp_path, "G1_A_001", "G1", "A", n=500, collapse_at=137)
    write_sample(tmp_path, tmp_path / "sample", stride=25)
    meta = json.loads(
        (tmp_path / "sample" / "runs" / "G1" / "G1_A_001.json").read_text())
    assert meta["collapse_timestep"] == 137


def test_sample_run_identity_columns_are_not_averaged(tmp_path) -> None:
    _write_run(tmp_path, "G1_A_001", "G1", "A", n=500)
    write_sample(tmp_path, tmp_path / "sample", stride=25)
    out = pd.read_csv(tmp_path / "sample" / "runs" / "G1" / "G1_A_001.csv")
    assert set(out["run_id"]) == {"G1_A_001"}
    assert set(out["governance_condition"]) == {"G1"}
    assert set(out["n_agents"]) == {50}
