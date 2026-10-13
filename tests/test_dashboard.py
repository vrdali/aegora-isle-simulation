"""Tests for the dashboard's non-UI layers (spec §7).

``app.py`` itself is thin Streamlit glue and is verified in a browser, not
here. Everything with real logic lives in ``data_access`` and
``process_control``, neither of which imports Streamlit -- so these run
without a server.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import pytest

from aegora_isle.dashboard.data_access import (
    KPI_COLUMNS,
    add_rolling_kpis,
    collapse_timestep,
    compute_progress,
    discover_runs,
    parse_run_id,
    runs_dataframe,
    timestep_stride,
)
from aegora_isle.dashboard.process_control import (
    ProcessHandle,
    build_command,
    is_running,
    pause,
    reset,
    resume,
)


def _write_run(root: Path, run_id: str, gov: str, scen: str, n: int = 200,
               *, collapse_t: int | None = None, with_json: bool = True) -> None:
    d = root / "runs" / gov
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "timestep": np.arange(1, n + 1),
        "gini_coefficient": np.linspace(0.4, 0.6, n),
        "hdi_proxy": np.linspace(0.7, 0.8, n),
        "gdp_equivalent": np.linspace(1.0, 2.0, n),
        "depletion_rate": np.zeros(n),
        "resource_stock": np.linspace(9000, 8000, n),
        "collapse_event": False,
        "governance_condition": gov, "scenario": scen, "run_id": run_id,
        "random_seed": 1, "n_agents": 50,
        "gini_raw": np.linspace(0.4, 0.6, n),
        "gdp_raw": np.full(n, 10.0),
    }).to_csv(d / f"{run_id}.csv", index=False)
    if with_json:
        (d / f"{run_id}.json").write_text(json.dumps({
            "run_id": run_id, "random_seed": 1, "governance_condition": gov,
            "scenario": scen, "n_agents": 50, "scenario_params": {"critical_min": 100.0},
            "collapse_occurred": collapse_t is not None,
            "collapse_timestep": collapse_t,
        }))


# ------------------------------------------------------------ data_access


def test_parse_run_id_matches_runner_format() -> None:
    assert parse_run_id("G1_A_007") == ("G1", "A", 7)
    assert parse_run_id("G3_D_030") == ("G3", "D", 30)


def test_discover_runs_finds_pairs_sorted(tmp_path) -> None:
    _write_run(tmp_path, "G1_A_002", "G1", "A")
    _write_run(tmp_path, "G1_A_001", "G1", "A")
    runs = discover_runs(tmp_path)
    assert [r.run_id for r in runs] == ["G1_A_001", "G1_A_002"]
    assert runs[0].governance == "G1" and runs[0].replication == 1


def test_discover_runs_skips_csv_without_json(tmp_path) -> None:
    """A CSV with no sidecar is a half-written run -- it must not be
    selectable, since the JSON is what makes it reproducible."""
    _write_run(tmp_path, "G1_A_001", "G1", "A", with_json=True)
    _write_run(tmp_path, "G1_A_002", "G1", "A", with_json=False)
    assert [r.run_id for r in discover_runs(tmp_path)] == ["G1_A_001"]


def test_discover_runs_empty_when_no_data_dir(tmp_path) -> None:
    assert discover_runs(tmp_path / "nope") == []


def test_runs_dataframe_joins_status_when_summary_exists(tmp_path) -> None:
    _write_run(tmp_path, "G1_A_001", "G1", "A")
    pd.DataFrame([{
        "run_id": "G1_A_001", "governance": "G1", "scenario": "A",
        "replication": 1, "seed": 1, "attempts_used": 1, "final_status": "CONVERGED",
    }]).to_csv(tmp_path / "attempts_summary.csv", index=False)
    df = runs_dataframe(tmp_path)
    assert df.loc[0, "final_status"] == "CONVERGED"


def test_runs_dataframe_blank_status_mid_run(tmp_path) -> None:
    """attempts_summary.csv only exists once a batch finishes. Mid-run the
    status must be blank, not fabricated."""
    _write_run(tmp_path, "G1_A_001", "G1", "A")
    df = runs_dataframe(tmp_path)
    assert df.loc[0, "final_status"] == ""


def test_add_rolling_kpis_adds_a_column_per_kpi(tmp_path) -> None:
    _write_run(tmp_path, "G1_A_001", "G1", "A", n=100)
    df = pd.read_csv(tmp_path / "runs" / "G1" / "G1_A_001.csv")
    out = add_rolling_kpis(df, window=50)
    for kpi in KPI_COLUMNS:
        assert f"{kpi}_rolling" in out.columns
    assert len(out) == len(df)


def test_rolling_uses_min_periods_so_no_leading_nans(tmp_path) -> None:
    """The chart must start at timestep 1, not at timestep 50."""
    _write_run(tmp_path, "G1_A_001", "G1", "A", n=100)
    df = pd.read_csv(tmp_path / "runs" / "G1" / "G1_A_001.csv")
    out = add_rolling_kpis(df, window=50)
    assert out["gini_coefficient_rolling"].notna().all()


def test_timestep_stride_is_one_at_full_resolution(tmp_path) -> None:
    _write_run(tmp_path, "G1_A_001", "G1", "A", n=100)
    df = pd.read_csv(tmp_path / "runs" / "G1" / "G1_A_001.csv")
    assert timestep_stride(df) == 1


def test_timestep_stride_detects_downsampled_frame() -> None:
    df = pd.DataFrame({"timestep": np.arange(1, 1000, 25)})
    assert timestep_stride(df) == 25


def test_rolling_window_is_timesteps_not_rows(tmp_path) -> None:
    """Spec §7.1 asks for a 50-*step* average. On a stride-25 distribution
    copy, a 50-row window would smooth 1,250 timesteps -- 25x too much."""
    _write_run(tmp_path, "G1_A_001", "G1", "A", n=1000)
    full = pd.read_csv(tmp_path / "runs" / "G1" / "G1_A_001.csv")
    sample = full.iloc[::25].reset_index(drop=True)

    # 50 steps / stride 25 => 2 rows, so an abrupt jump must survive
    # rather than being averaged away across 50 sample rows.
    sample.loc[:, "gini_coefficient"] = 0.0
    sample.loc[20:, "gini_coefficient"] = 1.0
    out = add_rolling_kpis(sample, window=50)
    assert out["gini_coefficient_rolling"].iloc[-1] == pytest.approx(1.0)


def test_rolling_handles_single_row_frame() -> None:
    """A run that collapsed on its first logged step must not divide by zero."""
    df = pd.DataFrame({"timestep": [1], "gini_coefficient": [0.5]})
    out = add_rolling_kpis(df, columns=("gini_coefficient",))
    assert out["gini_coefficient_rolling"].iloc[0] == pytest.approx(0.5)


def test_collapse_timestep_reads_sidecar(tmp_path) -> None:
    _write_run(tmp_path, "G1_A_001", "G1", "A", collapse_t=123)
    meta = json.loads((tmp_path / "runs" / "G1" / "G1_A_001.json").read_text())
    assert collapse_timestep(meta) == 123


def test_collapse_timestep_none_when_no_collapse(tmp_path) -> None:
    _write_run(tmp_path, "G1_A_001", "G1", "A", collapse_t=None)
    meta = json.loads((tmp_path / "runs" / "G1" / "G1_A_001.json").read_text())
    assert collapse_timestep(meta) is None


def test_compute_progress_counts_completed_against_expected(tmp_path) -> None:
    for rep in (1, 2, 3):
        _write_run(tmp_path, f"G1_A_{rep:03d}", "G1", "A")
    p = compute_progress(tmp_path, governance="G1", scenario="A", n_reps=10)
    assert (p.completed, p.total) == (3, 10)
    assert p.fraction == pytest.approx(0.3)


def test_compute_progress_all_expands_to_full_grid(tmp_path) -> None:
    _write_run(tmp_path, "G1_A_001", "G1", "A")
    p = compute_progress(tmp_path, governance="all", scenario="all", n_reps=30)
    assert p.total == 3 * 4 * 30      # 360
    assert p.completed == 1


def test_progress_fraction_safe_when_total_zero(tmp_path) -> None:
    assert compute_progress(tmp_path, "G1", "A", 0).fraction == 0.0


# -------------------------------------------------------- process_control


def test_build_command_uses_current_interpreter_and_module() -> None:
    cmd = build_command({"governance": "G1", "n_reps": 3})
    assert cmd[0] == sys.executable
    assert cmd[1:3] == ["-m", "aegora_isle.experiments.run_experiment"]
    assert "--governance" in cmd and "G1" in cmd
    assert "--n_reps" in cmd and "3" in cmd


def test_build_command_ignores_unknown_keys() -> None:
    """A typo in the UI must not produce an unparseable command line."""
    cmd = build_command({"governance": "G1", "not_a_flag": "boom"})
    assert "--not_a_flag" not in cmd
    assert "boom" not in cmd


def test_build_command_skips_none_values() -> None:
    cmd = build_command({"governance": "G1", "seed_offset": None})
    assert "--seed_offset" not in cmd


class _FakeProc:
    """Minimal psutil.Process stand-in recording what was called on it."""

    def __init__(self, pid: int, children: list["_FakeProc"] | None = None,
                 running: bool = True, status: str = psutil.STATUS_RUNNING) -> None:
        self.pid = pid
        self._children = children or []
        self._running = running
        self._status = status
        self.calls: list[str] = []

    def children(self, recursive: bool = False): return list(self._children)
    def is_running(self): return self._running
    def status(self): return self._status
    def suspend(self): self.calls.append("suspend")
    def resume(self): self.calls.append("resume")
    def terminate(self): self.calls.append("terminate")
    def kill(self): self.calls.append("kill")


def _patch_process(monkeypatch, proc: _FakeProc | None) -> None:
    def factory(pid):
        if proc is None:
            raise psutil.NoSuchProcess(pid)
        return proc
    monkeypatch.setattr(psutil, "Process", factory)


def test_pause_suspends_children_as_well_as_parent(monkeypatch) -> None:
    """run_experiment dispatches through joblib -- suspending only the parent
    would leave the workers computing and the 'paused' state would be false."""
    child = _FakeProc(2)
    parent = _FakeProc(1, children=[child])
    _patch_process(monkeypatch, parent)
    assert pause(ProcessHandle(pid=1)) is True
    assert "suspend" in parent.calls
    assert "suspend" in child.calls


def test_resume_restores_parent_and_children(monkeypatch) -> None:
    child = _FakeProc(2)
    parent = _FakeProc(1, children=[child])
    _patch_process(monkeypatch, parent)
    assert resume(ProcessHandle(pid=1)) is True
    assert "resume" in parent.calls and "resume" in child.calls


def test_reset_resumes_before_terminating(monkeypatch) -> None:
    """A suspended process cannot handle SIGTERM. Without the resume first,
    Pause-then-Reset would strand workers holding CPU."""
    parent = _FakeProc(1)
    _patch_process(monkeypatch, parent)
    monkeypatch.setattr(psutil, "wait_procs", lambda procs, timeout=None: (procs, []))
    assert reset(ProcessHandle(pid=1)) is True
    assert parent.calls.index("resume") < parent.calls.index("terminate")


def test_reset_kills_survivors_of_terminate(monkeypatch) -> None:
    parent = _FakeProc(1)
    _patch_process(monkeypatch, parent)
    monkeypatch.setattr(psutil, "wait_procs", lambda procs, timeout=None: ([], procs))
    reset(ProcessHandle(pid=1))
    assert "kill" in parent.calls


@pytest.mark.parametrize("fn", [pause, resume, reset])
def test_controls_are_noops_on_an_exited_process(monkeypatch, fn) -> None:
    """The user can press Pause after a run finishes on its own -- that must
    be a no-op, not a crash."""
    _patch_process(monkeypatch, None)
    assert fn(ProcessHandle(pid=999999)) is False


def test_is_running_false_for_none_and_for_zombie(monkeypatch) -> None:
    assert is_running(None) is False
    _patch_process(monkeypatch, _FakeProc(1, status=psutil.STATUS_ZOMBIE))
    assert is_running(ProcessHandle(pid=1)) is False
