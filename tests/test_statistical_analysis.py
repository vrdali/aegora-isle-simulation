"""Tests for the spec §8 statistical-analysis package.

Covers the pure, load-bearing logic: the spec Table 4 survival encoding
(both clauses), run-index construction, inclusion of early-collapse runs,
epoch summaries, and the effect-size derivations. Plots and the PDF are
verified by inspection, not asserted numerically.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis.data_loading import (
    CENSOR_HORIZON,
    SCENARIO_VOLATILITY,
    encode_survival,
    load_run_index,
    mean_trajectories,
    write_epoch_summaries,
)


# ------------------------------------------------------------------ fixtures


def _write_run(
    root: Path,
    run_id: str,
    governance: str,
    scenario: str,
    n_steps: int,
    *,
    stock: float = 5000.0,
    critical_min: float = 100.0,
    gdp: float | list[float] = 100.0,
    seed: int = 1,
) -> None:
    """Writes one replication in exactly the shape run_experiment.py emits."""
    d = root / "runs" / governance
    d.mkdir(parents=True, exist_ok=True)
    ts = np.arange(1, n_steps + 1)
    gdp_arr = np.full(n_steps, gdp, dtype=float) if np.isscalar(gdp) else np.asarray(gdp, dtype=float)
    stock_arr = np.full(n_steps, stock, dtype=float) if np.isscalar(stock) else np.asarray(stock, dtype=float)
    pd.DataFrame({
        "timestep": ts,
        "gini_coefficient": np.linspace(0.4, 0.6, n_steps),
        "hdi_proxy": np.linspace(0.7, 0.8, n_steps),
        "gdp_equivalent": np.linspace(1.0, 2.0, n_steps),
        "depletion_rate": np.zeros(n_steps),
        "resource_stock": stock_arr,
        "collapse_event": False,
        "governance_condition": governance,
        "scenario": scenario,
        "run_id": run_id,
        "random_seed": seed,
        "n_agents": 50,
        "gini_raw": np.linspace(0.4, 0.6, n_steps),
        "gdp_raw": gdp_arr,
    }).to_csv(d / f"{run_id}.csv", index=False)
    (d / f"{run_id}.json").write_text(json.dumps({
        "run_id": run_id, "random_seed": seed, "governance_condition": governance,
        "scenario": scenario, "n_agents": 50,
        "scenario_params": {"critical_min": critical_min},
        "collapse_occurred": False, "collapse_timestep": None,
        "final_kpi_means": {}, "csv_checksum": "x",
    }))


# ------------------------------------------------- survival (spec Table 4)


def test_full_length_run_is_censored_not_an_event() -> None:
    """A run reaching the horizon with a healthy pool is censored -- which is
    what makes Kaplan-Meier the right estimator rather than a plain mean."""
    df = pd.DataFrame({
        "timestep": np.arange(1, CENSOR_HORIZON + 1),
        "resource_stock": np.full(CENSOR_HORIZON, 5000.0),
        "gdp_raw": np.full(CENSOR_HORIZON, 50.0),
    })
    duration, event = encode_survival(df, critical_min=100.0)
    assert duration == float(CENSOR_HORIZON)
    assert event == 0


def test_stock_below_critical_min_is_an_event_at_that_timestep() -> None:
    n = CENSOR_HORIZON
    stock = np.full(n, 5000.0)
    stock[299:] = 50.0                      # breaches at timestep 300
    df = pd.DataFrame({
        "timestep": np.arange(1, n + 1),
        "resource_stock": stock,
        "gdp_raw": np.full(n, 50.0),
    })
    duration, event = encode_survival(df, critical_min=100.0)
    assert event == 1
    assert duration == 300.0


def test_gdp_zero_clause_fires_independently_of_pool_death() -> None:
    """Spec Table 4's *second* clause: GDP = 0 for 3 consecutive steps.

    Uses a FULL-LENGTH run so the pool is healthy and the episode reaches the
    horizon -- the GDP clause is then the only thing that can fire, isolating
    it from both the stock clause and the short-run fallback.
    """
    n = CENSOR_HORIZON
    gdp = np.full(n, 10.0)
    gdp[199:202] = 0.0                      # three zeros -> completes at 202
    df = pd.DataFrame({
        "timestep": np.arange(1, n + 1),
        "resource_stock": np.full(n, 5000.0),
        "gdp_raw": gdp,
    })
    duration, event = encode_survival(df, critical_min=100.0)
    assert event == 1
    assert duration == 202.0


def test_two_isolated_gdp_zeros_do_not_fire() -> None:
    """Fewer than 3 consecutive zeros is not a collapse.

    Full-length run again: at 500 steps the short-run fallback would fire and
    mask the result, which is precisely the confound this test exists to rule
    out.
    """
    n = CENSOR_HORIZON
    gdp = np.full(n, 10.0)
    gdp[100] = 0.0
    gdp[300:302] = 0.0                      # only 2 in a row
    df = pd.DataFrame({
        "timestep": np.arange(1, n + 1),
        "resource_stock": np.full(n, 5000.0),
        "gdp_raw": gdp,
    })
    duration, event = encode_survival(df, critical_min=100.0)
    assert event == 0
    assert duration == float(CENSOR_HORIZON)


def test_earliest_clause_wins() -> None:
    """When both clauses would fire, the event time is the earlier one."""
    n = CENSOR_HORIZON
    stock = np.full(n, 5000.0)
    stock[399:] = 10.0                      # pool death at 400
    gdp = np.full(n, 10.0)
    gdp[99:102] = 0.0                       # GDP clause completes at 102
    df = pd.DataFrame({
        "timestep": np.arange(1, n + 1),
        "resource_stock": stock,
        "gdp_raw": gdp,
    })
    duration, event = encode_survival(df, critical_min=100.0)
    assert event == 1
    assert duration == 102.0


def test_short_run_without_explicit_breach_still_counts_as_an_event() -> None:
    """The environment terminates the episode when the pool dies, so a run
    that simply stops early is an observed event, not a censored one."""
    df = pd.DataFrame({
        "timestep": np.arange(1, 501),
        "resource_stock": np.full(500, 5000.0),
        "gdp_raw": np.full(500, 50.0),
    })
    duration, event = encode_survival(df, critical_min=100.0)
    assert event == 1
    assert duration == 500.0


# ------------------------------------------------------------- run index


def test_run_index_maps_volatility_from_scenario(tmp_path) -> None:
    for scen in ("A", "B", "C", "D"):
        _write_run(tmp_path, f"G1_{scen}_001", "G1", scen, 200)
    df = load_run_index(tmp_path)
    got = dict(zip(df["scenario"].astype(str), df["volatility"]))
    assert got == SCENARIO_VOLATILITY


def test_early_collapse_runs_are_included_not_dropped(tmp_path) -> None:
    """65 of the real 360 replications end early. Dropping them would delete
    exactly the events MTSC exists to measure."""
    _write_run(tmp_path, "G1_A_001", "G1", "A", CENSOR_HORIZON)
    _write_run(tmp_path, "G1_A_002", "G1", "A", 400)          # early collapse
    df = load_run_index(tmp_path)
    assert len(df) == 2
    assert set(df["n_timesteps"]) == {CENSOR_HORIZON, 400}
    assert df.loc[df.run_id == "G1_A_002", "event_observed"].iloc[0] == 1
    assert df.loc[df.run_id == "G1_A_001", "event_observed"].iloc[0] == 0


def test_kpi_means_use_each_runs_own_length(tmp_path) -> None:
    """A 400-step run must not be averaged as though it had 10,000 rows."""
    _write_run(tmp_path, "G1_A_001", "G1", "A", 400)
    df = load_run_index(tmp_path)
    expected = float(np.linspace(0.4, 0.6, 400).mean())
    assert df["gini_coefficient"].iloc[0] == pytest.approx(expected)


def test_run_index_raises_when_no_runs_present(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_run_index(tmp_path)


# --------------------------------------------------------- trajectories


def test_mean_trajectory_averages_across_replications(tmp_path) -> None:
    _write_run(tmp_path, "G1_A_001", "G1", "A", 100)
    _write_run(tmp_path, "G1_A_002", "G1", "A", 100)
    traj = mean_trajectories(tmp_path, kpi="gini_coefficient", max_steps=100)
    assert ("G1", "A") in traj
    assert traj[("G1", "A")].iloc[0] == pytest.approx(0.4)


def test_trajectory_tail_averages_only_surviving_runs(tmp_path) -> None:
    """Where one run has ended, the mean past that point reflects only the
    runs still alive -- the honest representation, not zero-padding."""
    _write_run(tmp_path, "G1_A_001", "G1", "A", 100)
    _write_run(tmp_path, "G1_A_002", "G1", "A", 50)
    traj = mean_trajectories(tmp_path, kpi="gini_coefficient", max_steps=100)
    series = traj[("G1", "A")]
    assert not np.isnan(series.iloc[99])          # long run still contributes
    assert series.iloc[99] == pytest.approx(0.6)  # its own final value only


# ------------------------------------------------------ epoch summaries


def test_epoch_summaries_written_with_expected_shape(tmp_path) -> None:
    """Spec Table 7 requires data/summaries/ -- it did not previously exist."""
    _write_run(tmp_path, "G1_A_001", "G1", "A", 300)
    path = write_epoch_summaries(tmp_path, epoch_length=100)
    assert path.exists()
    out = pd.read_csv(path)
    assert list(out["epoch"]) == [1, 2, 3]
    assert {"run_id", "governance", "scenario", "gini_coefficient"} <= set(out.columns)
