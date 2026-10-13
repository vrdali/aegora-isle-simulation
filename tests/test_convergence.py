"""Unit tests for aegora_isle.kpi.convergence.ConvergenceMonitor.

Covers M2 Sprint 2.4 acceptance criteria (AC07):
- Synthetic flat-loss trajectory -> CONVERGED at epoch 50.
- Synthetic noisy trajectory -> NON_CONVERGED.
- Fewer than 50 epochs -> CONVERGING.
- Pre-registered constants (0.01 threshold, 20-epoch window) never drift.
- log_failure writes the expected CSV schema.
"""

from __future__ import annotations

import csv as csv_module

import numpy as np
import pytest

from aegora_isle.kpi.convergence import (
    CONVERGENCE_FAILURES_COLUMNS,
    CONVERGENCE_THRESHOLD,
    CONVERGENCE_WINDOW,
    MIN_EPOCHS_BEFORE_CHECK,
    ConvergenceMonitor,
    ConvergenceStatus,
)


# ------------------------------------------------------------- pre-registered


def test_pre_registered_constants_locked() -> None:
    """Drift detector: these must never change *silently* (CLAUDE.md, spec §6.3).

    The window and minimum-epoch count remain at their pre-registered values.
    The threshold was recalibrated once, on 2026-08-10, by explicit client
    direction after the corrected criterion proved unable to fail anything:
    all 921 measurements in the 360-run dataset passed 0.01, with the maximum
    observed variance 30x below it. The new value is Tukey's far fence
    (Q3 + 3*IQR) over that dataset's own distribution. See DECISIONS.md
    "Convergence threshold recalibrated from data".

    This test's purpose is unchanged -- it still fails loudly on any further
    unannounced edit.
    """
    assert CONVERGENCE_THRESHOLD == 2.5e-4   # was 0.01 pre-2026-08-10
    assert CONVERGENCE_WINDOW == 20          # pre-registered, never changed
    assert MIN_EPOCHS_BEFORE_CHECK == 50     # pre-registered, never changed


# ----------------------------------------------------------------- CONVERGING


def test_converging_before_50_epochs() -> None:
    mon = ConvergenceMonitor(agent_types=("extractor",))
    for _ in range(49):
        mon.record_epoch_loss("extractor", 1.0)
    assert mon.check_convergence("extractor") is ConvergenceStatus.CONVERGING


def test_converging_with_zero_epochs() -> None:
    mon = ConvergenceMonitor(agent_types=("extractor",))
    assert mon.check_convergence("extractor") is ConvergenceStatus.CONVERGING


# ------------------------------------------------------------------ CONVERGED


def test_flat_loss_converges_at_epoch_50() -> None:
    """A perfectly flat trajectory has zero variance -> CONVERGED."""
    mon = ConvergenceMonitor(agent_types=("extractor",))
    for _ in range(50):
        mon.record_epoch_loss("extractor", 0.5)
    assert mon.check_convergence("extractor") is ConvergenceStatus.CONVERGED
    assert mon.measurement_window_start_epoch("extractor") == 51


def test_low_noise_loss_converges() -> None:
    """Small jitter well under the 0.01 variance threshold -> CONVERGED."""
    rng = np.random.default_rng(0)
    mon = ConvergenceMonitor(agent_types=("extractor",))
    for _ in range(50):
        mon.record_epoch_loss("extractor", 0.5 + rng.normal(0, 0.01))
    assert mon.check_convergence("extractor") is ConvergenceStatus.CONVERGED


def test_measurement_window_start_epoch_none_before_convergence() -> None:
    mon = ConvergenceMonitor(agent_types=("extractor",))
    for _ in range(30):
        mon.record_epoch_loss("extractor", 1.0)
    assert mon.measurement_window_start_epoch("extractor") is None


# -------------------------------------------------------------- NON_CONVERGED


def test_noisy_loss_does_not_converge() -> None:
    """High-variance trajectory -> NON_CONVERGED."""
    rng = np.random.default_rng(1)
    mon = ConvergenceMonitor(agent_types=("extractor",))
    for _ in range(50):
        mon.record_epoch_loss("extractor", rng.uniform(0, 10))
    assert mon.check_convergence("extractor") is ConvergenceStatus.NON_CONVERGED
    assert mon.measurement_window_start_epoch("extractor") is None


def test_current_variance_none_before_window_filled() -> None:
    mon = ConvergenceMonitor(agent_types=("extractor",))
    for _ in range(19):
        mon.record_epoch_loss("extractor", 1.0)
    assert mon.current_variance("extractor") is None


def test_current_variance_matches_numpy() -> None:
    mon = ConvergenceMonitor(agent_types=("extractor",))
    values = [float(i) for i in range(25)]
    for v in values:
        mon.record_epoch_loss("extractor", v)
    expected = float(np.var(values[-CONVERGENCE_WINDOW:]))
    assert mon.current_variance("extractor") == pytest.approx(expected)


# --------------------------------------------------------------- rolling window


def test_rolling_window_uses_most_recent_20_only() -> None:
    """A noisy start followed by 20 flat epochs must still converge —
    proves the variance window is trailing, not over the whole history."""
    mon = ConvergenceMonitor(agent_types=("extractor",))
    rng = np.random.default_rng(2)
    for _ in range(30):
        mon.record_epoch_loss("extractor", rng.uniform(0, 10))
    for _ in range(20):
        mon.record_epoch_loss("extractor", 0.5)
    assert mon.check_convergence("extractor") is ConvergenceStatus.CONVERGED


def test_status_can_regress_after_convergence() -> None:
    """Continuous monitoring: a later noisy stretch can flip CONVERGED back
    to NON_CONVERGED, but the historical measurement_window_start_epoch
    is NOT cleared (see ConvergenceMonitor.record_epoch_loss docstring)."""
    mon = ConvergenceMonitor(agent_types=("extractor",))
    for _ in range(50):
        mon.record_epoch_loss("extractor", 0.5)
    assert mon.check_convergence("extractor") is ConvergenceStatus.CONVERGED
    start = mon.measurement_window_start_epoch("extractor")

    rng = np.random.default_rng(3)
    for _ in range(20):
        mon.record_epoch_loss("extractor", rng.uniform(0, 10))
    assert mon.check_convergence("extractor") is ConvergenceStatus.NON_CONVERGED
    assert mon.measurement_window_start_epoch("extractor") == start  # unchanged


# ------------------------------------------------------------ multi-agent-type


def test_independent_tracking_per_agent_type() -> None:
    mon = ConvergenceMonitor(agent_types=("extractor", "producer", "trader"))
    for _ in range(50):
        mon.record_epoch_loss("extractor", 0.5)  # flat -> converges
    rng = np.random.default_rng(4)
    for _ in range(50):
        mon.record_epoch_loss("producer", rng.uniform(0, 10))  # noisy -> doesn't
    for _ in range(30):
        mon.record_epoch_loss("trader", 0.5)  # too few epochs -> converging

    assert mon.check_convergence("extractor") is ConvergenceStatus.CONVERGED
    assert mon.check_convergence("producer") is ConvergenceStatus.NON_CONVERGED
    assert mon.check_convergence("trader") is ConvergenceStatus.CONVERGING
    assert mon.all_converged() is False


def test_all_converged_true_when_every_type_converges() -> None:
    mon = ConvergenceMonitor(agent_types=("extractor", "producer"))
    for _ in range(50):
        mon.record_epoch_loss("extractor", 0.5)
        mon.record_epoch_loss("producer", 0.3)
    assert mon.all_converged() is True


def test_unknown_agent_type_raises() -> None:
    mon = ConvergenceMonitor(agent_types=("extractor",))
    with pytest.raises(ValueError):
        mon.record_epoch_loss("governor", 1.0)
    with pytest.raises(ValueError):
        mon.check_convergence("governor")


# ------------------------------------------------------------------- log_failure


def test_log_failure_writes_expected_schema(tmp_path) -> None:
    mon = ConvergenceMonitor(agent_types=("extractor",))
    rng = np.random.default_rng(5)
    for _ in range(50):
        mon.record_epoch_loss("extractor", rng.uniform(0, 10))

    out_path = tmp_path / "convergence_failures.csv"
    mon.log_failure(
        agent_type="extractor",
        run_id="G2_A_001",
        governance="G2",
        scenario="A",
        replication=1,
        path=out_path,
    )

    with out_path.open() as f:
        reader = csv_module.DictReader(f)
        assert reader.fieldnames == CONVERGENCE_FAILURES_COLUMNS
        rows = list(reader)
    assert len(rows) == 1
    assert rows[0]["run_id"] == "G2_A_001"
    assert rows[0]["governance"] == "G2"
    assert rows[0]["agent_type"] == "extractor"


def test_log_failure_appends_not_overwrites(tmp_path) -> None:
    mon = ConvergenceMonitor(agent_types=("extractor",))
    for _ in range(50):
        mon.record_epoch_loss("extractor", 1.0)
    out_path = tmp_path / "convergence_failures.csv"

    mon.log_failure("extractor", "run_1", "G1", "A", 1, path=out_path)
    mon.log_failure("extractor", "run_2", "G1", "B", 2, path=out_path)

    with out_path.open() as f:
        rows = list(csv_module.DictReader(f))
    assert len(rows) == 2
    assert [r["run_id"] for r in rows] == ["run_1", "run_2"]


def test_log_failure_accepts_arbitrary_agent_type_label(tmp_path) -> None:
    """ConvergenceMonitor and log_failure's CSV schema are fully generic
    over the agent_type label -- which is what lets the observational
    value-loss monitor reuse the class unmodified via distinct
    '{kind}_value' labels (see run_experiment._write_loss_diagnostics).
    Originally written for the since-retired 'extractor_kpi' label; kept
    pointed at a live label so the generic path stays covered."""
    mon = ConvergenceMonitor(agent_types=("extractor_value",))
    rng = np.random.default_rng(6)
    for _ in range(50):
        mon.record_epoch_loss("extractor_value", rng.uniform(0, 5))

    out_path = tmp_path / "convergence_failures.csv"
    mon.log_failure(
        agent_type="extractor_value",
        run_id="G3_B_001",
        governance="G3",
        scenario="B",
        replication=1,
        path=out_path,
    )

    with out_path.open() as f:
        reader = csv_module.DictReader(f)
        assert reader.fieldnames == CONVERGENCE_FAILURES_COLUMNS
        rows = list(reader)
    assert len(rows) == 1
    assert rows[0]["agent_type"] == "extractor_value"
    assert rows[0]["run_id"] == "G3_B_001"
