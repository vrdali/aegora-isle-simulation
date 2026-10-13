"""Convergence diagnostics per spec §6.3 (verbatim-confirmed protocol).

Protocol (pre-registered in the doctoral methodology, spec §6.3 — the
threshold and window "must not be changed", per the spec's own wording):

- Run the first 50 epochs (100 steps/epoch = 5,000 timesteps) unconditionally.
- From epoch 50 onward, compute the rolling variance of policy loss over
  the most recent 20 epochs.
- ``variance < 0.01`` -> CONVERGED; the KPI measurement window begins at
  epoch 51.
- ``variance >= 0.01`` after epoch 50 -> NON_CONVERGED; log to
  ``data/convergence_failures.csv``.

The spec's own wording ("policy loss", singular) most naturally describes
one run-level judgement, but the M2 architecture trains one PPO model per
agent type (Extractor/Producer/Trader) with independent loss curves. This
module tracks convergence **per agent type** (more granular, individually
testable and debuggable) and exposes :meth:`ConvergenceMonitor.all_converged`
for the run-level decision spec §6.3 actually cares about: a run's data is
only trustworthy once *every* trained policy has converged.

No ``numpy.random.Generator`` is needed — like ``kpi_logger.py``, this is
pure bookkeeping over externally-supplied loss floats, nothing stochastic
happens here.
"""

from __future__ import annotations

import csv
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np


# Spec §6.3 constants. The window and minimum-epoch count are pre-registered
# and unchanged. The threshold was **recalibrated by explicit client
# direction** (2026-08-10) -- see below and DECISIONS.md "Convergence
# threshold recalibrated from data". This is NOT a silent re-tune: the
# client asked for an evidence-derived value after the corrected criterion
# proved unable to fail anything.
#
# Original pre-registered value: 0.01. That number was never calibrated
# against this implementation -- measured on the combined loss it ran ~337x
# too tight, and on the spec's actual quantity (policy loss) ~30x too loose.
# Across the full 360-run dataset every one of 921 measurements passed, with
# the maximum observed variance 30x BELOW the threshold.
#
# Recalibrated value derived from that dataset's own distribution using
# Tukey's far fence (Q3 + 3*IQR) -- the conventional definition of an
# extreme outlier:
#     Q3 = 1.143e-04, IQR = 4.53e-05  ->  Q3 + 3*IQR = 2.509e-04  ~ 2.5e-4
#
# Honest caveat recorded with it: the distribution is unimodal with
# max/median = 3.7x and no bimodality, so this flags distributional outliers
# rather than demonstrated training failures. A genuine converged/non-
# converged split would separate by orders of magnitude. See DECISIONS.md.
CONVERGENCE_THRESHOLD: float = 2.5e-4
CONVERGENCE_WINDOW: int = 20          # pre-registered, unchanged
MIN_EPOCHS_BEFORE_CHECK: int = 50     # pre-registered, unchanged

DEFAULT_AGENT_TYPES: tuple[str, ...] = ("extractor", "producer", "trader")

CONVERGENCE_FAILURES_COLUMNS: list[str] = [
    "run_id",
    "governance",
    "scenario",
    "replication",
    "agent_type",
    "max_variance",
]


class ConvergenceStatus(Enum):
    CONVERGING = "CONVERGING"
    CONVERGED = "CONVERGED"
    NON_CONVERGED = "NON_CONVERGED"


class ConvergenceMonitor:
    """Tracks per-agent-type rolling policy-loss variance.

    Args:
        agent_types: The agent types to track. Defaults to the three
            PPO-trained types (Extractor/Producer/Trader) — the G3
            Governor's SAC loss is tracked separately, self-contained
            inside ``AdaptiveRLGovernor`` (Sprint 2.5), not here.
    """

    def __init__(self, agent_types: tuple[str, ...] = DEFAULT_AGENT_TYPES) -> None:
        self._agent_types = tuple(agent_types)
        self._loss_history: dict[str, list[float]] = {t: [] for t in self._agent_types}
        self._measurement_window_start: dict[str, int | None] = {
            t: None for t in self._agent_types
        }

    def _check_known_type(self, agent_type: str) -> None:
        if agent_type not in self._loss_history:
            raise ValueError(
                f"unknown agent_type {agent_type!r}; tracked types: {self._agent_types}"
            )

    def record_epoch_loss(self, agent_type: str, epoch_loss: float) -> None:
        """Append one epoch's loss value (called once per 100-step epoch)."""
        self._check_known_type(agent_type)
        self._loss_history[agent_type].append(float(epoch_loss))

        status = self.check_convergence(agent_type)
        if status is ConvergenceStatus.CONVERGED and self._measurement_window_start[agent_type] is None:
            # First epoch at which convergence was achieved; cached
            # permanently even if a later epoch's status regresses to
            # NON_CONVERGED (the historical fact that this epoch existed
            # doesn't change). Resolving what to do operationally about a
            # run that later un-converges is M3's reseed protocol, out of
            # scope here.
            self._measurement_window_start[agent_type] = len(self._loss_history[agent_type]) + 1

    def current_variance(self, agent_type: str) -> float | None:
        """Rolling variance over the trailing CONVERGENCE_WINDOW epochs.

        None if fewer than CONVERGENCE_WINDOW epochs have been recorded.
        """
        self._check_known_type(agent_type)
        history = self._loss_history[agent_type]
        if len(history) < CONVERGENCE_WINDOW:
            return None
        return float(np.var(history[-CONVERGENCE_WINDOW:]))

    def check_convergence(self, agent_type: str) -> ConvergenceStatus:
        """Current convergence status for one agent type."""
        self._check_known_type(agent_type)
        history = self._loss_history[agent_type]
        if len(history) < MIN_EPOCHS_BEFORE_CHECK:
            return ConvergenceStatus.CONVERGING
        variance = self.current_variance(agent_type)
        assert variance is not None  # guaranteed: len >= 50 > CONVERGENCE_WINDOW
        return (
            ConvergenceStatus.CONVERGED
            if variance < CONVERGENCE_THRESHOLD
            else ConvergenceStatus.NON_CONVERGED
        )

    def measurement_window_start_epoch(self, agent_type: str) -> int | None:
        """First epoch (1-indexed) at which this agent type converged, plus
        one — i.e. 51 if convergence was first achieved at epoch 50. None
        if never yet converged.
        """
        self._check_known_type(agent_type)
        return self._measurement_window_start[agent_type]

    def all_converged(self) -> bool:
        """Run-level judgement: every tracked agent type currently CONVERGED."""
        return all(
            self.check_convergence(t) is ConvergenceStatus.CONVERGED for t in self._agent_types
        )

    def log_failure(
        self,
        agent_type: str,
        run_id: str,
        governance: str,
        scenario: str,
        replication: int,
        path: str | Path = "data/convergence_failures.csv",
    ) -> None:
        """Append one row to the non-convergence log (spec §6.2 reseed protocol).

        Creates the file with a header on first write; appends thereafter.
        """
        variance = self.current_variance(agent_type)
        row: dict[str, Any] = {
            "run_id": run_id,
            "governance": governance,
            "scenario": scenario,
            "replication": replication,
            "agent_type": agent_type,
            "max_variance": variance if variance is not None else "",
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists()
        with path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CONVERGENCE_FAILURES_COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
