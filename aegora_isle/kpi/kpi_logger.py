"""KPI computation, CSV writer, and JSON state writer.

Per spec §3/§3.1/§3.2 (verbatim-extracted from the spec docx — see
``DECISIONS.md`` "KPI formulas" for the full derivation and the sign-off
items where the spec under-specifies a sub-formula). Every formula below
is a **pure function** operating on primitives (wealth lists, stock
histories, trade-log dicts) — no class-instance coupling, so each one has
a direct known-input/known-output test in ``tests/test_kpi.py``.

None of these functions take a ``numpy.random.Generator``: KPI computation
is deterministic given state, so — like :class:`G1Handler` — there is no
randomness to seed (CLAUDE.md's seeding contract applies to stochastic
components; this module has none).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


# CSV column order — locked, exact, per CLAUDE.md. Never rename, never add.
CSV_COLUMNS: list[str] = [
    "timestep",
    "gini_coefficient",
    "hdi_proxy",
    "gdp_equivalent",
    "depletion_rate",
    "resource_stock",
    "collapse_event",
    "governance_condition",
    "scenario",
    "run_id",
    "random_seed",
    "n_agents",
    "gini_raw",
    "gdp_raw",
]

# JSON state fields — locked, exact, per spec §3.2 (verbatim-confirmed).
JSON_FIELDS: list[str] = [
    "run_id",
    "random_seed",
    "governance_condition",
    "scenario",
    "governance_params",
    "scenario_params",
    "n_agents",
    "agent_type_distribution",
    "rl_hyperparams",
    "collapse_occurred",
    "collapse_timestep",
    "final_kpi_means",
    "csv_checksum",
]

ROLLING_WINDOW: int = 50
EPOCH_LENGTH: int = 100

# Per DECISIONS.md sign-off: professor's subsistence collapse definition.
DEFAULT_SUBSISTENCE_FRACTION: float = 0.5
DEFAULT_BOTTOM_FRACTION: float = 0.25

# Mirrors aegora_isle.environment.aegora_env.MAX_PRODUCE_AMOUNT.
#
# Deliberately duplicated rather than imported: kpi/ must not depend on
# environment/ (environment/ already depends on kpi/ for live KPI
# injection — importing the other way would be circular). Same pattern as
# rule_based.py's MAX_EXTRACT_AMOUNT duplication. A drift-detector test in
# tests/test_kpi.py asserts these two constants stay equal.
DEFAULT_MAX_PRODUCE_AMOUNT: float = 3.0


# --------------------------------------------------------------- pure formulas


def compute_gini_raw(wealths: Sequence[float]) -> float:
    """Lorenz-curve-area Gini coefficient over a wealth array.

    ``gini = (n + 1 - 2*sum(cumsum(sorted_wealth))/sum(wealth)) / n``.
    Returns 0.0 for an empty array, a single agent, or all-zero wealth
    (no inequality to measure).
    """
    arr = np.sort(np.asarray(wealths, dtype=float))
    n = len(arr)
    total = arr.sum()
    if n == 0 or total == 0:
        return 0.0
    cumsum = np.cumsum(arr)
    return float((n + 1 - 2 * cumsum.sum() / total) / n)


def compute_depletion_rate_raw(stock_t: float, stock_t_minus_1: float) -> float:
    """Per-step percentage depletion: ``(stock_{t-1} - stock_t) / stock_{t-1} * 100``.

    Positive = depletion, negative = regeneration, per the CSV schema's
    Notes column (the sign is flipped relative to the spec's bare
    ``Δ(stock_t - stock_{t-1})`` shorthand to match that explicit
    convention — see DECISIONS.md "Depletion-rate sign convention").
    Returns 0.0 if ``stock_t_minus_1`` is 0 (no prior stock to deplete from).
    """
    if stock_t_minus_1 == 0:
        return 0.0
    return float((stock_t_minus_1 - stock_t) / stock_t_minus_1 * 100.0)


def compute_subsistence_collapse(
    wealths: Sequence[float],
    initial_mean_wealth: float,
    subsistence_fraction: float = DEFAULT_SUBSISTENCE_FRACTION,
    bottom_fraction: float = DEFAULT_BOTTOM_FRACTION,
) -> bool:
    """Professor's subsistence collapse definition (DECISIONS.md, 2026-06-12).

    True when the mean wealth of the bottom ``bottom_fraction`` of agents
    (by wealth) falls below ``subsistence_fraction * initial_mean_wealth``.
    Returns False if there are no agents or ``initial_mean_wealth <= 0``
    (no baseline to compare against).
    """
    arr = np.sort(np.asarray(wealths, dtype=float))
    n = len(arr)
    if n == 0 or initial_mean_wealth <= 0:
        return False
    bottom_n = max(1, int(np.ceil(n * bottom_fraction)))
    bottom_mean = float(arr[:bottom_n].mean())
    threshold = subsistence_fraction * initial_mean_wealth
    return bottom_mean < threshold


def compute_wealth_histogram(wealths: Sequence[float], bins: int = 10) -> list[float]:
    """Population-fraction histogram of wealth, normalised to sum to 1.0.

    Matches :class:`aegora_isle.agents.governor.Governor`'s observation
    bounds (each bin in ``[0, 1]``).
    """
    arr = np.asarray(wealths, dtype=float)
    n = len(arr)
    if n == 0:
        return [0.0] * bins
    upper = float(arr.max()) if arr.max() > 0 else 1.0
    counts, _ = np.histogram(arr, bins=bins, range=(0.0, upper))
    return (counts / n).tolist()


def compute_hdi_proxy(
    resource_access: float,
    capability_utilisation: float,
    agent_longevity: float,
) -> float:
    """HDI proxy per spec §3 (verbatim): linear composite, weights 0.4/0.4/0.2.

    Each sub-term must already be clipped to ``[0, 1]`` by the caller —
    see DECISIONS.md "HDI proxy sub-term operationalisation" for how
    ``resource_access``/``capability_utilisation``/``agent_longevity``
    (named by the spec, not formula-defined by it) are computed from env
    state.
    """
    return float(
        0.4 * resource_access + 0.4 * capability_utilisation + 0.2 * agent_longevity
    )


def compute_resource_access(mean_wealth: float, initial_mean_wealth: float) -> float:
    """HDI sub-term: current mean wealth relative to the run's starting mean."""
    if initial_mean_wealth <= 0:
        return 0.0
    return float(np.clip(mean_wealth / initial_mean_wealth, 0.0, 1.0))


def compute_capability_utilisation(
    cycle_production_value: float,
    n_producers: int,
    max_produce_amount: float,
    mean_price: float,
) -> float:
    """HDI sub-term: actual vs. theoretical-max production value this cycle."""
    theoretical_max = n_producers * max_produce_amount * mean_price
    if theoretical_max <= 0:
        return 0.0
    return float(np.clip(cycle_production_value / theoretical_max, 0.0, 1.0))


def compute_agent_longevity(mean_agent_age: float, current_timestep: int) -> float:
    """HDI sub-term: mean agent age relative to elapsed run time.

    Honest caveat (DECISIONS.md): with no agent-death mechanic implemented,
    this term is ~1.0 for nearly the entire run, since every agent persists
    from ``reset()``. Documented as a known limitation, not a bug.
    """
    if current_timestep <= 0:
        return 0.0
    return float(np.clip(mean_agent_age / current_timestep, 0.0, 1.0))


def _rolling_mean(values: Sequence[float], window: int = ROLLING_WINDOW) -> list[float]:
    """Trailing simple moving average; window adapts near the start of the series."""
    s = pd.Series(values, dtype=float)
    return s.rolling(window=window, min_periods=1).mean().tolist()


# --------------------------------------------------------------------- logger


class KPILogger:
    """Accumulates per-cycle KPI rows and writes the spec-locked CSV + JSON.

    Args:
        env: The :class:`AegoraEnv` instance this logger is attached to.
            Read-only — the logger never mutates the env.
        run_id: Spec format ``{governance}_{scenario}_{replication_index}``,
            e.g. ``"G1_A_001"``. Not derivable from the env; caller-supplied.
        scenario: One of ``"A"``, ``"B"``, ``"C"``, ``"D"``. Not derivable
            from the env (the env only knows raw scenario *parameters*, not
            a letter); caller-supplied.
        subsistence_fraction: See :func:`compute_subsistence_collapse`.
        bottom_fraction: See :func:`compute_subsistence_collapse`.
        max_produce_amount: Must match the env's ``MAX_PRODUCE_AMOUNT``;
            overridable for testing in isolation from the env.
    """

    def __init__(
        self,
        env: Any,
        run_id: str,
        scenario: str,
        subsistence_fraction: float = DEFAULT_SUBSISTENCE_FRACTION,
        bottom_fraction: float = DEFAULT_BOTTOM_FRACTION,
        max_produce_amount: float = DEFAULT_MAX_PRODUCE_AMOUNT,
    ) -> None:
        self.env = env
        self.run_id = run_id
        self.scenario = scenario
        self.subsistence_fraction = subsistence_fraction
        self.bottom_fraction = bottom_fraction
        self.max_produce_amount = max_produce_amount

        self._rows: list[dict[str, Any]] = []
        self._initial_mean_wealth: float | None = None
        self._collapsed_at: int | None = None  # first timestep collapse fired
        self._was_collapsed_last_row: bool = False

    def log_step(self) -> dict[str, Any]:
        """Build and append one CSV row from the env's current state.

        Call once per completed cycle (i.e. once per ``self.env._timestep``
        increment), mirroring how :class:`GovernanceModule.on_step` is
        invoked in ``aegora_env.py``.
        """
        env = self.env
        prices = {item: env.market.get_price(item) for item in env.market.base_prices}
        wealths = [agent.wealth(prices) for agent in env._agents_by_id.values()]

        if self._initial_mean_wealth is None:
            self._initial_mean_wealth = float(np.mean(wealths)) if wealths else 0.0

        gini_raw = compute_gini_raw(wealths)

        collapsed_now = compute_subsistence_collapse(
            wealths,
            self._initial_mean_wealth,
            self.subsistence_fraction,
            self.bottom_fraction,
        )
        # Rising-edge pulse: True only on the timestep collapse is first
        # detected (spec: "True on the timestep... False otherwise").
        collapse_event = collapsed_now and not self._was_collapsed_last_row
        if collapse_event and self._collapsed_at is None:
            self._collapsed_at = env._timestep
        self._was_collapsed_last_row = collapsed_now

        row = {
            "timestep": env._timestep,
            "gini_coefficient": None,  # filled by write_csv's rolling-mean pass
            "hdi_proxy": None,  # filled below if at an epoch boundary, else NaN
            "gdp_equivalent": None,  # filled by write_csv's epoch-0-index pass
            "depletion_rate": None,  # filled by write_csv's rolling-mean pass
            "resource_stock": env.resource_pool.stock,
            "collapse_event": collapse_event,
            "governance_condition": env.governance_module.regime,
            "scenario": self.scenario,
            "run_id": self.run_id,
            "random_seed": env.seed,
            "n_agents": env.n_agents,
            "gini_raw": gini_raw,
            "gdp_raw": getattr(env, "_last_cycle_gdp_raw", 0.0),
        }

        if env._timestep > 0 and env._timestep % EPOCH_LENGTH == 0:
            mean_wealth = float(np.mean(wealths)) if wealths else 0.0
            # Output-goods price only — excludes "raw" (an input cost, not
            # what gets produced) from the theoretical-max production value.
            output_prices = [prices[k] for k in ("food", "tool") if k in prices]
            mean_price = float(np.mean(output_prices)) if output_prices else 0.0
            n_producers = sum(
                1 for aid in env._agents_by_id if aid.startswith("producer")
            )
            resource_access = compute_resource_access(mean_wealth, self._initial_mean_wealth)
            # _last_cycle_production_value (not _cycle_production_value):
            # by the time log_step() is called here, the cycle-boundary
            # reset in AegoraEnv.step() has already zeroed the live
            # accumulator for the next cycle. _last_cycle_production_value
            # is captured just before that reset -- see DECISIONS.md
            # "KPILogger capability_utilisation read-after-reset fix".
            capability_utilisation = compute_capability_utilisation(
                getattr(env, "_last_cycle_production_value", 0.0),
                n_producers,
                self.max_produce_amount,
                mean_price,
            )
            mean_age = float(np.mean([a.age for a in env._agents_by_id.values()]))
            agent_longevity = compute_agent_longevity(mean_age, env._timestep)
            row["hdi_proxy"] = compute_hdi_proxy(
                resource_access, capability_utilisation, agent_longevity
            )
        else:
            row["hdi_proxy"] = float("nan")

        self._rows.append(row)
        return row

    # -------------------------------------------------------------- writers

    def write_csv(self, path: str | Path) -> Path:
        """Apply deferred rolling/indexing passes, then write the locked CSV."""
        path = Path(path)
        df = pd.DataFrame(self._rows)

        df["gini_coefficient"] = _rolling_mean(df["gini_raw"])

        stocks = df["resource_stock"].tolist()
        depletion_raw = [0.0] + [
            compute_depletion_rate_raw(stocks[i], stocks[i - 1])
            for i in range(1, len(stocks))
        ]
        df["depletion_rate"] = _rolling_mean(depletion_raw)

        epoch0 = df[(df["timestep"] >= 0) & (df["timestep"] < EPOCH_LENGTH)]
        baseline = float(epoch0["gdp_raw"].mean()) if len(epoch0) else float("nan")
        if baseline and baseline > 0:
            df["gdp_equivalent"] = df["gdp_raw"] / baseline
        else:
            df["gdp_equivalent"] = float("nan")

        df = df[CSV_COLUMNS]
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        return path

    def write_json_state(
        self,
        csv_path: str | Path,
        governance_params: dict[str, Any],
        scenario_params: dict[str, Any],
        agent_type_distribution: dict[str, float],
        rl_hyperparams: dict[str, Any],
    ) -> Path:
        """Write the spec §3.2 JSON state file (13 fields, incl. CSV SHA-256)."""
        csv_path = Path(csv_path)
        csv_bytes = csv_path.read_bytes()
        checksum = hashlib.sha256(csv_bytes).hexdigest()

        final_kpi_means: dict[str, float] = {}
        if self._rows:
            df = pd.DataFrame(self._rows)
            for col in ("gini_raw", "gdp_raw", "resource_stock"):
                if col in df.columns:
                    final_kpi_means[col] = float(df[col].mean())

        state = {
            "run_id": self.run_id,
            "random_seed": self.env.seed,
            "governance_condition": self.env.governance_module.regime,
            "scenario": self.scenario,
            "governance_params": governance_params,
            "scenario_params": scenario_params,
            "n_agents": self.env.n_agents,
            "agent_type_distribution": agent_type_distribution,
            "rl_hyperparams": rl_hyperparams,
            "collapse_occurred": self._collapsed_at is not None,
            "collapse_timestep": self._collapsed_at,
            "final_kpi_means": final_kpi_means,
            "csv_checksum": checksum,
        }
        assert set(state.keys()) == set(JSON_FIELDS)

        json_path = csv_path.with_suffix(".json")
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(state, indent=2))
        return json_path
