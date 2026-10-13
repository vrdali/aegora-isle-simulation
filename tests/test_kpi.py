"""Unit tests for aegora_isle.kpi.kpi_logger.

Covers M2 Sprint 2.3 acceptance criteria (AC05):
- Known-input/known-output tests for every formula, against the
  verbatim-extracted spec §3 text (see M2_PLAN.md / DECISIONS.md).
- CSV schema: exact 14 locked columns, in order.
- JSON schema: exact 12 locked fields.
- Drift detector: kpi_logger's duplicated MAX_PRODUCE_AMOUNT mirrors the
  env's constant (same pattern as rule_based.py's MAX_EXTRACT_AMOUNT test).
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from aegora_isle.environment.aegora_env import MAX_PRODUCE_AMOUNT as ENV_MAX_PRODUCE_AMOUNT
from aegora_isle.kpi.kpi_logger import (
    CSV_COLUMNS,
    DEFAULT_MAX_PRODUCE_AMOUNT,
    JSON_FIELDS,
    KPILogger,
    compute_agent_longevity,
    compute_capability_utilisation,
    compute_depletion_rate_raw,
    compute_gini_raw,
    compute_hdi_proxy,
    compute_resource_access,
    compute_subsistence_collapse,
    compute_wealth_histogram,
)


# ----------------------------------------------------------------- drift detector


def test_max_produce_amount_mirrors_env_constant() -> None:
    assert DEFAULT_MAX_PRODUCE_AMOUNT == ENV_MAX_PRODUCE_AMOUNT


# --------------------------------------------------------------------- gini


def test_gini_all_equal_is_zero() -> None:
    assert compute_gini_raw([100, 100, 100, 100]) == pytest.approx(0.0)


def test_gini_known_inequality_case() -> None:
    """Standard textbook case: [0,0,0,100] -> 0.75."""
    assert compute_gini_raw([0, 0, 0, 100]) == pytest.approx(0.75)


def test_gini_empty_is_zero() -> None:
    assert compute_gini_raw([]) == 0.0


def test_gini_all_zero_is_zero() -> None:
    assert compute_gini_raw([0, 0, 0]) == 0.0


def test_gini_single_agent_is_zero() -> None:
    assert compute_gini_raw([50.0]) == pytest.approx(0.0)


def test_gini_bounded_zero_to_one() -> None:
    rng = np.random.default_rng(0)
    for _ in range(20):
        wealths = rng.uniform(0, 1000, size=30)
        g = compute_gini_raw(wealths)
        assert 0.0 <= g <= 1.0


# ------------------------------------------------------------- depletion rate


def test_depletion_rate_sign_positive_on_decline() -> None:
    """Stock falling 100->95 is depletion -> positive value, per the CSV
    schema's explicit Notes column ('positive = depletion')."""
    assert compute_depletion_rate_raw(stock_t=95.0, stock_t_minus_1=100.0) == pytest.approx(5.0)


def test_depletion_rate_sign_negative_on_regen() -> None:
    assert compute_depletion_rate_raw(stock_t=105.0, stock_t_minus_1=100.0) == pytest.approx(-5.0)


def test_depletion_rate_synthetic_series() -> None:
    """stock=[100,95,90,85]: a constant 5-unit absolute drop yields a
    slightly *increasing* percentage each step, since the formula divides
    by the shrinking stock_{t-1} base — (100-95)/100=5.0%,
    (95-90)/95=5.263...%, (90-85)/90=5.555...%. Positive = depletion."""
    stocks = [100.0, 95.0, 90.0, 85.0]
    rates = [
        compute_depletion_rate_raw(stocks[i], stocks[i - 1]) for i in range(1, len(stocks))
    ]
    assert rates == pytest.approx([5.0, 5.0 / 0.95, 5.0 / 0.90])


def test_depletion_rate_zero_prior_stock() -> None:
    assert compute_depletion_rate_raw(stock_t=10.0, stock_t_minus_1=0.0) == 0.0


# ------------------------------------------------------- subsistence collapse


def test_subsistence_collapse_triggers_below_threshold() -> None:
    # bottom 25% of 8 agents = 2 agents; mean of bottom 2 = 1.0;
    # threshold = 0.5 * initial_mean_wealth(10.0) = 5.0; 1.0 < 5.0 -> True
    wealths = [1.0, 1.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0]
    assert compute_subsistence_collapse(wealths, initial_mean_wealth=10.0) is True


def test_subsistence_collapse_false_when_above_threshold() -> None:
    wealths = [10.0] * 8
    assert compute_subsistence_collapse(wealths, initial_mean_wealth=10.0) is False


def test_subsistence_collapse_false_with_no_baseline() -> None:
    assert compute_subsistence_collapse([1.0, 2.0], initial_mean_wealth=0.0) is False


def test_subsistence_collapse_false_empty() -> None:
    assert compute_subsistence_collapse([], initial_mean_wealth=10.0) is False


def test_subsistence_collapse_custom_fractions() -> None:
    wealths = [0.0, 0.0, 0.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    # bottom 50% = 4 agents -> mean = (0+0+0+100)/4 = 25; threshold 0.5*10=5
    assert compute_subsistence_collapse(
        wealths, initial_mean_wealth=10.0, subsistence_fraction=0.5, bottom_fraction=0.5
    ) is False  # 25 >= 5
    # bottom 25% = 2 agents -> mean = 0; below threshold 5 -> True
    assert compute_subsistence_collapse(
        wealths, initial_mean_wealth=10.0, subsistence_fraction=0.5, bottom_fraction=0.25
    ) is True


# ------------------------------------------------------------- wealth histogram


def test_wealth_histogram_sums_to_one() -> None:
    wealths = [1.0, 5.0, 10.0, 50.0, 100.0]
    hist = compute_wealth_histogram(wealths, bins=10)
    assert len(hist) == 10
    assert sum(hist) == pytest.approx(1.0)


def test_wealth_histogram_empty() -> None:
    assert compute_wealth_histogram([], bins=10) == [0.0] * 10


def test_wealth_histogram_all_same_value() -> None:
    hist = compute_wealth_histogram([10.0] * 5, bins=10)
    assert sum(hist) == pytest.approx(1.0)


# ------------------------------------------------------------------- HDI proxy


def test_hdi_proxy_weighted_composite() -> None:
    """0.4*resource_access + 0.4*capability_utilisation + 0.2*agent_longevity."""
    assert compute_hdi_proxy(1.0, 1.0, 1.0) == pytest.approx(1.0)
    assert compute_hdi_proxy(0.0, 0.0, 0.0) == pytest.approx(0.0)
    assert compute_hdi_proxy(1.0, 0.0, 0.0) == pytest.approx(0.4)
    assert compute_hdi_proxy(0.0, 1.0, 0.0) == pytest.approx(0.4)
    assert compute_hdi_proxy(0.0, 0.0, 1.0) == pytest.approx(0.2)


def test_resource_access_clips_to_unit_interval() -> None:
    assert compute_resource_access(mean_wealth=20.0, initial_mean_wealth=10.0) == pytest.approx(1.0)
    assert compute_resource_access(mean_wealth=5.0, initial_mean_wealth=10.0) == pytest.approx(0.5)
    assert compute_resource_access(mean_wealth=5.0, initial_mean_wealth=0.0) == 0.0


def test_capability_utilisation_clips_to_unit_interval() -> None:
    # actual=30, theoretical_max = 5 producers * 3.0 max * 10.0 price = 150
    assert compute_capability_utilisation(30.0, 5, 3.0, 10.0) == pytest.approx(0.2)
    # over the theoretical max -> clipped to 1.0
    assert compute_capability_utilisation(300.0, 5, 3.0, 10.0) == pytest.approx(1.0)
    assert compute_capability_utilisation(30.0, 0, 3.0, 10.0) == 0.0


def test_agent_longevity_clips_to_unit_interval() -> None:
    assert compute_agent_longevity(mean_agent_age=50.0, current_timestep=100) == pytest.approx(0.5)
    assert compute_agent_longevity(mean_agent_age=150.0, current_timestep=100) == pytest.approx(1.0)
    assert compute_agent_longevity(mean_agent_age=50.0, current_timestep=0) == 0.0


# ----------------------------------------------------------------- CSV schema


class _StubMarket:
    base_prices = {"food": 10.0, "tool": 20.0}

    def get_price(self, item: str) -> float:
        return self.base_prices[item]


class _StubGovernanceModule:
    regime = "G1"


class _StubAgent:
    def __init__(self, agent_id: str, cash: float = 10.0, age: int = 0) -> None:
        self.agent_id = agent_id
        self.inventory = {"raw": 0.0, "food": 0.0, "tool": 0.0}
        self.cash = cash
        self.age = age

    def wealth(self, prices: dict[str, float]) -> float:
        return (
            self.inventory["raw"]
            + self.inventory["food"] * prices.get("food", 0.0)
            + self.inventory["tool"] * prices.get("tool", 0.0)
            + self.cash
        )


class _StubEnv:
    """Minimal duck-typed env for KPILogger tests, independent of AegoraEnv."""

    def __init__(self, n_agents: int = 10, seed: int = 42) -> None:
        self.n_agents = n_agents
        self.seed = seed
        self.market = _StubMarket()
        self.governance_module = _StubGovernanceModule()
        self._agents_by_id = {
            f"extractor_{i}": _StubAgent(f"extractor_{i}") for i in range(n_agents)
        }
        self._timestep = 0
        self._cycle_production_value = 0.0
        self._last_cycle_gdp_raw = 0.0
        self._last_cycle_production_value = 0.0

    def advance(self, timestep: int, stock: float, gdp_raw: float = 0.0) -> None:
        self._timestep = timestep
        self.resource_pool = type("RP", (), {"stock": stock})()
        self._last_cycle_gdp_raw = gdp_raw


def test_csv_schema_exact_columns(tmp_path) -> None:
    env = _StubEnv()
    logger = KPILogger(env, run_id="G1_A_001", scenario="A")
    for t in range(1, 6):
        env.advance(t, stock=1000.0 - t)
        logger.log_step()
    csv_path = logger.write_csv(tmp_path / "run.csv")

    import pandas as pd

    df = pd.read_csv(csv_path)
    assert list(df.columns) == CSV_COLUMNS
    assert len(CSV_COLUMNS) == 14


def test_csv_dtypes_and_values(tmp_path) -> None:
    env = _StubEnv()
    logger = KPILogger(env, run_id="G1_A_001", scenario="A")
    for t in range(1, 4):
        env.advance(t, stock=1000.0 - t)
        logger.log_step()
    csv_path = logger.write_csv(tmp_path / "run.csv")

    import pandas as pd

    df = pd.read_csv(csv_path)
    assert df["governance_condition"].eq("G1").all()
    assert df["scenario"].eq("A").all()
    assert df["run_id"].eq("G1_A_001").all()
    assert df["random_seed"].eq(42).all()
    assert df["n_agents"].eq(10).all()
    assert df["timestep"].tolist() == [1, 2, 3]


def test_hdi_proxy_nan_between_epochs(tmp_path) -> None:
    env = _StubEnv()
    logger = KPILogger(env, run_id="G1_A_001", scenario="A")
    for t in range(1, 5):
        env.advance(t, stock=1000.0)
        row = logger.log_step()
        assert math.isnan(row["hdi_proxy"])
    env.advance(100, stock=1000.0)
    row = logger.log_step()
    assert not math.isnan(row["hdi_proxy"])


def test_hdi_mean_price_excludes_raw() -> None:
    """capability_utilisation's theoretical-max denominator must average
    food/tool (output-goods) prices only -- not "raw" (an input cost),
    even once raw becomes a real market item (see DECISIONS.md "Producer
    raw-material acquisition — resolved"). Uses a wildly different raw
    price so any accidental inclusion would be obvious."""
    env = _StubEnv(n_agents=4)
    env.market.base_prices = {"food": 10.0, "tool": 20.0, "raw": 1000.0}
    env._agents_by_id["producer_0"] = _StubAgent("producer_0")
    env._last_cycle_production_value = 30.0  # nonzero, to exercise the ratio
    env.advance(100, stock=1000.0)

    logger = KPILogger(env, run_id="G1_A_001", scenario="A")
    row = logger.log_step()

    # If raw's price (1000.0) were included in the mean, capability_utilisation
    # would be driven toward 0 (theoretical_max would be huge). Excluding it,
    # mean_price = mean(10.0, 20.0) = 15.0, theoretical_max = 1*3.0*15.0 = 45.0,
    # capability_utilisation = 30.0/45.0 ≈ 0.667 -> HDI's 0.4 term ≈ 0.267,
    # nonzero and well above what a raw-inflated denominator would produce.
    assert not math.isnan(row["hdi_proxy"])
    assert row["hdi_proxy"] > 0.1  # would be ~0 if raw's price polluted the mean


def test_capability_utilisation_ignores_already_reset_live_accumulator() -> None:
    """Regression test for DECISIONS.md "KPILogger capability_utilisation
    read-after-reset fix": in the real env, AegoraEnv.step() resets
    _cycle_production_value to 0.0 for the next cycle in the very same
    call that also finalizes the cycle that just ended, before
    KPILogger.log_step() is ever invoked -- so log_step() must read the
    separately-captured _last_cycle_production_value, not the live
    (already-reset) accumulator. Simulates exactly that real-world
    ordering: _cycle_production_value sits at its post-reset 0.0 while
    _last_cycle_production_value holds the real finished-cycle value."""
    env = _StubEnv(n_agents=4)
    env._agents_by_id["producer_0"] = _StubAgent("producer_0")
    env._cycle_production_value = 0.0  # already reset, as it would be for real
    env._last_cycle_production_value = 30.0  # the real value from the cycle that just ended
    env.advance(100, stock=1000.0)

    logger = KPILogger(env, run_id="G1_A_001", scenario="A")
    row = logger.log_step()

    assert not math.isnan(row["hdi_proxy"])
    assert row["hdi_proxy"] > 0.6, (
        "hdi_proxy fell back to the always-zero-capability_utilisation "
        "plateau (~0.6) -- log_step() is reading the reset live "
        "accumulator instead of _last_cycle_production_value"
    )


def test_collapse_event_is_rising_edge_pulse() -> None:
    """Fires once at first detection, not on every subsequent below-threshold row."""
    env = _StubEnv(n_agents=4)
    logger = KPILogger(env, run_id="G1_A_001", scenario="A")
    env.advance(1, stock=1000.0)
    row1 = logger.log_step()
    assert row1["collapse_event"] is False  # cash=10 each, at baseline, not below

    # Force collapse by draining cash on all agents.
    for a in env._agents_by_id.values():
        a.cash = 0.0
    env.advance(2, stock=1000.0)
    row2 = logger.log_step()
    assert row2["collapse_event"] is True  # rising edge

    env.advance(3, stock=1000.0)
    row3 = logger.log_step()
    assert row3["collapse_event"] is False  # still collapsed, but not a NEW pulse


# ---------------------------------------------------------------- JSON schema


def test_json_schema_exact_fields(tmp_path) -> None:
    env = _StubEnv()
    logger = KPILogger(env, run_id="G1_A_001", scenario="A")
    for t in range(1, 4):
        env.advance(t, stock=1000.0 - t)
        logger.log_step()
    csv_path = logger.write_csv(tmp_path / "run.csv")
    json_path = logger.write_json_state(
        csv_path,
        governance_params={"regime": "G1"},
        scenario_params={"initial_stock": 1000.0},
        agent_type_distribution={"extractor": 1.0},
        rl_hyperparams={},
    )
    state = json.loads(json_path.read_text())
    assert set(state.keys()) == set(JSON_FIELDS)
    # Spec §3.2's bulleted field list has 13 entries when actually counted
    # verbatim (an earlier research pass mis-tallied this as 12 — see
    # DECISIONS.md "JSON state field count correction").
    assert len(JSON_FIELDS) == 13


def test_json_csv_checksum_matches_file(tmp_path) -> None:
    import hashlib

    env = _StubEnv()
    logger = KPILogger(env, run_id="G1_A_001", scenario="A")
    env.advance(1, stock=999.0)
    logger.log_step()
    csv_path = logger.write_csv(tmp_path / "run.csv")
    json_path = logger.write_json_state(
        csv_path, governance_params={}, scenario_params={}, agent_type_distribution={}, rl_hyperparams={}
    )
    state = json.loads(json_path.read_text())
    expected = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    assert state["csv_checksum"] == expected


def test_json_collapse_fields_reflect_logger_state(tmp_path) -> None:
    env = _StubEnv(n_agents=4)
    logger = KPILogger(env, run_id="G1_A_001", scenario="A")
    env.advance(1, stock=1000.0)
    logger.log_step()
    for a in env._agents_by_id.values():
        a.cash = 0.0
    env.advance(5, stock=1000.0)
    logger.log_step()
    csv_path = logger.write_csv(tmp_path / "run.csv")
    json_path = logger.write_json_state(
        csv_path, governance_params={}, scenario_params={}, agent_type_distribution={}, rl_hyperparams={}
    )
    state = json.loads(json_path.read_text())
    assert state["collapse_occurred"] is True
    assert state["collapse_timestep"] == 5
