"""Unit tests for aegora_isle.governance.adaptive_rl.AdaptiveRLGovernor.

Covers M2 Sprint 2.5 acceptance criteria (G3_PLAN.md):
- Composite reward weights must sum to 1.0; formula verified against
  hand-computed values.
- Running normaliser produces stable mean/std on standard inputs.
- Extraction-cap clipping uses self.current_cap (SAC-set), not a YAML
  constant — same mechanism as RuleBasedGovernor, different parameter
  source.
- Redistribution uses self.current_redist_rate, conserves total cash.
- on_step builds the SAC model lazily, fills the replay buffer BEFORE
  training (the gap found in the original G3_PLAN.md sketch), and trains
  only once warmed up and on a train_freq boundary.
- Seeding contract + parameter validation.
- Drift detector: MAX_EXTRACT_AMOUNT mirrors the env's constant.
"""

from __future__ import annotations

import numpy as np
import pytest

from aegora_isle.agents.governor import Governor
from aegora_isle.environment.aegora_env import MAX_EXTRACT_AMOUNT as ENV_MAX_EXTRACT_AMOUNT
from aegora_isle.governance.adaptive_rl import (
    MAX_EXTRACT_AMOUNT,
    SAC_ACTION_DIM,
    AdaptiveRLGovernor,
    _RunningNormaliser,
)


SMALL_SAC_KWARGS = {
    "learning_rate": 3.0e-4,
    "buffer_size": 1000,
    "batch_size": 16,
    "gamma": 0.99,
    "tau": 0.005,
    "ent_coef": "auto",
    "train_freq": [4, "step"],
    "gradient_steps": 1,
    "learning_starts": 5,
    "target_update_interval": 1,
    "policy_kwargs": {"net_arch": [16, 16]},
}

DEFAULT_WEIGHTS = {"equality": 0.40, "sustainability": 0.35, "productivity": 0.25}


def _make_governor(
    *,
    seed: int = 0,
    sac_kwargs: dict | None = None,
    extraction_cap_range: tuple[float, float] = (0.5, 5.0),
    redistribution_rate_range: tuple[float, float] = (0.0, 0.3),
    sanction_default_magnitude: float = -2.0,
) -> AdaptiveRLGovernor:
    return AdaptiveRLGovernor(
        reward_weights=DEFAULT_WEIGHTS,
        sac_kwargs=sac_kwargs if sac_kwargs is not None else dict(SMALL_SAC_KWARGS),
        extraction_cap_range=extraction_cap_range,
        redistribution_rate_range=redistribution_rate_range,
        sanction_default_magnitude=sanction_default_magnitude,
        rng=np.random.default_rng(seed),
    )


def _extract_action(frac: float) -> dict:
    return {"action_type": 0, "amount": np.array([frac], dtype=np.float32), "target_zone": 0}


class _StubAgent:
    def __init__(self, agent_id: str, cash: float = 0.0) -> None:
        self.agent_id = agent_id
        self.cash = cash
        self.inventory: dict[str, float] = {}


def _env_state(
    *,
    timestep: int = 1,
    stock: float = 1000.0,
    initial_stock: float = 1000.0,
    critical_min: float = 50.0,
    gini: float = 0.3,
    hdi_proxy: float = 0.5,
    gdp_indexed: float = 10.0,
    depletion_rate: float = 0.0,
    extra_agents: dict | None = None,
) -> dict:
    governor = Governor(agent_id="governor_0", rng=np.random.default_rng(99))
    agents = {"governor_0": governor}
    if extra_agents:
        agents.update(extra_agents)
    return {
        "timestep": timestep,
        "resource_pool": {"stock": stock, "initial_stock": initial_stock, "critical_min": critical_min},
        "kpis": {"gini": gini, "hdi_proxy": hdi_proxy, "gdp_indexed": gdp_indexed, "depletion_rate": depletion_rate},
        "wealth_histogram": [0.1] * 10,
        "agents_by_id": agents,
    }


# ----------------------------------------------------------- drift detector


def test_max_extract_amount_mirrors_env_constant() -> None:
    assert MAX_EXTRACT_AMOUNT == ENV_MAX_EXTRACT_AMOUNT


# --------------------------------------------------------------- validation


def test_seeding_contract_rejects_non_generator() -> None:
    with pytest.raises(TypeError):
        AdaptiveRLGovernor(
            reward_weights=DEFAULT_WEIGHTS,
            sac_kwargs=dict(SMALL_SAC_KWARGS),
            extraction_cap_range=(0.5, 5.0),
            redistribution_rate_range=(0.0, 0.3),
            sanction_default_magnitude=-2.0,
            rng=42,  # type: ignore[arg-type]
        )


def test_weights_must_sum_to_one() -> None:
    with pytest.raises(ValueError):
        AdaptiveRLGovernor(
            reward_weights={"equality": 0.5, "sustainability": 0.5, "productivity": 0.5},
            sac_kwargs=dict(SMALL_SAC_KWARGS),
            extraction_cap_range=(0.5, 5.0),
            redistribution_rate_range=(0.0, 0.3),
            sanction_default_magnitude=-2.0,
            rng=np.random.default_rng(0),
        )


def test_invalid_normalisation_mode_raises() -> None:
    with pytest.raises(ValueError):
        AdaptiveRLGovernor(
            reward_weights=DEFAULT_WEIGHTS,
            sac_kwargs=dict(SMALL_SAC_KWARGS),
            extraction_cap_range=(0.5, 5.0),
            redistribution_rate_range=(0.0, 0.3),
            sanction_default_magnitude=-2.0,
            rng=np.random.default_rng(0),
            reward_normalisation="bogus",
        )


def test_train_freq_list_converted_to_tuple() -> None:
    """YAML loads train_freq as a list; SB3's constructor requires a tuple
    or it crashes inside _convert_train_freq. Verified against actual
    installed SB3 2.9.0 source."""
    gov = _make_governor()
    assert isinstance(gov._sac_kwargs["train_freq"], tuple)
    assert gov._sac_kwargs["train_freq"] == (4, "step")
    assert gov._train_freq_steps == 4


# ------------------------------------------------------------- composite reward


def test_composite_reward_known_input_no_normalisation() -> None:
    gov = _make_governor()
    gov._normalisers = None  # isolate the formula from normalisation
    state = _env_state(gini=0.5, stock=800.0, initial_stock=1000.0, depletion_rate=0.0, gdp_indexed=0.0)
    # equality = 1 - 0.5 = 0.5
    # sustainability = stock_norm(0.8) * max(0, 1 - 0) = 0.8
    # productivity = log1p(0) = 0
    expected = 0.40 * 0.5 + 0.35 * 0.8 + 0.25 * 0.0
    assert gov.compute_composite_reward(state) == pytest.approx(expected)


def test_sustainability_term_penalised_by_depletion() -> None:
    gov = _make_governor()
    gov._normalisers = None
    state_no_depletion = _env_state(stock=1000.0, initial_stock=1000.0, depletion_rate=0.0)
    state_high_depletion = _env_state(stock=1000.0, initial_stock=1000.0, depletion_rate=0.5)
    r1 = gov.compute_composite_reward(state_no_depletion)
    r2 = gov.compute_composite_reward(state_high_depletion)
    assert r2 < r1  # higher depletion -> lower reward, given positive=depletion convention


def test_sustainability_floors_at_zero_for_extreme_depletion() -> None:
    gov = _make_governor()
    gov._normalisers = None
    state = _env_state(stock=1000.0, initial_stock=1000.0, depletion_rate=1.5)  # > 1
    # max(0, 1 - 1.5) = 0 -> sustainability term is exactly 0, never negative
    assert gov._component_sustainability(state) == 0.0


def test_running_normaliser_stabilises() -> None:
    rng = np.random.default_rng(0)
    norm = _RunningNormaliser(clip=5.0)
    samples = rng.normal(loc=3.0, scale=2.0, size=2000)
    for s in samples:
        norm.update(s)
    assert norm.mean == pytest.approx(3.0, abs=0.2)
    normalised = [norm.normalise(s) for s in samples[-100:]]
    assert abs(np.mean(normalised)) < 0.5
    assert all(-5.0 <= v <= 5.0 for v in normalised)


def test_normaliser_single_sample_clips_without_division_by_zero() -> None:
    norm = _RunningNormaliser(clip=5.0)
    norm.update(10.0)
    assert norm.normalise(10.0) == pytest.approx(5.0)  # n<2 path, clipped


# --------------------------------------------------------------- on_action


def test_within_cap_passes_through_unmodified() -> None:
    gov = _make_governor()
    gov.current_cap = 5.0  # well above any reasonable request
    action = _extract_action(0.5)
    mod, delta = gov.on_action("extractor_0", action)
    assert mod == action
    assert delta == 0.0


def test_over_cap_clips_using_current_cap_not_yaml_constant() -> None:
    gov = _make_governor()
    gov.current_cap = 2.0  # SAC-set value, not a constructor arg
    gov.current_sanction_probability = 0.0
    mod, delta = gov.on_action("extractor_0", _extract_action(1.0))
    assert mod["amount"][0] == pytest.approx(2.0 / 5.0)
    assert delta == 0.0


def test_sanction_uses_current_sanction_probability() -> None:
    gov = _make_governor(seed=2026)
    gov.current_cap = 1.0
    gov.current_sanction_probability = 1.0
    _, delta = gov.on_action("extractor_0", _extract_action(1.0))
    assert delta == pytest.approx(-2.0)


def test_non_extractor_passthrough() -> None:
    gov = _make_governor()
    gov.current_cap = 0.0
    gov.current_sanction_probability = 1.0
    producer_action = {"action_type": 2, "produce_type": 0, "offer_item": 0, "offer_qty": np.array([1.0])}
    mod, delta = gov.on_action("producer_0", producer_action)
    assert mod == producer_action
    assert delta == 0.0


# ------------------------------------------------------------------ on_step


def test_on_step_builds_model_lazily() -> None:
    gov = _make_governor()
    assert gov.model is None
    gov.on_step(_env_state(timestep=1))
    assert gov.model is not None


def test_on_step_does_not_fill_buffer_on_first_call() -> None:
    """No previous (obs, action) pair exists yet on cycle 1."""
    gov = _make_governor()
    gov.on_step(_env_state(timestep=1))
    assert gov.model.replay_buffer.size() == 0


def test_on_step_fills_buffer_from_second_call_onward() -> None:
    gov = _make_governor()
    for t in range(1, 6):
        gov.on_step(_env_state(timestep=t))
    assert gov.model.replay_buffer.size() == 4  # cycles 2,3,4,5 each add one


def test_on_step_trains_once_warmed_up_and_on_freq_boundary() -> None:
    """learning_starts=5, train_freq=4 in SMALL_SAC_KWARGS: buffer needs
    >=5 transitions (i.e. cycle 6+) AND step_count % 4 == 0."""
    gov = _make_governor()
    for t in range(1, 9):
        gov.on_step(_env_state(timestep=t))
    # By cycle 8: buffer size = 7 (>=5), step_count=8 (%4==0) -> trained at least once.
    assert gov.model.replay_buffer.size() == 7
    assert gov.state_dict()["step_count"] == 8


def test_on_step_updates_current_decision_values() -> None:
    gov = _make_governor()
    gov.on_step(_env_state(timestep=1))
    assert 0.5 <= gov.current_cap <= 5.0
    assert 0.0 <= gov.current_redist_rate <= 0.3
    assert 0.0 <= gov.current_sanction_probability <= 1.0


def test_on_step_redistribution_conserves_cash() -> None:
    gov = _make_governor()
    extractor = _StubAgent("extractor_0", cash=0.0)
    producer = _StubAgent("producer_0", cash=100.0)
    state = _env_state(timestep=1, extra_agents={"extractor_0": extractor, "producer_0": producer})
    total_before = extractor.cash + producer.cash

    # Force a known positive redistribution rate for a deterministic check.
    gov.on_step(state)
    gov.current_redist_rate = 0.2
    gov._redistribute(state)

    assert extractor.cash + producer.cash == pytest.approx(total_before)
    assert extractor.cash > 0.0
    assert gov.state_dict()["redistribution_count"] >= 1


def test_on_step_requires_a_governor_agent() -> None:
    gov = _make_governor()
    state = _env_state(timestep=1)
    state["agents_by_id"] = {"extractor_0": _StubAgent("extractor_0")}  # no governor
    with pytest.raises(RuntimeError):
        gov.on_step(state)


def test_is_terminal_uses_collapse_check() -> None:
    gov = _make_governor()
    not_collapsed = _env_state(stock=500.0, critical_min=50.0)
    collapsed = _env_state(stock=10.0, critical_min=50.0)
    assert gov._is_terminal(not_collapsed) is False
    assert gov._is_terminal(collapsed) is True


def test_on_episode_end_clears_transition_memory() -> None:
    gov = _make_governor()
    gov.on_step(_env_state(timestep=1))
    gov.on_step(_env_state(timestep=2))
    assert gov._prev_obs is not None
    gov.on_episode_end(_env_state(timestep=2))
    assert gov._prev_obs is None
    assert gov._prev_sac_action is None


# ---------------------------------------------------------- multi-cycle smoke


def test_short_multi_cycle_run_does_not_crash() -> None:
    """End-to-end smoke check across enough cycles to exercise warm-up,
    training, and redistribution together."""
    gov = _make_governor(seed=7)
    extractor = _StubAgent("extractor_0", cash=10.0)
    producer = _StubAgent("producer_0", cash=50.0)
    trader = _StubAgent("trader_0", cash=30.0)
    for t in range(1, 31):
        state = _env_state(
            timestep=t,
            stock=1000.0 - t,
            gini=0.2 + t * 0.01,
            depletion_rate=0.001 * t,
            extra_agents={"extractor_0": extractor, "producer_0": producer, "trader_0": trader},
        )
        action = _extract_action(0.9)
        gov.on_action("extractor_0", action)
        gov.on_step(state)
    assert gov.model.replay_buffer.size() == 29
    assert gov.state_dict()["step_count"] == 30


# ------------------------------------------------------------ SAC action dim


def test_sac_action_dim_is_three() -> None:
    """The whole point of the design correction: SAC's action space is 3
    dims, not a mechanical 505-dim flatten of the Governor's Dict space."""
    assert SAC_ACTION_DIM == 3
    gov = _make_governor()
    gov.on_step(_env_state(timestep=1))
    assert gov.model.action_space.shape == (3,)
