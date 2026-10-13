"""Tests for spec §2.3.1's three environmental event types.

Only ``disruption`` was implemented before; ``market_volatility`` and
``agent_withdrawal`` were missing entirely (no code, no docs, no
DECISIONS.md entry). See DECISIONS.md "Spec §2.3.1 events -- the two
missing types".

Design under test:
- ``ResourcePool`` rolls all three independently at ``event_rate/100`` and
  applies only ``disruption`` (it owns the stock).
- ``AegoraEnv`` routes the other two, because it is the only layer holding
  both the Market and the agent registry.
"""

from __future__ import annotations

import numpy as np
import pytest

from aegora_isle.environment.aegora_env import AegoraEnv
from aegora_isle.environment.market import Market
from aegora_isle.environment.resource_pool import (
    MARKET_VOLATILITY_DURATION_RANGE,
    MARKET_VOLATILITY_MULTIPLIER_RANGE,
    ResourcePool,
)
from aegora_isle.governance.governance_module import G1Handler


SCENARIO = {
    "initial_stock": 10000.0,
    "regeneration_rate": 0.05,
    "event_rate": 0.0,
    "critical_min": 50.0,
    "scarcity_alpha": 0.5,
    "base_prices": {"food": 10.0, "tool": 20.0, "raw": 1.0},
}


def _pool(seed: int = 7, event_rate: float = 5.0) -> ResourcePool:
    return ResourcePool(
        initial_stock=1000.0, regeneration_rate=0.0, critical_min=10.0,
        event_rate=event_rate, rng=np.random.default_rng(seed),
    )


# ------------------------------------------------------------ pool emission


def test_pool_emits_all_three_event_types() -> None:
    pool = _pool()
    for _ in range(5000):
        pool.step()
    counts = pool.event_counts_by_type
    assert counts["disruption"] > 0
    assert counts["market_volatility"] > 0
    assert counts["agent_withdrawal"] > 0


def test_pool_applies_only_disruption_itself() -> None:
    """The pool owns stock, so it applies disruption. The other two are
    emitted for the env to route -- they must not touch stock."""
    pool = _pool(event_rate=0.0)
    before = pool.stock
    pool.inject_market_volatility()
    pool.inject_agent_withdrawal()
    assert pool.stock == before


def test_emitted_volatility_within_documented_ranges() -> None:
    pool = _pool()
    lo, hi = MARKET_VOLATILITY_MULTIPLIER_RANGE
    d_lo, d_hi = MARKET_VOLATILITY_DURATION_RANGE
    for _ in range(200):
        e = pool.inject_market_volatility()
        assert lo <= e["multiplier"] <= hi
        assert d_lo <= e["duration"] <= d_hi
        assert e["type"] == "market_volatility"


def test_events_are_reproducible_from_seed() -> None:
    a, b = _pool(seed=99), _pool(seed=99)
    for _ in range(500):
        a.step()
        b.step()
    assert a.event_counts_by_type == b.event_counts_by_type
    assert a.stock == b.stock


# ------------------------------------------------------------ market effect


def test_volatility_multiplier_scales_price_then_expires() -> None:
    pool = _pool(event_rate=0.0)
    market = Market(pool, {"food": 10.0}, scarcity_alpha=0.0,
                    rng=np.random.default_rng(0))
    base = market.get_price("food")

    market.apply_volatility_episode(multiplier=1.5, duration=3)
    assert market.get_price("food") == pytest.approx(base * 1.5)

    for _ in range(3):
        market.tick_volatility()
    assert market.get_price("food") == pytest.approx(base)
    assert market.volatility_multiplier == 1.0


def test_volatility_episode_replaces_rather_than_compounds() -> None:
    """Coincident episodes must not multiply -- that would run prices away
    exponentially, which the spec's 'temporary price multiplier' does not
    describe."""
    pool = _pool(event_rate=0.0)
    market = Market(pool, {"food": 10.0}, scarcity_alpha=0.0,
                    rng=np.random.default_rng(0))
    market.apply_volatility_episode(multiplier=1.5, duration=10)
    market.apply_volatility_episode(multiplier=1.2, duration=10)
    assert market.volatility_multiplier == pytest.approx(1.2)


def test_volatility_reset_clears_episode() -> None:
    pool = _pool(event_rate=0.0)
    market = Market(pool, {"food": 10.0}, scarcity_alpha=0.0,
                    rng=np.random.default_rng(0))
    market.apply_volatility_episode(multiplier=1.4, duration=99)
    market.reset()
    assert market.volatility_multiplier == 1.0
    assert market.get_price("food") == pytest.approx(10.0)


# -------------------------------------------------------- agent withdrawal


def _env(seed: int = 3) -> AegoraEnv:
    env = AegoraEnv(seed=seed, n_agents=20, governance_module=G1Handler(),
                    scenario_params=SCENARIO, max_timesteps=1000)
    env.reset()
    return env


def test_withdrawn_agent_action_has_no_effect() -> None:
    env = _env()
    extractor = next(a for a in env.agents if a.startswith("extractor"))
    agent = env._agents_by_id[extractor]
    env._withdrawal_remaining[extractor] = 5

    stock_before = env.resource_pool.stock
    env._apply_action(agent, {"action_type": 0,
                              "amount": np.array([1.0], dtype=np.float32),
                              "target_zone": 0})
    assert env.resource_pool.stock == stock_before          # no extraction
    assert agent.inventory["raw"] == pytest.approx(0.0)
    assert env._step_log[extractor] == {"withdrawn": True}


def test_withdrawal_expires_after_duration() -> None:
    env = _env()
    aid = next(a for a in env.agents if a.startswith("extractor"))
    env._withdrawal_remaining[aid] = 3
    assert env.is_withdrawn(aid)
    for _ in range(3):
        env._tick_agent_withdrawals()
    assert not env.is_withdrawn(aid)


def test_withdrawal_never_targets_the_governor() -> None:
    """Withdrawing the policy-setter would silently disable the regime under
    test and confound the comparison the experiment exists to make."""
    from pathlib import Path as _Path

    from aegora_isle.governance.governance_module import GovernanceModule

    cfg = str(_Path(__file__).resolve().parent.parent
              / "aegora_isle" / "governance" / "configs" / "g2_rule_based.yaml")
    env = AegoraEnv(seed=1, n_agents=20,
                    governance_module=GovernanceModule(cfg, np.random.default_rng(1)),
                    scenario_params=SCENARIO, max_timesteps=1000)
    env.reset()
    assert any(a.startswith("governor") for a in env.agents), "fixture needs a Governor"
    for _ in range(300):
        env._start_agent_withdrawal(5, {})
    assert not any(a.startswith("governor") for a in env._withdrawal_remaining)


def test_env_routes_pool_events_to_market_and_agents() -> None:
    """End-to-end: an env stepping with a high event rate must actually show
    volatility episodes and withdrawals taking effect."""
    env = AegoraEnv(seed=5, n_agents=20, governance_module=G1Handler(),
                    scenario_params={**SCENARIO, "event_rate": 40.0},
                    max_timesteps=400)
    env.reset()
    saw_volatility = False
    saw_withdrawal = False
    steps = 0
    while env.agents and env._timestep < 300 and steps < 300 * env.n_agents:
        aid = env.agent_selection
        if env.terminations[aid] or env.truncations[aid]:
            env.step(None)
        else:
            env.step(env.action_space(aid).sample())
        steps += 1
        if env.market is not None and env.market.volatility_multiplier != 1.0:
            saw_volatility = True
        if any(v > 0 for v in env._withdrawal_remaining.values()):
            saw_withdrawal = True
    assert saw_volatility, "no market-volatility episode ever became active"
    assert saw_withdrawal, "no agent was ever withdrawn"
