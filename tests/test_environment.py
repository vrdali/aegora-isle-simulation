"""Integration tests for the AegoraEnv (Sprint 1.4 / AC01 + AC02).

- ``pettingzoo.test.api_test`` must pass cleanly (AC01).
- 4 agent types instantiate at default n=50 (AC02 with Governor present
  via a forced G2 stub; G1 yields 3 types per DECISIONS.md).
- Per-agent obs/action shapes match Sprint 1.3 expectations.
- Seeded determinism: two envs with identical seed yield identical
  ``agent_selection`` sequences over 200 steps.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pettingzoo.test import api_test

from aegora_isle.agents.extractor import Extractor
from aegora_isle.agents.governor import Governor
from aegora_isle.agents.producer import Producer
from aegora_isle.agents.trader import Trader
from aegora_isle.environment.aegora_env import AegoraEnv


class _ForcedRegimeStub:
    """Like GovernanceStub but with configurable ``regime`` for tests."""

    def __init__(self, regime: str = "G2") -> None:
        self.regime = regime

    def on_action(self, agent_id: str, action: Any) -> tuple[Any, float]:
        return action, 0.0

    def on_step(self, env_state: dict) -> None:
        return None


def test_api_test_passes_default() -> None:
    """AC01 — PettingZoo AEC compliance."""
    env = AegoraEnv(seed=42)
    api_test(env, num_cycles=200, verbose_progress=False)


def test_api_test_passes_with_governor() -> None:
    env = AegoraEnv(seed=42, governance_module=_ForcedRegimeStub("G2"))
    api_test(env, num_cycles=200, verbose_progress=False)


def test_ac02_four_types_with_governor() -> None:
    """AC02 — 4 agent types when Governor is enabled."""
    env = AegoraEnv(seed=42, n_agents=50, governance_module=_ForcedRegimeStub("G2"))
    env.reset()
    types = {a.split("_")[0] for a in env.possible_agents}
    assert types == {"extractor", "producer", "trader", "governor"}


def test_g1_has_three_types_no_governor() -> None:
    """Per DECISIONS.md, G1 drops Governor → 17/17/16 at n=50."""
    env = AegoraEnv(seed=42, n_agents=50)  # default stub is G1
    env.reset()
    types = {a.split("_")[0] for a in env.possible_agents}
    assert types == {"extractor", "producer", "trader"}
    counts = {kind: sum(1 for a in env.possible_agents if a.startswith(kind)) for kind in types}
    assert counts == {"extractor": 17, "producer": 17, "trader": 16}


def test_g2_split_at_n50() -> None:
    env = AegoraEnv(seed=42, n_agents=50, governance_module=_ForcedRegimeStub("G2"))
    env.reset()
    counts = {kind: sum(1 for a in env.possible_agents if a.startswith(kind))
              for kind in ("extractor", "producer", "trader", "governor")}
    # Per DECISIONS.md: 12/13/13/12.
    assert counts == {"extractor": 12, "producer": 13, "trader": 13, "governor": 12}


def test_agent_classes_match_per_id() -> None:
    env = AegoraEnv(seed=42, governance_module=_ForcedRegimeStub("G2"))
    env.reset()
    for aid, agent in env._agents_by_id.items():
        kind = aid.split("_")[0]
        if kind == "extractor":
            assert isinstance(agent, Extractor)
        elif kind == "producer":
            assert isinstance(agent, Producer)
        elif kind == "trader":
            assert isinstance(agent, Trader)
        elif kind == "governor":
            assert isinstance(agent, Governor)


def test_obs_in_space_for_every_agent_at_reset() -> None:
    env = AegoraEnv(seed=42, governance_module=_ForcedRegimeStub("G2"))
    env.reset()
    for aid in env.agents:
        obs = env.observe(aid)
        assert env.observation_space(aid).contains(obs), aid


def test_seeded_determinism_agent_selection_sequence() -> None:
    """Two envs with identical seed must yield identical agent_selection sequences."""
    e1 = AegoraEnv(seed=2026)
    e2 = AegoraEnv(seed=2026)
    e1.reset()
    e2.reset()
    seq1: list[str] = []
    seq2: list[str] = []
    for _ in range(200):
        seq1.append(e1.agent_selection)
        seq2.append(e2.agent_selection)
        a1 = e1.action_space(e1.agent_selection).sample()
        a2 = e2.action_space(e2.agent_selection).sample()
        # action_space.sample() of identical seeds isn't itself seeded — but
        # we only compare agent_selection here, which is deterministic from
        # the agent_iter order regardless of actions.
        e1.step(a1)
        e2.step(a2)
        if not e1.agents or not e2.agents:
            break
    assert seq1 == seq2


def test_seeded_determinism_resource_pool_under_fixed_actions() -> None:
    """Same env seed + locally-seeded action spaces → identical pool trajectory.

    ``action_space.sample()`` reads from Gymnasium's global RNG, which is
    advanced by every other test in the suite. We seed each agent's action
    space explicitly so the trajectory is reproducible regardless of test
    ordering.
    """

    def run(env_seed: int, action_seed: int = 7) -> list[float]:
        env = AegoraEnv(seed=env_seed, max_timesteps=50)
        env.reset()
        for aid in env.agents:
            env.action_space(aid).seed(action_seed)
        stocks: list[float] = []
        for _ in range(50 * env.n_agents):
            aid = env.agent_selection
            if env.terminations[aid] or env.truncations[aid]:
                env.step(None)
            else:
                env.step(env.action_space(aid).sample())
            stocks.append(env.resource_pool.stock)
            if not env.agents:
                break
        return stocks

    s1 = run(env_seed=99)
    s2 = run(env_seed=99)
    assert s1 == s2


def test_last_returns_5_tuple() -> None:
    env = AegoraEnv(seed=42)
    env.reset()
    obs, reward, term, trunc, info = env.last()
    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(term, bool)
    assert isinstance(trunc, bool)
    assert isinstance(info, dict)


def test_run_many_cycles_no_error() -> None:
    env = AegoraEnv(seed=42, max_timesteps=200)
    env.reset()
    for _ in range(2000):
        aid = env.agent_selection
        if env.terminations[aid] or env.truncations[aid]:
            env.step(None)
        else:
            env.step(env.action_space(aid).sample())
        if not env.agents:
            break


def test_collapse_truncates_all_agents() -> None:
    """Force a tiny pool — collapse triggers truncation flags."""
    env = AegoraEnv(
        seed=42,
        scenario_params={
            "initial_stock": 50.0,
            "regeneration_rate": 0.0,
            "event_rate": 0.0,
            "critical_min": 49.0,  # collapses almost immediately
            "scarcity_alpha": 0.0,
            "base_prices": {"food": 10.0, "tool": 20.0},
        },
        max_timesteps=1000,
    )
    env.reset()
    for _ in range(500):
        aid = env.agent_selection
        if env.terminations[aid] or env.truncations[aid]:
            env.step(None)
        else:
            env.step(env.action_space(aid).sample())
        if all(env.truncations.values()) and len(env.truncations) > 0:
            break
    # After enough random extraction, the pool falls below 49 → trunc.
    assert any(env.truncations.values())
