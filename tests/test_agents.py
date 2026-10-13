"""Unit tests for agent classes.

Verifies M1 Sprint 1.3 acceptance criteria:
- Each agent's ``get_observation`` lands inside its declared ``observation_space``.
- Each agent's ``action_space.sample()`` is consumed by ``compute_reward`` without error.
- No file in ``aegora_isle/agents/`` imports from ``aegora_isle.governance``
  (CLAUDE.md non-negotiable).
- All four agents accept the seeding-contract Generator.
"""

from __future__ import annotations

import pathlib
import subprocess
from typing import Any

import numpy as np
import pytest

from aegora_isle.agents.base import AgentBase
from aegora_isle.agents.extractor import Extractor
from aegora_isle.agents.governor import Governor
from aegora_isle.agents.producer import Producer
from aegora_isle.agents.trader import Trader


AGENT_CLASSES = [Extractor, Producer, Trader, Governor]


def _dummy_env_state() -> dict[str, Any]:
    return {
        "resource_pool": {"stock": 750.0, "initial_stock": 1000.0, "critical_min": 100.0},
        "prices": {"food": 10.0, "tool": 20.0},
        "production_capacity_remaining": 0.5,
        "last_5_trade_volumes": [1.0, 2.0, 3.0, 4.0, 5.0],
        "last_extracted_amount": 1.5,
        "last_depletion_penalty": 0.0,
        "last_produced": {"food": 1.0, "tool": 0.0},
        "last_trade_profit": 0.25,
        "kpis": {
            "gini": 0.4,
            "hdi_proxy": 0.7,
            "gdp_indexed": 1.2,
            "depletion_rate": -0.01,
        },
        "wealth_histogram": [0.1] * 10,
    }


@pytest.mark.parametrize("cls", AGENT_CLASSES)
def test_observation_in_space(cls: type[AgentBase]) -> None:
    rng = np.random.default_rng(0)
    agent = cls(agent_id=f"{cls.__name__.lower()}_0", rng=rng)
    obs = agent.get_observation(_dummy_env_state())
    assert agent.observation_space.contains(obs), (
        f"{cls.__name__} observation {obs} not in space {agent.observation_space}"
    )


@pytest.mark.parametrize("cls", AGENT_CLASSES)
def test_action_space_sample_is_consumable(cls: type[AgentBase]) -> None:
    rng = np.random.default_rng(0)
    agent = cls(agent_id=f"{cls.__name__.lower()}_0", rng=rng)
    for _ in range(20):
        action = agent.action_space.sample()
        assert agent.action_space.contains(action)
        reward = agent.compute_reward(_dummy_env_state(), action)
        assert isinstance(reward, float)


@pytest.mark.parametrize("cls", AGENT_CLASSES)
def test_seeding_contract_rejects_non_generator(cls: type[AgentBase]) -> None:
    with pytest.raises(TypeError):
        cls(agent_id="x", rng=42)  # type: ignore[arg-type]


def test_governor_reward_is_zero_in_m1() -> None:
    """Per DECISIONS.md, Governor reward is 0.0 placeholder until M2."""
    rng = np.random.default_rng(0)
    gov = Governor(agent_id="governor_0", rng=rng)
    action = gov.action_space.sample()
    assert gov.compute_reward(_dummy_env_state(), action) == 0.0


def test_wealth_computation() -> None:
    rng = np.random.default_rng(0)
    ex = Extractor(agent_id="extractor_0", rng=rng)
    ex.inventory = {"raw": 5.0, "food": 2.0, "tool": 1.0}
    ex.cash = 10.0
    prices = {"food": 10.0, "tool": 20.0}
    # raw(5) + food(2*10) + tool(1*20) + cash(10) = 55
    assert ex.wealth(prices) == pytest.approx(55.0)


def test_governance_isolation_grep() -> None:
    """Hard CLAUDE.md rule: no governance imports inside aegora_isle/agents/.

    Matches only ``from aegora_isle.governance`` / ``import aegora_isle.governance``
    at the start of a line in ``.py`` files (skipping docstrings, comments,
    and ``__pycache__``).
    """
    agents_dir = pathlib.Path(__file__).resolve().parent.parent / "aegora_isle" / "agents"
    result = subprocess.run(
        [
            "grep",
            "-rnE",
            "--include=*.py",
            r"^\s*(from|import)\s+aegora_isle\.governance",
            str(agents_dir),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1, (
        f"Governance leakage into agents/:\n{result.stdout}"
    )


def test_agents_module_does_not_import_governance() -> None:
    """Defence in depth: walk every .py file and parse-check imports."""
    import ast
    agents_dir = pathlib.Path(__file__).resolve().parent.parent / "aegora_isle" / "agents"
    for py in agents_dir.glob("*.py"):
        tree = ast.parse(py.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert node.module is None or "governance" not in node.module, (
                    f"{py.name} imports governance: {node.module}"
                )
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "governance" not in alias.name, (
                        f"{py.name} imports {alias.name}"
                    )
