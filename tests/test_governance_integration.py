"""End-to-end integration tests: AegoraEnv wired with a real GovernanceModule.

Previews AC04 (same seed, different governance config -> different runs)
ahead of the M2 KPI logger landing in Sprint 2.3. The CSV-level AC04 check
happens in Sprint 2.6; this file checks the underlying mechanism — that a
GovernanceModule-G2 instance actually changes env dynamics relative to G1,
not just that the unit-level RuleBasedGovernor behaves correctly in
isolation (that's tests/test_rule_based.py's job).
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from aegora_isle.environment.aegora_env import AegoraEnv
from aegora_isle.governance.governance_module import GovernanceModule


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
CONFIGS_DIR = REPO_ROOT / "aegora_isle" / "governance" / "configs"


def _config(name: str) -> str:
    return str(CONFIGS_DIR / name)


def _run_fixed_overcap_extraction(governance_module, *, seed: int = 7, n_cycles: int = 50) -> float:
    """Run an env where every Extractor always requests 100% of
    MAX_EXTRACT_AMOUNT. Returns the final resource_pool.stock.

    Forcing a fixed, deterministic, maximal action isolates the
    governance effect (cap clipping) from action-sampling noise.
    """
    env = AegoraEnv(
        seed=seed,
        n_agents=20,
        governance_module=governance_module,
        scenario_params={
            "initial_stock": 1000.0,
            "regeneration_rate": 0.0,
            "event_rate": 0.0,
            "critical_min": 0.0,
            "scarcity_alpha": 0.0,
            "base_prices": {"food": 10.0, "tool": 20.0},
        },
        max_timesteps=n_cycles,
    )
    env.reset()
    for _ in range(n_cycles * env.n_agents):
        aid = env.agent_selection
        if env.terminations[aid] or env.truncations[aid]:
            env.step(None)
            continue
        kind = aid.split("_")[0]
        if kind == "extractor":
            action = {
                "action_type": 0,
                "amount": np.array([1.0], dtype=np.float32),
                "target_zone": 0,
            }
        else:
            action = {"action_type": 1}  # idle-equivalent for non-extractors; spaces differ
            # Non-extractor spaces don't share a uniform idle encoding, so
            # sample within-space instead — only Extractor behaviour is
            # under test here.
            action = env.action_space(aid).sample()
        env.step(action)
        if not env.agents:
            break
    return env.resource_pool.stock


def test_g2_extraction_cap_slows_depletion_relative_to_g1() -> None:
    """Under identical seeds and identical (maximal) Extractor actions,
    G2's extraction_cap=3.0 must leave MORE stock remaining than G1's
    uncapped extraction (cap = 3.0 < MAX_EXTRACT_AMOUNT = 5.0).
    """
    g1 = GovernanceModule(_config("g1_ungoverned.yaml"), np.random.default_rng(0))
    g2 = GovernanceModule(_config("g2_rule_based.yaml"), np.random.default_rng(0))

    stock_g1 = _run_fixed_overcap_extraction(g1, seed=7)
    stock_g2 = _run_fixed_overcap_extraction(g2, seed=7)

    assert stock_g2 > stock_g1, (stock_g2, stock_g1)


def test_g1_and_g2_same_seed_diverge() -> None:
    """Minimal AC04 precursor: same seed, different governance_module ->
    different resource_pool trajectory."""
    g1 = GovernanceModule(_config("g1_ungoverned.yaml"), np.random.default_rng(0))
    g2 = GovernanceModule(_config("g2_rule_based.yaml"), np.random.default_rng(0))

    stock_g1 = _run_fixed_overcap_extraction(g1, seed=99)
    stock_g2 = _run_fixed_overcap_extraction(g2, seed=99)

    assert stock_g1 != pytest.approx(stock_g2)
