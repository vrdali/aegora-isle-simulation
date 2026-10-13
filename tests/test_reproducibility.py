"""AC06 — load JSON state -> rerun -> byte-identical CSV.

Per CLAUDE.md's testing rules, this test runs in CI on every push. It
proves the JSON state file written by KPILogger.write_json_state contains
enough information (random_seed + scenario_params) to exactly reproduce a
run's CSV, not just that calling the same Python function twice gives the
same result.

Action-space sampling is NOT seeded by AegoraEnv(seed=...) — confirmed
during M1 (see DECISIONS.md). Each agent's action space must be explicitly
seeded for the action stream itself to be reproducible; this test does so.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from aegora_isle.environment.aegora_env import AegoraEnv
from aegora_isle.governance.governance_module import GovernanceModule
from aegora_isle.kpi.kpi_logger import KPILogger


REPO_ROOT = Path(__file__).resolve().parent.parent
G1_CONFIG = str(REPO_ROOT / "aegora_isle" / "governance" / "configs" / "g1_ungoverned.yaml")
G2_CONFIG = str(REPO_ROOT / "aegora_isle" / "governance" / "configs" / "g2_rule_based.yaml")

ACTION_SEED = 7
N_AGENTS = 12
MAX_TIMESTEPS = 60
SCENARIO_PARAMS = {
    "initial_stock": 1000.0,
    "regeneration_rate": 0.02,
    "event_rate": 1.0,
    "critical_min": 50.0,
    "scarcity_alpha": 0.5,
    "base_prices": {"food": 10.0, "tool": 20.0},
}


def _run(
    config_path: str,
    seed: int,
    scenario_params: dict,
    out_csv: Path,
    run_id: str = "G1_A_001",
    scenario: str = "A",
) -> Path:
    governance = GovernanceModule(config_path, np.random.default_rng(seed))
    env = AegoraEnv(
        seed=seed,
        n_agents=N_AGENTS,
        governance_module=governance,
        scenario_params=scenario_params,
        max_timesteps=MAX_TIMESTEPS,
    )
    env.reset()
    for aid in env.agents:
        env.action_space(aid).seed(ACTION_SEED)

    logger = KPILogger(env, run_id=run_id, scenario=scenario)
    while env.agents:
        cycle_agents = list(env.agents)
        for _ in cycle_agents:
            if not env.agents:
                break
            aid = env.agent_selection
            if env.terminations[aid] or env.truncations[aid]:
                env.step(None)
            else:
                env.step(env.action_space(aid).sample())
        if env.agents:
            logger.log_step()
        if env._timestep >= MAX_TIMESTEPS:
            break

    csv_path = logger.write_csv(out_csv)
    logger.write_json_state(
        csv_path,
        governance_params=governance.state_dict(),
        scenario_params=scenario_params,
        agent_type_distribution={"extractor": 1.0},
        rl_hyperparams={},
    )
    return csv_path


def test_ac06_byte_identical_reproduction_g1(tmp_path: Path) -> None:
    csv1 = _run(G1_CONFIG, seed=2026, scenario_params=SCENARIO_PARAMS, out_csv=tmp_path / "run1.csv")
    json1 = csv1.with_suffix(".json")
    state = json.loads(json1.read_text())

    # Reconstruct entirely from the JSON state file's stored values, not
    # the original Python variables — proves the JSON is self-sufficient.
    csv2 = _run(
        G1_CONFIG,
        seed=state["random_seed"],
        scenario_params=state["scenario_params"],
        out_csv=tmp_path / "run2.csv",
    )

    assert csv1.read_bytes() == csv2.read_bytes()


def test_ac06_byte_identical_reproduction_g2(tmp_path: Path) -> None:
    """Same check under G2 — proves reproducibility holds with the
    RuleBasedGovernor's own RNG (sanction rolls) in the loop too."""
    csv1 = _run(G2_CONFIG, seed=99, scenario_params=SCENARIO_PARAMS, out_csv=tmp_path / "g2_run1.csv")
    json1 = csv1.with_suffix(".json")
    state = json.loads(json1.read_text())

    csv2 = _run(
        G2_CONFIG,
        seed=state["random_seed"],
        scenario_params=state["scenario_params"],
        out_csv=tmp_path / "g2_run2.csv",
    )

    assert csv1.read_bytes() == csv2.read_bytes()


def test_ac06_different_seed_diverges(tmp_path: Path) -> None:
    """Sanity check on the test itself: different seeds must NOT match,
    or the byte-identical assertions above would be vacuously true."""
    csv1 = _run(G1_CONFIG, seed=1, scenario_params=SCENARIO_PARAMS, out_csv=tmp_path / "a.csv")
    csv2 = _run(G1_CONFIG, seed=2, scenario_params=SCENARIO_PARAMS, out_csv=tmp_path / "b.csv")
    assert csv1.read_bytes() != csv2.read_bytes()
