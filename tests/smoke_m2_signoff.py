"""M2 Sprint 2.6 sign-off pass — AC04/AC05 spot-check across all 3 regimes.

Generates one CSV+JSON pair per regime (G1/G2/G3) at a matched scenario
and seed, using identically-seeded action streams (per
test_governance_integration.py's pattern) so governance is the only
variable that differs between runs. This is the underlying verification
the Sprint 2.6 checklist in M2_PLAN.md calls for; it's written as a direct
script rather than `python -m ...train_agents --governance ... --output_dir
...` because that CLI is `run_experiment.py`'s job (M3 Sprint 3.1) — out
of M2's scope. `ManualPPOCollector` (Sprint 2.5) trains the PPO models in
the loop too, so this also doubles as a multi-regime training smoke check.

Run from repo root::

    uv run python tests/smoke_m2_signoff.py
"""

from __future__ import annotations

import pathlib

import numpy as np

from aegora_isle.environment.aegora_env import AegoraEnv
from aegora_isle.experiments.train_agents import ManualPPOCollector
from aegora_isle.governance.governance_module import GovernanceModule
from aegora_isle.kpi.kpi_logger import KPILogger


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
CONFIGS_DIR = REPO_ROOT / "aegora_isle" / "governance" / "configs"
OUT_DIR = REPO_ROOT / "data" / "smoke" / "m2_signoff"

SEED = 42
ACTION_SEED = 7
N_AGENTS = 20
MAX_TIMESTEPS = 300
SMALL_PPO_KWARGS = {"policy_kwargs": {"net_arch": [16, 16]}}
SCENARIO_A = {
    "initial_stock": 10000.0,
    "regeneration_rate": 0.05,
    "event_rate": 0.5,
    "critical_min": 100.0,
    "scarcity_alpha": 0.5,
    "base_prices": {"food": 10.0, "tool": 20.0},
}


def run_regime(config_name: str, scenario: str) -> pathlib.Path:
    gm = GovernanceModule(str(CONFIGS_DIR / config_name), np.random.default_rng(0))
    env = AegoraEnv(
        seed=SEED,
        n_agents=N_AGENTS,
        governance_module=gm,
        scenario_params=SCENARIO_A,
        max_timesteps=MAX_TIMESTEPS,
    )
    env.reset()
    for aid in env.agents:
        env.action_space(aid).seed(ACTION_SEED)

    logger = KPILogger(env, run_id=f"{gm.regime}_{scenario}_001", scenario=scenario)
    collector = ManualPPOCollector(env, n_steps=32, ppo_kwargs=dict(SMALL_PPO_KWARGS))

    last_timestep = 0
    while env.agents and env._timestep < MAX_TIMESTEPS:
        agent_id = env.agent_selection
        kind = agent_id.split("_")[0]
        obs, reward, terminated, truncated, _info = env.last()
        if kind in collector.models:
            collector._consume_pending_reward(kind, agent_id, obs, reward)
        if terminated or truncated:
            env.step(None)
        elif kind in collector.models:
            env.step(collector._act_and_record(kind, agent_id, obs))
        else:
            env.step(env.action_space(agent_id).sample())
        if env._timestep != last_timestep:
            logger.log_step()
            last_timestep = env._timestep

    out_dir = OUT_DIR / gm.regime
    csv_path = logger.write_csv(out_dir / f"{gm.regime}_{scenario}.csv")
    logger.write_json_state(
        csv_path,
        governance_params=gm.state_dict(),
        scenario_params=SCENARIO_A,
        agent_type_distribution={"extractor": 0.25, "producer": 0.25, "trader": 0.25, "governor": 0.25},
        rl_hyperparams=SMALL_PPO_KWARGS,
    )
    print(f"{gm.regime}: wrote {csv_path} ({env._timestep} timesteps, {len(env.agents)} agents alive)")
    return csv_path


def main() -> int:
    paths = {}
    for config_name in ("g1_ungoverned.yaml", "g2_rule_based.yaml", "g3_adaptive_rl.yaml"):
        csv_path = run_regime(config_name, scenario="A")
        regime = config_name.split("_")[0].upper()
        paths[regime] = csv_path

    g1_bytes = paths["G1"].read_bytes()
    g2_bytes = paths["G2"].read_bytes()
    g3_bytes = paths["G3"].read_bytes()

    print()
    print("AC04 check (same seed, different regime -> different CSV):")
    print("  G1 == G2:", g1_bytes == g2_bytes, "(expect False)")
    print("  G1 == G3:", g1_bytes == g3_bytes, "(expect False)")
    print("  G2 == G3:", g2_bytes == g3_bytes, "(expect False)")

    ok = (g1_bytes != g2_bytes) and (g1_bytes != g3_bytes) and (g2_bytes != g3_bytes)
    print("AC04:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
