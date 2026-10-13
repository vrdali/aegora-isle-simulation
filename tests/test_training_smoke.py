"""M2 Sprint 2.5 smoke test — the SB3 manual rollout collector.

Per the agreed validation scope: a short smoke-scale pilot proving the
bridge (action codec, rollout buffer bookkeeping, PPO training calls)
works end-to-end without crashing and produces non-degenerate state —
not a tuned/convergent training run. Real training scale and
hyperparameter tuning are M3's job.
"""

from __future__ import annotations

import numpy as np
import pytest

from aegora_isle.environment.aegora_env import AegoraEnv
from aegora_isle.experiments.train_agents import ManualPPOCollector
from aegora_isle.governance.governance_module import GovernanceModule
from aegora_isle.kpi.convergence import ConvergenceMonitor


REPO_ROOT = __file__.rsplit("/tests/", 1)[0]
CONFIGS_DIR = f"{REPO_ROOT}/aegora_isle/governance/configs"

SMALL_SCENARIO = {
    "initial_stock": 1000.0,
    "regeneration_rate": 0.02,
    "event_rate": 1.0,
    "critical_min": 50.0,
    "scarcity_alpha": 0.5,
    "base_prices": {"food": 10.0, "tool": 20.0},
}

SMALL_PPO_KWARGS = {"policy_kwargs": {"net_arch": [16, 16]}}


def _make_env(config_name: str, *, seed: int = 42, n_agents: int = 20, max_timesteps: int = 250) -> AegoraEnv:
    gm = GovernanceModule(f"{CONFIGS_DIR}/{config_name}", np.random.default_rng(0))
    env = AegoraEnv(
        seed=seed,
        n_agents=n_agents,
        governance_module=gm,
        scenario_params=SMALL_SCENARIO,
        max_timesteps=max_timesteps,
    )
    env.reset()
    return env


@pytest.mark.parametrize(
    "config_name",
    ["g1_ungoverned.yaml", "g2_rule_based.yaml", "g3_adaptive_rl.yaml"],
)
def test_short_pilot_runs_without_crashing(config_name: str) -> None:
    """Regime-agnostic: identical driver code, all three regimes."""
    env = _make_env(config_name)
    collector = ManualPPOCollector(env, n_steps=32, ppo_kwargs=dict(SMALL_PPO_KWARGS))
    collector.run(total_cycles=200)
    assert env._timestep > 0


def test_collector_trains_only_ppo_types() -> None:
    env = _make_env("g2_rule_based.yaml")
    collector = ManualPPOCollector(env, n_steps=32, ppo_kwargs=dict(SMALL_PPO_KWARGS))
    assert set(collector.models.keys()) == {"extractor", "producer", "trader"}
    assert "governor" not in collector.models


def test_collector_does_not_branch_on_regime() -> None:
    """Same ManualPPOCollector code path for G1 (no Governor) and G3 (SAC
    Governor self-trained elsewhere) -- proves no `if regime == ...`
    branching leaked into the training driver."""
    import inspect

    import aegora_isle.experiments.train_agents as train_agents_module

    source = inspect.getsource(train_agents_module)
    for forbidden in ('"G1"', '"G2"', '"G3"', "'G1'", "'G2'", "'G3'"):
        assert forbidden not in source, f"regime literal {forbidden} found in train_agents.py"


def test_buffer_fills_and_trains_multiple_times() -> None:
    """With n_steps=32 and a 17-extractor G1 population, 200 cycles should
    trigger several full buffer/train cycles, not just zero or one."""
    env = _make_env("g1_ungoverned.yaml")
    collector = ManualPPOCollector(env, n_steps=32, ppo_kwargs=dict(SMALL_PPO_KWARGS))
    collector.run(total_cycles=200)
    extractor_timesteps = collector.state_dict()["extractor"]["num_timesteps"]
    # 17 extractors * 200 cycles ~= 3400 transitions; at n_steps=32 that's
    # ~100 buffer-fill-train cycles -- assert comfortably more than one.
    assert extractor_timesteps > 32 * 3


def test_num_timesteps_matches_transitions_pushed() -> None:
    env = _make_env("g1_ungoverned.yaml", n_agents=8, max_timesteps=50)
    collector = ManualPPOCollector(env, n_steps=16, ppo_kwargs=dict(SMALL_PPO_KWARGS))
    collector.run(total_cycles=50)
    n_extractors = sum(1 for a in env.possible_agents if a.startswith("extractor"))
    # Each extractor contributes one transition per completed cycle, minus
    # the very first cycle (no prior obs/action exists yet to pair with
    # the first reward) -- so timesteps == n_extractors * (cycles_run - 1)
    # within a small tolerance for truncation/timing edge effects.
    expected = n_extractors * (env._timestep - 1)
    actual = collector.state_dict()["extractor"]["num_timesteps"]
    assert abs(actual - expected) <= n_extractors


def test_convergence_monitor_receives_epoch_losses() -> None:
    env = _make_env("g1_ungoverned.yaml", max_timesteps=350)
    monitor = ConvergenceMonitor()
    collector = ManualPPOCollector(
        env, n_steps=32, ppo_kwargs=dict(SMALL_PPO_KWARGS), convergence_monitor=monitor
    )
    collector.run(total_cycles=350)
    for kind in ("extractor", "producer", "trader"):
        assert len(monitor._loss_history[kind]) >= 1
        # Far short of the 50-epoch minimum at this scale -> still CONVERGING.
        assert monitor.check_convergence(kind).name == "CONVERGING"


# ------------------------------------------------------- Sprint 3.3 normalisation fix


def test_obs_and_reward_normalisers_built_per_type() -> None:
    env = _make_env("g1_ungoverned.yaml")
    collector = ManualPPOCollector(env, n_steps=32, ppo_kwargs=dict(SMALL_PPO_KWARGS))
    assert set(collector._obs_normalisers.keys()) == {"extractor", "producer", "trader"}
    assert set(collector._reward_normalisers.keys()) == {"extractor", "producer", "trader"}
    for kind in ("extractor", "producer", "trader"):
        expected_dim = env.observation_space(next(a for a in env.possible_agents if a.startswith(kind))).shape[0]
        assert collector._obs_normalisers[kind].dim == expected_dim
        assert collector._reward_normalisers[kind].dim == 1


def test_normalisers_persist_across_rollout_resets() -> None:
    """The running mean/std must track the whole run, not reset when
    SB3's fixed-size rollout buffer resets after each train() call."""
    env = _make_env("g1_ungoverned.yaml")
    collector = ManualPPOCollector(env, n_steps=32, ppo_kwargs=dict(SMALL_PPO_KWARGS))
    collector.run(total_cycles=200)  # triggers several buffer fill/reset cycles
    assert collector._obs_normalisers["extractor"].n > 32 * 3


def test_disabling_normalisation_reproduces_raw_unbounded_input() -> None:
    """Negative control: with normalisation disabled, a large-magnitude
    synthetic observation is passed straight through, unclipped -- proves
    normalise_observations=False genuinely bypasses the fix (documents the
    old pathology is still reachable for ablation), and that the enabled
    path actually changes behaviour rather than being a no-op."""
    env = _make_env("g1_ungoverned.yaml", n_agents=8, max_timesteps=30)
    agent_id = next(a for a in env.possible_agents if a.startswith("extractor"))
    obs_dim = env.observation_space(agent_id).shape[0]
    huge_obs = np.zeros(obs_dim, dtype=np.float32)
    huge_obs[3] = 5_000_000.0  # own_cash, far past obs_clip

    disabled = ManualPPOCollector(
        env, n_steps=16, ppo_kwargs=dict(SMALL_PPO_KWARGS), normalise_observations=False
    )
    disabled._act_and_record("extractor", agent_id, huge_obs)
    stored_raw = disabled._pending["extractor"][agent_id][0]
    assert stored_raw[3] == pytest.approx(5_000_000.0)

    env2 = _make_env("g1_ungoverned.yaml", n_agents=8, max_timesteps=30)
    enabled = ManualPPOCollector(
        env2, n_steps=16, ppo_kwargs=dict(SMALL_PPO_KWARGS), normalise_observations=True
    )
    enabled._act_and_record("extractor", agent_id, huge_obs)
    stored_normalised = enabled._pending["extractor"][agent_id][0]
    assert abs(stored_normalised[3]) <= enabled.obs_clip


def test_large_synthetic_cash_does_not_produce_nan_or_inf_loss() -> None:
    """Direct regression test for the Sprint 3.3 pilot bug: an Extractor
    observation sequence whose cash dimension grows unboundedly (mimicking
    unbounded wealth accumulation under G1 at real scale) must not produce
    a NaN/Inf training loss, with normalisation enabled (the default)."""
    env = _make_env("g1_ungoverned.yaml", n_agents=8, max_timesteps=50)
    collector = ManualPPOCollector(env, n_steps=16, ppo_kwargs=dict(SMALL_PPO_KWARGS))
    kind = "extractor"
    agent_id = next(a for a in env.possible_agents if a.startswith(kind))
    obs_dim = env.observation_space(agent_id).shape[0]

    for i in range(200):
        cash = 10.0 * (1.3**i)  # grows past 1e6 well within 200 iterations
        obs = np.zeros(obs_dim, dtype=np.float32)
        obs[3] = cash  # own_cash, per Extractor.get_observation
        reward = float(cash % 5.0)
        collector._consume_pending_reward(kind, agent_id, obs, reward)
        collector._act_and_record(kind, agent_id, obs)

    loss = collector.models[kind].logger.name_to_value.get("train/loss")
    assert loss is not None
    assert np.isfinite(loss)


def test_decoded_actions_satisfy_real_action_space() -> None:
    """Every action the collector emits to env.step() must be valid
    against the agent's real Dict action space (round-trip correctness
    of the action codec, exercised under real PPO-sampled flat actions,
    not just hand-built test vectors)."""
    env = _make_env("g1_ungoverned.yaml", n_agents=8, max_timesteps=30)
    collector = ManualPPOCollector(env, n_steps=16, ppo_kwargs=dict(SMALL_PPO_KWARGS))

    checked = 0
    for _ in range(30 * env.n_agents):
        if not env.agents:
            break
        agent_id = env.agent_selection
        kind = agent_id.split("_")[0]
        obs, reward, terminated, truncated, _info = env.last()
        if kind in collector.models:
            collector._consume_pending_reward(kind, agent_id, obs, reward)
        if terminated or truncated:
            env.step(None)
            continue
        if kind in collector.models:
            action = collector._act_and_record(kind, agent_id, obs)
            assert env.action_space(agent_id).contains(action), (agent_id, action)
            checked += 1
        else:
            action = env.action_space(agent_id).sample()
        env.step(action)
    assert checked > 0
