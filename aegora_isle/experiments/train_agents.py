"""Manual PPO rollout collector for Extractor/Producer/Trader.

Per M2_PLAN.md Sprint 2.5 / DECISIONS.md: no SuperSuit, one PPO model per
agent type (parameter-shared across all agents of that type), manual
rollout collection by walking ``env.agent_iter()`` directly rather than
using SB3's ``.learn()`` (which assumes it owns the env's stepping cadence
exclusively — incompatible with interleaving several independently-driven
models against one shared multi-agent env).

**Regime-agnostic by design**: this module never branches on G1/G2/G3.
G3's SAC training is entirely self-contained inside
``AdaptiveRLGovernor.on_step()`` (Sprint 2.5's other half), invisible to
this driver — it only ever sees Extractor/Producer/Trader turns, under
any regime, identically. This preserves CLAUDE.md's "no regime branching
outside governance/" rule even in the training script.

Implementation corrections from a design review against the actual
installed SB3 2.9.0 source (see ``DECISIONS.md`` "manual rollout collector
corrections"):

- Actions/values/log-probs come from ``model.policy(obs_tensor)`` —
  **not** ``model.predict()``, which cannot supply value/log_prob at all
  (confirmed: ``BasePolicy.predict()`` returns only ``(actions, state)``).
- ``model.num_timesteps`` is incremented manually per transition pushed
  (``collect_rollouts()`` does this as a side effect we must replicate).
- ``episode_start`` is tracked **per agent type**, true only for that
  type's very first pushed transition in this run — multi-episode
  resets mid-collection are out of scope for this smoke-validation pass
  (see module docstring "Known simplifications").
- ``model._current_progress_remaining`` is held constant (LR/clip-range
  schedules don't anneal) — an accepted, documented M2 simplification,
  not an oversight.

Known simplifications (acceptable for this sprint's validation scope —
a short smoke pilot proving the bridge works, not a tuned training run;
real-scale training is M3's job):

- Single continuous episode per ``run()`` call — no mid-collection
  ``env.reset()`` handling.

Convergence-loss recording reads PPO's ``train/policy_gradient_loss``
scalar from ``model.logger.name_to_value`` after each ``train()`` call —
i.e. **policy loss**, exactly as spec §6.3 specifies ("rolling mean policy
loss variance"). This previously read the combined ``train/loss`` scalar
(policy + value*vf_coef + entropy*ent_coef), which was a spec deviation
with real consequences: the value-loss term is dominated by how
*predictable* an agent's reward is, not by whether its policy has settled.
Producer's reward is irreducibly noisy (random counterparty via
``_pick_counterparty``, partial fills), so its value loss stayed high
forever and swamped the measurement — producing a spurious 0%-convergence
result across G2/G3. See DECISIONS.md "Convergence measurement — two
defects corrected".

``train/value_loss`` is still recorded, but into a **separate,
purely-observational** monitor that never gates convergence status (client
request: keep value-function instability visible without letting it decide
outcomes).
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch as th
from gymnasium import spaces

from aegora_isle.agents.action_codec import decode_action, flatten_action_space
from aegora_isle.experiments.normalisation import RunningNormaliser
from aegora_isle.kpi.convergence import ConvergenceMonitor

DEFAULT_AGENT_TYPES: tuple[str, ...] = ("extractor", "producer", "trader")
EPOCH_LENGTH: int = 100  # mirrors kpi_logger.EPOCH_LENGTH / convergence.py's epoch unit


class _DummyTypeEnv(gym.Env):
    """gym.Env shell for one agent type, used only for SB3 space
    introspection at PPO construction time. Never stepped."""

    metadata: dict[str, Any] = {}

    def __init__(self, observation_space: spaces.Box, action_space: spaces.Box) -> None:
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        return self.observation_space.sample(), {}

    def step(self, action: Any):
        return self.observation_space.sample(), 0.0, False, False, {}


class ManualPPOCollector:
    """Drives one shared PPO model per agent type against a real AegoraEnv.

    Args:
        env: A reset ``AegoraEnv`` instance (any governance regime).
        agent_types: Types to train. Defaults to the three PPO-trained
            types; Governor is intentionally excluded (G1/G2: not
            trained here; G3: trained separately inside
            ``AdaptiveRLGovernor``).
        n_steps: Rollout buffer size per model before each ``train()``
            call. Kept small by default for smoke-scale validation runs
            (a real M3 run would use SB3's usual 2048).
        ppo_kwargs: Forwarded to every ``PPO`` constructor (shared across
            types; per-type overrides aren't needed for this sprint).
        convergence_monitor: Optional; if given, ``run()`` records one
            epoch loss per type every ``EPOCH_LENGTH`` env timesteps.
        normalise_observations: Whether to feed each type's policy/value
            network a running-normalised observation instead of the raw
            one. Default True -- confirmed by the Milestone 3 Sprint 3.3
            pilot that raw, unbounded observations (cash/inventory can
            grow unboundedly over a long run, especially under G1 where no
            governance caps exist) destabilise PPO training at real scale.
            See DECISIONS.md "PPO observation/reward normalisation --
            confirmed gap, fixed".
        normalise_rewards: Same idea, for the scalar reward fed into the
            rollout buffer. Default True, applied uniformly across all
            three PPO-trained types (Extractor's reward is bounded and
            unaffected in practice; Producer/Trader's are not confirmed
            bounded).
        obs_clip: Clip bound for normalised observations. Matches SB3's
            own ``VecNormalize`` default (``clip_obs=10.0``) -- this fix is
            reinstating what ``VecNormalize`` would have supplied had this
            project used SB3's own ``.learn()``.
        reward_clip: Clip bound for normalised rewards. Matches the
            already-accepted ``_RunningNormaliser`` default used for G3's
            SAC composite reward elsewhere in this codebase.
    """

    def __init__(
        self,
        env: Any,
        agent_types: tuple[str, ...] = DEFAULT_AGENT_TYPES,
        n_steps: int = 128,
        ppo_kwargs: dict[str, Any] | None = None,
        convergence_monitor: ConvergenceMonitor | None = None,
        value_diagnostic_monitor: ConvergenceMonitor | None = None,
        normalise_observations: bool = True,
        normalise_rewards: bool = True,
        obs_clip: float = 10.0,
        reward_clip: float = 5.0,
    ) -> None:
        self.env = env
        self.agent_types = tuple(t for t in agent_types if self._type_present(env, t))
        self.n_steps = int(n_steps)
        self.convergence_monitor = convergence_monitor
        self.value_diagnostic_monitor = value_diagnostic_monitor
        self.normalise_observations = bool(normalise_observations)
        self.normalise_rewards = bool(normalise_rewards)
        self.obs_clip = float(obs_clip)
        self.reward_clip = float(reward_clip)

        self.models: dict[str, Any] = {}
        self._real_action_space: dict[str, spaces.Dict] = {}
        self._obs_normalisers: dict[str, RunningNormaliser] = {}
        self._reward_normalisers: dict[str, RunningNormaliser] = {}
        for kind in self.agent_types:
            self.models[kind] = self._build_model(kind, dict(ppo_kwargs or {}))

        # Per-(type, agent_id) pending transition awaiting its delayed
        # AEC reward: (obs, flat_action, value, log_prob).
        self._pending: dict[str, dict[str, tuple]] = {t: {} for t in self.agent_types}
        self._first_push: dict[str, bool] = {t: True for t in self.agent_types}
        self._last_epoch_recorded: dict[str, int] = {t: 0 for t in self.agent_types}
        # Separate epoch cursor for the observational value-loss series, so
        # it can never interfere with the gating convergence series above.
        self._last_value_epoch_recorded: dict[str, int] = {t: 0 for t in self.agent_types}

    @staticmethod
    def _type_present(env: Any, kind: str) -> bool:
        return any(a.startswith(kind) for a in env.possible_agents)

    def _build_model(self, kind: str, ppo_kwargs: dict[str, Any]) -> Any:
        from stable_baselines3 import PPO
        from stable_baselines3.common.logger import configure

        sample_aid = next(a for a in self.env.possible_agents if a.startswith(kind))
        real_space = self.env.action_space(sample_aid)
        self._real_action_space[kind] = real_space
        flat_action_space = flatten_action_space(real_space)
        obs_space = self.env.observation_space(sample_aid)

        dummy_env = _DummyTypeEnv(obs_space, flat_action_space)
        kwargs = dict(ppo_kwargs)
        kwargs.setdefault("n_steps", self.n_steps)
        kwargs.setdefault("batch_size", min(64, self.n_steps))
        model = PPO(policy="MlpPolicy", env=dummy_env, verbose=0, **kwargs)
        # .learn() normally configures this; we bypass it entirely.
        model.set_logger(configure(folder=None, format_strings=[]))

        self._obs_normalisers[kind] = RunningNormaliser(dim=int(obs_space.shape[0]), clip=self.obs_clip)
        self._reward_normalisers[kind] = RunningNormaliser(dim=1, clip=self.reward_clip)
        return model

    # ------------------------------------------------------------------ run

    def run(self, total_cycles: int) -> None:
        """Drive the env for up to ``total_cycles`` cycles (env timesteps),
        collecting rollouts and training every ``n_steps`` transitions per
        type. Non-tracked agent types (e.g. Governor under G1/G2) act
        randomly — this driver only trains Extractor/Producer/Trader.
        """
        env = self.env
        max_iters = total_cycles * env.n_agents
        for _ in range(max_iters):
            if not env.agents:
                break
            agent_id = env.agent_selection
            kind = agent_id.split("_")[0]
            obs, reward, terminated, truncated, _info = env.last()

            if kind in self.models:
                self._consume_pending_reward(kind, agent_id, obs, reward)

            if terminated or truncated:
                env.step(None)
                continue

            if kind in self.models:
                action = self._act_and_record(kind, agent_id, obs)
            else:
                action = env.action_space(agent_id).sample()
            env.step(action)

            if self.convergence_monitor is not None:
                self._maybe_record_epoch_loss()

    def _consume_pending_reward(self, kind: str, agent_id: str, obs: np.ndarray, reward: float) -> None:
        pending = self._pending[kind].pop(agent_id, None)
        if pending is None:
            return
        prev_obs, flat_action, value, log_prob = pending  # prev_obs already normalised, if enabled
        model = self.models[kind]
        episode_start = self._first_push[kind]
        self._first_push[kind] = False

        if self.normalise_rewards:
            r_norm = self._reward_normalisers[kind]
            raw_r = np.array([reward], dtype=np.float64)
            r_norm.update(raw_r)
            reward_to_store = float(r_norm.normalise(raw_r)[0])
        else:
            reward_to_store = float(reward)

        model.rollout_buffer.add(
            np.asarray(prev_obs, dtype=np.float32).reshape(1, -1),
            flat_action.reshape(1, -1),
            np.array([reward_to_store], dtype=np.float32),
            np.array([episode_start]),
            value,
            log_prob,
        )
        model.num_timesteps += 1

        if model.rollout_buffer.full:
            self._finish_rollout(kind, obs)

    def _act_and_record(self, kind: str, agent_id: str, obs: np.ndarray) -> dict[str, Any]:
        from stable_baselines3.common.utils import obs_as_tensor

        model = self.models[kind]
        raw_obs = np.asarray(obs, dtype=np.float32).reshape(-1)

        if self.normalise_observations:
            normaliser = self._obs_normalisers[kind]
            normaliser.update(raw_obs)
            model_input = normaliser.normalise(raw_obs).astype(np.float32)
        else:
            model_input = raw_obs

        obs_tensor = obs_as_tensor(model_input.reshape(1, -1), model.device)
        with th.no_grad():
            flat_action_t, value, log_prob = model.policy(obs_tensor)
        flat_action = flat_action_t.cpu().numpy().flatten()

        # Store the NORMALISED obs, not raw -- must match what produced
        # this action/value/log_prob, or train() later re-evaluates the
        # policy on a different input distribution than the one that
        # generated this transition.
        self._pending[kind][agent_id] = (model_input, flat_action, value, log_prob)

        real_space = self._real_action_space[kind]
        return decode_action(real_space, flat_action)

    def _finish_rollout(self, kind: str, bootstrap_obs: np.ndarray) -> None:
        """Buffer full: bootstrap, compute GAE, train, reset for the next batch."""
        from stable_baselines3.common.utils import obs_as_tensor

        model = self.models[kind]
        raw_boot = np.asarray(bootstrap_obs, dtype=np.float32).reshape(-1)

        if self.normalise_observations:
            # normalise() ONLY -- no update(). This exact raw observation
            # is about to be update()'d for real on the very next
            # _act_and_record call for this same agent (unless it just
            # terminated); calling update() here too would double-count it.
            boot_input = self._obs_normalisers[kind].normalise(raw_boot).astype(np.float32)
        else:
            boot_input = raw_boot

        obs_tensor = obs_as_tensor(boot_input.reshape(1, -1), model.device)
        with th.no_grad():
            last_values = model.policy.predict_values(obs_tensor)
        # Single shared episode for this smoke-validation scope (see module
        # docstring) — never treat the bootstrap point as terminal.
        model.rollout_buffer.compute_returns_and_advantage(last_values=last_values, dones=np.array([False]))
        model.train()
        model.rollout_buffer.reset()

    def _maybe_record_epoch_loss(self) -> None:
        """Records this epoch's **policy loss** into the gating convergence
        monitor (spec §6.3), and — separately and observationally only —
        this epoch's value loss into ``value_diagnostic_monitor``.

        The two are deliberately independent: the value series never
        influences convergence classification. See the module docstring.
        """
        timestep = self.env._timestep
        epoch = timestep // EPOCH_LENGTH
        for kind, model in self.models.items():
            if epoch == 0:
                continue
            name_to_value = model.logger.name_to_value

            if epoch > self._last_epoch_recorded[kind]:
                policy_loss = name_to_value.get("train/policy_gradient_loss")
                if policy_loss is not None:
                    self.convergence_monitor.record_epoch_loss(kind, float(policy_loss))
                    self._last_epoch_recorded[kind] = epoch

            if (
                self.value_diagnostic_monitor is not None
                and epoch > self._last_value_epoch_recorded[kind]
            ):
                value_loss = name_to_value.get("train/value_loss")
                if value_loss is not None:
                    self.value_diagnostic_monitor.record_epoch_loss(
                        f"{kind}_value", float(value_loss)
                    )
                    self._last_value_epoch_recorded[kind] = epoch

    # ---------------------------------------------------------- bookkeeping

    def state_dict(self) -> dict[str, Any]:
        return {
            kind: {
                "num_timesteps": model.num_timesteps,
                "rollout_buffer_pos": model.rollout_buffer.pos,
            }
            for kind, model in self.models.items()
        }
