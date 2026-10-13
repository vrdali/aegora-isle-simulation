"""G3: adaptive reinforcement-learning governance via SAC.

Per CLAUDE.md's single-entry-point rule, this module is imported **only**
from ``governance_module.py`` (and from tests). The env never imports it
directly.

Design rationale lives in ``G3_PLAN.md``; this implementation incorporates
corrections from a design review against the actual installed SB3 2.9.0
source (see ``DECISIONS.md`` "G3 SAC action space" and "SAC replay buffer
fix"):

- SAC's action space is the **3-dim** ``Box(0,1,(3,))`` =
  ``[cap_norm, redist_norm, sanction_probability]`` already specified in
  ``G3_PLAN.md``'s own "Action (SAC output)" table — **not** a mechanical
  flatten of the Governor agent's full ``Dict{action_type, value,
  target_agent_idx}`` (505 dims once one-hot-expanded), which would bury
  the 3 dims that matter inside ~500 nearly-always-irrelevant dims and
  badly condition the critic. "Which agent to sanction" is handled the
  same way G2 handles cap violations: reactively, per-Extractor, inside
  ``on_action`` — never SAC-learned targeting. The Governor *agent's* own
  Dict action space in ``governor.py`` is unaffected by this; individual
  Governor agents' emitted actions are not consumed by this handler.
- ``on_step`` fills ``model.replay_buffer`` via ``replay_buffer.add(...)``
  *before* calling ``model.train()`` — a gap in the original design sketch
  that would otherwise have trained on an empty buffer (no crash, a
  real-looking loss number, zero actual learning).
- The composite reward's sustainability term divides ``depletion_rate`` by
  its now-fractional, positive-means-depletion convention (Sprint 2.3),
  not the old ``max(0, 1 + depletion_rate)`` which assumed the opposite
  sign and percentage units.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


# Mirrors aegora_isle.environment.aegora_env.MAX_EXTRACT_AMOUNT.
# Deliberately duplicated, not imported — same pattern and rationale as
# rule_based.py's MAX_EXTRACT_AMOUNT. Drift-detector test in
# tests/test_adaptive_rl.py asserts the two stay equal.
MAX_EXTRACT_AMOUNT: float = 5.0

_ACTION_EXTRACT: int = 0

# SAC's 3-dim macro-action space. Order is locked: cap, redistribution,
# sanction probability.
SAC_ACTION_DIM: int = 3


class _RunningNormaliser:
    """Welford online mean/std, for reward-component normalisation.

    No ``numpy.random.Generator`` needed — this is deterministic
    bookkeeping, nothing stochastic happens here.
    """

    def __init__(self, clip: float = 5.0) -> None:
        self.mean: float = 0.0
        self._m2: float = 0.0
        self.n: int = 0
        self.clip = float(clip)

    def update(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self._m2 += delta * (x - self.mean)

    def normalise(self, x: float) -> float:
        if self.n < 2:
            return float(np.clip(x, -self.clip, self.clip))
        variance = self._m2 / (self.n - 1)
        std = max(1e-6, variance**0.5)
        return float(np.clip((x - self.mean) / std, -self.clip, self.clip))


class _DummyGovernorEnv(gym.Env):
    """Minimal gymnasium.Env, used only so SB3 can introspect
    observation_space/action_space when constructing the SAC model. Never
    stepped — all real transitions are collected manually in on_step().
    Must be a real gym.Env subclass: SB3's _patch_env rejects anything
    that merely duck-types the interface.
    """

    metadata: dict[str, Any] = {}

    def __init__(self, observation_space: spaces.Box, action_space: spaces.Box) -> None:
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        return self.observation_space.sample(), {}

    def step(self, action: Any):
        obs = self.observation_space.sample()
        return obs, 0.0, False, False, {}


class AdaptiveRLGovernor:
    """G3: SAC-trained Governor with composite reward per spec §2.2.2.

    Args:
        reward_weights: ``{"equality": ..., "sustainability": ..., "productivity": ...}``,
            must sum to 1.0 (validated against spec-confirmed defaults
            0.40/0.35/0.25 — see DECISIONS.md).
        sac_kwargs: Forwarded to ``stable_baselines3.SAC``'s constructor
            (learning_rate, buffer_size, batch_size, gamma, tau, ent_coef,
            train_freq, gradient_steps, learning_starts,
            target_update_interval, policy_kwargs). ``train_freq`` is
            converted from the YAML's list form to the tuple SB3 requires.
        extraction_cap_range: ``(lo, hi)`` raw units; SAC's ``cap_norm``
            output in [0,1] maps linearly into this range.
        redistribution_rate_range: ``(lo, hi)``; SAC's ``redist_norm``
            output in [0,1] maps linearly into this range.
        sanction_default_magnitude: Reward delta applied when a sanction
            fires (mirrors G2's ``sanction_magnitude``).
        rng: Externally-seeded ``numpy.random.Generator`` (CLAUDE.md
            seeding contract). Used for the warm-up random-action phase
            and the sanction roll — SAC's own internal torch RNG is a
            separate concern SB3 manages itself.
        reward_normalisation: ``"running"`` (Welford per-component) or
            ``"none"``.
        reward_clip: Clip bound for normalised components.
        max_extract_amount: Must match the env's ``MAX_EXTRACT_AMOUNT``.
    """

    regime: str = "G3"
    # Extractor over-cap requests are sanctioned by SAC's own
    # ``current_sanction_probability`` (dynamic, but still a coin-flip),
    # independent of policy quality -- same rationale as G2's flag. Like
    # G2's, this no longer selects a convergence signal: the
    # harvested-amount override was retired 2026-08-09 (see DECISIONS.md
    # "Convergence measurement -- two defects corrected"). Under G3 that
    # override was actively harmful -- it measured this Governor's own cap
    # adaptivity and penalised the regime for adapting.
    has_stochastic_extractor_sanction: bool = True

    def __init__(
        self,
        reward_weights: dict[str, float],
        sac_kwargs: dict[str, Any],
        extraction_cap_range: tuple[float, float],
        redistribution_rate_range: tuple[float, float],
        sanction_default_magnitude: float,
        rng: np.random.Generator,
        reward_normalisation: str = "running",
        reward_clip: float = 5.0,
        max_extract_amount: float = MAX_EXTRACT_AMOUNT,
    ) -> None:
        if not isinstance(rng, np.random.Generator):
            raise TypeError("AdaptiveRLGovernor requires a numpy.random.Generator")
        if abs(sum(reward_weights.values()) - 1.0) > 1e-6:
            raise ValueError(f"reward_weights must sum to 1.0, got {reward_weights}")
        if reward_normalisation not in ("running", "none"):
            raise ValueError(f"unknown reward_normalisation {reward_normalisation!r}")

        self.weights = dict(reward_weights)
        self.cap_lo, self.cap_hi = extraction_cap_range
        self.redist_lo, self.redist_hi = redistribution_rate_range
        self.sanction_default_magnitude = float(sanction_default_magnitude)
        self.rng = rng
        self.max_extract_amount = float(max_extract_amount)

        self._sac_kwargs = dict(sac_kwargs)
        if "train_freq" in self._sac_kwargs:
            tf = self._sac_kwargs["train_freq"]
            self._sac_kwargs["train_freq"] = tuple(tf) if isinstance(tf, list) else tf
        self._train_freq_steps = (
            self._sac_kwargs["train_freq"][0]
            if isinstance(self._sac_kwargs.get("train_freq"), tuple)
            else int(self._sac_kwargs.get("train_freq", 4))
        )
        self._learning_starts = int(self._sac_kwargs.get("learning_starts", 1000))
        self._gradient_steps = int(self._sac_kwargs.get("gradient_steps", 1))

        self._normalisers: dict[str, _RunningNormaliser] | None = (
            {k: _RunningNormaliser(reward_clip) for k in reward_weights}
            if reward_normalisation == "running"
            else None
        )

        # Mirrors G2's structural defaults until SAC's first decision lands.
        self.current_cap: float = float(np.mean(extraction_cap_range))
        self.current_redist_rate: float = 0.0
        self.current_sanction_probability: float = 0.0

        self.model: Any = None  # built lazily on first on_step (needs obs space)
        self._prev_obs: np.ndarray | None = None
        self._prev_sac_action: np.ndarray | None = None
        self._step_count: int = 0
        self._sanction_count: int = 0
        self._redistribution_count: int = 0

    # ----------------------------------------------------- composite reward

    def _component_equality(self, kpis: dict[str, float]) -> float:
        return 1.0 - float(kpis.get("gini", 0.0))

    def _component_sustainability(self, env_state: dict[str, Any]) -> float:
        """``stock_norm * max(0, 1 - depletion_rate)``.

        ``depletion_rate`` here is the live, fractional, positive-means-
        depletion value injected by ``aegora_env._update_live_kpis``
        (Sprint 2.3) — NOT the CSV's percentage column. A subtraction
        (not the addition an earlier sketch used) is correct for that
        sign convention: more depletion -> lower sustainability.
        """
        pool = env_state["resource_pool"]
        stock_norm = pool["stock"] / pool["initial_stock"] if pool["initial_stock"] > 0 else 0.0
        stock_norm = float(np.clip(stock_norm, 0.0, 1.0))
        depletion = float(env_state.get("kpis", {}).get("depletion_rate", 0.0))
        return stock_norm * max(0.0, 1.0 - depletion)

    def _component_productivity(self, kpis: dict[str, float]) -> float:
        return float(np.log1p(max(0.0, float(kpis.get("gdp_indexed", 0.0)))))

    def compute_composite_reward(self, env_state: dict[str, Any]) -> float:
        kpis = env_state.get("kpis", {})
        raw = {
            "equality": self._component_equality(kpis),
            "sustainability": self._component_sustainability(env_state),
            "productivity": self._component_productivity(kpis),
        }
        if self._normalisers is not None:
            for key, value in raw.items():
                self._normalisers[key].update(value)
            scored = {key: self._normalisers[key].normalise(value) for key, value in raw.items()}
        else:
            scored = raw
        return float(sum(self.weights[key] * scored[key] for key in raw))

    # ----------------------------------------------------- env-facing hooks

    def on_action(self, agent_id: str, action: dict[str, Any]) -> tuple[dict[str, Any], float]:
        """Identical structure to G2's on_action, but cap/sanction-probability
        are SAC's current macro-decision (mutable), not YAML constants."""
        if not agent_id.startswith("extractor"):
            return action, 0.0
        if int(action["action_type"]) != _ACTION_EXTRACT:
            return action, 0.0

        frac = float(np.asarray(action["amount"]).reshape(-1)[0])
        requested = frac * self.max_extract_amount
        if requested <= self.current_cap:
            return action, 0.0

        cap_frac = self.current_cap / self.max_extract_amount
        modified = dict(action)
        modified["amount"] = np.array([cap_frac], dtype=np.float32)

        reward_delta = 0.0
        if self.rng.random() < self.current_sanction_probability:
            reward_delta = self.sanction_default_magnitude
            self._sanction_count += 1

        return modified, reward_delta

    def on_step(self, env_state: dict[str, Any]) -> None:
        """End-of-cycle hook: fill the replay buffer, sample a new macro
        decision, apply redistribution, train SAC every _train_freq_steps
        cycles once warmed up.
        """
        self._step_count += 1

        governor = self._find_governor(env_state)
        obs = governor.get_observation(env_state)
        reward = self.compute_composite_reward(env_state)
        done = self._is_terminal(env_state)

        if self.model is None:
            self._build_model(governor.observation_space)

        if self._prev_obs is not None and self._prev_sac_action is not None:
            self.model.replay_buffer.add(
                self._prev_obs.reshape(1, -1).astype(np.float32),
                obs.reshape(1, -1).astype(np.float32),
                self._prev_sac_action.reshape(1, -1).astype(np.float32),
                np.array([reward], dtype=np.float32),
                np.array([done]),
                [{}],
            )
            self.model.num_timesteps += 1

        sac_action = self._sample_action(obs)
        self._prev_obs = None if done else obs
        self._prev_sac_action = None if done else sac_action

        self.current_cap = self._unbox(sac_action[0], self.cap_lo, self.cap_hi)
        self.current_redist_rate = self._unbox(sac_action[1], self.redist_lo, self.redist_hi)
        self.current_sanction_probability = float(np.clip(sac_action[2], 0.0, 1.0))

        if self.current_redist_rate > 0.0:
            self._redistribute(env_state)

        buffer_size = self.model.replay_buffer.size()
        if buffer_size >= self._learning_starts and self._step_count % self._train_freq_steps == 0:
            self.model.train(gradient_steps=self._gradient_steps)

    def on_episode_end(self, env_state: dict[str, Any]) -> None:
        self._prev_obs = None
        self._prev_sac_action = None

    # -------------------------------------------------------------- helpers

    def _find_governor(self, env_state: dict[str, Any]) -> Any:
        """Reuses a live Governor agent's own get_observation/observation_space
        — same construction Governor already declares and tests, rather
        than duplicating the KPI-vector assembly (and its exact, non-
        uniform per-dimension bounds) here."""
        agents = env_state.get("agents_by_id", {})
        governor = next((a for aid, a in agents.items() if aid.startswith("governor")), None)
        if governor is None:
            raise RuntimeError(
                "AdaptiveRLGovernor requires at least one Governor agent in the "
                "population (G3 population split includes Governors per DECISIONS.md)"
            )
        return governor

    def _is_terminal(self, env_state: dict[str, Any]) -> bool:
        """Best-available terminal signal from env_state alone: the env's
        own collapse check. Horizon-truncation isn't detectable here
        (max_timesteps isn't in the snapshot) — but treating a horizon
        cutoff as non-terminal (done=False, always bootstrap) is the
        conventionally correct treatment of censored data anyway, not
        just a simplification. See DECISIONS.md "SAC done-flag scope".
        """
        pool = env_state["resource_pool"]
        return bool(pool["stock"] < pool["critical_min"])

    def _sample_action(self, obs: np.ndarray) -> np.ndarray:
        """Random during SAC's warm-up (mirrors SB3's own .learn() warm-up
        behaviour), policy-sampled thereafter."""
        if self.model.num_timesteps < self._learning_starts:
            action = self.model.action_space.sample()
        else:
            action, _ = self.model.predict(obs, deterministic=False)
        return np.asarray(action, dtype=np.float32)

    def _build_model(self, governor_observation_space: spaces.Box) -> None:
        from stable_baselines3 import SAC
        from stable_baselines3.common.logger import configure

        sac_action_space = spaces.Box(low=0.0, high=1.0, shape=(SAC_ACTION_DIM,), dtype=np.float32)
        dummy_env = _DummyGovernorEnv(governor_observation_space, sac_action_space)
        self.model = SAC(policy="MlpPolicy", env=dummy_env, verbose=0, **self._sac_kwargs)
        # .learn() normally configures this; since we bypass it entirely
        # (manual replay-buffer + train() calls), train()'s internal
        # self.logger.record(...) calls need a logger to exist at all.
        # Empty format_strings -> no-op, nothing is actually written.
        self.model.set_logger(configure(folder=None, format_strings=[]))

    def _redistribute(self, env_state: dict[str, Any]) -> None:
        """Identical mechanism to RuleBasedGovernor._redistribute, reading
        self.current_redist_rate (SAC-set) instead of a YAML constant."""
        agents = env_state.get("agents_by_id")
        if not agents:
            return
        donors = [a for aid, a in agents.items() if aid.startswith(("producer", "trader"))]
        recipients = [a for aid, a in agents.items() if aid.startswith("extractor")]
        if not donors or not recipients:
            return

        total_pool = 0.0
        for donor in donors:
            contribution = float(donor.cash) * self.current_redist_rate
            donor.cash = float(donor.cash) - contribution
            total_pool += contribution

        if total_pool <= 0.0:
            return

        per_recipient = total_pool / len(recipients)
        for recipient in recipients:
            recipient.cash = float(recipient.cash) + per_recipient
        self._redistribution_count += 1

    @staticmethod
    def _unbox(box_value: float, lo: float, hi: float) -> float:
        return lo + float(np.clip(box_value, 0.0, 1.0)) * (hi - lo)

    # ---------------------------------------------------------- bookkeeping

    def state_dict(self) -> dict[str, Any]:
        return {
            "regime": "G3",
            "weights": self.weights,
            "current_cap": self.current_cap,
            "current_redist_rate": self.current_redist_rate,
            "current_sanction_probability": self.current_sanction_probability,
            "step_count": self._step_count,
            "sanction_count": self._sanction_count,
            "redistribution_count": self._redistribution_count,
            "model_built": self.model is not None,
        }
