"""Governance module: the single point of regime-dependent behaviour.

``GovernanceModule`` is the **only** class other code may import from this
package (CLAUDE.md "single entry point" rule — never import ``rule_based``
or ``adaptive_rl`` directly outside this file). It loads a YAML config,
determines the active regime, and dispatches ``on_action`` / ``on_step`` to
the regime-specific handler:

    - G1 (ungoverned):  internal :class:`G1Handler` — no parameters, no RNG.
    - G2 (rule-based):  :class:`aegora_isle.governance.rule_based.RuleBasedGovernor`
    - G3 (adaptive RL): :class:`aegora_isle.governance.adaptive_rl.AdaptiveRLGovernor`
      (SAC-trained; see ``G3_PLAN.md`` and ``DECISIONS.md`` "G3 SAC action space")

The env never imports ``rule_based`` or ``adaptive_rl`` directly; it only
ever sees a ``GovernanceModule`` instance. This is what lets the same
``AegoraEnv`` codebase run all three regimes unmodified (spec AC04).

Locked interface (every regime handler implements this; do not change
without updating all three):

    on_action(agent_id, action) -> (action, reward_shaping_delta)
    on_step(env_state) -> None
    on_episode_end(env_state) -> None
    state_dict() -> dict
"""

from __future__ import annotations

from typing import Any

import numpy as np
import yaml


SUPPORTED_REGIMES: tuple[str, ...] = ("G1", "G2", "G3")


class G1Handler:
    """Ungoverned baseline. Identity on actions, no-op on world ticks.

    No randomness and no parameters — exists so :class:`AegoraEnv` has a
    zero-config default and so G1 flows through the same dispatcher
    interface as G2/G3 rather than being special-cased in the env (see
    ``G1_PLAN.md`` "why G1 still exists as a module").
    """

    regime: str = "G1"
    has_stochastic_extractor_sanction: bool = False

    def on_action(self, agent_id: str, action: Any) -> tuple[Any, float]:
        return action, 0.0

    def on_step(self, env_state: dict[str, Any]) -> None:
        return None

    def on_episode_end(self, env_state: dict[str, Any]) -> None:
        return None

    def state_dict(self) -> dict[str, Any]:
        return {"regime": "G1", "kind": "ungoverned"}


class GovernanceModule:
    """Regime dispatcher loaded from YAML. The only public governance API.

    Args:
        config_path: Path to a regime YAML (e.g.
            ``aegora_isle/governance/configs/g2_rule_based.yaml``). Must
            contain a top-level ``regime`` key with value ``G1``, ``G2``,
            or ``G3``.
        rng: Externally-seeded ``numpy.random.Generator``, passed straight
            through to whichever regime handler needs randomness (G2's
            sanction rolls, G3's SAC). Required even for G1, to keep the
            constructor signature uniform across regimes — G1 simply never
            reads it (CLAUDE.md seeding contract: no module creates its
            own Generator).
    """

    def __init__(self, config_path: str, rng: np.random.Generator) -> None:
        if not isinstance(rng, np.random.Generator):
            raise TypeError("GovernanceModule requires a numpy.random.Generator")

        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        if not cfg or "regime" not in cfg:
            raise ValueError(f"{config_path} missing required 'regime' key")

        regime = cfg["regime"]
        if regime not in SUPPORTED_REGIMES:
            raise ValueError(
                f"unknown regime {regime!r} in {config_path}; "
                f"expected one of {SUPPORTED_REGIMES}"
            )

        self._config_path = config_path
        self._cfg = cfg
        self._handler = self._build_handler(regime, cfg, rng)

    @staticmethod
    def _build_handler(regime: str, cfg: dict[str, Any], rng: np.random.Generator) -> Any:
        if regime == "G1":
            return G1Handler()

        if regime == "G2":
            # Local import: governance_module.py is the ONLY file allowed
            # to import rule_based (CLAUDE.md single-entry-point rule).
            from aegora_isle.governance.rule_based import RuleBasedGovernor

            return RuleBasedGovernor(
                extraction_cap=cfg["extraction_cap"],
                sanction_probability=cfg["sanction_probability"],
                sanction_magnitude=cfg["sanction_magnitude"],
                redistribution_period=cfg["redistribution_period"],
                redistribution_rate=cfg["redistribution_rate"],
                rng=rng,
            )

        if regime == "G3":
            # Local import: governance_module.py is the ONLY file allowed
            # to import adaptive_rl (CLAUDE.md single-entry-point rule).
            from aegora_isle.governance.adaptive_rl import AdaptiveRLGovernor

            return AdaptiveRLGovernor(
                reward_weights=cfg["reward_weights"],
                sac_kwargs=cfg["sac"],
                extraction_cap_range=tuple(cfg["extraction_cap_range"]),
                redistribution_rate_range=tuple(cfg["redistribution_rate_range"]),
                sanction_default_magnitude=cfg["sanction_default_magnitude"],
                rng=rng,
                reward_normalisation=cfg.get("reward_normalisation", "running"),
                reward_clip=cfg.get("reward_clip", 5.0),
            )

        raise AssertionError(f"unreachable: regime {regime!r}")  # validated in __init__

    @property
    def regime(self) -> str:
        return self._handler.regime

    @property
    def has_stochastic_extractor_sanction(self) -> bool:
        """Whether this regime's over-cap enforcement injects randomness
        into the Extractor's reward independent of policy quality (a
        coin-flip sanction).

        **No longer consulted by the experiment runner.** It formerly
        selected an alternative ("harvested-amount") convergence signal for
        the Extractor under G2/G3; that override was retired by client
        decision on 2026-08-09 and convergence is now judged by a single
        policy-loss criterion in every regime (see DECISIONS.md
        "Convergence measurement -- two defects corrected"). Retained
        because it remains a true, tested statement about each regime's
        mechanism and is useful to future analysis, but nothing in the
        pipeline branches on it today."""
        return self._handler.has_stochastic_extractor_sanction

    def on_action(self, agent_id: str, action: Any) -> tuple[Any, float]:
        return self._handler.on_action(agent_id, action)

    def on_step(self, env_state: dict[str, Any]) -> None:
        self._handler.on_step(env_state)

    def on_episode_end(self, env_state: dict[str, Any]) -> None:
        self._handler.on_episode_end(env_state)

    def state_dict(self) -> dict[str, Any]:
        return self._handler.state_dict()
