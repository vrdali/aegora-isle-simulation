"""G2: rule-based static governance.

Per CLAUDE.md's single-entry-point rule, this module is imported **only**
from ``governance_module.py`` (and from tests). The env never imports it
directly.

Design rationale lives in ``G2_PLAN.md``. Short version: G2 is the
"non-AI bureaucracy" baseline — fixed extraction caps, scheduled
redistribution, probabilistic sanctions. It must be a credible static
rule-set (not a strawman) so that G3's eventual advantage (spec H3) is a
real finding rather than beating a deliberately weak baseline.
"""

from __future__ import annotations

from typing import Any

import numpy as np


# Mirrors aegora_isle.environment.aegora_env.MAX_EXTRACT_AMOUNT.
#
# Deliberately duplicated rather than imported: governance/ must not depend
# on environment/ (environment depends on governance, not vice versa — see
# MERMAID.md's module dependency graph). A drift-detector test in
# tests/test_rule_based.py asserts these two constants stay equal.
MAX_EXTRACT_AMOUNT: float = 5.0

# Extractor action_type discriminator values (see aegora_isle/agents/extractor.py).
_ACTION_EXTRACT: int = 0


class RuleBasedGovernor:
    """Static G2 enforcer: extraction caps, redistribution, sanctions.

    Args:
        extraction_cap: Max raw units an Extractor may pull in one step.
        sanction_probability: P(sanction fires | requested amount > cap).
        sanction_magnitude: Reward delta applied when a sanction fires
            (typically negative).
        redistribution_period: Apply redistribution every N env timesteps.
        redistribution_rate: Fraction of (Producer + Trader) cash collected
            and redistributed equally among Extractors at each
            redistribution event.
        rng: Externally-seeded ``numpy.random.Generator`` (CLAUDE.md
            seeding contract — this class never creates its own).
        max_extract_amount: Must match the env's ``MAX_EXTRACT_AMOUNT``;
            overridable for testing in isolation from the env.
    """

    regime: str = "G2"
    # Extractor over-cap requests are sanctioned by a coin-flip
    # (``sanction_probability``), independent of policy quality. NOTE: this
    # formerly told the M3 experiment runner to substitute a
    # harvested-amount convergence signal for the Extractor; that override
    # was retired 2026-08-09 by client decision and nothing reads this for
    # convergence any more (see DECISIONS.md "Convergence measurement --
    # two defects corrected"). Kept as a true, tested regime capability.
    has_stochastic_extractor_sanction: bool = True

    def __init__(
        self,
        extraction_cap: float,
        sanction_probability: float,
        sanction_magnitude: float,
        redistribution_period: int,
        redistribution_rate: float,
        rng: np.random.Generator,
        max_extract_amount: float = MAX_EXTRACT_AMOUNT,
    ) -> None:
        if not isinstance(rng, np.random.Generator):
            raise TypeError("RuleBasedGovernor requires a numpy.random.Generator")
        if extraction_cap < 0:
            raise ValueError("extraction_cap must be >= 0")
        if not 0.0 <= sanction_probability <= 1.0:
            raise ValueError("sanction_probability must be in [0, 1]")
        if redistribution_period <= 0:
            raise ValueError("redistribution_period must be > 0")
        if not 0.0 <= redistribution_rate <= 1.0:
            raise ValueError("redistribution_rate must be in [0, 1]")

        self.extraction_cap = float(extraction_cap)
        self.sanction_probability = float(sanction_probability)
        self.sanction_magnitude = float(sanction_magnitude)
        self.redistribution_period = int(redistribution_period)
        self.redistribution_rate = float(redistribution_rate)
        self.rng = rng
        self.max_extract_amount = float(max_extract_amount)

        self._sanction_count: int = 0
        self._redistribution_count: int = 0

    # ----------------------------------------------------------- on_action

    def on_action(
        self, agent_id: str, action: dict[str, Any]
    ) -> tuple[dict[str, Any], float]:
        """Clip over-cap extraction; probabilistically sanction the attempt.

        Non-Extractor agents and non-extract actions pass through
        unmodified. Within-cap extraction passes through unmodified.
        """
        if not agent_id.startswith("extractor"):
            return action, 0.0
        if int(action["action_type"]) != _ACTION_EXTRACT:
            return action, 0.0

        frac = float(np.asarray(action["amount"]).reshape(-1)[0])
        requested = frac * self.max_extract_amount
        if requested <= self.extraction_cap:
            return action, 0.0

        cap_frac = self.extraction_cap / self.max_extract_amount
        modified = dict(action)
        modified["amount"] = np.array([cap_frac], dtype=np.float32)

        reward_delta = 0.0
        if self.rng.random() < self.sanction_probability:
            reward_delta = self.sanction_magnitude
            self._sanction_count += 1

        return modified, reward_delta

    # ------------------------------------------------------------- on_step

    def on_step(self, env_state: dict[str, Any]) -> None:
        """End-of-cycle hook. Redistributes cash every ``redistribution_period``
        env timesteps, reading the live agent registry from ``env_state``.
        """
        timestep = int(env_state.get("timestep", 0))
        if timestep <= 0 or timestep % self.redistribution_period != 0:
            return
        self._redistribute(env_state)

    def _redistribute(self, env_state: dict[str, Any]) -> None:
        agents = env_state.get("agents_by_id")
        if not agents:
            return

        donors = [a for aid, a in agents.items() if aid.startswith(("producer", "trader"))]
        recipients = [a for aid, a in agents.items() if aid.startswith("extractor")]
        if not donors or not recipients:
            return

        total_pool = 0.0
        for donor in donors:
            contribution = float(donor.cash) * self.redistribution_rate
            donor.cash = float(donor.cash) - contribution
            total_pool += contribution

        if total_pool <= 0.0:
            return

        per_recipient = total_pool / len(recipients)
        for recipient in recipients:
            recipient.cash = float(recipient.cash) + per_recipient

        self._redistribution_count += 1

    # ---------------------------------------------------------- lifecycle

    def on_episode_end(self, env_state: dict[str, Any]) -> None:
        return None

    # ---------------------------------------------------------- bookkeeping

    def state_dict(self) -> dict[str, Any]:
        return {
            "regime": "G2",
            "extraction_cap": self.extraction_cap,
            "sanction_probability": self.sanction_probability,
            "sanction_magnitude": self.sanction_magnitude,
            "redistribution_period": self.redistribution_period,
            "redistribution_rate": self.redistribution_rate,
            "sanction_count": self._sanction_count,
            "redistribution_count": self._redistribution_count,
        }
