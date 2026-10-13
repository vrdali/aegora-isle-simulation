"""Governor agent. Sets caps, redistribution, sanctions.

Per DECISIONS.md and CLAUDE.md, the Governor class is **governance-agnostic**:
it just emits actions. The interpretation of those actions (and any rule
shaping) happens inside :class:`aegora_isle.governance.GovernanceModule`.

M1 reward is a 0.0 placeholder; the composite reward
(w_equality * equality + w_sustainability * sustainability +
w_productivity * productivity) per spec §2.2.2 lands in M2 alongside
``kpi_logger``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces

from aegora_isle.agents.base import AgentBase, MAX_AGENTS


# Wealth histogram bin count for the Governor observation.
WEALTH_BINS: int = 10


class Governor(AgentBase):
    """Action space (spec §2.1):

        Dict{
            action_type:      Discrete(4)   # 0=set_cap, 1=set_redistribution,
                                            # 2=sanction, 3=idle
            value:            Box(0.0, 1.0, (1,))   # used by set_cap, set_redist
            target_agent_idx: Discrete(MAX_AGENTS)  # used by sanction;
                                                    # out-of-range = no-op
        }

    Observation: ``[gini, hdi_proxy, gdp_indexed, depletion_rate,
                    resource_stock_norm, *wealth_histogram (10 bins)]``
    → dim 5 + 10 = 15.
    Reward: 0.0 placeholder until M2 composite reward lands.
    """

    OBS_DIM = 5 + WEALTH_BINS

    @property
    def observation_space(self) -> spaces.Space:
        return spaces.Box(
            low=np.concatenate(
                [
                    np.array([0.0, 0.0, 0.0, -1.0, 0.0], dtype=np.float32),
                    np.zeros(WEALTH_BINS, dtype=np.float32),
                ]
            ),
            high=np.concatenate(
                [
                    np.array([1.0, 1.0, 1e6, 1.0, 1.0], dtype=np.float32),
                    np.full(WEALTH_BINS, 1.0, dtype=np.float32),
                ]
            ),
            dtype=np.float32,
        )

    @property
    def action_space(self) -> spaces.Space:
        return spaces.Dict(
            {
                "action_type": spaces.Discrete(4),
                "value": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                "target_agent_idx": spaces.Discrete(MAX_AGENTS),
            }
        )

    def get_observation(self, env_state: dict[str, Any]) -> np.ndarray:
        kpis = env_state.get("kpis", {})
        pool = env_state["resource_pool"]
        stock_norm = float(pool["stock"] / pool["initial_stock"]) if pool["initial_stock"] > 0 else 0.0
        stock_norm = max(0.0, min(1.0, stock_norm))

        head = np.array(
            [
                float(kpis.get("gini", 0.0)),
                float(kpis.get("hdi_proxy", 0.0)),
                float(kpis.get("gdp_indexed", 0.0)),
                float(kpis.get("depletion_rate", 0.0)),
                stock_norm,
            ],
            dtype=np.float32,
        )
        hist = env_state.get("wealth_histogram", [0.0] * WEALTH_BINS)
        tail = (list(hist) + [0.0] * WEALTH_BINS)[:WEALTH_BINS]
        return np.concatenate([head, np.array(tail, dtype=np.float32)])

    def compute_reward(self, env_state: dict[str, Any], last_action: Any) -> float:
        # M1 placeholder per DECISIONS.md. M2 implements spec §2.2.2 composite.
        return 0.0
