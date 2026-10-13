"""Extractor agent. Harvests raw resource from the lagoon."""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces

from aegora_isle.agents.base import AgentBase, N_ZONES


class Extractor(AgentBase):
    """Action space (spec §2.1, literal Dict encoding from DECISIONS.md):

        Dict{
            action_type: Discrete(3)   # 0=extract, 1=idle, 2=move_to_zone
            amount:      Box(0.0, 1.0, (1,))   # fraction of max_extract
            target_zone: Discrete(N_ZONES)
        }

    Observation: ``[own_raw, own_food, own_tool, own_cash,
                    zone_stock_norm, last_depletion_penalty]``
    Reward: raw resource units actually extracted this step.
    """

    OBS_DIM = 6

    @property
    def observation_space(self) -> spaces.Space:
        return spaces.Box(
            low=np.array([0.0, 0.0, 0.0, -1e6, 0.0, -1e6], dtype=np.float32),
            high=np.array([1e6, 1e6, 1e6, 1e6, 1.0, 1e6], dtype=np.float32),
            dtype=np.float32,
        )

    @property
    def action_space(self) -> spaces.Space:
        return spaces.Dict(
            {
                "action_type": spaces.Discrete(3),
                "amount": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                "target_zone": spaces.Discrete(N_ZONES),
            }
        )

    def get_observation(self, env_state: dict[str, Any]) -> np.ndarray:
        pool = env_state["resource_pool"]
        zone_norm = float(pool["stock"] / pool["initial_stock"]) if pool["initial_stock"] > 0 else 0.0
        zone_norm = max(0.0, min(1.0, zone_norm))
        return np.array(
            [
                self.inventory.get("raw", 0.0),
                self.inventory.get("food", 0.0),
                self.inventory.get("tool", 0.0),
                self.cash,
                zone_norm,
                float(env_state.get("last_depletion_penalty", 0.0)),
            ],
            dtype=np.float32,
        )

    def compute_reward(self, env_state: dict[str, Any], last_action: Any) -> float:
        return float(env_state.get("last_extracted_amount", 0.0))
