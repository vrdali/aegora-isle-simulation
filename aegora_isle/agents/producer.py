"""Producer agent. Transforms raw resource into food/tool goods."""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces

from aegora_isle.agents.base import AgentBase


class Producer(AgentBase):
    """Action space (spec §2.1):

        Dict{
            action_type:  Discrete(3)   # 0=produce, 1=store, 2=trade_offer
            produce_type: Discrete(2)   # 0=food, 1=tool (read when action_type=0)
            offer_item:   Discrete(2)   # 0=food, 1=tool (read when action_type=2)
            offer_qty:    Box(0.0, 1.0, (1,))   # fraction of inventory to offer
        }

    Observation: ``[raw, food, tool, cash, food_price, tool_price,
                    production_capacity_remaining]``
    Reward: total value of goods manufactured this step at current prices.
    """

    OBS_DIM = 7

    @property
    def observation_space(self) -> spaces.Space:
        return spaces.Box(
            low=np.array([0.0, 0.0, 0.0, -1e6, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

    @property
    def action_space(self) -> spaces.Space:
        return spaces.Dict(
            {
                "action_type": spaces.Discrete(3),
                "produce_type": spaces.Discrete(2),
                "offer_item": spaces.Discrete(2),
                "offer_qty": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            }
        )

    def get_observation(self, env_state: dict[str, Any]) -> np.ndarray:
        prices = env_state.get("prices", {})
        return np.array(
            [
                self.inventory.get("raw", 0.0),
                self.inventory.get("food", 0.0),
                self.inventory.get("tool", 0.0),
                self.cash,
                float(prices.get("food", 0.0)),
                float(prices.get("tool", 0.0)),
                float(env_state.get("production_capacity_remaining", 1.0)),
            ],
            dtype=np.float32,
        )

    def compute_reward(self, env_state: dict[str, Any], last_action: Any) -> float:
        produced = env_state.get("last_produced", {})  # e.g. {"food": 2.5, "tool": 0.0}
        prices = env_state.get("prices", {})
        return float(
            sum(qty * float(prices.get(item, 0.0)) for item, qty in produced.items())
        )
