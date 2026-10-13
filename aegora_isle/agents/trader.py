"""Trader agent. Buys and sells produced goods on the market."""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces

from aegora_isle.agents.base import AgentBase


class Trader(AgentBase):
    """Action space (spec §2.1):

        Dict{
            action_type: Discrete(3)   # 0=buy, 1=sell, 2=idle
            item:        Discrete(2)   # 0=food, 1=tool
            qty:         Box(0.0, 1.0, (1,))   # fraction of capital (buy)
                                                # or inventory (sell)
        }

    Observation: ``[food, tool, cash, food_price, tool_price,
                    *last_5_trade_volumes]``  → dim 5 + 5 = 10.
    Reward: net trade profit this step.
    """

    OBS_DIM = 10

    @property
    def observation_space(self) -> spaces.Space:
        return spaces.Box(
            low=np.concatenate(
                [
                    np.array([0.0, 0.0, -1e6, 0.0, 0.0], dtype=np.float32),
                    np.zeros(5, dtype=np.float32),
                ]
            ),
            high=np.concatenate(
                [
                    np.array([1e6, 1e6, 1e6, 1e6, 1e6], dtype=np.float32),
                    np.full(5, 1e6, dtype=np.float32),
                ]
            ),
            dtype=np.float32,
        )

    @property
    def action_space(self) -> spaces.Space:
        return spaces.Dict(
            {
                "action_type": spaces.Discrete(3),
                "item": spaces.Discrete(2),
                "qty": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            }
        )

    def get_observation(self, env_state: dict[str, Any]) -> np.ndarray:
        prices = env_state.get("prices", {})
        recent = env_state.get("last_5_trade_volumes", [0.0] * 5)
        # Pad / clip to 5 entries.
        vols = (list(recent) + [0.0] * 5)[:5]
        head = np.array(
            [
                self.inventory.get("food", 0.0),
                self.inventory.get("tool", 0.0),
                self.cash,
                float(prices.get("food", 0.0)),
                float(prices.get("tool", 0.0)),
            ],
            dtype=np.float32,
        )
        tail = np.array(vols, dtype=np.float32)
        return np.concatenate([head, tail])

    def compute_reward(self, env_state: dict[str, Any], last_action: Any) -> float:
        return float(env_state.get("last_trade_profit", 0.0))
