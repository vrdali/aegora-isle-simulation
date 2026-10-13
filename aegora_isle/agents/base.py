"""Base class for all Aegora Isle agents.

Per CLAUDE.md, agents are **governance-agnostic**. No imports from
``aegora_isle.governance`` are permitted in this module or any subclass.
All regime-dependent behaviour flows through ``GovernanceModule`` externally.

Per spec §11.2, ``MAX_AGENTS`` is fixed at 500 so the Governor's
``target_agent_idx`` space is forward-compatible with 500-agent runs without
schema changes.

Per DECISIONS.md, ``N_ZONES`` is fixed at 4.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from gymnasium import spaces


MAX_AGENTS: int = 500
N_ZONES: int = 4
ITEMS: tuple[str, ...] = ("food", "tool")


class AgentBase(ABC):
    """Common state and interface for every agent type.

    Each agent owns:
        - ``agent_id``: string identifier (e.g. ``"extractor_0"``).
        - ``inventory``: dict mapping resource/item names to float quantities.
            Always contains ``"raw"``, ``"food"``, ``"tool"`` keys.
        - ``cash``: liquid currency balance.
        - ``alive``: termination flag.
        - ``age``: steps since reset.
        - ``rng``: injected per-agent generator (CLAUDE.md contract).

    Subclasses must define ``observation_space``, ``action_space``,
    ``get_observation``, and ``compute_reward``.
    """

    def __init__(
        self,
        agent_id: str,
        rng: np.random.Generator,
        initial_cash: float = 10.0,
    ) -> None:
        if not isinstance(rng, np.random.Generator):
            raise TypeError("AgentBase requires a numpy.random.Generator")
        self.agent_id = agent_id
        self.rng = rng
        self.inventory: dict[str, float] = {"raw": 0.0, "food": 0.0, "tool": 0.0}
        self.cash: float = float(initial_cash)
        self.alive: bool = True
        self.age: int = 0

    # ------------------------------------------------------------------ spaces

    @property
    @abstractmethod
    def observation_space(self) -> spaces.Space: ...

    @property
    @abstractmethod
    def action_space(self) -> spaces.Space: ...

    # -------------------------------------------------------------- behaviour

    @abstractmethod
    def get_observation(self, env_state: dict[str, Any]) -> np.ndarray: ...

    @abstractmethod
    def compute_reward(
        self, env_state: dict[str, Any], last_action: Any
    ) -> float: ...

    # -------------------------------------------------------------- utilities

    def wealth(self, prices: dict[str, float]) -> float:
        """Total wealth at current prices: raw + food*p + tool*p + cash.

        Raw resource priced at 1.0 internally (no market for it).
        """
        return (
            float(self.inventory.get("raw", 0.0))
            + float(self.inventory.get("food", 0.0)) * float(prices.get("food", 0.0))
            + float(self.inventory.get("tool", 0.0)) * float(prices.get("tool", 0.0))
            + float(self.cash)
        )

    def reset(self, initial_cash: float = 10.0) -> None:
        self.inventory = {"raw": 0.0, "food": 0.0, "tool": 0.0}
        self.cash = float(initial_cash)
        self.alive = True
        self.age = 0
