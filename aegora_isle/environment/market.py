"""Scarcity-priced peer-to-peer market for produced goods.

Per spec §2.1 and DECISIONS.md, the market handles **produced goods only**
(``food``, ``tool``). The raw resource is tracked separately by
:class:`aegora_isle.environment.resource_pool.ResourcePool`.

Pricing formula (locked):

    price(item) = base_prices[item] * (1 + alpha * (1 - stock / initial_stock))

So price rises smoothly from ``base`` (full lagoon) toward ``base * (1+alpha)``
(empty lagoon), reflecting downstream production cost.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np

from aegora_isle.environment.resource_pool import ResourcePool


# Sentinel return for invalid trades. See ``execute_trade`` docstring.
INVALID_TRADE: dict[str, Any] | None = None

# Market-maker bid-ask spread earned by the Trader role (EXPERIMENTAL).
#
# The spec defines the Trader as "Market Exchange" with reward "net trade
# profit", and gives it an observation of market prices + trade history --
# but never defines any mechanism by which intermediation can be profitable.
# With a single global price, buying and selling are wealth-neutral by
# construction, so Trader wealth is pinned at its starting endowment and
# decays under volatility (measured: 10.00 -> 3.41 over 1500 steps, while
# Producers grew 2882x). See DECISIONS.md "Trader bid-ask spread".
#
# This implements the standard market-maker mechanism: the Trader buys at
# the bid (below mid) and sells at the ask (above mid), capturing 2x the
# spread per round trip as compensation for bearing inventory price risk.
#
# NOT in the specification, and **DISABLED BY DEFAULT (0.0)**.
#
# Validated on a 144-run pilot at 0.05: it works -- Trader wealth rose from
# 7.88 to 582.90 (74x) at t=2000 and compounds, with convergence unchanged
# (144/144). But it also **flipped the sign of the H3 effect**: G3-minus-G2
# GDP-equivalent went from +0.179 (supports "adaptive beats rule-based") to
# -0.098 (contradicts it). Neither is significant at n=12 (p=0.057, p=0.251),
# but the direction reverses, and at the real n=30 that could resolve either
# way -- whichever we bake in.
#
# A non-spec mechanism that reverses a pre-registered hypothesis is not ours
# to adopt: we would be engineering the answer rather than measuring it. Left
# implemented and fully tested so the client can enable it by changing this
# one constant if they decide the Trader role must be economically real.
# See DECISIONS.md "Trader bid-ask spread -- implemented, disabled by default".
TRADER_SPREAD: float = 0.0


class Market:
    """Scarcity-adjusted peer-to-peer market for produced goods.

    Args:
        resource_pool: The lagoon pool whose ``stock / initial_stock`` ratio
            drives the scarcity multiplier.
        base_prices: Mapping ``{"food": 10.0, "tool": 20.0}`` (or similar).
            All keys are valid trade items.
        scarcity_alpha: Non-negative scarcity sensitivity. ``0`` disables
            scarcity adjustment.
        rng: Injected ``numpy.random.Generator`` (required by CLAUDE.md
            seeding contract, even if the market is currently deterministic).
        timestep_fn: Optional callable returning the current env timestep,
            used to stamp trade-log entries. Defaults to ``lambda: 0``.
    """

    def __init__(
        self,
        resource_pool: ResourcePool,
        base_prices: dict[str, float],
        scarcity_alpha: float,
        rng: np.random.Generator,
        timestep_fn: Optional[Callable[[], int]] = None,
    ) -> None:
        if not isinstance(rng, np.random.Generator):
            raise TypeError("Market requires a numpy.random.Generator")
        if scarcity_alpha < 0:
            raise ValueError("scarcity_alpha must be >= 0")
        if not base_prices:
            raise ValueError("base_prices must be non-empty")
        for item, price in base_prices.items():
            if price <= 0:
                raise ValueError(f"base price for {item!r} must be > 0")

        self.resource_pool = resource_pool
        self.base_prices: dict[str, float] = {k: float(v) for k, v in base_prices.items()}
        self.scarcity_alpha = float(scarcity_alpha)
        self.rng = rng
        self._timestep_fn = timestep_fn if timestep_fn is not None else (lambda: 0)

        self._trade_log: list[dict[str, Any]] = []

        # Spec §2.3.1 market-volatility episode: a temporary multiplier applied
        # to all trades for N consecutive timesteps. Injected by ResourcePool,
        # routed here by AegoraEnv (this class must not reach back into either).
        # 1.0 = no episode active.
        self._volatility_multiplier: float = 1.0
        self._volatility_steps_remaining: int = 0

    # -------------------------------------------------------------- volatility

    def apply_volatility_episode(self, multiplier: float, duration: int) -> None:
        """Start a market-volatility episode (spec §2.3.1).

        A later episode overwrites an in-flight one rather than compounding —
        multiplying them would let coincident events run prices away
        exponentially, which the spec's "temporary price multiplier" wording
        does not describe.
        """
        if duration <= 0 or multiplier <= 0:
            return
        self._volatility_multiplier = float(multiplier)
        self._volatility_steps_remaining = int(duration)

    def tick_volatility(self) -> None:
        """Advance the episode clock one timestep; expire it at zero.

        Called once per cycle by ``AegoraEnv``, alongside the pool's own step.
        """
        if self._volatility_steps_remaining > 0:
            self._volatility_steps_remaining -= 1
            if self._volatility_steps_remaining == 0:
                self._volatility_multiplier = 1.0

    @property
    def volatility_multiplier(self) -> float:
        """Current episode multiplier; 1.0 when no episode is active."""
        return self._volatility_multiplier

    # ------------------------------------------------------------------ pricing

    def _scarcity_multiplier(self) -> float:
        """Returns ``1 + alpha * (1 - stock/initial)``, clamped at >= 1.0.

        Clamped because logistic regeneration can in principle overshoot the
        carrying capacity in degenerate parameterisations — we never want a
        negative-scarcity (subsidised) price.
        """
        ratio = self.resource_pool.stock / self.resource_pool.initial_stock
        # ratio in [0, 1] under nominal operation; clamp to [0, 1] for safety.
        ratio = max(0.0, min(1.0, ratio))
        return 1.0 + self.scarcity_alpha * (1.0 - ratio)

    def get_price(self, item: str) -> float:
        """Scarcity-adjusted price, further scaled by any active
        market-volatility episode (spec §2.3.1)."""
        if item not in self.base_prices:
            raise KeyError(f"unknown item {item!r}; known items: {list(self.base_prices)}")
        return (
            self.base_prices[item]
            * self._scarcity_multiplier()
            * self._volatility_multiplier
        )

    # -------------------------------------------------------------- execution

    def _execution_price(self, buyer_id: str, seller_id: str, item: str) -> float:
        """Mid price, adjusted by the Trader's bid-ask spread (EXPERIMENTAL).

        Standard market-maker convention: the Trader quotes a two-sided
        market and earns the spread from whoever crosses it.

        - Trader is the **buyer**  -> pays the *bid*, ``mid * (1 - spread)``.
        - Trader is the **seller** -> receives the *ask*, ``mid * (1 + spread)``.
        - Trader on **both** sides -> mid; two market makers trading with each
          other have no reason to hand one the spread.
        - **No** Trader involved -> mid, exactly as before.

        Role is read from the agent-id prefix, matching this codebase's
        existing convention (``_pick_counterparty(..., ("producer", "trader"))``
        in ``aegora_env``). This is role dispatch, not governance-regime
        branching -- CLAUDE.md's prohibition concerns the latter.

        With ``TRADER_SPREAD = 0.0`` this returns the mid price for every
        pairing, i.e. exactly the pre-existing single-price behaviour.
        """
        mid = self.get_price(item)
        if TRADER_SPREAD <= 0.0:
            return mid
        buyer_is_trader = buyer_id.startswith("trader")
        seller_is_trader = seller_id.startswith("trader")
        if buyer_is_trader and not seller_is_trader:
            return mid * (1.0 - TRADER_SPREAD)
        if seller_is_trader and not buyer_is_trader:
            return mid * (1.0 + TRADER_SPREAD)
        return mid

    def execute_trade(
        self,
        buyer_id: str,
        seller_id: str,
        item: str,
        qty: float,
        agent_registry: Callable[[str], Any],
    ) -> dict[str, Any] | None:
        """Execute a peer-to-peer trade.

        ``agent_registry(agent_id)`` must return an object with at least
        ``inventory: dict[str, float]`` and ``cash: float`` attributes
        (mutable). Conservation is checked: total cash + total inventory
        value before and after the trade are equal.

        Returns the trade record dict on success, or ``None`` if the trade
        is invalid (unknown item, non-positive qty, self-trade, seller lacks
        inventory, buyer lacks cash). Invalid trades are silently ignored —
        no exception — which lets agents propose trades freely without
        the env having to pre-validate.
        """
        if item not in self.base_prices:
            return INVALID_TRADE
        if qty <= 0:
            return INVALID_TRADE
        if buyer_id == seller_id:
            return INVALID_TRADE

        buyer = agent_registry(buyer_id)
        seller = agent_registry(seller_id)
        if buyer is None or seller is None:
            return INVALID_TRADE

        seller_qty = float(seller.inventory.get(item, 0.0))
        if seller_qty <= 0.0:
            return INVALID_TRADE
        traded_qty = float(min(qty, seller_qty))

        price = self._execution_price(buyer_id, seller_id, item)
        value = price * traded_qty
        if buyer.cash < value:
            # Buyer can only afford a fraction.
            if buyer.cash <= 0.0:
                return INVALID_TRADE
            traded_qty = float(buyer.cash / price)
            value = price * traded_qty

        # Effect the transfer atomically.
        seller.inventory[item] = seller_qty - traded_qty
        buyer.inventory[item] = float(buyer.inventory.get(item, 0.0)) + traded_qty
        seller.cash = float(seller.cash) + value
        buyer.cash = float(buyer.cash) - value

        record = {
            "timestep": int(self._timestep_fn()),
            "buyer": buyer_id,
            "seller": seller_id,
            "item": item,
            "qty": traded_qty,
            "price": price,
            "value": value,
        }
        self._trade_log.append(record)
        return record

    # ------------------------------------------------------------- bookkeeping

    def get_trade_log(self) -> list[dict[str, Any]]:
        return list(self._trade_log)

    def reset(self) -> None:
        self._trade_log.clear()
        self._volatility_multiplier = 1.0
        self._volatility_steps_remaining = 0

    def state_dict(self) -> dict[str, Any]:
        return {
            "base_prices": dict(self.base_prices),
            "scarcity_alpha": self.scarcity_alpha,
            "trade_count": len(self._trade_log),
            "current_prices": {item: self.get_price(item) for item in self.base_prices},
            "volatility_multiplier": self._volatility_multiplier,
            "volatility_steps_remaining": self._volatility_steps_remaining,
        }
