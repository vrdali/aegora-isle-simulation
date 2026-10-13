"""Shared raw-resource pool for the Aegora Isle lagoon.

Per spec §2.3 and DECISIONS.md, the pool models a single raw resource:
- a finite initial stock that regenerates logistically toward carrying capacity,
- extraction draws against current stock,
- the three environmental event types of spec §2.3.1, rolled here because the
  spec assigns event injection to this module ("Events are injected by the
  resource_pool.py module using the run's random seed for reproducibility").

Spec §2.3.1's three event types:

1. ``disruption``      -- cuts stock by 10-30% (magnitudes given by the spec).
2. ``market_volatility`` -- a temporary price multiplier applied to all trades
   for N consecutive timesteps. The pool only *emits* this event; applying it
   is :class:`~aegora_isle.environment.market.Market`'s job, routed by
   ``AegoraEnv``. The pool must not import Market -- the dependency runs the
   other way (Market takes a ResourcePool).
3. ``agent_withdrawal`` -- one random agent forced idle for M timesteps. Again
   emitted here, applied by ``AegoraEnv``, which owns the agent registry.

Each type is rolled **independently** at ``event_rate / 100`` per step. The
spec fixes the rate ("events per 100 timesteps") but never says how it splits
across the three types; rolling independently keeps disruption frequency
identical to the single-event implementation that preceded this, so the
Scenario C/D dynamics tuned with the client remain valid. See DECISIONS.md
"Spec §2.3.1 events -- the two missing types".

Magnitudes for types 2 and 3 are **not given by the spec** and are resolved
here as documented defaults (see the module constants below).

All randomness flows from an injected ``numpy.random.Generator``. The class
never creates its own RNG (see CLAUDE.md "Random seeding contract").
"""

from __future__ import annotations

from typing import Any

import numpy as np


# Spec §2.3.1 gives magnitudes only for `disruption` (10-30% stock cut). The
# two constants below resolve what it leaves open for the other two event
# types. Chosen so the events are actually consequential -- in particular the
# multiplier is two-directional, so a Trader (whose spec observation is
# `market prices` + `trade history`) has genuine price movement to time, which
# a one-directional or very narrow band would not provide. Documented as
# spec-ambiguity resolutions in DECISIONS.md, not silent tuning.
MARKET_VOLATILITY_MULTIPLIER_RANGE: tuple[float, float] = (0.7, 1.5)
MARKET_VOLATILITY_DURATION_RANGE: tuple[int, int] = (10, 50)
AGENT_WITHDRAWAL_DURATION_RANGE: tuple[int, int] = (10, 50)


class ResourcePool:
    """A logistically-regenerating raw resource pool with random disruptions.

    Args:
        initial_stock: Carrying capacity (upper bound on stock). Immutable.
        regeneration_rate: Fractional logistic growth per step. ``stock``
            grows by ``regeneration_rate * stock * (1 - stock/initial_stock)``.
            A rate of ``0.0`` disables regeneration.
        critical_min: Stock level below which ``is_collapsed`` returns True.
            (M1 placeholder; M2's kpi_logger replaces with the subsistence
            definition — see DECISIONS.md.)
        event_rate: Disruption events expected per 100 timesteps. The per-step
            probability is ``event_rate / 100``.
        rng: An externally-seeded ``numpy.random.Generator``. **Required.**
    """

    def __init__(
        self,
        initial_stock: float,
        regeneration_rate: float,
        critical_min: float,
        event_rate: float,
        rng: np.random.Generator,
    ) -> None:
        if not isinstance(rng, np.random.Generator):
            raise TypeError(
                "ResourcePool requires a numpy.random.Generator; "
                "see CLAUDE.md seeding contract."
            )
        if initial_stock <= 0:
            raise ValueError("initial_stock must be > 0")
        if regeneration_rate < 0:
            raise ValueError("regeneration_rate must be >= 0")
        if event_rate < 0:
            raise ValueError("event_rate must be >= 0")

        self._initial_stock = float(initial_stock)
        self._regeneration_rate = float(regeneration_rate)
        self._critical_min = float(critical_min)
        self._event_rate = float(event_rate)
        self.rng = rng

        self.stock: float = self._initial_stock
        self._event_count: int = 0
        # Per-type counts. `event_count` is the total across all three spec
        # §2.3.1 types; these break it down, and let tests assert the
        # per-type rate (the actual spec contract) rather than only the sum.
        self._event_counts_by_type: dict[str, int] = {
            "disruption": 0, "market_volatility": 0, "agent_withdrawal": 0,
        }
        # Events emitted on the most recent step(), drained by AegoraEnv.
        self._pending_events: list[dict[str, Any]] = []

    @property
    def initial_stock(self) -> float:
        return self._initial_stock

    @property
    def critical_min(self) -> float:
        return self._critical_min

    @property
    def event_count(self) -> int:
        return self._event_count

    def extract(self, amount: float) -> float:
        """Subtract from stock, clipping to availability.

        Returns the actual amount extracted.
        """
        if amount < 0:
            raise ValueError("extract amount must be >= 0")
        actual = float(min(amount, self.stock))
        self.stock -= actual
        return actual

    def step(self) -> dict[str, Any] | None:
        """Advance one timestep: regenerate, then roll each event type.

        Returns the **disruption** event dict if one fired (preserving this
        method's original contract for existing callers), else ``None``.
        Every event that fired this step -- including the two the pool only
        emits rather than applies -- is available via :meth:`pending_events`,
        which ``AegoraEnv`` drains and routes.
        """
        # Logistic regeneration capped at carrying capacity.
        if self._regeneration_rate > 0.0 and self.stock > 0.0:
            growth = self._regeneration_rate * self.stock * (
                1.0 - self.stock / self._initial_stock
            )
            self.stock = min(self.stock + growth, self._initial_stock)

        # Spec §2.3.1: each of the three event types rolls independently at
        # event_rate/100 (see module docstring for why independently).
        p_event = self._event_rate / 100.0
        self._pending_events = []
        disruption: dict[str, Any] | None = None
        if p_event > 0.0:
            if self.rng.random() < p_event:
                disruption = self.inject_disruption()
                self._pending_events.append(disruption)
            if self.rng.random() < p_event:
                self._pending_events.append(self.inject_market_volatility())
            if self.rng.random() < p_event:
                self._pending_events.append(self.inject_agent_withdrawal())
        return disruption

    @property
    def event_counts_by_type(self) -> dict[str, int]:
        """Per-type event tallies; sums to :attr:`event_count`."""
        return dict(self._event_counts_by_type)

    def pending_events(self) -> list[dict[str, Any]]:
        """Events emitted on the most recent :meth:`step`, drained by the env.

        The pool applies ``disruption`` itself (it owns the stock). The other
        two are applied by ``AegoraEnv`` -- it owns the Market and the agent
        registry, and this module must not import either.
        """
        return list(self._pending_events)

    def inject_disruption(self) -> dict[str, Any]:
        """Apply a random 10-30% stock cut. Returns the event record."""
        cut_pct = float(self.rng.uniform(0.10, 0.30))
        prev_stock = self.stock
        self.stock = max(0.0, self.stock * (1.0 - cut_pct))
        self._event_count += 1
        self._event_counts_by_type["disruption"] += 1
        return {
            "type": "disruption",
            "cut_pct": cut_pct,
            "stock_before": prev_stock,
            "stock_after": self.stock,
            "timestep": None,  # env fills this in
        }

    def inject_market_volatility(self) -> dict[str, Any]:
        """Emit a spec §2.3.1 market-volatility episode.

        The pool does not apply this -- it has no reference to the Market
        (and must not: Market depends on ResourcePool, not the reverse).
        ``AegoraEnv`` routes it to ``Market.apply_volatility_episode``.
        """
        lo, hi = MARKET_VOLATILITY_MULTIPLIER_RANGE
        d_lo, d_hi = MARKET_VOLATILITY_DURATION_RANGE
        multiplier = float(self.rng.uniform(lo, hi))
        duration = int(self.rng.integers(d_lo, d_hi + 1))
        self._event_count += 1
        self._event_counts_by_type["market_volatility"] += 1
        return {
            "type": "market_volatility",
            "multiplier": multiplier,
            "duration": duration,
            "timestep": None,  # env fills this in
        }

    def inject_agent_withdrawal(self) -> dict[str, Any]:
        """Emit a spec §2.3.1 agent-withdrawal episode.

        Names no agent: the pool has no view of the population. ``AegoraEnv``
        picks the target from its own registry using the same seeded
        Generator lineage.
        """
        d_lo, d_hi = AGENT_WITHDRAWAL_DURATION_RANGE
        duration = int(self.rng.integers(d_lo, d_hi + 1))
        self._event_count += 1
        self._event_counts_by_type["agent_withdrawal"] += 1
        return {
            "type": "agent_withdrawal",
            "duration": duration,
            "timestep": None,  # env fills this in
        }

    def is_collapsed(self) -> bool:
        """True when stock falls below ``critical_min`` (M1 placeholder)."""
        return self.stock < self._critical_min

    def state_dict(self) -> dict[str, Any]:
        """Serialisable snapshot of pool state."""
        return {
            "stock": self.stock,
            "initial_stock": self._initial_stock,
            "critical_min": self._critical_min,
            "regeneration_rate": self._regeneration_rate,
            "event_rate": self._event_rate,
            "event_count": self._event_count,
            "event_counts_by_type": dict(self._event_counts_by_type),
        }
