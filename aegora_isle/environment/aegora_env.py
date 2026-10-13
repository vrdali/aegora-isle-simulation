"""Aegora Isle PettingZoo AEC environment.

Per CLAUDE.md, **AEC API only** — this module does not export a Parallel
variant. All randomness flows from a single seed via ``np.random.default_rng``
which is passed through to :class:`ResourcePool`, :class:`Market`, and each
agent (CLAUDE.md "Random seeding contract").

The env is governance-agnostic: when no ``governance_module`` is supplied,
it defaults to :class:`G1Handler` (the ungoverned baseline). Real regime
configuration goes through :class:`GovernanceModule`, which this env accepts
without modification thanks to the locked
``on_action(agent_id, action) -> (action, reward_delta)`` interface — see
``aegora_isle/governance/governance_module.py`` and ``G1_PLAN.md`` /
``G2_PLAN.md`` / ``G3_PLAN.md``.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from gymnasium import spaces
from pettingzoo.utils import AECEnv, AgentSelector

from aegora_isle.agents.base import AgentBase
from aegora_isle.agents.extractor import Extractor
from aegora_isle.agents.governor import Governor
from aegora_isle.agents.producer import Producer
from aegora_isle.agents.trader import Trader
from aegora_isle.environment.market import Market
from aegora_isle.environment.resource_pool import ResourcePool
from aegora_isle.governance.governance_module import G1Handler
from aegora_isle.kpi import kpi_logger as _kpi


DEFAULT_SCENARIO_PARAMS: dict[str, Any] = {
    "initial_stock": 1000.0,
    "regeneration_rate": 0.01,
    "event_rate": 1.0,
    "critical_min": 50.0,
    "scarcity_alpha": 0.5,
    "base_prices": {"food": 10.0, "tool": 20.0, "raw": 1.0},
}

# Per-step extraction ceiling: agent's continuous Box amount in [0,1] is
# multiplied by this to get raw resource units requested.
MAX_EXTRACT_AMOUNT: float = 5.0

# Per-step production ceiling: 1 raw -> 1 good (food or tool), capped per step.
MAX_PRODUCE_AMOUNT: float = 3.0

# Fraction of current cash an Extractor spends on food/tool each cycle,
# recirculating cash back into the economy (fixes the one-way cash sink
# where Extractors only ever receive cash from Producers' raw purchases
# and never spend it -- see DECISIONS.md "Extractor cash-sink fix").
# This was briefly lowered to 0.05 during the convergence investigation, on
# the theory that gentler per-cycle demand would calm Producer cash swings.
# That premise was DISPROVEN -- a controlled 0.05-vs-0.15 pilot produced an
# identical stuck set (same 28 run_ids), and the real cause was a pair of
# measurement defects (DECISIONS.md "Convergence measurement -- two defects
# corrected"). The value has no bearing on convergence either way; it sets
# only how much cash recirculates, and therefore how much economic activity
# gdp_raw/hdi_proxy record (0.15 measured ~250-800 gdp_raw per cycle vs
# 0.05's ~163-408). Restored to 0.15 by explicit decision: more circulation
# means more signal in exactly the two KPIs this fix exists to rescue for
# H2/H3.
EXTRACTOR_SPEND_FRACTION: float = 0.15


class AegoraEnv(AECEnv):
    """PettingZoo AECEnv for the Aegora Isle synthetic economy.

    Args:
        n_agents: Total population. Per DECISIONS.md, agent-type proportions
            with Governor are 12/13/13/12; without (G1) 17/17/16. Other
            ``n_agents`` values are rounded analogously.
        governance_module: Object implementing ``on_action`` and ``on_step``.
            Default :class:`G1Handler` (ungoverned baseline, regime ``"G1"``).
            For G2/G3, pass a configured :class:`GovernanceModule` instance.
        scenario_params: Per-scenario overrides for resource pool + market.
            Falls back to :data:`DEFAULT_SCENARIO_PARAMS`.
        seed: Master seed. Threaded through ``np.random.default_rng`` to
            every stochastic component.
        max_timesteps: Truncation horizon.
    """

    metadata = {"render_modes": ["human"], "name": "aegora_isle_v0", "is_parallelizable": False}

    def __init__(
        self,
        n_agents: int = 50,
        governance_module: Optional[Any] = None,
        scenario_params: Optional[dict[str, Any]] = None,
        seed: int = 0,
        max_timesteps: int = 10_000,
    ) -> None:
        super().__init__()
        if n_agents < 4:
            raise ValueError("n_agents must be >= 4")
        self.n_agents = int(n_agents)
        self.governance_module = governance_module if governance_module is not None else G1Handler()
        self.scenario_params = {**DEFAULT_SCENARIO_PARAMS, **(scenario_params or {})}
        self.seed = int(seed)
        self.max_timesteps = int(max_timesteps)

        # State populated in reset().
        self.rng: np.random.Generator | None = None
        self.resource_pool: ResourcePool | None = None
        self.market: Market | None = None
        self._agents_by_id: dict[str, AgentBase] = {}
        self._agent_selector: AgentSelector | None = None
        self._timestep: int = 0
        self._last_event: dict[str, Any] | None = None
        self._last_5_trade_volumes: list[float] = [0.0] * 5
        self._pending_governor_actions: dict[str, dict[str, Any]] = {}
        self._step_log: dict[str, dict[str, Any]] = {}
        # Spec §2.3.1 agent-withdrawal: agent_id -> cycles still withdrawn.
        self._withdrawal_remaining: dict[str, int] = {}

        # KPI bookkeeping (Sprint 2.3). Production/trade value accumulate
        # during a cycle and are consumed (then reset) at end-of-cycle.
        # _stock_history feeds the live depletion-rate calc; capped to the
        # last 51 entries (50-step window + current).
        self._stock_history: list[float] = []
        self._cycle_production_value: float = 0.0
        self._cycle_trade_value: float = 0.0
        self._last_cycle_gdp_raw: float = 0.0
        # Production value of the cycle that just finished, captured before
        # _cycle_production_value resets for the next cycle -- see
        # DECISIONS.md "KPILogger capability_utilisation read-after-reset
        # fix". _cycle_production_value itself stays a live, in-progress
        # accumulator (read directly by _snapshot_state for per-step
        # Governor observations); this is the finalized, once-per-cycle
        # counterpart, mirroring _last_cycle_gdp_raw exactly.
        self._last_cycle_production_value: float = 0.0
        self._initial_mean_wealth: float = 0.0
        self._kpis: dict[str, float] = {
            "gini": 0.0,
            "hdi_proxy": 0.0,
            "gdp_indexed": 0.0,
            "depletion_rate": 0.0,
        }
        self._wealth_histogram: list[float] = [0.0] * 10

        # Compute possible_agents under the current regime.
        self.possible_agents: list[str] = self._compute_possible_agents()
        self.agents: list[str] = []

        # Per-agent space caches (filled by reset()).
        self._observation_spaces: dict[str, spaces.Space] = {}
        self._action_spaces: dict[str, spaces.Space] = {}

        # AEC bookkeeping.
        self.rewards: dict[str, float] = {}
        self._cumulative_rewards: dict[str, float] = {}
        self.terminations: dict[str, bool] = {}
        self.truncations: dict[str, bool] = {}
        self.infos: dict[str, dict[str, Any]] = {}
        self.agent_selection: str = ""

    # ------------------------------------------------------------ population

    def _agent_counts(self) -> dict[str, int]:
        """Per DECISIONS.md rounding. With Governor: 12/13/13/12 at n=50."""
        n = self.n_agents
        regime = getattr(self.governance_module, "regime", "G1")
        if regime == "G1":
            # 17/17/16 at n=50 → proportions roughly 1/3 each, last absorbs remainder.
            base = n // 3
            rem = n - 3 * base
            ex = base + (1 if rem >= 1 else 0)
            pr = base + (1 if rem >= 2 else 0)
            tr = base
            return {"extractor": ex, "producer": pr, "trader": tr, "governor": 0}
        # With Governor (G2/G3): 12/13/13/12 at n=50.
        base = n // 4
        rem = n - 4 * base
        # Distribute remainder: producer, trader, extractor, governor in order.
        counts = {"extractor": base, "producer": base, "trader": base, "governor": base}
        order = ["producer", "trader", "extractor", "governor"]
        for i in range(rem):
            counts[order[i % 4]] += 1
        return counts

    def _compute_possible_agents(self) -> list[str]:
        counts = self._agent_counts()
        ids: list[str] = []
        for kind, n in counts.items():
            for i in range(n):
                ids.append(f"{kind}_{i}")
        return ids

    def _instantiate_agents(self) -> dict[str, AgentBase]:
        assert self.rng is not None
        out: dict[str, AgentBase] = {}
        for aid in self.possible_agents:
            kind = aid.split("_")[0]
            child_rng = np.random.default_rng(self.rng.integers(0, 2**31 - 1))
            cls: type[AgentBase]
            if kind == "extractor":
                cls = Extractor
            elif kind == "producer":
                cls = Producer
            elif kind == "trader":
                cls = Trader
            elif kind == "governor":
                cls = Governor
            else:
                raise ValueError(f"unknown agent kind {kind!r}")
            out[aid] = cls(agent_id=aid, rng=child_rng)
        return out

    # ------------------------------------------------------------------ spaces

    def observation_space(self, agent: str) -> spaces.Space:
        return self._observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space:
        return self._action_spaces[agent]

    # -------------------------------------------------------------- lifecycle

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> None:
        if seed is not None:
            self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

        # Rebuild stochastic substrates with fresh child RNGs.
        pool_rng = np.random.default_rng(self.rng.integers(0, 2**31 - 1))
        market_rng = np.random.default_rng(self.rng.integers(0, 2**31 - 1))

        self.resource_pool = ResourcePool(
            initial_stock=float(self.scenario_params["initial_stock"]),
            regeneration_rate=float(self.scenario_params["regeneration_rate"]),
            critical_min=float(self.scenario_params["critical_min"]),
            event_rate=float(self.scenario_params["event_rate"]),
            rng=pool_rng,
        )
        self.market = Market(
            resource_pool=self.resource_pool,
            base_prices=dict(self.scenario_params["base_prices"]),
            scarcity_alpha=float(self.scenario_params["scarcity_alpha"]),
            rng=market_rng,
            timestep_fn=lambda: self._timestep,
        )

        # Possible agents can shift if governance_module.regime changed.
        self.possible_agents = self._compute_possible_agents()
        self._agents_by_id = self._instantiate_agents()
        self.agents = list(self.possible_agents)

        # Cache spaces.
        self._observation_spaces = {
            aid: agent.observation_space for aid, agent in self._agents_by_id.items()
        }
        self._action_spaces = {
            aid: agent.action_space for aid, agent in self._agents_by_id.items()
        }

        # AEC dicts.
        self.rewards = {aid: 0.0 for aid in self.agents}
        self._cumulative_rewards = {aid: 0.0 for aid in self.agents}
        self.terminations = {aid: False for aid in self.agents}
        self.truncations = {aid: False for aid in self.agents}
        self.infos = {aid: {} for aid in self.agents}

        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        self._timestep = 0
        self._last_event = None
        self._last_5_trade_volumes = [0.0] * 5
        self._pending_governor_actions = {}
        self._step_log = {aid: {} for aid in self.agents}
        self._withdrawal_remaining = {}

        # KPI bookkeeping reset. Initial mean wealth is captured once here
        # as the baseline for resource_access / subsistence-collapse checks.
        prices0 = {item: self.market.get_price(item) for item in self.market.base_prices}
        wealths0 = [a.wealth(prices0) for a in self._agents_by_id.values()]
        self._initial_mean_wealth = float(np.mean(wealths0)) if wealths0 else 0.0
        self._stock_history = [self.resource_pool.stock]
        self._cycle_production_value = 0.0
        self._cycle_trade_value = 0.0
        self._last_cycle_gdp_raw = 0.0
        self._last_cycle_production_value = 0.0
        self._kpis = {"gini": 0.0, "hdi_proxy": 0.0, "gdp_indexed": 0.0, "depletion_rate": 0.0}
        self._wealth_histogram = [0.0] * 10

    def close(self) -> None:
        return None

    def render(self) -> None:
        return None

    # -------------------------------------------------------------- snapshot

    def _snapshot_state(
        self,
        *,
        acting_agent: Optional[str] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        assert self.resource_pool is not None and self.market is not None
        prices = {item: self.market.get_price(item) for item in self.market.base_prices}
        log = self._step_log.get(acting_agent, {}) if acting_agent is not None else {}
        snap: dict[str, Any] = {
            "timestep": self._timestep,
            "resource_pool": self.resource_pool.state_dict(),
            "prices": prices,
            "production_capacity_remaining": 1.0,
            "last_5_trade_volumes": list(self._last_5_trade_volumes),
            "last_extracted_amount": float(log.get("extracted", 0.0)),
            "last_depletion_penalty": 0.0,
            "last_produced": dict(log.get("produced", {})),
            "last_trade_profit": float(log.get("trade_profit", 0.0)),
            # Live KPI cache, updated once per cycle by _update_live_kpis().
            # Deliberately differs from kpi_logger.KPILogger's CSV columns:
            # gini/depletion_rate here are instantaneous (not rolling-mean
            # smoothed), hdi_proxy is recomputed every step (not NaN-gated
            # to 100-step epochs), gdp_indexed is the raw per-cycle GDP
            # value (not indexed to the epoch-0 baseline), and
            # depletion_rate is a FRACTION (not the CSV's percentage) to
            # fit Governor.observation_space's declared [-1, 1] bound.
            # See DECISIONS.md "Live vs CSV KPI values".
            "kpis": dict(self._kpis),
            "wealth_histogram": list(self._wealth_histogram),
            # Governance-only keys: live, mutable references. Agents never
            # read these via get_observation (their dummy test states omit
            # them); only GovernanceModule handlers may use them. G2's
            # redistribution mutates agent.cash through agents_by_id;
            # pending_governor_actions is reserved for G3 (Sprint 2.5).
            "agents_by_id": self._agents_by_id,
            "pending_governor_actions": dict(self._pending_governor_actions),
        }
        if extra:
            snap.update(extra)
        return snap

    def observe(self, agent: str) -> np.ndarray:
        a = self._agents_by_id[agent]
        return a.get_observation(self._snapshot_state(acting_agent=agent))

    # ------------------------------------------------------------- helpers

    def _agent_registry(self, aid: str) -> Optional[AgentBase]:
        return self._agents_by_id.get(aid)

    def _other_agent_ids(self, excluding: str, kinds: tuple[str, ...]) -> list[str]:
        return [
            a
            for a in self.agents
            if a != excluding and a.split("_")[0] in kinds
        ]

    def _pick_counterparty(
        self, excluding: str, kinds: tuple[str, ...]
    ) -> Optional[str]:
        assert self.rng is not None
        choices = self._other_agent_ids(excluding, kinds)
        if not choices:
            return None
        return str(self.rng.choice(choices))

    # ----------------------------------------------------------- action apply

    # -------------------------------------------------- environmental events

    def _apply_environmental_event(self, event: dict[str, Any]) -> None:
        """Applies the two spec §2.3.1 event types the pool only emits.

        ``disruption`` is already applied by ``ResourcePool`` itself and is
        skipped here. Dispatch is on the event's own ``type`` field -- data
        selection, not behaviour branching on a governance regime.
        """
        kind = event.get("type")
        if kind == "market_volatility" and self.market is not None:
            self.market.apply_volatility_episode(
                multiplier=float(event["multiplier"]),
                duration=int(event["duration"]),
            )
        elif kind == "agent_withdrawal":
            self._start_agent_withdrawal(int(event["duration"]), event)

    def _start_agent_withdrawal(self, duration: int, event: dict[str, Any]) -> None:
        """Forces one random live agent idle for ``duration`` timesteps.

        Governor agents are excluded: withdrawing the policy-setter would
        silently disable the governance regime under test, confounding the
        very comparison the experiment exists to make. The spec's stated
        intent is "simulating agent failure or exit" among the economic
        actors.
        """
        assert self.rng is not None
        candidates = [
            aid for aid in self.agents
            if not aid.startswith("governor") and self._withdrawal_remaining.get(aid, 0) <= 0
        ]
        if not candidates:
            event["agent_id"] = None
            return
        target = str(self.rng.choice(candidates))
        self._withdrawal_remaining[target] = duration
        event["agent_id"] = target

    def _tick_agent_withdrawals(self) -> None:
        """Decrements every active withdrawal by one cycle."""
        for aid, remaining in list(self._withdrawal_remaining.items()):
            if remaining > 0:
                self._withdrawal_remaining[aid] = remaining - 1

    def is_withdrawn(self, agent_id: str) -> bool:
        """Whether this agent is currently withdrawn (spec §2.3.1)."""
        return self._withdrawal_remaining.get(agent_id, 0) > 0

    def _apply_extractor(self, agent: Extractor, action: dict[str, Any]) -> None:
        assert self.resource_pool is not None
        assert self.market is not None
        log: dict[str, Any] = {"extracted": 0.0}
        if int(action["action_type"]) == 0:  # extract
            amt = float(np.asarray(action["amount"]).reshape(-1)[0]) * MAX_EXTRACT_AMOUNT
            got = self.resource_pool.extract(amt)
            agent.inventory["raw"] = float(agent.inventory.get("raw", 0.0)) + got
            log["extracted"] = got
        # action_type 1 (idle) and 2 (move_to_zone) are no-ops at the env layer.

        # Passive cash recycling, independent of action_type: Extractors
        # only ever receive cash (from Producers' raw purchases) and never
        # spend it, which is a one-way cash sink -- see DECISIONS.md
        # "Extractor cash-sink fix". This closes the loop by spending a
        # fraction of current cash on food/tool each cycle, mirroring
        # _apply_trader's buy branch exactly. Not tied to a chosen action
        # (no new action_type), so it isn't something the RL policy needs
        # to learn -- reward stays "raw units extracted" only.
        if agent.cash > 0.0:
            item = "food" if int(self.rng.integers(0, 2)) == 0 else "tool"
            price = self.market.get_price(item)
            spend = float(agent.cash) * EXTRACTOR_SPEND_FRACTION
            qty = spend / price if price > 0 else 0.0
            if qty > 0.0:
                seller_id = self._pick_counterparty(agent.agent_id, ("producer", "trader"))
                if seller_id is not None:
                    rec = self.market.execute_trade(
                        buyer_id=agent.agent_id,
                        seller_id=seller_id,
                        item=item,
                        qty=qty,
                        agent_registry=self._agent_registry,
                    )
                    if rec is not None:
                        self._cycle_trade_value += float(rec["value"])
                        log["consumed"] = {"item": item, "qty": rec["qty"], "value": rec["value"]}

        self._step_log[agent.agent_id] = log

    def _apply_producer(self, agent: Producer, action: dict[str, Any]) -> None:
        assert self.market is not None
        log: dict[str, Any] = {"produced": {"food": 0.0, "tool": 0.0}, "trade_profit": 0.0}
        atype = int(action["action_type"])
        if atype == 0:  # produce
            ptype = int(action["produce_type"])
            item = "food" if ptype == 0 else "tool"
            available = float(agent.inventory.get("raw", 0.0))
            # Auto-source any shortfall from a random Extractor counterparty
            # before producing — mirrors _apply_trader's buy branch exactly.
            # Scarcity-priced (Market.get_price("raw")), paid from cash.
            # execute_trade already clips to whatever the seller has and the
            # buyer can afford, so a Producer that's short on cash/supply
            # simply buys and produces less, never crashes. See DECISIONS.md
            # "Producer raw-material acquisition — resolved".
            shortfall = MAX_PRODUCE_AMOUNT - available
            if shortfall > 0.0:
                seller_id = self._pick_counterparty(agent.agent_id, ("extractor",))
                if seller_id is not None:
                    rec = self.market.execute_trade(
                        buyer_id=agent.agent_id,
                        seller_id=seller_id,
                        item="raw",
                        qty=shortfall,
                        agent_registry=self._agent_registry,
                    )
                    if rec is not None:
                        self._cycle_trade_value += float(rec["value"])
                    available = float(agent.inventory.get("raw", 0.0))
            qty = min(MAX_PRODUCE_AMOUNT, available)
            if qty > 0.0:
                agent.inventory["raw"] = available - qty
                agent.inventory[item] = float(agent.inventory.get(item, 0.0)) + qty
                log["produced"][item] = qty
                # GDP-equivalent (spec §3): production value is part of
                # gdp_raw alongside trade value. Valued at the price in
                # effect at production time.
                self._cycle_production_value += qty * self.market.get_price(item)
        elif atype == 2:  # trade_offer → sell a fraction of one good to a trader
            item = "food" if int(action["offer_item"]) == 0 else "tool"
            frac = float(np.asarray(action["offer_qty"]).reshape(-1)[0])
            qty = float(agent.inventory.get(item, 0.0)) * frac
            if qty > 0.0:
                buyer_id = self._pick_counterparty(agent.agent_id, ("trader",))
                if buyer_id is not None:
                    cash_before = agent.cash
                    rec = self.market.execute_trade(
                        buyer_id=buyer_id,
                        seller_id=agent.agent_id,
                        item=item,
                        qty=qty,
                        agent_registry=self._agent_registry,
                    )
                    if rec is not None:
                        log["trade_profit"] = float(agent.cash - cash_before)
                        self._last_5_trade_volumes = (
                            self._last_5_trade_volumes[1:] + [float(rec["qty"])]
                        )
                        self._cycle_trade_value += float(rec["value"])
        # action_type 1 (store) is no-op.
        self._step_log[agent.agent_id] = log

    def _apply_trader(self, agent: Trader, action: dict[str, Any]) -> None:
        assert self.market is not None
        log: dict[str, Any] = {"trade_profit": 0.0}
        atype = int(action["action_type"])
        item = "food" if int(action["item"]) == 0 else "tool"
        frac = float(np.asarray(action["qty"]).reshape(-1)[0])
        cash_before = agent.cash
        inv_value_before = (
            float(agent.inventory.get(item, 0.0)) * self.market.get_price(item)
        )
        if atype == 0:  # buy from a producer or another trader
            price = self.market.get_price(item)
            spend = float(agent.cash) * frac
            qty = spend / price if price > 0 else 0.0
            if qty > 0.0:
                seller_id = self._pick_counterparty(agent.agent_id, ("producer", "trader"))
                if seller_id is not None:
                    rec = self.market.execute_trade(
                        buyer_id=agent.agent_id,
                        seller_id=seller_id,
                        item=item,
                        qty=qty,
                        agent_registry=self._agent_registry,
                    )
                    if rec is not None:
                        self._last_5_trade_volumes = (
                            self._last_5_trade_volumes[1:] + [float(rec["qty"])]
                        )
                        self._cycle_trade_value += float(rec["value"])
        elif atype == 1:  # sell
            qty = float(agent.inventory.get(item, 0.0)) * frac
            if qty > 0.0:
                buyer_id = self._pick_counterparty(agent.agent_id, ("trader", "producer"))
                if buyer_id is not None:
                    rec = self.market.execute_trade(
                        buyer_id=buyer_id,
                        seller_id=agent.agent_id,
                        item=item,
                        qty=qty,
                        agent_registry=self._agent_registry,
                    )
                    if rec is not None:
                        self._last_5_trade_volumes = (
                            self._last_5_trade_volumes[1:] + [float(rec["qty"])]
                        )
                        self._cycle_trade_value += float(rec["value"])
        inv_value_after = (
            float(agent.inventory.get(item, 0.0)) * self.market.get_price(item)
        )
        log["trade_profit"] = float(
            (agent.cash - cash_before) + (inv_value_after - inv_value_before)
        )
        self._step_log[agent.agent_id] = log

    def _apply_governor(self, agent: Governor, action: dict[str, Any]) -> None:
        # Defer to end-of-cycle. G1/G2 ignore it; G3 (Sprint 2.5) reads it
        # from `pending_governor_actions` via the snapshot passed to on_step.
        self._pending_governor_actions[agent.agent_id] = dict(action)
        self._step_log[agent.agent_id] = {}

    def _apply_action(self, agent: AgentBase, action: dict[str, Any]) -> None:
        # Spec §2.3.1 agent withdrawal: a withdrawn agent is "set to idle",
        # i.e. its action has no environmental effect for the episode's
        # duration. Enforced here rather than by rewriting the action, so it
        # holds for every agent type without needing a per-type idle encoding
        # -- and so the policy still sees the action it chose (this simulates
        # agent failure/exit, not a change of mind).
        if self.is_withdrawn(agent.agent_id):
            self._step_log[agent.agent_id] = {"withdrawn": True}
            return
        if isinstance(agent, Extractor):
            self._apply_extractor(agent, action)
        elif isinstance(agent, Producer):
            self._apply_producer(agent, action)
        elif isinstance(agent, Trader):
            self._apply_trader(agent, action)
        elif isinstance(agent, Governor):
            self._apply_governor(agent, action)
        else:
            raise TypeError(f"unknown agent type for {agent.agent_id}")

    # --------------------------------------------------------------------- kpi

    def _update_live_kpis(self) -> None:
        """Recompute the live KPI cache after one cycle completes.

        Called once per cycle, before ``governance_module.on_step()`` so
        G2's/G3's handlers and every Governor's observation see the KPIs
        for the cycle that just finished. Uses the same pure formulas as
        ``kpi_logger.py``'s CSV writer, but the instantaneous/per-cycle
        forms — see the comment on ``_snapshot_state``'s "kpis" key for
        exactly how these differ from the CSV columns.
        """
        assert self.resource_pool is not None and self.market is not None
        prices = {item: self.market.get_price(item) for item in self.market.base_prices}
        wealths = [a.wealth(prices) for a in self._agents_by_id.values()]

        gini = _kpi.compute_gini_raw(wealths)

        stock_t = self._stock_history[-1]
        stock_t_minus_1 = self._stock_history[-2] if len(self._stock_history) >= 2 else stock_t
        # Fractional (not percentage) to fit Governor.observation_space's
        # declared depletion_rate bound of [-1, 1].
        depletion_pct = _kpi.compute_depletion_rate_raw(stock_t, stock_t_minus_1)
        depletion_fraction = float(np.clip(depletion_pct / 100.0, -1.0, 1.0))

        mean_wealth = float(np.mean(wealths)) if wealths else 0.0
        resource_access = _kpi.compute_resource_access(mean_wealth, self._initial_mean_wealth)

        n_producers = sum(1 for aid in self._agents_by_id if aid.startswith("producer"))
        # Output-goods price only — excludes "raw" (an input cost, not what
        # gets produced) from the theoretical-max production value below.
        output_prices = [prices[k] for k in ("food", "tool") if k in prices]
        mean_price = float(np.mean(output_prices)) if output_prices else 0.0
        capability_utilisation = _kpi.compute_capability_utilisation(
            self._cycle_production_value, n_producers, MAX_PRODUCE_AMOUNT, mean_price
        )

        ages = [a.age for a in self._agents_by_id.values()]
        mean_age = float(np.mean(ages)) if ages else 0.0
        agent_longevity = _kpi.compute_agent_longevity(mean_age, self._timestep)

        hdi = _kpi.compute_hdi_proxy(resource_access, capability_utilisation, agent_longevity)
        gdp_indexed_live = float(np.clip(self._last_cycle_gdp_raw, 0.0, 1e6))

        self._kpis = {
            "gini": gini,
            "hdi_proxy": hdi,
            "gdp_indexed": gdp_indexed_live,
            "depletion_rate": depletion_fraction,
        }
        self._wealth_histogram = _kpi.compute_wealth_histogram(wealths)

    # ------------------------------------------------------------------ step

    def step(self, action: Any) -> None:
        if self._agent_selector is None:
            raise RuntimeError("step() called before reset()")

        agent_id = self.agent_selection

        # Dead agents: PettingZoo convention — accept the step, advance.
        if self.terminations[agent_id] or self.truncations[agent_id]:
            self._was_dead_step(action)
            return

        agent = self._agents_by_id[agent_id]

        # Reset cumulative for the acting agent before assigning new reward.
        self._cumulative_rewards[agent_id] = 0.0

        # Governance modification (M1 stub is a no-op).
        mod_action, reward_delta = self.governance_module.on_action(agent_id, action)
        if mod_action is None:
            mod_action = action

        self._apply_action(agent, mod_action)

        snap = self._snapshot_state(acting_agent=agent_id)
        base_reward = float(agent.compute_reward(snap, mod_action))
        self.rewards[agent_id] = base_reward + float(reward_delta)

        agent.age += 1

        # End-of-cycle work: tick the world.
        if self._agent_selector.is_last():
            self._timestep += 1

            # GDP-equivalent (spec §3) = production value + trade value
            # accumulated during the cycle that just finished. Captured
            # before resetting the accumulators for the next cycle.
            self._last_cycle_gdp_raw = self._cycle_production_value + self._cycle_trade_value
            # HDI's capability_utilisation term needs the same finalized,
            # pre-reset value -- see DECISIONS.md "KPILogger
            # capability_utilisation read-after-reset fix".
            self._last_cycle_production_value = self._cycle_production_value
            self._cycle_production_value = 0.0
            self._cycle_trade_value = 0.0

            evt = self.resource_pool.step() if self.resource_pool is not None else None
            if evt is not None:
                evt["timestep"] = self._timestep
                self._last_event = evt

            # Spec §2.3.1: the pool rolls all three event types but only
            # applies `disruption` (it owns the stock). Route the other two
            # here -- this is the only layer that holds both the Market and
            # the agent registry.
            if self.resource_pool is not None:
                for pending in self.resource_pool.pending_events():
                    pending["timestep"] = self._timestep
                    self._apply_environmental_event(pending)
            if self.market is not None:
                self.market.tick_volatility()
            self._tick_agent_withdrawals()

            self._stock_history.append(self.resource_pool.stock)
            if len(self._stock_history) > 51:
                del self._stock_history[: len(self._stock_history) - 51]
            self._update_live_kpis()

            self.governance_module.on_step(self._snapshot_state())
            self._pending_governor_actions.clear()

            # Termination: env-wide collapse, or horizon.
            collapsed = self.resource_pool is not None and self.resource_pool.is_collapsed()
            at_horizon = self._timestep >= self.max_timesteps
            if collapsed or at_horizon:
                for a in self.agents:
                    self.truncations[a] = True

        self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()
