"""Unit tests for aegora_isle.environment.market.Market.

Covers M1 Sprint 1.2 acceptance criteria:
- Scarcity-adjusted price formula (worked example).
- Price monotonicity as stock depletes.
- Trade conservation (zero-sum cash + inventory value).
- Invalid trades return None without raising.
- Trade log accumulates correctly across calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from aegora_isle.environment.market import Market
from aegora_isle.environment.resource_pool import ResourcePool


@dataclass
class _StubAgent:
    """Minimal duck-typed agent for market tests."""

    agent_id: str
    cash: float = 0.0
    inventory: dict[str, float] = field(default_factory=dict)


def _make_market(
    *,
    seed: int = 0,
    initial_stock: float = 1000.0,
    scarcity_alpha: float = 0.5,
    timestep: int = 0,
) -> tuple[ResourcePool, Market]:
    rng = np.random.default_rng(seed)
    pool = ResourcePool(
        initial_stock=initial_stock,
        regeneration_rate=0.0,
        critical_min=0.0,
        event_rate=0.0,
        rng=rng,
    )
    holder = {"t": timestep}
    market = Market(
        resource_pool=pool,
        base_prices={"food": 10.0, "tool": 20.0},
        scarcity_alpha=scarcity_alpha,
        rng=rng,
        timestep_fn=lambda: holder["t"],
    )
    market._test_timestep_holder = holder  # type: ignore[attr-defined]
    return pool, market


# ----------------------------------------------------------------------- pricing


def test_price_formula_worked_example() -> None:
    """Per DECISIONS.md: price = base * (1 + 0.5 * (1 - stock/initial))."""
    pool, market = _make_market(initial_stock=1000.0, scarcity_alpha=0.5)
    # Full stock → multiplier 1.0
    pool.stock = 1000.0
    assert market.get_price("food") == pytest.approx(10.0)
    assert market.get_price("tool") == pytest.approx(20.0)
    # Half stock → multiplier 1 + 0.5 * 0.5 = 1.25
    pool.stock = 500.0
    assert market.get_price("food") == pytest.approx(12.5)
    assert market.get_price("tool") == pytest.approx(25.0)
    # Near-empty → multiplier 1 + 0.5 * 0.9 = 1.45
    pool.stock = 100.0
    assert market.get_price("food") == pytest.approx(14.5)
    assert market.get_price("tool") == pytest.approx(29.0)


def test_price_monotonic_as_stock_depletes() -> None:
    pool, market = _make_market(scarcity_alpha=0.5)
    last = -1.0
    for stock in range(1000, -1, -50):
        pool.stock = float(stock)
        p = market.get_price("food")
        assert p >= last, (stock, p, last)
        last = p


def test_price_clamped_no_subsidy_when_overshoot() -> None:
    """If stock somehow exceeds initial_stock, price doesn't go below base."""
    pool, market = _make_market(scarcity_alpha=0.5)
    pool.stock = 1500.0  # overshoot
    assert market.get_price("food") == pytest.approx(10.0)


def test_unknown_item_raises_keyerror() -> None:
    _, market = _make_market()
    with pytest.raises(KeyError):
        market.get_price("gold")


# ----------------------------------------------------------------------- trade


def _registry_factory(*agents: _StubAgent):
    table = {a.agent_id: a for a in agents}
    return lambda aid: table.get(aid)


def test_trade_conservation() -> None:
    pool, market = _make_market(initial_stock=1000.0, scarcity_alpha=0.0)
    pool.stock = 1000.0  # alpha=0 → price = base = 10.0
    buyer = _StubAgent("buyer", cash=100.0, inventory={"food": 0.0, "tool": 0.0})
    seller = _StubAgent("seller", cash=0.0, inventory={"food": 5.0, "tool": 0.0})
    reg = _registry_factory(buyer, seller)

    total_cash_before = buyer.cash + seller.cash
    total_food_before = buyer.inventory["food"] + seller.inventory["food"]

    record = market.execute_trade("buyer", "seller", "food", 3.0, reg)

    assert record is not None
    assert record["qty"] == pytest.approx(3.0)
    assert record["price"] == pytest.approx(10.0)
    assert record["value"] == pytest.approx(30.0)
    assert buyer.cash == pytest.approx(70.0)
    assert seller.cash == pytest.approx(30.0)
    assert buyer.inventory["food"] == pytest.approx(3.0)
    assert seller.inventory["food"] == pytest.approx(2.0)
    assert buyer.cash + seller.cash == pytest.approx(total_cash_before)
    assert buyer.inventory["food"] + seller.inventory["food"] == pytest.approx(
        total_food_before
    )


def test_execute_trade_supports_raw_item() -> None:
    """Market.execute_trade is fully generic over item name — confirmed
    working for "raw" (used by the Producer auto-purchase fix, see
    DECISIONS.md "Producer raw-material acquisition — resolved") with no
    code changes to Market itself, just an extra base_prices entry."""
    rng = np.random.default_rng(0)
    pool = ResourcePool(
        initial_stock=1000.0, regeneration_rate=0.0, critical_min=0.0, event_rate=0.0, rng=rng
    )
    market = Market(
        resource_pool=pool,
        base_prices={"food": 10.0, "tool": 20.0, "raw": 1.0},
        scarcity_alpha=0.0,
        rng=rng,
    )
    producer = _StubAgent("producer_0", cash=10.0, inventory={"raw": 0.0})
    extractor = _StubAgent("extractor_0", cash=0.0, inventory={"raw": 50.0})
    reg = _registry_factory(producer, extractor)

    record = market.execute_trade("producer_0", "extractor_0", "raw", 3.0, reg)

    assert record is not None
    assert record["item"] == "raw"
    assert record["qty"] == pytest.approx(3.0)
    assert record["price"] == pytest.approx(1.0)
    assert producer.inventory["raw"] == pytest.approx(3.0)
    assert extractor.inventory["raw"] == pytest.approx(47.0)
    assert producer.cash == pytest.approx(7.0)
    assert extractor.cash == pytest.approx(3.0)


def test_trade_clips_to_seller_inventory() -> None:
    pool, market = _make_market(scarcity_alpha=0.0)
    pool.stock = 1000.0
    buyer = _StubAgent("buyer", cash=10_000.0, inventory={"food": 0.0})
    seller = _StubAgent("seller", cash=0.0, inventory={"food": 2.0})
    record = market.execute_trade(
        "buyer", "seller", "food", qty=999.0, agent_registry=_registry_factory(buyer, seller)
    )
    assert record is not None
    assert record["qty"] == pytest.approx(2.0)
    assert seller.inventory["food"] == pytest.approx(0.0)


def test_trade_clips_to_buyer_cash() -> None:
    pool, market = _make_market(scarcity_alpha=0.0)
    pool.stock = 1000.0
    buyer = _StubAgent("buyer", cash=25.0, inventory={"food": 0.0})  # 25 / 10 = 2.5 units
    seller = _StubAgent("seller", cash=0.0, inventory={"food": 5.0})
    record = market.execute_trade(
        "buyer", "seller", "food", qty=5.0, agent_registry=_registry_factory(buyer, seller)
    )
    assert record is not None
    assert record["qty"] == pytest.approx(2.5)
    assert buyer.cash == pytest.approx(0.0)
    assert seller.cash == pytest.approx(25.0)
    assert seller.inventory["food"] == pytest.approx(2.5)


def test_invalid_trade_returns_none() -> None:
    """Unknown item, non-positive qty, self-trade, empty seller, broke buyer."""
    pool, market = _make_market(scarcity_alpha=0.0)
    pool.stock = 1000.0
    buyer = _StubAgent("buyer", cash=100.0, inventory={"food": 0.0})
    seller = _StubAgent("seller", cash=0.0, inventory={"food": 5.0})
    broke = _StubAgent("broke", cash=0.0, inventory={"food": 0.0})
    empty = _StubAgent("empty", cash=0.0, inventory={"food": 0.0})
    reg = _registry_factory(buyer, seller, broke, empty)

    assert market.execute_trade("buyer", "seller", "gold", 1.0, reg) is None
    assert market.execute_trade("buyer", "seller", "food", 0.0, reg) is None
    assert market.execute_trade("buyer", "seller", "food", -1.0, reg) is None
    assert market.execute_trade("buyer", "buyer", "food", 1.0, reg) is None
    assert market.execute_trade("buyer", "empty", "food", 1.0, reg) is None
    assert market.execute_trade("broke", "seller", "food", 1.0, reg) is None
    assert market.execute_trade("ghost", "seller", "food", 1.0, reg) is None  # unknown buyer


def test_trade_log_accumulates_and_resets() -> None:
    pool, market = _make_market(scarcity_alpha=0.0)
    pool.stock = 1000.0
    buyer = _StubAgent("buyer", cash=1000.0, inventory={"food": 0.0})
    seller = _StubAgent("seller", cash=0.0, inventory={"food": 100.0})
    reg = _registry_factory(buyer, seller)

    for _ in range(5):
        market.execute_trade("buyer", "seller", "food", 1.0, reg)
    log = market.get_trade_log()
    assert len(log) == 5
    assert all(r["item"] == "food" for r in log)

    market.reset()
    assert market.get_trade_log() == []


def test_trade_record_carries_timestep() -> None:
    pool, market = _make_market(scarcity_alpha=0.0)
    pool.stock = 1000.0
    buyer = _StubAgent("buyer", cash=100.0, inventory={"food": 0.0})
    seller = _StubAgent("seller", cash=0.0, inventory={"food": 5.0})
    reg = _registry_factory(buyer, seller)
    market._test_timestep_holder["t"] = 42  # type: ignore[attr-defined]
    record = market.execute_trade("buyer", "seller", "food", 1.0, reg)
    assert record is not None
    assert record["timestep"] == 42


def test_state_dict_keys() -> None:
    _, market = _make_market()
    sd = market.state_dict()
    assert set(sd.keys()) == {
        "base_prices",
        "scarcity_alpha",
        "trade_count",
        "current_prices",
        # spec §2.3.1 market-volatility episode state
        "volatility_multiplier",
        "volatility_steps_remaining",
    }
    assert set(sd["current_prices"].keys()) == {"food", "tool"}


def test_seeding_contract_rejects_non_generator() -> None:
    rng = np.random.default_rng(0)
    pool = ResourcePool(
        initial_stock=100.0,
        regeneration_rate=0.0,
        critical_min=0.0,
        event_rate=0.0,
        rng=rng,
    )
    with pytest.raises(TypeError):
        Market(
            resource_pool=pool,
            base_prices={"food": 1.0},
            scarcity_alpha=0.0,
            rng=42,  # type: ignore[arg-type]
        )


# ------------------------------------------------- Trader bid-ask spread


@pytest.fixture
def spread_enabled(monkeypatch):
    """TRADER_SPREAD ships at 0.0 (see market.py -- it flips H3, so it is not
    adopted by default). These tests exercise the *mechanism*, so they force a
    known non-zero spread rather than depending on the shipped default."""
    import aegora_isle.environment.market as market_mod

    monkeypatch.setattr(market_mod, "TRADER_SPREAD", 0.05)
    return 0.05


def test_trader_buys_at_bid_below_mid(spread_enabled) -> None:
    """Market-maker convention: Trader as buyer pays the bid."""
    _, market = _make_market()
    mid = market.get_price("food")
    assert market._execution_price("trader_0", "producer_0", "food") == pytest.approx(
        mid * (1.0 - spread_enabled)
    )


def test_trader_sells_at_ask_above_mid(spread_enabled) -> None:
    _, market = _make_market()
    mid = market.get_price("food")
    assert market._execution_price("extractor_0", "trader_0", "food") == pytest.approx(
        mid * (1.0 + spread_enabled)
    )


def test_non_trader_pairs_transact_at_mid(spread_enabled) -> None:
    """Producer <-> Extractor is unaffected: no Trader, no spread."""
    _, market = _make_market()
    mid = market.get_price("food")
    assert market._execution_price("producer_0", "extractor_1", "food") == pytest.approx(mid)


def test_trader_to_trader_transacts_at_mid(spread_enabled) -> None:
    """Two market makers have no reason to hand one the spread."""
    _, market = _make_market()
    mid = market.get_price("food")
    assert market._execution_price("trader_0", "trader_1", "food") == pytest.approx(mid)


def test_shipped_default_is_disabled() -> None:
    """The spread must ship OFF: enabling it flips the sign of the H3 effect
    (see market.py). This guards against it being switched on by accident."""
    import aegora_isle.environment.market as market_mod

    assert market_mod.TRADER_SPREAD == 0.0


def test_spread_disabled_restores_single_price_exactly() -> None:
    """TRADER_SPREAD = 0.0 must reproduce the pre-spread behaviour for every
    pairing -- the escape hatch for reverting this experiment."""
    import aegora_isle.environment.market as market_mod

    _, market = _make_market()
    original = market_mod.TRADER_SPREAD
    market_mod.TRADER_SPREAD = 0.0
    try:
        mid = market.get_price("food")
        for pair in (("trader_0", "producer_0"), ("extractor_0", "trader_0"),
                     ("producer_0", "extractor_0"), ("trader_0", "trader_1")):
            assert market._execution_price(*pair, "food") == pytest.approx(mid)
    finally:
        market_mod.TRADER_SPREAD = original


def test_spread_earns_the_trader_a_round_trip_profit(spread_enabled) -> None:
    """The point of the mechanism: buy at bid, sell at ask, keep 2x spread.

    Uses a stub registry so the round trip is deterministic and isolated
    from agent policies.
    """
    pool, market = _make_market()

    class _A:
        def __init__(self, cash, food):
            self.cash = cash
            self.inventory = {"food": food, "tool": 0.0, "raw": 0.0}

    agents = {"trader_0": _A(1000.0, 0.0),
              "producer_0": _A(0.0, 100.0),
              "extractor_0": _A(1000.0, 0.0)}
    mid = market.get_price("food")
    start_wealth = agents["trader_0"].cash + agents["trader_0"].inventory["food"] * mid

    market.execute_trade("trader_0", "producer_0", "food", 10.0, agents.get)
    market.execute_trade("extractor_0", "trader_0", "food", 10.0, agents.get)

    end_wealth = agents["trader_0"].cash + agents["trader_0"].inventory["food"] * mid
    assert end_wealth > start_wealth
    assert end_wealth - start_wealth == pytest.approx(2.0 * spread_enabled * mid * 10.0)
