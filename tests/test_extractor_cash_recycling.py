"""Tests for the Extractor cash-recycling fix.

Per DECISIONS.md "Extractor cash-sink fix -- Option 1 chosen": Extractors
sell raw material for cash but previously had no mechanism to spend it
back into the economy, so 100% of system cash migrated to Extractors by
~timestep 100 of every 10,000-step run, freezing all trade/production for
the rest of the run. The fix: each cycle, an Extractor with cash > 0
automatically spends ``EXTRACTOR_SPEND_FRACTION`` of it buying food/tool
from a random Producer/Trader counterparty via ``Market.execute_trade`` --
mirroring ``_apply_trader``'s buy branch exactly. No new action_type on
Extractor; the purchase lives entirely inside ``AegoraEnv._apply_extractor``.

These tests call ``env._apply_extractor(agent, action)`` directly
(bypassing the full AEC step cycle) so the Extractor/counterparty pairing
and starting cash/inventory can be set up deterministically, rather than
depending on ``_pick_counterparty``'s random selection landing on a
specific agent -- mirrors ``tests/test_producer_raw_sourcing.py`` exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

from aegora_isle.environment.aegora_env import EXTRACTOR_SPEND_FRACTION, AegoraEnv
from aegora_isle.environment.market import TRADER_SPREAD
from aegora_isle.governance.governance_module import G1Handler


SCENARIO = {
    "initial_stock": 1000.0,
    "regeneration_rate": 0.0,
    "event_rate": 0.0,
    "critical_min": 0.0,
    "scarcity_alpha": 0.5,
    "base_prices": {"food": 10.0, "tool": 20.0, "raw": 1.0},
}


def _make_env(n_agents: int = 20, seed: int = 42) -> AegoraEnv:
    env = AegoraEnv(
        seed=seed,
        n_agents=n_agents,
        governance_module=G1Handler(),
        scenario_params=SCENARIO,
        max_timesteps=1000,
    )
    env.reset()
    return env


def _idle_action() -> dict:
    return {
        "action_type": 1,
        "amount": np.array([0.0], dtype=np.float32),
        "target_zone": 0,
    }


def _first_of_kind(env: AegoraEnv, kind: str) -> str:
    return next(a for a in env.possible_agents if a.startswith(kind))


def _all_of_kind(env: AegoraEnv, kind: str) -> list:
    return [a for aid, a in env._agents_by_id.items() if aid.startswith(kind)]


def _set_all_producers_and_traders(env: AegoraEnv, *, cash: float, food: float, tool: float) -> None:
    """_pick_counterparty chooses randomly among ALL Producers/Traders, not
    a specific one -- set every one identically so the test's outcome
    doesn't depend on which one gets picked."""
    for agent in _all_of_kind(env, "producer") + _all_of_kind(env, "trader"):
        agent.cash = cash
        agent.inventory["food"] = food
        agent.inventory["tool"] = tool


def test_extractor_spends_fraction_of_cash_on_food_or_tool() -> None:
    """Extractor with cash, counterparties with inventory -> cash drops by
    the expected fraction, extractor's food/tool inventory increases,
    counterparty gains cash."""
    env = _make_env()
    extractor = env._agents_by_id[_first_of_kind(env, "extractor")]
    extractor.cash = 100.0
    _set_all_producers_and_traders(env, cash=0.0, food=50.0, tool=50.0)

    env._apply_extractor(extractor, _idle_action())

    # The Extractor sizes its order at the mid price, but pays the actual
    # execution price. If the randomly-picked counterparty is a Trader, the
    # Trader is the seller and quotes the ask, so the Extractor crosses the
    # spread and pays up to (1 + TRADER_SPREAD) x more. Either counterparty
    # is valid here, so assert the achievable band rather than a single
    # value. See DECISIONS.md "Trader bid-ask spread".
    expected_spend = 100.0 * EXTRACTOR_SPEND_FRACTION
    assert 100.0 - expected_spend * (1.0 + TRADER_SPREAD) - 1e-6 <= extractor.cash
    assert extractor.cash <= 100.0 - expected_spend + 1e-6
    assert extractor.inventory["food"] > 0.0 or extractor.inventory["tool"] > 0.0


def test_extractor_zero_cash_is_silent_noop() -> None:
    """Extractor with 0 cash -> no trade attempted, no crash."""
    env = _make_env()
    extractor = env._agents_by_id[_first_of_kind(env, "extractor")]
    extractor.cash = 0.0
    _set_all_producers_and_traders(env, cash=0.0, food=50.0, tool=50.0)

    env._apply_extractor(extractor, _idle_action())

    assert extractor.cash == pytest.approx(0.0)
    assert extractor.inventory["food"] == pytest.approx(0.0)
    assert extractor.inventory["tool"] == pytest.approx(0.0)


def test_extractor_no_producer_or_trader_in_population_is_silent_noop() -> None:
    """If _pick_counterparty finds no Producer/Trader at all, the spend
    step silently no-ops -- no crash."""
    env = _make_env()
    extractor_id = _first_of_kind(env, "extractor")
    extractor = env._agents_by_id[extractor_id]
    extractor.cash = 100.0

    for aid in list(env._agents_by_id):
        if aid.startswith("producer") or aid.startswith("trader"):
            del env._agents_by_id[aid]
    env.agents = [
        a for a in env.agents if not (a.startswith("producer") or a.startswith("trader"))
    ]

    env._apply_extractor(extractor, _idle_action())

    assert extractor.cash == pytest.approx(100.0)
    assert extractor.inventory["food"] == pytest.approx(0.0)
    assert extractor.inventory["tool"] == pytest.approx(0.0)


def test_extractor_seller_zero_inventory_is_silent_noop() -> None:
    """Every Producer/Trader has 0.0 of both goods -> execute_trade returns
    None for whichever item gets picked, spend silently no-ops."""
    env = _make_env()
    extractor = env._agents_by_id[_first_of_kind(env, "extractor")]
    extractor.cash = 100.0
    _set_all_producers_and_traders(env, cash=0.0, food=0.0, tool=0.0)

    env._apply_extractor(extractor, _idle_action())

    assert extractor.cash == pytest.approx(100.0)
    assert extractor.inventory["food"] == pytest.approx(0.0)
    assert extractor.inventory["tool"] == pytest.approx(0.0)


def test_extractor_spend_conserves_total_cash() -> None:
    """Conservation: total cash across the extractor and every producer/
    trader counterparty is unchanged by the spend step."""
    env = _make_env()
    extractor = env._agents_by_id[_first_of_kind(env, "extractor")]
    extractor.cash = 100.0
    _set_all_producers_and_traders(env, cash=0.0, food=50.0, tool=50.0)
    counterparties = _all_of_kind(env, "producer") + _all_of_kind(env, "trader")

    total_before = extractor.cash + sum(a.cash for a in counterparties)

    env._apply_extractor(extractor, _idle_action())

    total_after = extractor.cash + sum(a.cash for a in counterparties)
    assert total_after == pytest.approx(total_before)


def test_extractor_reward_unaffected_by_spend() -> None:
    """compute_reward still equals extracted amount only -- the spend step
    is passive background behaviour, not something the RL policy is
    rewarded for learning."""
    env = _make_env()
    extractor = env._agents_by_id[_first_of_kind(env, "extractor")]
    extractor.cash = 100.0
    _set_all_producers_and_traders(env, cash=0.0, food=50.0, tool=50.0)

    action = {
        "action_type": 0,
        "amount": np.array([1.0], dtype=np.float32),
        "target_zone": 0,
    }
    env._apply_extractor(extractor, action)

    env_state = {"last_extracted_amount": env._step_log[extractor.agent_id]["extracted"]}
    reward = extractor.compute_reward(env_state, action)
    assert reward == pytest.approx(env._step_log[extractor.agent_id]["extracted"])


def test_gdp_raw_and_producer_cash_stay_nonzero_late_in_a_run() -> None:
    """End-to-end regression check: a longer run with forced actions no
    longer freezes -- gdp_raw stays intermittently nonzero, and Producer
    cash doesn't permanently flatline at 0, well past the point where the
    pre-fix economy would have already fully drained (empirically ~step
    100 of 10,000; this run is long enough relative to n_agents to cross
    that same relative point)."""
    env = _make_env(n_agents=20)
    n_cycles = 800
    gdp_by_cycle: list[float] = []
    producer_cash_by_cycle: list[float] = []

    steps = 0
    max_steps = n_cycles * env.n_agents
    while steps < max_steps:
        if not env.agents:
            break
        aid = env.agent_selection
        kind = aid.split("_")[0]
        if env.terminations[aid] or env.truncations[aid]:
            env.step(None)
            steps += 1
            continue
        if kind == "extractor":
            action = {"action_type": 0, "amount": np.array([1.0], dtype=np.float32), "target_zone": 0}
        elif kind == "producer":
            action = {
                "action_type": 0,
                "produce_type": 0,
                "offer_item": 0,
                "offer_qty": np.array([0.0], dtype=np.float32),
            }
        else:
            action = env.action_space(aid).sample()
        env.step(action)
        steps += 1
        if env.agents and env._agent_selector.is_first():
            gdp_by_cycle.append(env._last_cycle_gdp_raw)
            producers = _all_of_kind(env, "producer")
            producer_cash_by_cycle.append(
                float(np.mean([p.cash for p in producers])) if producers else 0.0
            )

    back_half = gdp_by_cycle[len(gdp_by_cycle) // 2 :]
    assert any(v > 0.0 for v in back_half), "gdp_raw stayed at 0 for the entire back half of the run"

    back_half_cash = producer_cash_by_cycle[len(producer_cash_by_cycle) // 2 :]
    assert any(c > 0.0 for c in back_half_cash), (
        "producer cash stayed at 0 for the entire back half of the run -- "
        "the cash-sink freeze was not fixed, only delayed"
    )
