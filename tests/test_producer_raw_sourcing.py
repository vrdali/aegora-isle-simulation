"""Tests for the Producer raw-material auto-purchase fix.

Per DECISIONS.md "Producer raw-material acquisition — resolved": when a
Producer's ``produce`` action fires and it lacks enough raw material, it
automatically buys the shortfall from a randomly-picked Extractor
counterparty via ``Market.execute_trade`` (scarcity-priced, same formula
as food/tool) — mirroring ``_apply_trader``'s buy branch exactly. No new
action_type on either agent; the purchase lives entirely inside
``AegoraEnv._apply_producer``.

These tests call ``env._apply_producer(agent, action)`` directly (bypassing
the full AEC step cycle) so the Producer/Extractor pairing and their
starting cash/inventory can be set up deterministically, rather than
depending on ``_pick_counterparty``'s random selection landing on a
specific agent.
"""

from __future__ import annotations

import numpy as np
import pytest

from aegora_isle.environment.aegora_env import MAX_PRODUCE_AMOUNT, AegoraEnv
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


def _produce_action(produce_type: int = 0) -> dict:
    return {
        "action_type": 0,
        "produce_type": produce_type,
        "offer_item": 0,
        "offer_qty": np.array([0.0], dtype=np.float32),
    }


def _first_of_kind(env: AegoraEnv, kind: str) -> str:
    return next(a for a in env.possible_agents if a.startswith(kind))


def _all_of_kind(env: AegoraEnv, kind: str) -> list:
    return [a for aid, a in env._agents_by_id.items() if aid.startswith(kind)]


def _set_all_extractors(env: AegoraEnv, *, cash: float, raw: float) -> None:
    """_pick_counterparty chooses randomly among ALL Extractors, not a
    specific one -- set every Extractor identically so the test's outcome
    doesn't depend on which one gets picked."""
    for extractor in _all_of_kind(env, "extractor"):
        extractor.cash = cash
        extractor.inventory["raw"] = raw


def test_producer_buys_shortfall_and_produces() -> None:
    """Producer with zero raw, every Extractor with plenty -> full purchase,
    full production, cash moves Producer->Extractor, raw moves the other way.
    """
    env = _make_env()
    producer = env._agents_by_id[_first_of_kind(env, "producer")]
    producer.cash = 100.0
    producer.inventory["raw"] = 0.0
    _set_all_extractors(env, cash=0.0, raw=50.0)
    extractors = _all_of_kind(env, "extractor")

    total_cash_before = producer.cash + sum(e.cash for e in extractors)
    total_raw_before = producer.inventory["raw"] + sum(e.inventory["raw"] for e in extractors)

    env._apply_producer(producer, _produce_action(produce_type=0))

    # Fully filled: shortfall was MAX_PRODUCE_AMOUNT, every extractor had plenty.
    assert producer.inventory["raw"] == pytest.approx(0.0)  # bought, then consumed
    assert producer.inventory["food"] == pytest.approx(MAX_PRODUCE_AMOUNT)
    assert producer.cash < 100.0
    # Exactly one extractor (whichever was picked) lost MAX_PRODUCE_AMOUNT raw
    # and gained cash; the rest are untouched.
    sellers = [e for e in extractors if e.inventory["raw"] != pytest.approx(50.0)]
    assert len(sellers) == 1
    assert sellers[0].inventory["raw"] == pytest.approx(50.0 - MAX_PRODUCE_AMOUNT)
    assert sellers[0].cash > 0.0

    # Conservation: cash + raw (at flat 1.0 for this check) both preserved.
    assert producer.cash + sum(e.cash for e in extractors) == pytest.approx(total_cash_before)
    assert (
        producer.inventory["raw"] + sum(e.inventory["raw"] for e in extractors) + producer.inventory["food"]
        == pytest.approx(total_raw_before)
    )


def test_producer_partial_fill_when_cash_limited() -> None:
    """Producer can't afford the full shortfall -> buys what it can afford,
    produces a smaller amount than MAX_PRODUCE_AMOUNT."""
    env = _make_env()
    producer = env._agents_by_id[_first_of_kind(env, "producer")]

    # Raw price ~1.0 (scarcity_alpha=0.5, full stock -> multiplier 1.0).
    # cash=1.0 affords ~1.0 raw units, well under MAX_PRODUCE_AMOUNT=3.0.
    producer.cash = 1.0
    producer.inventory["raw"] = 0.0
    _set_all_extractors(env, cash=0.0, raw=50.0)

    env._apply_producer(producer, _produce_action(produce_type=0))

    assert producer.cash == pytest.approx(0.0, abs=1e-6)
    assert 0.0 < producer.inventory["food"] < MAX_PRODUCE_AMOUNT


def test_producer_partial_fill_when_extractor_supply_limited() -> None:
    """Every Extractor has less raw than the shortfall -> buys all that's
    available from whichever one gets picked, produces only that much."""
    env = _make_env()
    producer = env._agents_by_id[_first_of_kind(env, "producer")]
    producer.cash = 1000.0
    producer.inventory["raw"] = 0.0
    _set_all_extractors(env, cash=0.0, raw=1.5)  # less than MAX_PRODUCE_AMOUNT=3.0

    env._apply_producer(producer, _produce_action(produce_type=0))

    assert producer.inventory["food"] == pytest.approx(1.5)
    sellers = [e for e in _all_of_kind(env, "extractor") if e.inventory["raw"] != pytest.approx(1.5)]
    assert len(sellers) == 1
    assert sellers[0].inventory["raw"] == pytest.approx(0.0)


def test_producer_no_extractor_in_population_is_silent_noop() -> None:
    """If _pick_counterparty finds no Extractor at all, produce silently
    no-ops (matches pre-fix behaviour) -- no crash."""
    env = _make_env()
    producer_id = _first_of_kind(env, "producer")
    producer = env._agents_by_id[producer_id]
    producer.cash = 100.0
    producer.inventory["raw"] = 0.0

    # Remove every Extractor from the live registry so _pick_counterparty
    # finds no candidates.
    for aid in list(env._agents_by_id):
        if aid.startswith("extractor"):
            del env._agents_by_id[aid]
    env.agents = [a for a in env.agents if not a.startswith("extractor")]

    env._apply_producer(producer, _produce_action(produce_type=0))

    assert producer.inventory["food"] == pytest.approx(0.0)
    assert producer.cash == pytest.approx(100.0)


def test_producer_zero_raw_extractor_is_silent_noop() -> None:
    """Every Extractor has 0.0 raw -> execute_trade returns None,
    produce silently no-ops."""
    env = _make_env()
    producer = env._agents_by_id[_first_of_kind(env, "producer")]
    producer.cash = 100.0
    producer.inventory["raw"] = 0.0
    _set_all_extractors(env, cash=0.0, raw=0.0)

    env._apply_producer(producer, _produce_action(produce_type=0))

    assert producer.inventory["food"] == pytest.approx(0.0)
    assert producer.cash == pytest.approx(100.0)


def test_producer_with_partial_existing_raw_only_buys_shortfall() -> None:
    """Producer already has SOME raw -> only tops up to MAX_PRODUCE_AMOUNT,
    doesn't over-buy."""
    env = _make_env()
    producer = env._agents_by_id[_first_of_kind(env, "producer")]
    producer.cash = 1000.0
    producer.inventory["raw"] = 1.0  # already has 1.0 of the 3.0 needed
    _set_all_extractors(env, cash=0.0, raw=50.0)

    env._apply_producer(producer, _produce_action(produce_type=0))

    # Only the 2.0 shortfall should have been bought from whichever extractor
    # got picked.
    assert producer.inventory["food"] == pytest.approx(MAX_PRODUCE_AMOUNT)
    sellers = [e for e in _all_of_kind(env, "extractor") if e.inventory["raw"] != pytest.approx(50.0)]
    assert len(sellers) == 1
    assert sellers[0].inventory["raw"] == pytest.approx(50.0 - 2.0)


def test_gdp_raw_nonzero_after_fix_end_to_end() -> None:
    """End-to-end regression check: a short run with real (random) actions
    now produces nonzero gdp_raw at least once -- the actual symptom this
    fix resolves. Forces Extractors to always extract (maximal) and
    Producers to always attempt production, isolating the fix's effect
    from action-sampling luck."""
    env = _make_env(n_agents=20)
    gdp_values: list[float] = []
    for _ in range(100 * env.n_agents):
        if not env.agents:
            break
        aid = env.agent_selection
        kind = aid.split("_")[0]
        if env.terminations[aid] or env.truncations[aid]:
            env.step(None)
            continue
        if kind == "extractor":
            action = {"action_type": 0, "amount": np.array([1.0], dtype=np.float32), "target_zone": 0}
        elif kind == "producer":
            action = _produce_action(produce_type=0)
        else:
            action = env.action_space(aid).sample()
        env.step(action)
        if env.agents and env._agent_selector.is_first():
            gdp_values.append(env._last_cycle_gdp_raw)

    assert any(v > 0.0 for v in gdp_values), "gdp_raw stayed zero across the whole run"
