"""Unit tests for aegora_isle.governance.rule_based.RuleBasedGovernor.

Covers M2 Sprint 2.2 acceptance criteria (G2_PLAN.md):
- Extraction-cap clipping arithmetic.
- Sanction fires at the configured probability (statistical band check).
- No sanction / no modification when within cap.
- Non-Extractor and non-extract actions pass through unmodified.
- Redistribution conserves total cash and only fires on period boundaries.
- Seeding contract + parameter validation.
- Drift detector: MAX_EXTRACT_AMOUNT mirrors the env's constant exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from aegora_isle.environment.aegora_env import MAX_EXTRACT_AMOUNT as ENV_MAX_EXTRACT_AMOUNT
from aegora_isle.governance.rule_based import MAX_EXTRACT_AMOUNT, RuleBasedGovernor


@dataclass
class _StubAgent:
    """Minimal duck-typed agent for redistribution tests."""

    agent_id: str
    cash: float = 0.0
    inventory: dict[str, float] = field(default_factory=dict)


def _make_governor(
    *,
    extraction_cap: float = 3.0,
    sanction_probability: float = 0.85,
    sanction_magnitude: float = -2.0,
    redistribution_period: int = 100,
    redistribution_rate: float = 0.10,
    seed: int = 0,
) -> RuleBasedGovernor:
    return RuleBasedGovernor(
        extraction_cap=extraction_cap,
        sanction_probability=sanction_probability,
        sanction_magnitude=sanction_magnitude,
        redistribution_period=redistribution_period,
        redistribution_rate=redistribution_rate,
        rng=np.random.default_rng(seed),
    )


def _extract_action(frac: float) -> dict:
    return {
        "action_type": 0,
        "amount": np.array([frac], dtype=np.float32),
        "target_zone": 0,
    }


# ----------------------------------------------------------- drift detector


def test_max_extract_amount_mirrors_env_constant() -> None:
    """governance/rule_based.py duplicates this constant rather than
    importing aegora_env (layering: environment depends on governance, not
    vice versa). This test catches silent drift if either changes.
    """
    assert MAX_EXTRACT_AMOUNT == ENV_MAX_EXTRACT_AMOUNT


# --------------------------------------------------------------- on_action


def test_within_cap_passes_through_unmodified() -> None:
    gov = _make_governor(extraction_cap=5.0, sanction_probability=1.0)
    action = _extract_action(0.5)  # 0.5 * 5.0 = 2.5 raw <= cap of 5.0
    mod, delta = gov.on_action("extractor_0", action)
    assert mod == action
    assert delta == 0.0


def test_over_cap_clips_amount() -> None:
    """Requesting 100% of MAX_EXTRACT_AMOUNT (5.0) against a 3.0 cap clips
    the fraction to 3.0/5.0 = 0.6."""
    gov = _make_governor(extraction_cap=3.0, sanction_probability=0.0)
    mod, delta = gov.on_action("extractor_0", _extract_action(1.0))
    assert mod["amount"][0] == pytest.approx(0.6)
    assert mod["target_zone"] == 0  # other fields preserved
    assert delta == 0.0  # sanction_probability=0 -> never sanctioned


def test_sanction_fires_at_configured_rate() -> None:
    """Over 1000 oversized extracts at p=0.5, expect ~500 sanctions ±100
    (binomial std ≈ 15.8; 100 is a generous ±6σ band for CI stability)."""
    gov = _make_governor(extraction_cap=1.0, sanction_probability=0.5, seed=2026)
    deltas = [gov.on_action("extractor_0", _extract_action(1.0))[1] for _ in range(1000)]
    n_sanctions = sum(1 for d in deltas if d < 0)
    assert 400 <= n_sanctions <= 600, n_sanctions


def test_sanction_magnitude_applied_exactly() -> None:
    gov = _make_governor(extraction_cap=1.0, sanction_probability=1.0, sanction_magnitude=-7.5)
    _, delta = gov.on_action("extractor_0", _extract_action(1.0))
    assert delta == pytest.approx(-7.5)


def test_idle_and_move_actions_passthrough() -> None:
    gov = _make_governor(extraction_cap=0.0, sanction_probability=1.0)
    idle_action = {"action_type": 1, "amount": np.array([0.0], dtype=np.float32), "target_zone": 0}
    move_action = {"action_type": 2, "amount": np.array([0.0], dtype=np.float32), "target_zone": 2}
    assert gov.on_action("extractor_0", idle_action) == (idle_action, 0.0)
    assert gov.on_action("extractor_0", move_action) == (move_action, 0.0)


def test_non_extractor_actions_passthrough() -> None:
    """Producer/Trader/Governor actions are never subject to the cap."""
    gov = _make_governor(extraction_cap=0.0, sanction_probability=1.0)
    producer_action = {
        "action_type": 2,
        "produce_type": 0,
        "offer_item": 0,
        "offer_qty": np.array([1.0], dtype=np.float32),
    }
    mod, delta = gov.on_action("producer_3", producer_action)
    assert mod == producer_action
    assert delta == 0.0


# --------------------------------------------------------------- on_step


def _agents_by_id(*agents: _StubAgent) -> dict[str, _StubAgent]:
    return {a.agent_id: a for a in agents}


def test_redistribution_conserves_total_cash() -> None:
    gov = _make_governor(redistribution_period=100, redistribution_rate=0.10)
    extractor_a = _StubAgent("extractor_0", cash=10.0)
    extractor_b = _StubAgent("extractor_1", cash=20.0)
    producer = _StubAgent("producer_0", cash=100.0)
    trader = _StubAgent("trader_0", cash=50.0)
    agents = _agents_by_id(extractor_a, extractor_b, producer, trader)
    total_before = sum(a.cash for a in agents.values())

    gov.on_step({"timestep": 100, "agents_by_id": agents})

    total_after = sum(a.cash for a in agents.values())
    assert total_after == pytest.approx(total_before)
    # 10% of (100 + 50) = 15, split equally among 2 extractors = +7.5 each
    assert extractor_a.cash == pytest.approx(17.5)
    assert extractor_b.cash == pytest.approx(27.5)
    assert producer.cash == pytest.approx(90.0)
    assert trader.cash == pytest.approx(45.0)


def test_redistribution_only_fires_on_period_boundary() -> None:
    gov = _make_governor(redistribution_period=100, redistribution_rate=0.5)
    extractor = _StubAgent("extractor_0", cash=0.0)
    producer = _StubAgent("producer_0", cash=100.0)
    agents = _agents_by_id(extractor, producer)

    for t in (1, 50, 99, 101, 150, 199):
        gov.on_step({"timestep": t, "agents_by_id": agents})
    assert extractor.cash == pytest.approx(0.0)
    assert producer.cash == pytest.approx(100.0)

    gov.on_step({"timestep": 100, "agents_by_id": agents})
    assert extractor.cash == pytest.approx(50.0)
    assert producer.cash == pytest.approx(50.0)

    gov.on_step({"timestep": 200, "agents_by_id": agents})
    assert extractor.cash == pytest.approx(75.0)
    assert producer.cash == pytest.approx(25.0)


def test_redistribution_noop_with_no_donors_or_recipients() -> None:
    gov = _make_governor(redistribution_period=10, redistribution_rate=0.5)
    only_producer = _agents_by_id(_StubAgent("producer_0", cash=100.0))
    gov.on_step({"timestep": 10, "agents_by_id": only_producer})
    assert only_producer["producer_0"].cash == pytest.approx(100.0)  # no extractors to receive


def test_redistribution_noop_when_agents_by_id_missing() -> None:
    gov = _make_governor(redistribution_period=10, redistribution_rate=0.5)
    gov.on_step({"timestep": 10})  # no KeyError


def test_on_step_at_timestep_zero_does_not_redistribute() -> None:
    gov = _make_governor(redistribution_period=100, redistribution_rate=0.5)
    extractor = _StubAgent("extractor_0", cash=0.0)
    producer = _StubAgent("producer_0", cash=100.0)
    agents = _agents_by_id(extractor, producer)
    gov.on_step({"timestep": 0, "agents_by_id": agents})
    assert extractor.cash == pytest.approx(0.0)


# ---------------------------------------------------------- bookkeeping


def test_state_dict_tracks_counts() -> None:
    gov = _make_governor(extraction_cap=1.0, sanction_probability=1.0, redistribution_period=10)
    gov.on_action("extractor_0", _extract_action(1.0))
    agents = _agents_by_id(_StubAgent("extractor_0", cash=0.0), _StubAgent("producer_0", cash=10.0))
    gov.on_step({"timestep": 10, "agents_by_id": agents})
    sd = gov.state_dict()
    assert sd["regime"] == "G2"
    assert sd["sanction_count"] == 1
    assert sd["redistribution_count"] == 1


# ---------------------------------------------------------- validation


def test_seeding_contract_rejects_non_generator() -> None:
    with pytest.raises(TypeError):
        RuleBasedGovernor(
            extraction_cap=1.0,
            sanction_probability=0.5,
            sanction_magnitude=-1.0,
            redistribution_period=10,
            redistribution_rate=0.1,
            rng=42,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"extraction_cap": -1.0},
        {"sanction_probability": 1.5},
        {"sanction_probability": -0.1},
        {"redistribution_period": 0},
        {"redistribution_period": -5},
        {"redistribution_rate": 1.5},
        {"redistribution_rate": -0.1},
    ],
)
def test_invalid_parameters_raise(kwargs: dict) -> None:
    base = dict(
        extraction_cap=3.0,
        sanction_probability=0.5,
        sanction_magnitude=-2.0,
        redistribution_period=100,
        redistribution_rate=0.1,
        rng=np.random.default_rng(0),
    )
    base.update(kwargs)
    with pytest.raises(ValueError):
        RuleBasedGovernor(**base)
