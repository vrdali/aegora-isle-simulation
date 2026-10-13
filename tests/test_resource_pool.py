"""Unit tests for aegora_isle.environment.resource_pool.ResourcePool.

Verifies M1 Sprint 1.1 acceptance criteria:
- Determinism under shared seed.
- Logistic regeneration toward carrying capacity.
- Monotonic depletion when extraction exceeds regeneration.
- Disruption event probability matches event_rate over many steps.
- Extraction clipping when amount > stock.
- Seeding contract enforcement (rejects non-Generator inputs).
"""

from __future__ import annotations

import numpy as np
import pytest

from aegora_isle.environment.resource_pool import ResourcePool


def _make_pool(
    *,
    seed: int = 42,
    initial_stock: float = 1000.0,
    regeneration_rate: float = 0.01,
    critical_min: float = 100.0,
    event_rate: float = 0.0,
) -> ResourcePool:
    rng = np.random.default_rng(seed)
    return ResourcePool(
        initial_stock=initial_stock,
        regeneration_rate=regeneration_rate,
        critical_min=critical_min,
        event_rate=event_rate,
        rng=rng,
    )


def test_seeding_contract_rejects_non_generator() -> None:
    with pytest.raises(TypeError):
        ResourcePool(
            initial_stock=1000.0,
            regeneration_rate=0.01,
            critical_min=100.0,
            event_rate=0.0,
            rng=42,  # type: ignore[arg-type]
        )


def test_initial_state() -> None:
    pool = _make_pool()
    assert pool.stock == pytest.approx(1000.0)
    assert pool.initial_stock == pytest.approx(1000.0)
    assert pool.critical_min == pytest.approx(100.0)
    assert pool.event_count == 0
    assert pool.is_collapsed() is False


def test_determinism_same_seed() -> None:
    """Two pools with identical seed produce identical trajectories."""
    a = _make_pool(seed=12345, event_rate=5.0)
    b = _make_pool(seed=12345, event_rate=5.0)
    for _ in range(1000):
        a.extract(2.0)
        b.extract(2.0)
        a.step()
        b.step()
    assert a.state_dict() == b.state_dict()


def test_determinism_different_seed_diverges() -> None:
    a = _make_pool(seed=1, event_rate=5.0)
    b = _make_pool(seed=2, event_rate=5.0)
    for _ in range(500):
        a.extract(2.0)
        b.extract(2.0)
        a.step()
        b.step()
    assert a.stock != b.stock or a.event_count != b.event_count


def test_regeneration_grows_toward_capacity() -> None:
    """With no extraction and no events, stock should rise toward initial_stock."""
    pool = _make_pool(initial_stock=1000.0, regeneration_rate=0.05, event_rate=0.0)
    pool.stock = 500.0
    for _ in range(200):
        pool.step()
    # 200 steps of 5% logistic growth from 500 saturates very close to 1000.
    assert pool.stock > 990.0
    assert pool.stock <= 1000.0


def test_regeneration_capped_at_initial() -> None:
    pool = _make_pool(regeneration_rate=0.5, event_rate=0.0)
    pool.stock = 1000.0  # already at carrying capacity
    pool.step()
    assert pool.stock == pytest.approx(1000.0)


def test_depletion_when_extraction_exceeds_regen() -> None:
    pool = _make_pool(initial_stock=1000.0, regeneration_rate=0.01, event_rate=0.0)
    for _ in range(100):
        pool.extract(50.0)
        pool.step()
    assert pool.stock < 1000.0
    # Stock should be substantially drained.
    assert pool.stock < 100.0


def test_extraction_clips_to_availability() -> None:
    pool = _make_pool(initial_stock=10.0, regeneration_rate=0.0, event_rate=0.0)
    got = pool.extract(50.0)
    assert got == pytest.approx(10.0)
    assert pool.stock == pytest.approx(0.0)
    # Subsequent extracts return 0.
    assert pool.extract(5.0) == pytest.approx(0.0)


def test_extract_negative_raises() -> None:
    pool = _make_pool()
    with pytest.raises(ValueError):
        pool.extract(-1.0)


def test_event_probability_matches_rate() -> None:
    """Spec §2.3.1: event_rate is per-type, per 100 timesteps.

    Each of the three types rolls independently at event_rate/100, so over
    10,000 steps at event_rate=5 each type fires ~500 times and the total is
    ~1500. Asserting per-type (not just the sum) is what actually pins the
    spec contract -- and guarantees the `disruption` rate is unchanged from
    the single-event implementation, which is why the Scenario C/D tuning
    remains valid.
    """
    pool = _make_pool(seed=2026, event_rate=5.0, regeneration_rate=0.0)
    for _ in range(10_000):
        pool.step()
    counts = pool.event_counts_by_type
    # Poisson-ish: 5% chance per step over 10k steps ~ N(500, sqrt(475)≈22).
    # Allow 4σ band for CI stability.
    for kind in ("disruption", "market_volatility", "agent_withdrawal"):
        assert 400 <= counts[kind] <= 600, (kind, counts)
    assert pool.event_count == sum(counts.values())


def test_inject_disruption_cuts_10_to_30_pct() -> None:
    pool = _make_pool(initial_stock=1000.0, regeneration_rate=0.0, event_rate=0.0)
    for _ in range(50):
        before = pool.stock
        evt = pool.inject_disruption()
        cut_pct = evt["cut_pct"]
        assert 0.10 <= cut_pct <= 0.30
        assert pool.stock == pytest.approx(before * (1.0 - cut_pct))
    assert pool.event_count == 50


def test_is_collapsed_threshold() -> None:
    pool = _make_pool(initial_stock=1000.0, critical_min=100.0, regeneration_rate=0.0)
    assert pool.is_collapsed() is False
    pool.stock = 99.999
    assert pool.is_collapsed() is True


def test_state_dict_keys() -> None:
    pool = _make_pool()
    keys = set(pool.state_dict().keys())
    expected = {
        "stock",
        "initial_stock",
        "critical_min",
        "regeneration_rate",
        "event_rate",
        "event_count",
        # spec §2.3.1 per-type breakdown of event_count
        "event_counts_by_type",
    }
    assert keys == expected
