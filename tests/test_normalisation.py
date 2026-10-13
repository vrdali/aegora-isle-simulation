"""Unit tests for aegora_isle.experiments.normalisation.RunningNormaliser.

Vector generalisation of the already-tested scalar
aegora_isle.governance.adaptive_rl._RunningNormaliser (see
tests/test_adaptive_rl.py for the pattern this mirrors) -- covers the
Milestone 3 Sprint 3.3 fix for PPO training instability at real scale
(see DECISIONS.md "PPO observation/reward normalisation -- confirmed gap,
fixed").
"""

from __future__ import annotations

import numpy as np
import pytest

from aegora_isle.experiments.normalisation import RunningNormaliser


def test_running_normaliser_stabilises_per_dimension() -> None:
    rng = np.random.default_rng(0)
    dim = 3
    loc = np.array([3.0, -10.0, 1000.0])
    scale = np.array([2.0, 5.0, 50.0])
    norm = RunningNormaliser(dim=dim, clip=5.0)

    samples = rng.normal(loc=loc, scale=scale, size=(2000, dim))
    for s in samples:
        norm.update(s)

    assert norm.mean == pytest.approx(loc, abs=0.5)

    normalised = np.array([norm.normalise(s) for s in samples[-100:]])
    assert np.all(np.abs(normalised.mean(axis=0)) < 0.5)
    assert np.all(normalised >= -5.0) and np.all(normalised <= 5.0)


def test_single_sample_clips_without_division_by_zero() -> None:
    norm = RunningNormaliser(dim=2, clip=5.0)
    norm.update(np.array([10.0, -10.0]))
    result = norm.normalise(np.array([10.0, -10.0]))
    assert result == pytest.approx([5.0, -5.0])  # n<2 path, clipped


def test_zero_samples_clips_raw_value() -> None:
    norm = RunningNormaliser(dim=1, clip=8.0)
    assert norm.normalise(np.array([100.0])) == pytest.approx([8.0])
    assert norm.normalise(np.array([-100.0])) == pytest.approx([-8.0])


def test_large_magnitude_sequence_produces_bounded_output() -> None:
    """Direct regression test for the reported bug: a synthetic sequence
    growing from O(10) to O(1e9) (mimicking unbounded cash accumulation
    under G1) must never produce an out-of-bounds, NaN, or Inf normalised
    value, at any point in the sequence -- including early on, before the
    running statistics have stabilised."""
    norm = RunningNormaliser(dim=1, clip=10.0)
    magnitudes = [10.0 * (1.5**i) for i in range(60)]  # 10 -> ~1.9e10
    for m in magnitudes:
        raw = np.array([m])
        norm.update(raw)
        out = norm.normalise(raw)
        assert np.all(np.isfinite(out))
        assert np.all(out >= -10.0) and np.all(out <= 10.0)


def test_dim_mismatch_raises() -> None:
    norm = RunningNormaliser(dim=3)
    with pytest.raises(ValueError):
        norm.update(np.array([1.0, 2.0]))
    with pytest.raises(ValueError):
        norm.normalise(np.array([1.0, 2.0]))


def test_default_clip_matches_documented_default() -> None:
    """Drift detector: default clip is a documented, deliberate choice
    (8.0 for the class default -- see the module docstring for why
    ManualPPOCollector overrides this per-purpose: 10.0 for observations
    matching SB3's own VecNormalize default, 5.0 for rewards matching the
    existing _RunningNormaliser convention used for G3)."""
    norm = RunningNormaliser(dim=1)
    assert norm.clip == 8.0
