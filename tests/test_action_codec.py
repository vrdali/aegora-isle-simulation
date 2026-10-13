"""Round-trip tests for aegora_isle.agents.action_codec.

Covers M2 Sprint 2.5: flatten_action_space/encode_action/decode_action
must round-trip correctly for all four real agent action spaces, and
decode_action must use argmax (not gymnasium's unflatten, which silently
breaks on non-one-hot continuous policy output — see action_codec.py's
module docstring for the sourced reasoning).
"""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium import spaces

from aegora_isle.agents.action_codec import decode_action, encode_action, flatten_action_space
from aegora_isle.agents.extractor import Extractor
from aegora_isle.agents.governor import Governor
from aegora_isle.agents.producer import Producer
from aegora_isle.agents.trader import Trader


AGENT_CLASSES = [Extractor, Producer, Trader, Governor]


@pytest.mark.parametrize("cls", AGENT_CLASSES)
def test_round_trip_real_samples(cls) -> None:
    rng = np.random.default_rng(0)
    agent = cls(agent_id="x", rng=rng)
    space = agent.action_space
    space.seed(0)
    for _ in range(50):
        action = space.sample()
        flat = encode_action(space, action)
        decoded = decode_action(space, flat)
        for key, sub in space.spaces.items():
            if isinstance(sub, spaces.Discrete):
                assert decoded[key] == action[key]
            else:
                assert np.allclose(decoded[key], action[key], atol=1e-4)


@pytest.mark.parametrize("cls", AGENT_CLASSES)
def test_flattened_space_matches_encoded_sample(cls) -> None:
    rng = np.random.default_rng(0)
    agent = cls(agent_id="x", rng=rng)
    space = agent.action_space
    flat_space = flatten_action_space(space)
    space.seed(1)
    action = space.sample()
    flat = encode_action(space, action)
    assert flat.shape == flat_space.shape
    assert flat_space.contains(flat)


@pytest.mark.parametrize("cls", AGENT_CLASSES)
def test_flattened_dim_equals_sum_of_subspace_dims(cls) -> None:
    rng = np.random.default_rng(0)
    agent = cls(agent_id="x", rng=rng)
    space = agent.action_space
    flat_space = flatten_action_space(space)
    expected_dim = sum(
        int(sub.n) if isinstance(sub, spaces.Discrete) else int(np.prod(sub.shape))
        for sub in space.spaces.values()
    )
    assert flat_space.shape == (expected_dim,)


def test_decode_uses_argmax_not_first_nonzero() -> None:
    """The whole point of not using gymnasium's unflatten: must pick the
    LARGEST logit, not the first nonzero one."""
    space = spaces.Dict({"action_type": spaces.Discrete(3)})
    flat = np.array([0.1, 0.05, 0.3], dtype=np.float32)
    decoded = decode_action(space, flat)
    assert decoded["action_type"] == 2


def test_decode_handles_negative_logits() -> None:
    space = spaces.Dict({"action_type": spaces.Discrete(3)})
    flat = np.array([-1.0, -0.2, -3.0], dtype=np.float32)
    decoded = decode_action(space, flat)
    assert decoded["action_type"] == 1


def test_decode_clips_box_to_bounds() -> None:
    space = spaces.Dict({"amount": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)})
    flat = np.array([5.3], dtype=np.float32)
    decoded = decode_action(space, flat)
    assert decoded["amount"][0] == pytest.approx(1.0)

    flat_low = np.array([-2.0], dtype=np.float32)
    decoded_low = decode_action(space, flat_low)
    assert decoded_low["amount"][0] == pytest.approx(0.0)


def test_flatten_unsupported_subspace_raises() -> None:
    space = spaces.Dict({"bad": spaces.MultiBinary(3)})
    with pytest.raises(TypeError):
        flatten_action_space(space)


def test_encode_unsupported_subspace_raises() -> None:
    space = spaces.Dict({"bad": spaces.MultiBinary(3)})
    with pytest.raises(TypeError):
        encode_action(space, {"bad": np.array([1, 0, 1])})


def test_multi_field_governor_space_round_trips_independently() -> None:
    """Governor's three fields (Discrete(4), Box(1,), Discrete(500)) must
    decode independently without cross-talk between slices."""
    rng = np.random.default_rng(0)
    gov = Governor(agent_id="g", rng=rng)
    space = gov.action_space
    action = {
        "action_type": 2,
        "value": np.array([0.73], dtype=np.float32),
        "target_agent_idx": 417,
    }
    flat = encode_action(space, action)
    decoded = decode_action(space, flat)
    assert decoded["action_type"] == 2
    assert decoded["value"][0] == pytest.approx(0.73, abs=1e-4)
    assert decoded["target_agent_idx"] == 417
