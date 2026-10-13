"""Generic Dict<->Box action-space codec, for bridging agents to SB3.

Stable-Baselines3's PPO/SAC policies require ``Box``/``Discrete``/
``MultiDiscrete``/``MultiBinary`` action spaces — never ``Dict``. Every
Aegora Isle agent's action space is a ``gymnasium.spaces.Dict`` (per spec
§2.1's literal encoding, locked in DECISIONS.md). This module flattens any
such Dict (made of ``Discrete`` and ``Box`` sub-spaces) into a single
``Box`` for SB3 to train against, and converts samples back and forth.

Deliberately **not** using ``gymnasium.spaces.utils.flatten``/``unflatten``:
verified against the installed gymnasium 1.3.0 source, ``Discrete``'s
``unflatten`` takes the *first nonzero index* (``np.nonzero(x)[0][0]``),
not the largest — silently wrong when fed a continuous policy's raw
(non-one-hot) output, which is exactly what a trained PPO/SAC actor
produces. :func:`decode_action` uses ``argmax`` instead, which is correct
for that case.

Zero imports from ``governance/``, ``environment/``, or ``kpi/`` —
``agents/`` is the most primitive layer in the dependency graph (see
``MERMAID.md``); nothing here may depend on anything above it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces


# Bounds for the one-hot "logit" dims contributed by each Discrete(n)
# sub-space. Deliberately wide (not [0, 1]) so a continuous Gaussian
# policy head isn't artificially compressed — see DECISIONS.md
# "action_codec logit bounds".
LOGIT_LOW: float = -5.0
LOGIT_HIGH: float = 5.0

_SUPPORTED_SUBSPACES = (spaces.Discrete, spaces.Box)


def _check_supported(sub: spaces.Space, key: str) -> None:
    if not isinstance(sub, _SUPPORTED_SUBSPACES):
        raise TypeError(
            f"action_codec only supports Discrete/Box sub-spaces, "
            f"got {type(sub).__name__} for key {key!r}"
        )


def flatten_action_space(dict_space: spaces.Dict) -> spaces.Box:
    """Flatten a Dict of Discrete/Box sub-spaces into one Box.

    ``Discrete(n)`` contributes ``n`` "logit" dims (bounds
    ``[LOGIT_LOW, LOGIT_HIGH]``). ``Box(shape=(k,))`` contributes ``k``
    dims with its own real bounds, copied through unchanged.
    """
    lows: list[float] = []
    highs: list[float] = []
    for key, sub in dict_space.spaces.items():
        _check_supported(sub, key)
        if isinstance(sub, spaces.Discrete):
            lows.extend([LOGIT_LOW] * int(sub.n))
            highs.extend([LOGIT_HIGH] * int(sub.n))
        else:
            lows.extend(np.asarray(sub.low, dtype=np.float64).flatten().tolist())
            highs.extend(np.asarray(sub.high, dtype=np.float64).flatten().tolist())
    return spaces.Box(
        low=np.array(lows, dtype=np.float32),
        high=np.array(highs, dtype=np.float32),
        dtype=np.float32,
    )


def encode_action(dict_space: spaces.Dict, action: dict[str, Any]) -> np.ndarray:
    """Convert a real Dict action sample into its flat representation.

    ``Discrete`` fields become one-hot-ish: ``LOGIT_HIGH`` at the chosen
    index, ``LOGIT_LOW`` elsewhere (so round-tripping through
    :func:`decode_action`'s ``argmax`` recovers the same index). ``Box``
    fields are copied through directly.
    """
    parts: list[np.ndarray] = []
    for key, sub in dict_space.spaces.items():
        _check_supported(sub, key)
        value = action[key]
        if isinstance(sub, spaces.Discrete):
            one_hot = np.full(int(sub.n), LOGIT_LOW, dtype=np.float32)
            idx = int(value) - int(sub.start)
            one_hot[idx] = LOGIT_HIGH
            parts.append(one_hot)
        else:
            parts.append(np.asarray(value, dtype=np.float32).flatten())
    return np.concatenate(parts)


def decode_action(dict_space: spaces.Dict, flat: np.ndarray) -> dict[str, Any]:
    """Split a flat Box sample back into a real Dict action.

    ``Discrete`` slots decoded via ``argmax`` (not gymnasium's
    ``unflatten`` — see module docstring). ``Box`` slots clipped to the
    sub-space's declared bounds.
    """
    flat = np.asarray(flat, dtype=np.float32).flatten()
    out: dict[str, Any] = {}
    pos = 0
    for key, sub in dict_space.spaces.items():
        _check_supported(sub, key)
        if isinstance(sub, spaces.Discrete):
            n = int(sub.n)
            idx = int(np.argmax(flat[pos : pos + n]))
            out[key] = int(sub.start) + idx
            pos += n
        else:
            size = int(np.prod(sub.shape))
            chunk = flat[pos : pos + size].reshape(sub.shape)
            out[key] = np.clip(chunk, sub.low, sub.high).astype(sub.dtype)
            pos += size
    return out
