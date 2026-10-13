"""Running observation/reward normalisation for ManualPPOCollector.

Fixes a confirmed gap surfaced by the Milestone 3 Sprint 3.3 pilot run: at
real scale (50 agents, full 10,000-step horizon), Extractor/Producer/Trader's
raw, unnormalised observations (cash/inventory can grow unboundedly over a
long run, especially under G1 where no governance caps exist) fed directly
into PPO's policy/value network produced a genuine numerical blow-up
(logged loss variance in the trillions). ``ManualPPOCollector`` (unlike
SB3's own ``.learn()``, which normally gets this via ``VecNormalize``) had
no normalisation layer anywhere. See DECISIONS.md "PPO observation/reward
normalisation -- confirmed gap, fixed" for the full root-cause writeup.

``RunningNormaliser`` is a per-dimension Welford online mean/std with
clipping -- a vector generalisation of
:class:`aegora_isle.governance.adaptive_rl._RunningNormaliser` (identical
algorithm, identical ``n < 2`` cold-start handling). Deliberately **not**
shared/imported between the two modules: CLAUDE.md's single-entry-point
rule forbids ``experiments/`` from importing ``governance/adaptive_rl.py``,
and G3's SAC training is explicitly out of scope for this fix (already
correct, already tested).
"""

from __future__ import annotations

import numpy as np


class RunningNormaliser:
    """Per-dimension Welford online mean/std, with clipping.

    Used for both per-dimension observation vectors (``dim = OBS_DIM``)
    and scalar rewards (``dim = 1``). No ``numpy.random.Generator``
    needed -- deterministic bookkeeping, a pure function of the
    (deterministic, seed-driven) sequence of vectors fed to
    :meth:`update`.

    Args:
        dim: Number of dimensions to track independently.
        clip: Clip bound for normalised output, applied per-dimension.
    """

    def __init__(self, dim: int, clip: float = 8.0) -> None:
        self.dim = int(dim)
        self.mean = np.zeros(self.dim, dtype=np.float64)
        self._m2 = np.zeros(self.dim, dtype=np.float64)
        self.n = 0
        self.clip = float(clip)

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64).reshape(self.dim)
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self._m2 += delta * (x - self.mean)

    def normalise(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).reshape(self.dim)
        if self.n < 2:
            return np.clip(x, -self.clip, self.clip)
        variance = self._m2 / (self.n - 1)
        std = np.maximum(1e-6, np.sqrt(variance))
        return np.clip((x - self.mean) / std, -self.clip, self.clip)
