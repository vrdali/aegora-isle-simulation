"""M1 Sprint 1.5 smoke test — AC03.

Per spec §10, AC03 requires monotonic resource depletion under **G1-D**
(no governance + low abundance + high event frequency) when agents act
randomly. Output: ``data/smoke/g1_d_depletion.png``.

Run from repo root::

    uv run python tests/smoke_g1_depletion.py
"""

from __future__ import annotations

import pathlib
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from aegora_isle.environment.aegora_env import AegoraEnv
from aegora_isle.governance.governance_module import GovernanceModule


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SCENARIOS_PATH = REPO_ROOT / "aegora_isle" / "experiments" / "scenarios.yaml"
G1_CONFIG_PATH = REPO_ROOT / "aegora_isle" / "governance" / "configs" / "g1_ungoverned.yaml"
OUT_DIR = REPO_ROOT / "data" / "smoke"
OUT_PATH = OUT_DIR / "g1_d_depletion.png"

SEED = 12345


def main() -> int:
    with SCENARIOS_PATH.open() as f:
        scenarios = yaml.safe_load(f)
    sc_d = scenarios["D"]

    governance = GovernanceModule(str(G1_CONFIG_PATH), np.random.default_rng(SEED))
    env = AegoraEnv(
        n_agents=20,
        governance_module=governance,  # G1, via the real dispatcher (Sprint 2.1)
        scenario_params=sc_d,
        seed=SEED,
        max_timesteps=1000,
    )
    env.reset()

    stocks: list[float] = [env.resource_pool.stock]
    timesteps: list[int] = [0]
    t = 0
    while env.agents and t < 1000:
        # One full agent_iter cycle = one env timestep.
        cycle_agents = list(env.agents)
        for _ in cycle_agents:
            if not env.agents:
                break
            aid = env.agent_selection
            if env.terminations[aid] or env.truncations[aid]:
                env.step(None)
            else:
                env.step(env.action_space(aid).sample())
        t = env._timestep
        stocks.append(env.resource_pool.stock)
        timesteps.append(t)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(timesteps, stocks, lw=1.5, color="#b3543d")
    ax.axhline(sc_d["critical_min"], color="grey", ls="--", lw=0.8,
               label=f"critical_min = {sc_d['critical_min']}")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Resource stock")
    ax.set_title(
        "G1-D smoke test (no governance, low abundance, high events)\n"
        "20 random-action agents — stock should slope downward"
    )
    ax.legend(loc="upper right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    plt.close(fig)

    initial = sc_d["initial_stock"]
    final = stocks[-1]
    pct_remaining = 100.0 * final / initial
    print(f"Wrote {OUT_PATH}")
    print(f"Initial stock:  {initial:.1f}")
    print(f"Final stock:    {final:.1f}  ({pct_remaining:.1f}% remaining)")
    print(f"Events fired:   {env.resource_pool.event_count}")
    print(f"Timesteps run:  {t}")
    print(f"Collapsed:      {env.resource_pool.is_collapsed()}")

    # AC03 directional check: final < initial. Strict monotonicity isn't
    # required (small regen bumps are fine) — overall trend must be down.
    if final >= initial * 0.95:
        print("WARNING: stock did not deplete substantially — check regen rate.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
