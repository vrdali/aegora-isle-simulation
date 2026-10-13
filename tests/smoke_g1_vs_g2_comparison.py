"""M2 Sprint 2.1/2.2 illustration — G1 vs G2 resource trajectory, all scenarios.

Not an acceptance-criterion script (that's smoke_g1_depletion.py / AC03).
This one visualises *why* the governance dispatcher matters: for each of
the four scenarios (A/B/C/D), same seed + forced max-extraction Extractors
+ identically-seeded random non-extractor actions — the only thing that
differs between the G1 and G2 runs is which GovernanceModule is wired in.
G2's extraction_cap (3.0 of MAX_EXTRACT_AMOUNT=5.0) should visibly slow
depletion relative to G1's uncapped extraction, but note: G2 still
collapses in the low-abundance scenarios (C/D) under this worst-case
forced-max-extraction stress test — see G2_PLAN.md and the M2_PLAN.md
risk register for why a static cap throttles but does not guarantee
sustainability. That's expected, not a bug.

Run from repo root::

    uv run python tests/smoke_g1_vs_g2_comparison.py
"""

from __future__ import annotations

import pathlib

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from aegora_isle.environment.aegora_env import AegoraEnv
from aegora_isle.governance.governance_module import GovernanceModule


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SCENARIOS_PATH = REPO_ROOT / "aegora_isle" / "experiments" / "scenarios.yaml"
CONFIGS_DIR = REPO_ROOT / "aegora_isle" / "governance" / "configs"
OUT_DIR = REPO_ROOT / "data" / "smoke"
OUT_PATH = OUT_DIR / "g1_vs_g2_all_scenarios.png"

SEED = 12345
ACTION_SEED = 7  # shared across both runs so non-extractor noise is identical
N_AGENTS = 20
MAX_TIMESTEPS = 3000

SCENARIO_LABELS = {
    "A": "A — high abundance, low events",
    "B": "B — high abundance, high events",
    "C": "C — low abundance, low events",
    "D": "D — low abundance, high events",
}


def _forced_extract_action() -> dict:
    return {
        "action_type": 0,
        "amount": np.array([1.0], dtype=np.float32),
        "target_zone": 0,
    }


def run(config_name: str, scenario: dict) -> tuple[list[int], list[float], bool, int]:
    governance = GovernanceModule(
        str(CONFIGS_DIR / config_name), np.random.default_rng(SEED)
    )
    env = AegoraEnv(
        n_agents=N_AGENTS,
        governance_module=governance,
        scenario_params=scenario,
        seed=SEED,
        max_timesteps=MAX_TIMESTEPS,
    )
    env.reset()
    for aid in env.agents:
        env.action_space(aid).seed(ACTION_SEED)

    stocks: list[float] = [env.resource_pool.stock]
    timesteps: list[int] = [0]
    t = 0
    while env.agents and t < MAX_TIMESTEPS:
        cycle_agents = list(env.agents)
        for _ in cycle_agents:
            if not env.agents:
                break
            aid = env.agent_selection
            if env.terminations[aid] or env.truncations[aid]:
                env.step(None)
            elif aid.startswith("extractor"):
                env.step(_forced_extract_action())
            else:
                env.step(env.action_space(aid).sample())
        t = env._timestep
        stocks.append(env.resource_pool.stock)
        timesteps.append(t)

    return timesteps, stocks, env.resource_pool.is_collapsed(), t


def main() -> int:
    with SCENARIOS_PATH.open() as f:
        scenarios = yaml.safe_load(f)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes_flat = axes.flatten()

    for ax, key in zip(axes_flat, ("A", "B", "C", "D")):
        sc = scenarios[key]
        t_g1, s_g1, collapsed_g1, tend_g1 = run("g1_ungoverned.yaml", sc)
        t_g2, s_g2, collapsed_g2, tend_g2 = run("g2_rule_based.yaml", sc)

        label = SCENARIO_LABELS[key]
        print(f"--- Scenario {label} ---")
        print(f"  G1: final stock = {s_g1[-1]:.1f}  collapsed={collapsed_g1}  t={tend_g1}")
        print(f"  G2: final stock = {s_g2[-1]:.1f}  collapsed={collapsed_g2}  t={tend_g2}")

        ax.plot(t_g1, s_g1, lw=1.5, color="#b3543d", label="G1 — ungoverned")
        ax.plot(t_g2, s_g2, lw=1.5, color="#2e7d4f", label="G2 — rule-based (cap=3.0)")
        ax.axhline(sc["critical_min"], color="grey", ls="--", lw=0.7,
                   label=f"critical_min={sc['critical_min']}")
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Resource stock")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        "G1 vs G2 across all four scenarios\n"
        "Same seed, forced max-extraction Extractors, identical non-extractor action stream",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT_PATH, dpi=150)
    plt.close(fig)
    print(f"\nWrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
