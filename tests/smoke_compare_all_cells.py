"""Exploratory comparison across all 12 cells (3 governance x 4 scenarios).

Not an acceptance-criterion script (that's smoke_m2_signoff.py / AC04).
This one generates one short-pilot CSV per (governance, scenario) cell —
using ManualPPOCollector (Sprint 2.5) so PPO/SAC actually train during
the run, not just random actions — then prints a mean-KPI comparison
table and saves a grouped-bar-chart PNG.

Same seed across all 12 cells, identically-seeded action streams within
each governance regime, so any difference you see is attributable to
governance + scenario, not run-to-run action-sampling noise.

This is a SHORT pilot (a few hundred cycles), not the doctoral-grade
360-run x 30-replication comparison — that's M3's `run_experiment.py`,
not built yet. Treat this as "does the mechanism behave directionally
sensibly," not as a statistically validated finding.

Run from repo root::

    uv run python tests/smoke_compare_all_cells.py
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from aegora_isle.environment.aegora_env import AegoraEnv
from aegora_isle.experiments.train_agents import ManualPPOCollector
from aegora_isle.governance.governance_module import GovernanceModule
from aegora_isle.kpi.kpi_logger import KPILogger


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
CONFIGS_DIR = REPO_ROOT / "aegora_isle" / "governance" / "configs"
SCENARIOS_PATH = REPO_ROOT / "aegora_isle" / "experiments" / "scenarios.yaml"
OUT_DIR = REPO_ROOT / "data" / "smoke" / "all_cells"

SEED = 42
ACTION_SEED = 7
N_AGENTS = 20
MAX_TIMESTEPS = 300
SMALL_PPO_KWARGS = {"policy_kwargs": {"net_arch": [16, 16]}}

REGIME_CONFIGS = {
    "G1": "g1_ungoverned.yaml",
    "G2": "g2_rule_based.yaml",
    "G3": "g3_adaptive_rl.yaml",
}


def run_cell(regime: str, scenario_key: str, scenario_params: dict) -> pathlib.Path:
    gm = GovernanceModule(str(CONFIGS_DIR / REGIME_CONFIGS[regime]), np.random.default_rng(0))
    env = AegoraEnv(
        seed=SEED,
        n_agents=N_AGENTS,
        governance_module=gm,
        scenario_params=scenario_params,
        max_timesteps=MAX_TIMESTEPS,
    )
    env.reset()
    for aid in env.agents:
        env.action_space(aid).seed(ACTION_SEED)

    logger = KPILogger(env, run_id=f"{regime}_{scenario_key}_001", scenario=scenario_key)
    collector = ManualPPOCollector(env, n_steps=32, ppo_kwargs=dict(SMALL_PPO_KWARGS))

    last_timestep = 0
    while env.agents and env._timestep < MAX_TIMESTEPS:
        agent_id = env.agent_selection
        kind = agent_id.split("_")[0]
        obs, reward, terminated, truncated, _info = env.last()
        if kind in collector.models:
            collector._consume_pending_reward(kind, agent_id, obs, reward)
        if terminated or truncated:
            env.step(None)
        elif kind in collector.models:
            env.step(collector._act_and_record(kind, agent_id, obs))
        else:
            env.step(env.action_space(agent_id).sample())
        if env._timestep != last_timestep:
            logger.log_step()
            last_timestep = env._timestep

    out_dir = OUT_DIR / regime
    csv_path = logger.write_csv(out_dir / f"{regime}_{scenario_key}.csv")
    logger.write_json_state(
        csv_path,
        governance_params=gm.state_dict(),
        scenario_params=scenario_params,
        agent_type_distribution={"extractor": 0.25, "producer": 0.25, "trader": 0.25, "governor": 0.25},
        rl_hyperparams=SMALL_PPO_KWARGS,
    )
    return csv_path


def main() -> int:
    with SCENARIOS_PATH.open() as f:
        scenarios = yaml.safe_load(f)

    print("Running 12 cells (3 regimes x 4 scenarios) ...")
    for scenario_key, scenario_params in scenarios.items():
        for regime in REGIME_CONFIGS:
            csv_path = run_cell(regime, scenario_key, scenario_params)
            print(f"  {regime}-{scenario_key}: wrote {csv_path.relative_to(REPO_ROOT)}")

    dfs = [pd.read_csv(f) for f in OUT_DIR.glob("**/*.csv")]
    combined = pd.concat(dfs)

    summary = combined.groupby(["governance_condition", "scenario"])[
        ["gini_coefficient", "depletion_rate", "resource_stock"]
    ].mean()
    print("\nMean KPIs per cell:")
    print(summary)

    summary_path = OUT_DIR / "summary.csv"
    summary.to_csv(summary_path)
    print(f"\nWrote {summary_path.relative_to(REPO_ROOT)}")

    _plot(combined, OUT_DIR / "comparison.png")
    print(f"Wrote {(OUT_DIR / 'comparison.png').relative_to(REPO_ROOT)}")
    return 0


def _plot(combined: pd.DataFrame, out_path: pathlib.Path) -> None:
    kpis = ["gini_coefficient", "depletion_rate", "resource_stock"]
    scenario_order = ["A", "B", "C", "D"]
    regime_order = ["G1", "G2", "G3"]
    colors = {"G1": "#b3543d", "G2": "#2e7d4f", "G3": "#3d6cb3"}

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    means = combined.groupby(["scenario", "governance_condition"])[kpis].mean()

    width = 0.25
    x = np.arange(len(scenario_order))
    for ax, kpi in zip(axes, kpis):
        for i, regime in enumerate(regime_order):
            values = [means.loc[(s, regime), kpi] if (s, regime) in means.index else 0.0 for s in scenario_order]
            ax.bar(x + (i - 1) * width, values, width, label=regime, color=colors[regime])
        ax.set_xticks(x)
        ax.set_xticklabels(scenario_order)
        ax.set_xlabel("Scenario")
        ax.set_title(kpi)
        ax.grid(alpha=0.25, axis="y")
    axes[0].legend(title="Governance")
    fig.suptitle(
        "Mean KPIs per cell — 3 regimes x 4 scenarios (short pilot, not a validated finding)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
