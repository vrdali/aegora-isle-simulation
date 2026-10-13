"""Exploratory time-series comparison across all 12 cells, long enough for
G3's SAC to actually train.

Not an acceptance-criterion script. Builds on smoke_compare_all_cells.py's
mechanism (ManualPPOCollector so PPO trains for real; GovernanceModule so
G3's AdaptiveRLGovernor trains for real too) but runs 5,000 cycles per
cell instead of 300 — long enough to cross G3's SAC learning_starts=1000
(confirmed empirically: ~4ms/cycle, so 5,000 cycles ≈ 20s per cell, ~4min
for all 12) and to reach the spec's 50-epoch minimum before any
convergence check would apply.

Still a short pilot relative to M3's real 10,000-cycle x 30-replication
runs, and still using UNTUNED G2/G3 hyperparameters (see DECISIONS.md /
G2_PLAN.md / G3_PLAN.md — final values land in M3 Sprint 3.3). Treat every
chart here as "does the mechanism behave sensibly, and what does a
genuinely-training G3 look like," not a validated finding.

Produces:
- data/smoke/timeseries/{resource_stock,gini,depletion,gdp}_by_scenario.png
  -- one 2x2 (scenario) grid per KPI, 3 lines (G1/G2/G3) per subplot.
- data/smoke/timeseries/g3_decisions_by_scenario.png
  -- G3's own current_cap/current_redist_rate/current_sanction_probability
  over time, per scenario -- shows the exact cycle SAC crosses
  learning_starts and starts producing non-random decisions.
- data/smoke/timeseries/*.csv -- full per-cycle CSVs, one per cell.

Run from repo root::

    uv run python tests/smoke_compare_timeseries.py
"""

from __future__ import annotations

import pathlib
import time

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
OUT_DIR = REPO_ROOT / "data" / "smoke" / "timeseries"

SEED = 42
ACTION_SEED = 7
N_AGENTS = 20
MAX_TIMESTEPS = 5000  # crosses SAC learning_starts=1000; ~20s/cell measured
SMALL_PPO_KWARGS = {"policy_kwargs": {"net_arch": [16, 16]}}

REGIME_CONFIGS = {
    "G1": "g1_ungoverned.yaml",
    "G2": "g2_rule_based.yaml",
    "G3": "g3_adaptive_rl.yaml",
}
REGIME_COLORS = {"G1": "#b3543d", "G2": "#2e7d4f", "G3": "#3d6cb3"}
SCENARIO_ORDER = ["A", "B", "C", "D"]
SCENARIO_LABELS = {
    "A": "A — high abundance, low events",
    "B": "B — high abundance, high events",
    "C": "C — low abundance, low events",
    "D": "D — low abundance, high events",
}


def run_cell(regime: str, scenario_key: str, scenario_params: dict) -> tuple[pathlib.Path, pd.DataFrame | None]:
    """Runs one cell. Returns the written CSV path, plus a per-cycle
    DataFrame of G3's own decision variables (None for G1/G2)."""
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

    g3_decisions: list[dict] = []
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
            if regime == "G3":
                sd = env.governance_module.state_dict()
                g3_decisions.append(
                    {
                        "timestep": env._timestep,
                        "current_cap": sd["current_cap"],
                        "current_redist_rate": sd["current_redist_rate"],
                        "current_sanction_probability": sd["current_sanction_probability"],
                        "sac_trained": sd["step_count"] > 0
                        and env.governance_module._handler.model.num_timesteps >= 1000,
                    }
                )

    out_dir = OUT_DIR / regime
    csv_path = logger.write_csv(out_dir / f"{regime}_{scenario_key}.csv")
    logger.write_json_state(
        csv_path,
        governance_params=gm.state_dict(),
        scenario_params=scenario_params,
        agent_type_distribution={"extractor": 0.25, "producer": 0.25, "trader": 0.25, "governor": 0.25},
        rl_hyperparams=SMALL_PPO_KWARGS,
    )
    decisions_df = pd.DataFrame(g3_decisions) if g3_decisions else None
    return csv_path, decisions_df


def _plot_kpi_by_scenario(
    all_data: dict[tuple[str, str], pd.DataFrame],
    column: str,
    title: str,
    ylabel: str,
    out_path: pathlib.Path,
    rolling: int | None = None,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, scenario in zip(axes.flatten(), SCENARIO_ORDER):
        for regime in ("G1", "G2", "G3"):
            df = all_data.get((regime, scenario))
            if df is None:
                continue
            series = df[column]
            if rolling:
                series = series.rolling(rolling, min_periods=1).mean()
            ax.plot(df["timestep"], series, lw=1.2, color=REGIME_COLORS[regime], label=regime)
        ax.set_title(SCENARIO_LABELS[scenario], fontsize=10)
        ax.set_xlabel("Timestep")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    axes.flatten()[0].legend(loc="best", fontsize=9)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path.relative_to(REPO_ROOT)}")


def _plot_g3_decisions(
    g3_decisions: dict[str, pd.DataFrame], out_path: pathlib.Path
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, scenario in zip(axes.flatten(), SCENARIO_ORDER):
        df = g3_decisions.get(scenario)
        if df is None or df.empty:
            ax.set_title(f"{SCENARIO_LABELS[scenario]} (no data)", fontsize=10)
            continue
        ax.plot(df["timestep"], df["current_cap"], lw=1.2, label="extraction cap (raw units)", color="#b3543d")
        ax.plot(
            df["timestep"],
            df["current_redist_rate"] * 10,
            lw=1.2,
            label="redistribution rate x10 (for scale)",
            color="#2e7d4f",
        )
        ax.plot(
            df["timestep"],
            df["current_sanction_probability"],
            lw=1.2,
            label="sanction probability",
            color="#3d6cb3",
        )
        trained_mask = df["sac_trained"]
        if trained_mask.any():
            first_trained_t = df.loc[trained_mask, "timestep"].iloc[0]
            ax.axvline(
                first_trained_t, color="black", ls="--", lw=1.0,
                label=f"SAC starts training (t={int(first_trained_t)})",
            )
        ax.set_title(SCENARIO_LABELS[scenario], fontsize=10)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Decision value")
        ax.grid(alpha=0.25)
    axes.flatten()[0].legend(loc="best", fontsize=8)
    fig.suptitle(
        "G3's own Governor decisions over time — before/after SAC starts training\n"
        "(vertical dashed line = first cycle past learning_starts=1000)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path.relative_to(REPO_ROOT)}")


def main() -> int:
    with SCENARIOS_PATH.open() as f:
        scenarios = yaml.safe_load(f)

    all_data: dict[tuple[str, str], pd.DataFrame] = {}
    g3_decisions: dict[str, pd.DataFrame] = {}

    print(f"Running 12 cells at {MAX_TIMESTEPS} cycles each (long enough for G3's SAC to train) ...")
    t0 = time.perf_counter()
    for scenario_key in SCENARIO_ORDER:
        scenario_params = scenarios[scenario_key]
        for regime in REGIME_CONFIGS:
            cell_start = time.perf_counter()
            csv_path, decisions_df = run_cell(regime, scenario_key, scenario_params)
            elapsed = time.perf_counter() - cell_start
            df = pd.read_csv(csv_path)
            all_data[(regime, scenario_key)] = df
            if decisions_df is not None:
                g3_decisions[scenario_key] = decisions_df
            final_stock = df["resource_stock"].iloc[-1]
            collapsed = bool(df["collapse_event"].any())
            print(
                f"  {regime}-{scenario_key}: {elapsed:.1f}s, "
                f"{len(df)} cycles, final_stock={final_stock:.1f}, collapsed={collapsed}"
            )
    total_elapsed = time.perf_counter() - t0
    print(f"\nAll 12 cells done in {total_elapsed:.1f}s total.\n")

    _plot_kpi_by_scenario(
        all_data, "resource_stock",
        "Resource stock over time — 3 regimes x 4 scenarios\n(how much raw resource is left in the lagoon, per cycle)",
        "Resource stock",
        OUT_DIR / "resource_stock_by_scenario.png",
    )
    _plot_kpi_by_scenario(
        all_data, "gini_raw",
        "Wealth inequality (Gini) over time — 3 regimes x 4 scenarios\n(instantaneous, not the 50-step rolling CSV column)",
        "Gini coefficient",
        OUT_DIR / "gini_by_scenario.png",
    )
    _plot_kpi_by_scenario(
        all_data, "depletion_rate",
        "Depletion rate over time — 3 regimes x 4 scenarios\n(50-step rolling mean; positive = depleting, negative = regenerating)",
        "Depletion rate (%)",
        OUT_DIR / "depletion_rate_by_scenario.png",
    )
    _plot_kpi_by_scenario(
        all_data, "gdp_raw",
        "GDP-equivalent (raw) over time — 3 regimes x 4 scenarios\n10-cycle rolling mean for readability (per-cycle values are spiky)",
        "GDP-raw (rolling mean)",
        OUT_DIR / "gdp_by_scenario.png",
        rolling=10,
    )
    _plot_g3_decisions(g3_decisions, OUT_DIR / "g3_decisions_by_scenario.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
