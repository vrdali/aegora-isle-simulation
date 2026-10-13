"""Figure generation for the spec §8 analysis.

All figures are written as **300 DPI PNG** to ``analysis/outputs/figures/``
per spec §8.2. Matplotlib/seaborn only -- no interactive backends, so this
runs headless in CI or on a server.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # headless; must precede pyplot import

import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from analysis.data_loading import (  # noqa: E402
    GOVERNANCE_ORDER,
    KPI_COLUMNS,
    SCENARIO_ORDER,
    mean_trajectories,
)

DPI: int = 300  # spec §8.2

KPI_LABELS: dict[str, str] = {
    "gini_coefficient": "Gini coefficient",
    "hdi_proxy": "HDI proxy",
    "gdp_equivalent": "GDP-equivalent",
    "depletion_rate": "Depletion rate",
    "resource_stock": "Resource stock",
}

GOVERNANCE_LABELS: dict[str, str] = {
    "G1": "G1\nNo-Governance",
    "G2": "G2\nRule-Based",
    "G3": "G3\nLearning-Based",
}

PALETTE: dict[str, str] = {"G1": "#c44e52", "G2": "#4c72b0", "G3": "#55a868"}


def _save(fig: plt.Figure, path: Path) -> Path:
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def box_plots(runs, figures_dir: Path) -> list[Path]:
    """Spec §8.1: box plots per KPI, grouped by governance condition."""
    paths = []
    for kpi in KPI_COLUMNS:
        fig, ax = plt.subplots(figsize=(7, 5))
        data = runs.copy()
        data["governance"] = data["governance"].astype(str)
        sns.boxplot(data=data, x="governance", y=kpi, hue="governance",
                    order=list(GOVERNANCE_ORDER), palette=PALETTE, legend=False, ax=ax)
        sns.stripplot(data=data, x="governance", y=kpi, order=list(GOVERNANCE_ORDER),
                      color="black", alpha=0.25, size=3, ax=ax)
        ax.set_xticks(range(len(GOVERNANCE_ORDER)))
        ax.set_xticklabels([GOVERNANCE_LABELS[g] for g in GOVERNANCE_ORDER])
        ax.set_xlabel("")
        ax.set_ylabel(KPI_LABELS.get(kpi, kpi))
        ax.set_title(f"{KPI_LABELS.get(kpi, kpi)} by governance condition (n=360)")
        paths.append(_save(fig, figures_dir / f"box_{kpi}.png"))
    return paths


def kde_plots(runs, figures_dir: Path) -> list[Path]:
    """Spec §8.1: KDE plots per KPI, grouped by governance condition."""
    paths = []
    for kpi in KPI_COLUMNS:
        fig, ax = plt.subplots(figsize=(7, 5))
        for gov in GOVERNANCE_ORDER:
            v = runs.loc[runs["governance"] == gov, kpi].dropna()
            if v.nunique() > 1:
                sns.kdeplot(v, ax=ax, label=gov, color=PALETTE[gov], fill=True, alpha=0.25)
        ax.set_xlabel(KPI_LABELS.get(kpi, kpi))
        ax.set_ylabel("Density")
        ax.set_title(f"Distribution of {KPI_LABELS.get(kpi, kpi)} by governance")
        ax.legend(title="Regime")
        paths.append(_save(fig, figures_dir / f"kde_{kpi}.png"))
    return paths


def survival_curves(fitters: dict[str, Any], figures_dir: Path) -> Path:
    """Spec §8.1: Kaplan-Meier survival curves, one per governance regime."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for gov, kmf in fitters.items():
        kmf.plot_survival_function(ax=ax, color=PALETTE.get(gov), ci_show=True)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Survival probability")
    ax.set_title("Kaplan-Meier survival by governance (spec Table 4 event, censored at 10,000)")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    return _save(fig, figures_dir / "survival_kaplan_meier.png")


def trajectory_grid(data_dir: Path, figures_dir: Path, kpi: str = "gini_coefficient") -> Path:
    """Spec §8.1: mean KPI trajectory per cell, all 12 cells.

    Where a cell's replications collapse early, its mean is drawn only over
    the timesteps that still have surviving runs -- the line simply stops,
    rather than being padded with fabricated values.
    """
    traj = mean_trajectories(data_dir, kpi=kpi)
    fig, axes = plt.subplots(3, 4, figsize=(16, 9), sharex=True, sharey=True)
    for i, gov in enumerate(GOVERNANCE_ORDER):
        for j, scen in enumerate(SCENARIO_ORDER):
            ax = axes[i][j]
            series = traj.get((gov, scen))
            if series is not None:
                ax.plot(series.index, series.to_numpy(), color=PALETTE[gov], lw=1.2)
            if i == 0:
                ax.set_title(f"Scenario {scen}")
            if j == 0:
                ax.set_ylabel(f"{gov}\n{KPI_LABELS.get(kpi, kpi)}")
            ax.grid(alpha=0.25)
    for ax in axes[-1]:
        ax.set_xlabel("Timestep")
    fig.suptitle(
        f"Mean {KPI_LABELS.get(kpi, kpi)} trajectory across replications — all 12 cells",
        fontsize=13,
    )
    fig.tight_layout()
    return _save(fig, figures_dir / f"trajectories_{kpi}.png")


def render_all(runs, fitters: dict[str, Any], data_dir: Path, figures_dir: Path) -> dict[str, Any]:
    """Renders every spec §8.1 figure. Returns a name -> path map."""
    sns.set_theme(style="whitegrid")
    figures_dir = Path(figures_dir)
    out: dict[str, Any] = {
        "box": box_plots(runs, figures_dir),
        "kde": kde_plots(runs, figures_dir),
        "survival": survival_curves(fitters, figures_dir),
        "trajectories_gini": trajectory_grid(data_dir, figures_dir, "gini_coefficient"),
        "trajectories_stock": trajectory_grid(data_dir, figures_dir, "resource_stock"),
    }
    return out
