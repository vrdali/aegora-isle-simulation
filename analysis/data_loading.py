"""Dataset loading and derivation for the spec §8 statistical analysis.

Pure functions over the confirmatory dataset. No plotting, no Streamlit, no
writes to ``data/runs/`` -- everything here is read-only except the
spec Table 7 ``data/summaries/`` epoch aggregates, which are auto-generated
outputs.

Memory note: the dataset is 360 replications x up to 10,000 timesteps x 14
columns (~444 MB on disk, ~3.6 M rows). Nothing here materialises all of it
at once -- the run index and the trajectory means are both accumulated
cell-by-cell.

**MTSC event definition.** Spec Table 4 defines the collapse event as::

    stock < critical_min  OR  GDP = 0 for 3 consecutive steps
    (censored at 10,000 if no collapse)

Both clauses are implemented here. Note that the CSV's ``collapse_event``
column encodes a *different* concept -- the professor's subsistence
definition (see DECISIONS.md "Collapse definition"). The two disagree in
direction, and by explicit decision this module uses **spec Table 4** for
MTSC. See ``SUBSISTENCE_DISCREPANCY_NOTE``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

# Spec Table 4: censoring horizon for MTSC.
CENSOR_HORIZON: int = 10_000

# Spec Table 4, second collapse clause: GDP == 0 for this many consecutive steps.
GDP_ZERO_RUN_LENGTH: int = 3

GOVERNANCE_ORDER: tuple[str, ...] = ("G1", "G2", "G3")
SCENARIO_ORDER: tuple[str, ...] = ("A", "B", "C", "D")

# Spec §2.3: scenarios cross resource abundance with event frequency.
# Volatility here is the *event frequency* axis, which is what H3 manipulates.
SCENARIO_VOLATILITY: dict[str, str] = {"A": "low", "B": "high", "C": "low", "D": "high"}

# The five KPIs of spec Table 4. MTSC is derived (survival), not a CSV column,
# so it is handled separately; `resource_stock` is carried as a supplementary
# series because it is in the locked CSV schema and cheap to summarise.
KPI_COLUMNS: tuple[str, ...] = (
    "gini_coefficient",
    "hdi_proxy",
    "gdp_equivalent",
    "depletion_rate",
    "resource_stock",
)

SUBSISTENCE_DISCREPANCY_NOTE = """\
The delivered CSVs contain a `collapse_event` column that encodes the
professor's *subsistence* collapse definition (the bottom fraction of agents
falling below a subsistence share of initial mean wealth), not the spec
Table 4 MTSC definition used for the survival analysis in this report.

The two signals point in opposite directions. Under spec Table 4 (resource
pool below critical_min, or GDP zero for three consecutive steps),
No-Governance collapses in 70 of 120 replications (58.3%) against 5 of 120
for Rule-Based (4.2%) and 6 of 120 for Learning-Based (5.0%). Under the
subsistence definition the counts invert: No-Governance 20, Rule-Based 118,
Learning-Based 120.

Note that the governed regimes are not at exactly zero under spec Table 4.
Their handful of events come from the *second* clause -- GDP at zero for
three consecutive steps -- not from resource-pool death, which never occurs
under governance. Earlier summaries that reported "0%" for the governed
regimes had applied only the pool-death clause.

Both are real and mutually consistent: governance protects the *resource*
while concentrating wealth away from the poorest agents, which is the same
mechanism behind the H1 inversion. The survival analysis reported here uses
the specification's own MTSC definition. The subsistence column remains in
the delivered data and is noted here so the discrepancy is visible rather
than discovered.
"""


# --------------------------------------------------------------- discovery


def iter_run_files(data_dir: str | Path) -> Iterator[tuple[Path, Path]]:
    """Yields ``(csv_path, json_path)`` for every replication, sorted.

    Sorted so the run index is deterministic across machines.
    """
    runs_dir = Path(data_dir) / "runs"
    for csv_path in sorted(runs_dir.glob("*/*.csv")):
        json_path = csv_path.with_suffix(".json")
        if json_path.exists():
            yield csv_path, json_path


# ------------------------------------------------------- survival encoding


def _gdp_zero_streak_start(gdp_raw: pd.Series, run_length: int = GDP_ZERO_RUN_LENGTH) -> int | None:
    """First timestep index at which ``gdp_raw`` completes a zero-run.

    Spec Table 4's second collapse clause. Returns the 1-indexed timestep at
    which the ``run_length``-th consecutive zero occurs, or None.
    """
    zero = (gdp_raw.to_numpy() == 0.0).astype(int)
    if zero.sum() < run_length:
        return None
    streak = 0
    for i, z in enumerate(zero):
        streak = streak + 1 if z else 0
        if streak >= run_length:
            return i + 1  # timesteps are 1-indexed in the CSV
    return None


def encode_survival(
    df: pd.DataFrame, critical_min: float, censor_horizon: int = CENSOR_HORIZON
) -> tuple[float, int]:
    """Returns ``(duration, event_observed)`` per spec Table 4.

    Event fires at whichever comes first:
      * ``resource_stock < critical_min`` (the pool-death clause -- in
        practice the env terminates the episode at this point, so the run is
        short), or
      * ``gdp_raw == 0`` for :data:`GDP_ZERO_RUN_LENGTH` consecutive steps.

    A run reaching ``censor_horizon`` with neither condition is censored
    (``event_observed = 0``), which is what makes Kaplan-Meier the correct
    estimator here rather than a plain mean.
    """
    stock_breach = df.index[df["resource_stock"] < critical_min]
    t_stock = int(df.loc[stock_breach[0], "timestep"]) if len(stock_breach) else None
    t_gdp = _gdp_zero_streak_start(df["gdp_raw"])

    candidates = [t for t in (t_stock, t_gdp) if t is not None]
    if candidates:
        return float(min(candidates)), 1

    last_timestep = int(df["timestep"].iloc[-1])
    if last_timestep < censor_horizon:
        # Episode ended early without tripping either clause explicitly --
        # the environment terminated it, which is itself the pool-death
        # condition. Treat as an observed event at the point it stopped.
        return float(last_timestep), 1
    return float(censor_horizon), 0


# -------------------------------------------------------------- run index


def summarise_run(csv_path: Path, json_path: Path) -> dict[str, Any]:
    """One row of the run index: identifiers, per-KPI summaries, survival.

    KPI summaries use each run's **own** length. 65 of the 360 replications
    are early collapses that stop well before 10,000 steps; assuming a fixed
    length would silently mis-average them, and dropping them would delete
    precisely the events MTSC exists to measure.
    """
    meta = json.loads(json_path.read_text())
    df = pd.read_csv(csv_path)

    critical_min = float(meta["scenario_params"]["critical_min"])
    duration, event = encode_survival(df, critical_min)

    scenario = meta["scenario"]
    row: dict[str, Any] = {
        "run_id": meta["run_id"],
        "governance": meta["governance_condition"],
        "scenario": scenario,
        "volatility": SCENARIO_VOLATILITY[scenario],
        "replication": int(meta["run_id"].rsplit("_", 1)[1]),
        "random_seed": int(meta["random_seed"]),
        "n_agents": int(meta["n_agents"]),
        "n_timesteps": int(len(df)),
        "duration": duration,
        "event_observed": event,
        # No subsistence-collapse field here by design: `event_observed` above
        # is the spec Table 4 MTSC event and is the only collapse signal the
        # analysis uses. A parallel subsistence field previously sat alongside
        # it, unread by anything -- removed at the client's request so the
        # codebase cannot suggest an ambiguity that the analysis does not have.
        # The subsistence definition survives in the delivered CSVs'
        # `collapse_event` column and in the disclosure note above.
    }

    for col in KPI_COLUMNS:
        series = df[col].dropna()
        row[col] = float(series.mean()) if len(series) else np.nan
        row[f"{col}_final"] = float(series.iloc[-1]) if len(series) else np.nan
    return row


def load_run_index(data_dir: str | Path) -> pd.DataFrame:
    """One row per replication. Streams; never holds the full dataset."""
    rows = [summarise_run(c, j) for c, j in iter_run_files(data_dir)]
    if not rows:
        raise FileNotFoundError(f"no replications found under {Path(data_dir) / 'runs'}")
    df = pd.DataFrame(rows)
    df["governance"] = pd.Categorical(df["governance"], categories=GOVERNANCE_ORDER, ordered=True)
    df["scenario"] = pd.Categorical(df["scenario"], categories=SCENARIO_ORDER, ordered=True)
    return df.sort_values(["governance", "scenario", "replication"]).reset_index(drop=True)


# ---------------------------------------------------------- trajectories


def mean_trajectories(
    data_dir: str | Path, kpi: str = "gini_coefficient", max_steps: int = CENSOR_HORIZON
) -> dict[tuple[str, str], pd.Series]:
    """Mean KPI trajectory per (governance, scenario) cell across replications.

    Accumulates a running sum/count per timestep rather than concatenating
    360 frames. Replications that end early contribute only to the timesteps
    they actually reached, so the tail of a cell's mean is an average over
    the survivors -- which is the honest representation when 65 runs collapse
    early.
    """
    sums: dict[tuple[str, str], np.ndarray] = {}
    counts: dict[tuple[str, str], np.ndarray] = {}

    for csv_path, json_path in iter_run_files(data_dir):
        meta = json.loads(json_path.read_text())
        key = (meta["governance_condition"], meta["scenario"])
        series = pd.read_csv(csv_path, usecols=[kpi])[kpi].to_numpy(dtype=float)[:max_steps]
        if key not in sums:
            sums[key] = np.zeros(max_steps)
            counts[key] = np.zeros(max_steps)
        valid = ~np.isnan(series)
        idx = np.arange(len(series))[valid]
        sums[key][idx] += series[valid]
        counts[key][idx] += 1

    out: dict[tuple[str, str], pd.Series] = {}
    for key, total in sums.items():
        n = counts[key]
        with np.errstate(invalid="ignore"):
            mean = np.where(n > 0, total / np.maximum(n, 1), np.nan)
        out[key] = pd.Series(mean, index=np.arange(1, max_steps + 1), name=kpi)
    return out


# ------------------------------------------------------- epoch summaries


def write_epoch_summaries(
    data_dir: str | Path, out_dir: str | Path | None = None, epoch_length: int = 100
) -> Path:
    """Writes spec Table 7's ``data/summaries/`` epoch-level aggregates.

    Table 7 lists ``summaries/`` as "auto-generated: epoch-level aggregates
    for ANOVA input". The directory did not previously exist. One row per
    (run, epoch) with the mean of each KPI over that epoch.
    """
    data_dir = Path(data_dir)
    out_dir = Path(out_dir) if out_dir is not None else data_dir / "summaries"
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for csv_path, json_path in iter_run_files(data_dir):
        meta = json.loads(json_path.read_text())
        df = pd.read_csv(csv_path)
        df["epoch"] = ((df["timestep"] - 1) // epoch_length) + 1
        agg = df.groupby("epoch")[list(KPI_COLUMNS)].mean().reset_index()
        agg.insert(0, "run_id", meta["run_id"])
        agg.insert(1, "governance", meta["governance_condition"])
        agg.insert(2, "scenario", meta["scenario"])
        frames.append(agg)

    combined = pd.concat(frames, ignore_index=True)
    path = out_dir / "epoch_summaries.csv"
    combined.to_csv(path, index=False)
    return path
