"""Data access for the Streamlit dashboard (spec §7).

Deliberately free of any Streamlit import so every function here is
pytest-testable without a browser or a running server. ``app.py`` wraps
these in ``st.cache_data`` -- they are pure functions of a path, which is
exactly what that decorator needs.

Read-only: nothing in this module writes to ``data/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from aegora_isle.kpi.kpi_logger import ROLLING_WINDOW

# The five KPIs of spec Table 4 as they appear in the locked CSV schema.
# MTSC is a derived survival quantity, not a column, so the dashboard shows
# `resource_stock` in its place -- it is what makes collapse visible.
KPI_COLUMNS: tuple[str, ...] = (
    "gini_coefficient",
    "hdi_proxy",
    "gdp_equivalent",
    "depletion_rate",
    "resource_stock",
)

KPI_LABELS: dict[str, str] = {
    "gini_coefficient": "Gini coefficient",
    "hdi_proxy": "HDI proxy",
    "gdp_equivalent": "GDP-equivalent",
    "depletion_rate": "Depletion rate",
    "resource_stock": "Resource stock",
}

GOVERNANCE_LABELS: dict[str, str] = {
    "G1": "G1 — No-Governance",
    "G2": "G2 — Rule-Based",
    "G3": "G3 — Learning-Based",
}


@dataclass(frozen=True)
class RunMeta:
    """One discovered replication."""

    run_id: str
    governance: str
    scenario: str
    replication: int
    csv_path: Path
    json_path: Path


@dataclass(frozen=True)
class ProgressInfo:
    """Replication-level progress for the Experiment Progress view.

    Deliberately replication-level, not within-run: ``KPILogger`` writes each
    replication's CSV once, at the end, so there is no partial file to read
    mid-run. Reporting a fake intra-run curve would be worse than reporting
    honest completion counts. See DECISIONS.md.
    """

    completed: int
    total: int

    @property
    def fraction(self) -> float:
        return 0.0 if self.total <= 0 else min(1.0, self.completed / self.total)


# --------------------------------------------------------------- discovery


def parse_run_id(run_id: str) -> tuple[str, str, int]:
    """``"G1_A_007"`` -> ``("G1", "A", 7)``.

    Mirrors the ``{governance}_{scenario}_{replication:03d}`` format that
    ``run_experiment.py`` produces; parsed rather than re-derived so the two
    cannot drift apart.
    """
    governance, scenario, rep = run_id.split("_")
    return governance, scenario, int(rep)


def discover_runs(data_dir: str | Path) -> list[RunMeta]:
    """Every replication under ``data_dir/runs/``, sorted deterministically.

    A CSV without its JSON sidecar is skipped -- that pairing is what makes a
    replication reproducible, and a half-written run should not appear
    selectable.
    """
    runs_dir = Path(data_dir) / "runs"
    if not runs_dir.exists():
        return []
    out: list[RunMeta] = []
    for csv_path in sorted(runs_dir.glob("*/*.csv")):
        json_path = csv_path.with_suffix(".json")
        if not json_path.exists():
            continue
        try:
            governance, scenario, rep = parse_run_id(csv_path.stem)
        except ValueError:
            continue
        out.append(RunMeta(csv_path.stem, governance, scenario, rep, csv_path, json_path))
    return out


def runs_dataframe(data_dir: str | Path) -> pd.DataFrame:
    """Discovered runs as a table, with ``final_status`` joined where known.

    The status comes from ``attempts_summary.csv`` when the batch has
    finished writing it; mid-run that file does not exist yet, so status is
    left blank rather than fabricated.
    """
    runs = discover_runs(data_dir)
    if not runs:
        return pd.DataFrame(columns=["run_id", "governance", "scenario", "replication", "final_status"])
    df = pd.DataFrame([{
        "run_id": r.run_id, "governance": r.governance,
        "scenario": r.scenario, "replication": r.replication,
    } for r in runs])

    attempts = Path(data_dir) / "attempts_summary.csv"
    if attempts.exists():
        summary = pd.read_csv(attempts)[["run_id", "final_status", "attempts_used", "seed"]]
        df = df.merge(summary, on="run_id", how="left")
    else:
        df["final_status"] = ""
    return df.sort_values(["governance", "scenario", "replication"]).reset_index(drop=True)


# ------------------------------------------------------------------ loading


def load_run_csv(csv_path: str | Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def load_run_json(json_path: str | Path) -> dict[str, Any]:
    import json

    return json.loads(Path(json_path).read_text())


def timestep_stride(df: pd.DataFrame) -> int:
    """Timesteps between consecutive rows -- 1 for a full-resolution run.

    A downsampled distribution copy (see ``scripts/package_dataset.py``)
    keeps every Nth timestep, so one row no longer means one step.
    """
    if "timestep" not in df.columns or len(df) < 2:
        return 1
    delta = df["timestep"].diff().median()
    return max(1, int(delta)) if pd.notna(delta) else 1


def add_rolling_kpis(
    df: pd.DataFrame, window: int = ROLLING_WINDOW, columns: tuple[str, ...] = KPI_COLUMNS
) -> pd.DataFrame:
    """Adds ``{kpi}_rolling`` columns -- spec §7.1's "50-step rolling averages".

    ``window`` is in **timesteps, not rows**, and is converted to a row count
    using the frame's own stride. At full resolution the two are identical.
    On a downsampled copy they are not: a 50-row window over stride-25 data
    would average 1,250 timesteps and quietly over-smooth every panel by 25x,
    which is not what §7.1 asks for.

    Applied uniformly to all five KPIs for visual consistency. Note that
    ``gini_coefficient`` and ``depletion_rate`` are *already* rolling-averaged
    in the CSV itself (``kpi_logger.write_csv``); smoothing them again is
    harmless for display and keeps every panel on the same footing. This only
    affects in-memory display data -- the locked CSV schema is untouched.
    """
    out = df.copy()
    rows = max(1, round(window / timestep_stride(df)))
    for col in columns:
        if col in out.columns:
            out[f"{col}_rolling"] = out[col].rolling(window=rows, min_periods=1).mean()
    return out


def collapse_timestep(meta: dict[str, Any]) -> int | None:
    """Timestep of the collapse event, or None.

    Reads the JSON sidecar's ``collapse_timestep``. Note this is the
    *subsistence* collapse definition (the CSV's ``collapse_event`` column),
    not the spec Table 4 MTSC definition used by the statistical analysis --
    the two differ. For an observability dashboard the subsistence marker is
    the more useful one, since it is what the run itself recorded.
    """
    if not meta.get("collapse_occurred"):
        return None
    ts = meta.get("collapse_timestep")
    return int(ts) if ts is not None else None


# ----------------------------------------------------------------- progress


def compute_progress(
    data_dir: str | Path, governance: str, scenario: str, n_reps: int
) -> ProgressInfo:
    """Completed replications versus expected, for the progress bar.

    Counts JSON sidecars: the JSON is written *after* its CSV, so its
    presence means the replication is genuinely finished rather than
    mid-write.
    """
    expected_gov = list(GOVERNANCE_LABELS) if governance == "all" else [governance]
    expected_scen = ["A", "B", "C", "D"] if scenario == "all" else [scenario]
    total = len(expected_gov) * len(expected_scen) * int(n_reps)

    done = sum(
        1 for r in discover_runs(data_dir)
        if r.governance in expected_gov and r.scenario in expected_scen
    )
    return ProgressInfo(completed=done, total=total)
