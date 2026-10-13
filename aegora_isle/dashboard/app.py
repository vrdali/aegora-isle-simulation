"""Aegora Isle observability dashboard (spec §7).

Launch per spec Table 9::

    streamlit run aegora_isle/dashboard/app.py

Implements every view required by spec §7.1: Run Selector, KPI Time Series
(5 KPIs with 50-step rolling averages), Multi-Run Overlay, Experiment
Progress, Collapse Events, CSV Export, and Run/Pause/Reset.

This is a Phase 1 *observability* tool, not a publication surface -- the
publication figures come from ``analysis/statistical_analysis.py``. All
logic worth testing lives in ``data_access.py`` and ``process_control.py``;
this file is UI glue.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from aegora_isle.dashboard import process_control as pc
from aegora_isle.dashboard.data_access import (
    GOVERNANCE_LABELS,
    KPI_COLUMNS,
    KPI_LABELS,
    add_rolling_kpis,
    collapse_timestep,
    compute_progress,
    discover_runs,
    load_run_csv,
    load_run_json,
    runs_dataframe,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

# Where a dashboard-launched batch writes by default. Deliberately not the
# viewer's data directory: a batch overwrites same-named runs and rewrites the
# batch summary files, so defaulting it to the delivered dataset makes the
# most destructive option the easiest one to trigger.
DEFAULT_OUTPUT_DIR = "data/new_batch/"
PALETTE = {"G1": "#c44e52", "G2": "#4c72b0", "G3": "#55a868"}

st.set_page_config(page_title="Aegora Isle — Observability", layout="wide")


# Cached on path + mtime so a re-run of the same file is cheap, but a batch
# that overwrites a replication is still picked up.
@st.cache_data(show_spinner=False)
def _csv(path: str, _mtime: float) -> pd.DataFrame:
    return load_run_csv(path)


@st.cache_data(show_spinner=False)
def _json(path: str, _mtime: float) -> dict:
    return load_run_json(path)


@st.cache_data(show_spinner=False)
def _runs_df(data_dir: str, _n: int) -> pd.DataFrame:
    return runs_dataframe(data_dir)


def _load(meta) -> tuple[pd.DataFrame, dict]:
    df = _csv(str(meta.csv_path), meta.csv_path.stat().st_mtime)
    js = _json(str(meta.json_path), meta.json_path.stat().st_mtime)
    return df, js


def _mark_collapse(ax, ts: int | None) -> None:
    """Spec §7.1: vertical red line at the collapse timestep."""
    if ts is not None:
        ax.axvline(ts, color="red", linestyle="--", linewidth=1.4, alpha=0.9,
                   label=f"collapse @ {ts}")


# ----------------------------------------------------------------- sidebar

st.sidebar.title("Aegora Isle")
data_dir = st.sidebar.text_input("Data directory", value="data/")
runs = discover_runs(data_dir)
st.sidebar.caption(f"{len(runs)} replications discovered")

view = st.sidebar.radio("View", [
    "Run Selector", "KPI Time Series", "Multi-Run Overlay",
    "Collapse Events", "Experiment Progress", "CSV Export",
])

if not runs and view != "Experiment Progress":
    st.warning(f"No replications found under `{data_dir}runs/`. "
               "Point the sidebar at a data directory, or start a run from **Experiment Progress**.")
    st.stop()


def _pick_run(key: str):
    """Shared governance / scenario / replication selector (spec §7.1)."""
    govs = sorted({r.governance for r in runs})
    gov = st.selectbox("Governance", govs,
                       format_func=lambda g: GOVERNANCE_LABELS.get(g, g), key=f"{key}_gov")
    scens = sorted({r.scenario for r in runs if r.governance == gov})
    scen = st.selectbox("Scenario", scens, key=f"{key}_scen")
    reps = sorted(r.replication for r in runs if r.governance == gov and r.scenario == scen)
    rep = st.selectbox("Replication", reps, key=f"{key}_rep")
    return next(r for r in runs
                if r.governance == gov and r.scenario == scen and r.replication == rep)


# ------------------------------------------------------------ Run Selector

if view == "Run Selector":
    st.header("Run Selector")
    df = _runs_df(data_dir, len(runs))
    c1, c2, c3 = st.columns(3)
    gov_f = c1.multiselect("Governance", sorted(df.governance.unique()))
    scen_f = c2.multiselect("Scenario", sorted(df.scenario.unique()))
    status_opts = sorted(x for x in df.get("final_status", pd.Series(dtype=str)).dropna().unique() if x)
    status_f = c3.multiselect("Status", status_opts)

    view_df = df
    if gov_f:
        view_df = view_df[view_df.governance.isin(gov_f)]
    if scen_f:
        view_df = view_df[view_df.scenario.isin(scen_f)]
    if status_f:
        view_df = view_df[view_df.final_status.isin(status_f)]

    st.caption(f"{len(view_df)} of {len(df)} replications")
    st.dataframe(view_df, use_container_width=True, hide_index=True)

    if "final_status" in df.columns and df.final_status.astype(str).str.len().gt(0).any():
        st.subheader("Status by cell")
        st.dataframe(
            df.pivot_table(index="governance", columns="scenario",
                           values="final_status", aggfunc=lambda s: s.eq("CONVERGED").sum()),
            use_container_width=True,
        )
        st.caption("Counts show CONVERGED replications per cell.")


# -------------------------------------------------------- KPI Time Series

elif view == "KPI Time Series":
    st.header("KPI Time Series")
    meta = _pick_run("kpi")
    df, js = _load(meta)
    df = add_rolling_kpis(df)
    ts = collapse_timestep(js)

    st.caption(f"**{meta.run_id}** — seed {js.get('random_seed')}, "
               f"{len(df):,} timesteps. Solid line is the 50-step rolling average "
               "required by spec §7.1; the raw series is shown faintly behind it.")

    fig, axes = plt.subplots(len(KPI_COLUMNS), 1, figsize=(11, 2.1 * len(KPI_COLUMNS)), sharex=True)
    for ax, kpi in zip(axes, KPI_COLUMNS):
        ax.plot(df["timestep"], df[kpi], color="grey", alpha=0.28, lw=0.7)
        ax.plot(df["timestep"], df[f"{kpi}_rolling"], color=PALETTE.get(meta.governance), lw=1.3)
        ax.set_ylabel(KPI_LABELS[kpi], fontsize=8)
        ax.grid(alpha=0.25)
        _mark_collapse(ax, ts)
    axes[-1].set_xlabel("Timestep")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ------------------------------------------------------ Multi-Run Overlay

elif view == "Multi-Run Overlay":
    st.header("Multi-Run Overlay")
    st.caption("Overlay Gini trajectories across replications (spec §7.1).")

    c1, c2 = st.columns(2)
    govs = sorted({r.governance for r in runs})
    gov = c1.selectbox("Governance", govs, format_func=lambda g: GOVERNANCE_LABELS.get(g, g))
    scens = sorted({r.scenario for r in runs if r.governance == gov})
    scen = c2.selectbox("Scenario", scens)

    cell = [r for r in runs if r.governance == gov and r.scenario == scen]
    chosen = st.multiselect(
        "Replications", [r.run_id for r in cell],
        default=[r.run_id for r in cell[:5]],
    )
    kpi = st.selectbox("KPI", list(KPI_COLUMNS), index=0, format_func=lambda k: KPI_LABELS[k])

    if chosen:
        fig, ax = plt.subplots(figsize=(11, 5))
        for rid in chosen:
            m = next(r for r in cell if r.run_id == rid)
            d = add_rolling_kpis(_load(m)[0])
            ax.plot(d["timestep"], d[f"{kpi}_rolling"], lw=1.1, alpha=0.85, label=rid)
        ax.set_xlabel("Timestep")
        ax.set_ylabel(KPI_LABELS[kpi])
        ax.set_title(f"{GOVERNANCE_LABELS.get(gov, gov)} — Scenario {scen}")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, ncol=2)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    else:
        st.info("Select at least one replication.")


# ------------------------------------------------------- Collapse Events

elif view == "Collapse Events":
    st.header("Collapse Events")
    collapsed = []
    for r in runs:
        js = _json(str(r.json_path), r.json_path.stat().st_mtime)
        ts = collapse_timestep(js)
        if ts is not None:
            collapsed.append({"run_id": r.run_id, "governance": r.governance,
                              "scenario": r.scenario, "collapse_timestep": ts})

    if not collapsed:
        st.success("No collapse events recorded in this dataset.")
    else:
        cdf = pd.DataFrame(collapsed)
        st.caption(
            f"{len(cdf)} of {len(runs)} replications recorded a collapse. "
            "This uses the CSV's `collapse_event` (subsistence) definition — the "
            "statistical analysis uses the spec Table 4 MTSC definition instead, "
            "and the two differ. See the analysis report."
        )
        st.dataframe(cdf.groupby(["governance", "scenario"]).size()
                     .rename("collapses").reset_index(), hide_index=True)

        pick = st.selectbox("Inspect a collapsed replication", cdf.run_id.tolist())
        meta = next(r for r in runs if r.run_id == pick)
        df, js = _load(meta)
        df = add_rolling_kpis(df)
        ts = collapse_timestep(js)

        fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
        for ax, kpi in zip(axes, ("gini_coefficient", "resource_stock")):
            ax.plot(df["timestep"], df[f"{kpi}_rolling"],
                    color=PALETTE.get(meta.governance), lw=1.3)
            ax.set_ylabel(KPI_LABELS[kpi])
            ax.grid(alpha=0.25)
            _mark_collapse(ax, ts)
            ax.legend(fontsize=8)
        axes[-1].set_xlabel("Timestep")
        fig.suptitle(f"{pick} — collapse at timestep {ts}")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


# --------------------------------------------------- Experiment Progress

elif view == "Experiment Progress":
    st.header("Experiment Progress")
    handle: pc.ProcessHandle | None = st.session_state.get("handle")

    running = pc.is_running(handle)

    # Everything below is inside an st.form, and every widget carries an
    # explicit key, for one reason: a batch launched with the wrong output
    # directory overwrites real data irrecoverably.
    #
    # Outside a form, a Streamlit text_input only commits on Enter or blur.
    # Clicking Run with an uncommitted field therefore submitted the widget's
    # *default* rather than what was on screen -- and that default used to be
    # the live data directory. A form commits every field atomically on
    # submit, so what you see is what launches. The explicit keys stop values
    # reverting to defaults across the reruns each button triggers.
    with st.form("launch_config"):
        st.caption("Launch configuration")
        c1, c2, c3 = st.columns(3)
        cfg = {
            "governance": c1.selectbox("Governance", ["all", "G1", "G2", "G3"],
                                       key="cfg_governance"),
            "scenario": c1.selectbox("Scenario", ["all", "A", "B", "C", "D"],
                                     key="cfg_scenario"),
            "n_reps": c2.number_input("Replications per cell", 1, 100, 30,
                                      key="cfg_n_reps"),
            "n_agents": c2.number_input("Agents", 4, 500, 50, key="cfg_n_agents"),
            "max_timesteps": c3.number_input("Max timesteps", 100, 100_000, 10_000,
                                             step=100, key="cfg_max_timesteps"),
            "n_jobs": c3.number_input("Parallel workers", 1, 16, 6, key="cfg_n_jobs"),
            # Deliberately NOT defaulted to the sidebar's data directory: the
            # default target of a destructive action should never be the real
            # dataset.
            "output_dir": st.text_input("Output directory", value=DEFAULT_OUTPUT_DIR,
                                        key="cfg_output_dir"),
        }
        overwrite_ok = st.checkbox(
            "Allow writing into a directory that already contains runs",
            key="cfg_overwrite_ok",
            help="Required when the output directory is not empty. A batch "
                 "overwrites same-named runs and replaces the batch summary "
                 "files for whatever it does not cover.",
        )
        submitted = st.form_submit_button("▶ Run", disabled=running,
                                          use_container_width=True)

    if submitted:
        existing = len(discover_runs(cfg["output_dir"]))
        if existing and not overwrite_ok:
            st.error(
                f"`{cfg['output_dir']}` already contains **{existing} replications**. "
                "This batch would overwrite same-named runs and rewrite the batch "
                "summary files, discarding entries for anything outside its scope. "
                "Tick the confirmation box above to proceed, or choose another "
                "directory."
            )
        else:
            st.session_state["handle"] = pc.launch_experiment(cfg, REPO_ROOT)
            st.session_state["cfg"] = cfg
            st.rerun()

    b2, b3 = st.columns(2)
    if b2.button("⏸ Pause / Resume", disabled=not running, use_container_width=True):
        paused = st.session_state.get("paused", False)
        (pc.resume if paused else pc.pause)(handle)
        st.session_state["paused"] = not paused
        st.rerun()
    if b3.button("⏹ Reset", disabled=not running, use_container_width=True):
        pc.reset(handle)
        st.session_state["handle"] = None
        st.session_state["paused"] = False
        st.rerun()

    if running:
        state = "paused" if st.session_state.get("paused") else "running"
        st.info(f"Experiment {state} — PID {handle.pid}")
    else:
        st.caption("No experiment running.")

    active = st.session_state.get("cfg", cfg)
    prog = compute_progress(active["output_dir"], active["governance"],
                            active["scenario"], active["n_reps"])
    st.progress(prog.fraction, text=f"{prog.completed} / {prog.total} replications complete")

    st.caption(
        "Progress is **replication-level**. Each run's CSV is written once, when that "
        "replication finishes, so there is no partial file to plot mid-run — the live "
        "chart below shows KPIs from replications completed so far rather than a "
        "synthesised in-flight curve."
    )

    done = discover_runs(active["output_dir"])
    if done:
        # run_id breaks mtime ties: a copied or archived dataset can leave
        # hundreds of sidecars stamped to the same second, and an arbitrary
        # winner would make this label flicker between refreshes.
        latest = max(done, key=lambda r: (r.json_path.stat().st_mtime, r.run_id))
        d, _js = _load(latest)
        d = add_rolling_kpis(d)
        fig, ax = plt.subplots(figsize=(11, 3.6))
        ax.plot(d["timestep"], d["gini_coefficient_rolling"],
                color=PALETTE.get(latest.governance), lw=1.3)
        ax.set_title(f"Most recently completed: {latest.run_id}")
        ax.set_xlabel("Timestep")
        ax.set_ylabel(KPI_LABELS["gini_coefficient"])
        ax.grid(alpha=0.25)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    if st.checkbox("Auto-refresh every 10s", value=False) and running:
        import time
        time.sleep(10)
        st.rerun()


# -------------------------------------------------------------- CSV Export

elif view == "CSV Export":
    st.header("CSV Export")
    meta = _pick_run("exp")
    df, js = _load(meta)
    st.caption(f"**{meta.run_id}** — {len(df):,} rows x {len(df.columns)} columns.")
    st.dataframe(df.head(50), use_container_width=True, hide_index=True)

    st.download_button(
        f"Download {meta.run_id}.csv",
        data=meta.csv_path.read_bytes(),
        file_name=f"{meta.run_id}.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.download_button(
        f"Download {meta.run_id}.json (reproducibility state)",
        data=meta.json_path.read_bytes(),
        file_name=f"{meta.run_id}.json",
        mime="application/json",
        use_container_width=True,
    )
