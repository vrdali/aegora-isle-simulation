"""Factorial experiment runner CLI — spec §6.1/§6.2.

Milestone 3, data-generation only (no dashboard, no statistical analysis
script — both M4). Iterates governance x scenario x replication cells,
drives each through the same PPO/SAC training loop already proven in
``tests/smoke_compare_timeseries.py``, and writes the spec-locked CSV+JSON
pair per run via :class:`~aegora_isle.kpi.kpi_logger.KPILogger`.

Regime-agnostic by construction: the only place a governance letter maps to
anything is ``GOVERNANCE_CONFIG_FILES`` (data selection, not behaviour
branching) — same pattern as ``tests/smoke_compare_all_cells.py``'s
``REGIME_CONFIGS``.

Non-convergence handling (spec §6.2's "excluded and replaced" protocol):
per explicit client direction for the real 360-run confirmatory batch
(reversing an earlier two-retry/dual-check follow-up -- see DECISIONS.md
"Single-criterion exclude-and-replace-until-valid -- client-directed
reversal"), a run that reads ``NON_CONVERGED`` is excluded and replaced
with a fresh attempt at a new seed, repeated in rounds until every cell
has its full ``n_reps`` valid (``CONVERGED`` or ``CONVERGING``)
replications -- no fixed retry cap, up to ``DEFAULT_ROUND_CAP`` rounds as
a safety valve against a runaway process (see ``_run_convergence_rounds``).
No run is ever kept in the final dataset flagged as unconverged; hitting
the round cap aborts the batch loudly instead (``ConvergenceBudgetExceededError``)
-- with one narrow, client-approved exception (see ``CAP_EXCEEDED_KEEP_
PARTIAL``): the two Learning-Based cells the client pre-decided to accept
partial data for, if they exhaust the cap, are kept and documented as a
limitation rather than aborting the whole batch. Any other cell exhausting
the cap still aborts loudly, exactly as before.

Convergence is decided by a **single criterion for every agent type in
every regime**: policy-loss variance over the last 20 epochs vs the 0.01
threshold, per spec §6.3. The per-regime harvested-amount ("KPI-flatness")
override that previously replaced this for the Extractor under G2/G3 was
retired by explicit client decision (2026-08-09): it recorded extraction
*after* the Governor's cap clip, so under G3 -- where the cap moves by
design -- it measured Governor adaptivity rather than Extractor
convergence, and its original justification turned out to be an artifact
of a separate defect (this module previously fed the check SB3's combined
``train/loss`` instead of policy loss). Both are documented in DECISIONS.md
"Convergence measurement -- two defects corrected".

Value-function loss is still recorded, into a purely observational monitor
that never gates status, and written to ``value_loss_diagnostics.csv`` for
every run -- see ``_write_loss_diagnostics``.

Since ``run_id``/output paths are derived from ``(governance, scenario,
replication)`` only -- never from seed -- each replacement naturally
overwrites the prior attempt's CSV/JSON in place.
``data/convergence_failures.csv`` (existing, locked schema, untouched) is
the audit trail of what got excluded and why; ``attempts_summary.csv``
(per-replication) and ``cell_attempt_summary.csv`` (per-cell rollup, new)
report how many attempts each replication and each cell needed.

All governance parameters (extraction_cap, sanction settings, redistribution
rate, G3's SAC ranges, the sustainability reward formula) are frozen per
client sign-off -- see DECISIONS.md. This module reads them from the
existing YAML configs unmodified.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from joblib import Parallel, delayed
from tqdm import tqdm

from aegora_isle.environment.aegora_env import AegoraEnv
from aegora_isle.experiments.train_agents import (
    DEFAULT_AGENT_TYPES as TRAINED_AGENT_TYPES,
    ManualPPOCollector,
)
from aegora_isle.governance.governance_module import GovernanceModule
from aegora_isle.kpi.convergence import ConvergenceMonitor, ConvergenceStatus
from aegora_isle.kpi.kpi_logger import KPILogger

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIGS_DIR = REPO_ROOT / "aegora_isle" / "governance" / "configs"
SCENARIOS_PATH = Path(__file__).resolve().parent / "scenarios.yaml"

# Data selection only (CLAUDE.md forbids regime branching outside
# governance/; this is a lookup table, not an if/elif on regime identity).
GOVERNANCE_ORDER: tuple[str, ...] = ("G1", "G2", "G3")
SCENARIO_ORDER: tuple[str, ...] = ("A", "B", "C", "D")
GOVERNANCE_CONFIG_FILES: dict[str, str] = {
    "G1": "g1_ungoverned.yaml",
    "G2": "g2_rule_based.yaml",
    "G3": "g3_adaptive_rl.yaml",
}

PPO_N_STEPS: int = 128
# 2026-08-07: tested halving to 64 ("Option A" pilot, client-approved) on
# the 28 replications stuck identically across both EXTRACTOR_SPEND_FRACTION
# pilots -- 0 of 28 converged, no improvement at all. Reverted per the
# test's own pre-stated condition. See DECISIONS.md "PPO_N_STEPS=64 pilot
# -- no effect, reverted."
RETRY_SEED_DELTA: int = 1000  # per-round seed offset -- not a convergence
# constant (CLAUDE.md's "never re-tune 0.01/20" rule doesn't apply here),
# just the replacement-round seed offset. Round k's seed_override is
# base_seed + (k-1)*RETRY_SEED_DELTA (round 1 uses the base seed as-is).
DEFAULT_ROUND_CAP: int = 10  # safety valve, not a convergence constant --
# see DECISIONS.md "Single-criterion exclude-and-replace-until-valid --
# client-directed reversal" for the sizing math (client-confirmed).
MIN_TIMESTEPS_FOR_CONVERGENCE_CHECK: int = 5_000  # 50 epochs x 100 steps

# Client direction (2026-08-02, "M3 Addendum -- Confirmed, one
# scarcity-contrast check, cleared to launch"): originally a pre-decided
# fallback for G3-A/G3-B only, since the harvested-amount fix showed
# 0-of-6 improvement for Learning-Based non-convergence (vs 4-of-6 for
# Rule-Based). Extended to G3-C/G3-D (client follow-up, real-batch abort
# after 6h26m/10 rounds: 34 replications, all Learning-Based C/D) once the
# same instability turned out not to be scenario-specific -- the Scenario
# C/D resource-pool fix let those cells finally run to full length, and
# the same Governor non-stationarity showed up there too. Same underlying
# reason as the original A/B approval, now confirmed to generalise across
# all four scenarios. If any of these four cells exhausts the round cap in
# the real run, keep whatever valid replications it got instead of
# aborting the whole batch. Any other cell exhausting the cap still aborts
# loudly, exactly as before -- this is a named exception for these four
# cells specifically (a scenario-design decision), not a general change to
# the abort mechanism, and not a branch on governance regime identity.
CAP_EXCEEDED_KEEP_PARTIAL: frozenset[tuple[str, str]] = frozenset({
    ("G3", "A"), ("G3", "B"), ("G3", "C"), ("G3", "D"),
})


class ConvergenceBudgetExceededError(RuntimeError):
    """Raised when one or more replications are still NON_CONVERGED/ERROR
    after DEFAULT_ROUND_CAP replacement rounds, outside the client's
    pre-approved partial-data exception (see ``CAP_EXCEEDED_KEEP_PARTIAL``).
    Deliberately NOT caught and silently
    absorbed into a "kept, flagged unconverged" result for any other cell
    -- per explicit client direction, no run outside that named exception
    is ever retained in the final dataset labelled unconverged. This is a
    loud, unrecoverable abort that names exactly which (governance,
    scenario, replication) triples never resolved, so a human looks at it
    rather than the batch silently completing with bad data or looping
    forever."""


# --------------------------------------------------------------- pure helpers


def compute_seed(governance: str, scenario: str, replication: int, seed_offset: int) -> int:
    """seed = seed_offset + gov_idx*1000 + scenario_idx*100 + replication.

    Fixed pre-launch (see DECISIONS.md "Seed-collision bug fixed before
    the real 360-run launch"): spec §6.2's literal formula
    (``gov_idx*100 + scenario_idx*10 + replication``) treated replication
    as a single decimal digit, so any ``replication >= 10`` carried into
    the scenario digit column and collided with a different
    (scenario, replication) pair -- e.g. the old formula gave
    ``compute_seed("G1","A",11,0) == compute_seed("G1","B",1,0)``. With
    ``n_reps=30`` this affected 60 of 120 (scenario, replication) pairs per
    governance condition -- half the confirmatory dataset seeded
    identically to a different, supposedly-independent replication.

    This widens each digit block by one order of magnitude (replication
    now supports up to 999 without colliding into scenario's column,
    scenario up to 9 without colliding into governance's), preserving the
    spec formula's structure and intent -- still additive, still
    deterministic, still a strict place-value composition of
    seed_offset/governance/scenario/replication -- just with enough
    headroom for the real n_reps=30 batch (and well beyond).
    """
    return (
        seed_offset
        + GOVERNANCE_ORDER.index(governance) * 1000
        + SCENARIO_ORDER.index(scenario) * 100
        + replication
    )


def build_cells(governance: str, scenario: str, n_reps: int) -> list[dict[str, Any]]:
    """Cartesian product, governance-major -> scenario -> replication
    (1-indexed, matching every existing run_id convention in this codebase),
    per MASTER_PLAN.md Sprint 3.1's literal iteration order."""
    governances = list(GOVERNANCE_ORDER) if governance == "all" else [governance]
    scenarios = list(SCENARIO_ORDER) if scenario == "all" else [scenario]
    return [
        {"governance": g, "scenario": s, "replication": r}
        for g in governances
        for s in scenarios
        for r in range(1, n_reps + 1)
    ]


def _load_scenario_params(scenario: str) -> dict[str, Any]:
    with SCENARIOS_PATH.open() as f:
        return yaml.safe_load(f)[scenario]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m aegora_isle.experiments.run_experiment",
        description="Aegora Isle factorial experiment runner (spec §6.1).",
    )
    parser.add_argument("--governance", choices=[*GOVERNANCE_ORDER, "all"], default="all")
    parser.add_argument("--scenario", choices=[*SCENARIO_ORDER, "all"], default="all")
    parser.add_argument("--n_reps", type=int, default=30)
    parser.add_argument("--n_agents", type=int, default=50)
    parser.add_argument("--max_timesteps", type=int, default=10_000)
    parser.add_argument("--seed_offset", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="data/")
    parser.add_argument("--n_jobs", type=int, default=1)
    return parser


# ---------------------------------------------------------------- worker (core)


def _classify_convergence(
    monitor: ConvergenceMonitor,
    agent_types: tuple[str, ...],
) -> tuple[str, list[str]]:
    """Classifies a run's convergence status, one check per agent type.

    **Single criterion, all regimes**: policy-loss variance over the last
    20 epochs vs the 0.01 threshold, exactly as spec §6.3 specifies. The
    per-regime harvested-amount override that previously substituted an
    ``"extractor_kpi"`` check for G2/G3 was retired by explicit client
    decision (2026-08-09) -- it measured the Governor's cap adaptivity
    rather than Extractor convergence, and its original justification
    turned out to be an artifact of the combined-vs-policy loss defect
    fixed alongside it. See DECISIONS.md "Convergence measurement -- two
    defects corrected".

    CONVERGED only if every agent type reads CONVERGED. Any NON_CONVERGED
    is retry-eligible; CONVERGING with none non-converged is the
    early-collapse/short-run outcome, never retried.
    """
    checks: dict[str, ConvergenceStatus] = {
        t: monitor.check_convergence(t) for t in agent_types
    }
    non_converged = [t for t, s in checks.items() if s is ConvergenceStatus.NON_CONVERGED]
    still_converging = [t for t, s in checks.items() if s is ConvergenceStatus.CONVERGING]
    if non_converged:
        return "NON_CONVERGED", non_converged
    if still_converging:
        return "CONVERGING", []
    return "CONVERGED", []


def run_single_cell(
    governance: str,
    scenario: str,
    replication: int,
    n_agents: int = 50,
    max_timesteps: int = 10_000,
    seed_offset: int = 0,
    output_dir: str | Path = "data",
    track_convergence: bool = True,
    seed_override: int | None = None,
    torch_threads: int | None = None,
) -> dict[str, Any]:
    """Runs exactly one (governance, scenario, replication) cell end-to-end.

    Pure w.r.t. its arguments (same args -> same seed -> same output, modulo
    filesystem side effects) -- directly pytest-testable, and the unit
    ``joblib.Parallel`` dispatches. ``seed_override`` is what a replacement
    round supplies (``base_seed + (round-1)*RETRY_SEED_DELTA``); the same
    worker serves every round, unaware which one it is.

    The harvested-amount (KPI-flatness) monitor is always constructed
    whenever ``track_convergence`` is on -- there is no manual opt-in flag.
    Whether it's *authoritative* for the Extractor (vs. loss-variance) is
    decided automatically from ``gm.has_stochastic_extractor_sanction``
    (see ``_classify_convergence``), so tracking it is unconditional
    (cheap, passive, no training effect) but only regimes with a stochastic
    sanction actually use it to gate exclude/replace decisions.

    ``torch_threads``, if given, pins this process's PyTorch intra-op
    thread pool to that count (``torch.set_num_threads``) before touching
    any SB3 model. Measured directly (Sprint 3.4 compute-budget check):
    with ``--n_jobs > 1``, each worker process otherwise spawns its own
    multi-threaded torch thread pool, and N worker processes all doing that
    concurrently on the same machine oversubscribes the CPU -- real
    parallel speedup comes out worse than ``total_time / n_jobs`` would
    suggest. ``main()`` sets this to ``1`` per worker whenever
    ``--n_jobs != 1``; left at the default (``None`` -> torch's own
    multi-threaded default) for direct callers like the test suite and for
    ``--n_jobs 1``, where there's no cross-process contention to avoid.
    """
    if torch_threads is not None:
        import torch

        torch.set_num_threads(torch_threads)

    t0 = time.perf_counter()
    seed = (
        seed_override
        if seed_override is not None
        else compute_seed(governance, scenario, replication, seed_offset)
    )
    run_id = f"{governance}_{scenario}_{replication:03d}"
    output_dir = Path(output_dir)
    csv_path = output_dir / "runs" / governance / f"{run_id}.csv"

    result: dict[str, Any] = {
        "run_id": run_id,
        "governance": governance,
        "scenario": scenario,
        "replication": replication,
        "seed": seed,
        "is_retry": seed_override is not None,
        "csv_path": str(csv_path),
        "json_path": str(csv_path.with_suffix(".json")),
        "status": None,
        "converged": None,
        "non_converged_types": [],
        "policy_loss_variance": {},
        "value_loss_variance": {},
        "error": None,
        "wall_time_s": None,
    }

    try:
        scenario_params = _load_scenario_params(scenario)
        config_path = CONFIGS_DIR / GOVERNANCE_CONFIG_FILES[governance]

        # Governance RNG seeded from this run's own seed -- not a shared
        # fixed constant -- so governance stochasticity is genuinely
        # distinct per replication.
        gm = GovernanceModule(str(config_path), np.random.default_rng(seed))
        env = AegoraEnv(
            n_agents=n_agents,
            governance_module=gm,
            scenario_params=scenario_params,
            seed=seed,
            max_timesteps=max_timesteps,
        )
        env.reset()
        for aid in env.agents:
            env.action_space(aid).seed(seed)

        monitor = ConvergenceMonitor() if track_convergence else None
        # Purely observational value-function-loss series (client request,
        # 2026-08-09): keeps value-function instability visible in the
        # output without ever gating convergence status, so the corrected
        # re-run can show whether the policy-loss threshold needs
        # tightening. Reuses ConvergenceMonitor unmodified -- it's generic
        # over the agent_type label. NEVER passed to _classify_convergence.
        value_monitor = (
            ConvergenceMonitor(
                agent_types=tuple(f"{t}_value" for t in TRAINED_AGENT_TYPES)
            )
            if track_convergence
            else None
        )
        logger = KPILogger(env, run_id=run_id, scenario=scenario)
        # ppo_kwargs={"seed": seed} is required for determinism: SB3's
        # _setup_model() seeds global RNG state (which the policy's weight
        # init draws from) before this loop begins. Without it, identical
        # args would not reproduce a byte-identical CSV.
        collector = ManualPPOCollector(
            env,
            n_steps=PPO_N_STEPS,
            ppo_kwargs={"seed": seed},
            convergence_monitor=monitor,
            value_diagnostic_monitor=value_monitor,
        )

        last_timestep = 0
        while env.agents and env._timestep < max_timesteps:
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
                if monitor is not None:
                    collector._maybe_record_epoch_loss()

        csv_path = logger.write_csv(csv_path)
        counts = env._agent_counts()
        agent_type_distribution = {k: v / env.n_agents for k, v in counts.items()}
        logger.write_json_state(
            csv_path,
            governance_params=gm.state_dict(),
            scenario_params=scenario_params,
            agent_type_distribution=agent_type_distribution,
            rl_hyperparams={"n_steps": PPO_N_STEPS, "seed": seed},
        )
        result["csv_path"] = str(csv_path)
        result["json_path"] = str(csv_path.with_suffix(".json"))

        if monitor is not None:
            status, non_converged = _classify_convergence(monitor, collector.agent_types)
            result["non_converged_types"] = non_converged
            result["converged"] = status == "CONVERGED"
            result["status"] = status
            # Observational only -- captured for every run regardless of
            # status, never consulted when deciding it (client request).
            result["policy_loss_variance"] = {
                t: monitor.current_variance(t) for t in collector.agent_types
            }
            if value_monitor is not None:
                result["value_loss_variance"] = {
                    f"{t}_value": value_monitor.current_variance(f"{t}_value")
                    for t in collector.agent_types
                }
            if status == "NON_CONVERGED":
                for agent_type in non_converged:
                    monitor.log_failure(
                        agent_type=agent_type,
                        run_id=run_id,
                        governance=governance,
                        scenario=scenario,
                        replication=replication,
                        path=output_dir / "convergence_failures.csv",
                    )
            # status == "CONVERGING": ran out of cycles (often a legitimate
            # early collapse, e.g. Scenario C/D) before reaching the
            # 50-epoch convergence-check minimum. A valid, keepable data
            # point (see the CSV's collapse_event column) -- not a training
            # failure, not retried. See DECISIONS.md "Convergence-retry
            # protocol only applies to true NON_CONVERGED, not early
            # collapse."
        else:
            result["status"] = "SKIPPED_CONVERGENCE_CHECK"

    except Exception as exc:  # one crashed cell must not abort the whole batch
        result["status"] = "ERROR"
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        result["wall_time_s"] = time.perf_counter() - t0

    return result


# ------------------------------------------------------------------- orchestration


def _run_batch(jobs: list[dict[str, Any]], n_jobs: int, desc: str) -> list[dict[str, Any]]:
    """Runs jobs = [kwargs, ...] in parallel, streaming completions through
    tqdm as they finish (return_as="generator" reflects real completion
    order, not mere dispatch order)."""
    tasks = (delayed(run_single_cell)(**kwargs) for kwargs in jobs)
    results_iter = Parallel(n_jobs=n_jobs, return_as="generator")(tasks)
    return list(tqdm(results_iter, total=len(jobs), desc=desc))


def _merge_retries(
    results: list[dict[str, Any]], retries: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Replaces each original NON_CONVERGED/ERROR result with its
    replacement-round result, matched by (governance, scenario,
    replication). Extracted as a standalone pure function so the merge
    logic is unit-testable without mocking joblib.Parallel. Reused
    unchanged across every round of ``_run_convergence_rounds``."""
    retry_by_key = {(r["governance"], r["scenario"], r["replication"]): r for r in retries}
    merged = []
    for r in results:
        key = (r["governance"], r["scenario"], r["replication"])
        merged.append(retry_by_key.get(key, r))
    return merged


def _write_attempts_summary(results: list[dict[str, Any]], path: str | Path) -> Path:
    """One row per replication (final outcome only, not one row per
    attempt): run_id, governance, scenario, replication, seed,
    attempts_used, final_status. Client's ask: "log how many attempts
    each run needed" -- ``final_status`` is CONVERGED or CONVERGING for
    every ordinary replication, since ``_run_convergence_rounds`` doesn't
    return until each one resolves (or the batch aborts without writing
    this file at all -- see ``ConvergenceBudgetExceededError``). The one
    documented exception is ``"NON_CONVERGED_CAP_EXCEEDED_KEPT"`` for a
    replication in the client's pre-approved partial-data
    fallback (see ``CAP_EXCEEDED_KEEP_PARTIAL``) -- a deliberate, labelled
    limitation, not a silent unconverged result. A new, separate artifact
    -- never touches kpi_logger.py's locked 14-column CSV / 13-field
    JSON."""
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["run_id", "governance", "scenario", "replication", "seed", "attempts_used", "final_status"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "run_id": r["run_id"],
                    "governance": r["governance"],
                    "scenario": r["scenario"],
                    "replication": r["replication"],
                    "seed": r["seed"],
                    "attempts_used": r["attempt"],
                    "final_status": r["status"],
                }
            )
    return path


def _needs_replacement(r: dict[str, Any]) -> bool:
    """A replication needs a fresh replacement attempt if it's genuinely
    NON_CONVERGED, or if it crashed (ERROR is folded into the same
    replace-and-retry bucket -- confirmed with the user: leaving an
    errored slot unreplaced would permanently leave that cell short of
    its n_reps valid replications, a bigger problem than what's being
    fixed). CONVERGING (early collapse/short run) and CONVERGED are both
    valid, done -- never replaced."""
    return r["status"] in ("NON_CONVERGED", "ERROR")


def _build_cell_attempt_rollup(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pure aggregation (no filesystem/joblib) -- one row per (governance,
    scenario) cell. Client's ask: "log every exclusion and replacement
    attempt per cell... how many replacement attempts each cell needed to
    reach its 30 valid replications." Independently unit-testable with
    plain dicts, same rationale as _merge_retries."""
    cells = sorted({(r["governance"], r["scenario"]) for r in results})
    rows = []
    for governance, scenario in cells:
        cell = [r for r in results if r["governance"] == governance and r["scenario"] == scenario]
        attempts = [r["attempt"] for r in cell]
        total_replications = len(cell)
        total_attempts = sum(attempts)
        cap_exceeded_kept_count = sum(
            1 for r in cell if r["status"] == "NON_CONVERGED_CAP_EXCEEDED_KEPT"
        )
        rows.append(
            {
                "governance": governance,
                "scenario": scenario,
                "total_replications": total_replications,
                "total_attempts_across_cell": total_attempts,
                "total_replacement_attempts_across_cell": total_attempts - total_replications,
                "max_attempts_for_a_single_replication": max(attempts) if attempts else 0,
                "mean_attempts_per_replication": round(total_attempts / total_replications, 2)
                if total_replications
                else 0.0,
                # Client's pre-approved partial-data fallback (see
                # CAP_EXCEEDED_KEEP_PARTIAL): 0 for every cell unless this
                # one actually exhausted the round cap and was kept as a
                # documented limitation rather than aborting the batch.
                "cap_exceeded_kept_count": cap_exceeded_kept_count,
            }
        )
    return rows


def _write_loss_diagnostics(results: list[dict[str, Any]], path: str | Path) -> Path:
    """Writes the per-run, per-agent-type loss-variance diagnostic.

    Client request (2026-08-09): after retiring the harvested-amount
    override, keep value-function instability *visible* without letting it
    gate convergence, so the corrected re-run can show empirically whether
    the 0.01 policy-loss threshold needs tightening. Emitted for **every**
    run, not just failures -- a run passing comfortably vs passing by a
    hair is exactly the distinction this is here to expose.

    ``policy_loss_variance`` is the gating signal (spec §6.3);
    ``value_loss_variance`` is observational only. A new artifact,
    alongside ``attempts_summary.csv`` / ``cell_attempt_summary.csv`` --
    the spec-locked KPI CSV/JSON schemas and
    ``CONVERGENCE_FAILURES_COLUMNS`` are untouched.
    """
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "run_id", "governance", "scenario", "replication", "final_status",
            "agent_type", "policy_loss_variance", "value_loss_variance",
        ])
        for r in results:
            policy = r.get("policy_loss_variance") or {}
            value = r.get("value_loss_variance") or {}
            for agent_type in sorted(policy):
                writer.writerow([
                    r["run_id"], r["governance"], r["scenario"], r["replication"],
                    r.get("status"), agent_type,
                    policy.get(agent_type, ""),
                    value.get(f"{agent_type}_value", ""),
                ])
    return path


def _write_cell_attempt_summary(results: list[dict[str, Any]], path: str | Path) -> Path:
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = _build_cell_attempt_rollup(results)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    return path


def _print_summary(results: list[dict[str, Any]], total: int, max_timesteps: int) -> None:
    converged_first_try = sum(1 for r in results if r["status"] == "CONVERGED" and r["attempt"] == 1)
    still_converging_count = sum(1 for r in results if r["status"] == "CONVERGING")
    replaced = sum(1 for r in results if r["attempt"] > 1)
    total_attempts = sum(r["attempt"] for r in results)
    cap_exceeded_kept = [r for r in results if r["status"] == "NON_CONVERGED_CAP_EXCEEDED_KEPT"]
    print(f"\n{'=' * 60}")
    print(f"Run summary: {total}/{total} replications resolved (converged, valid short run, or documented limitation)")
    print(f"  Converged on the first attempt: {converged_first_try}")
    print(
        f"  Ended before the convergence-check minimum (often a legitimate "
        f"early collapse -- not a training failure): {still_converging_count}"
    )
    print(f"  Needed at least one replacement (excluded and retried at a new seed): {replaced}")
    print(f"  Total attempts across the whole batch: {total_attempts} (vs. {total} if every one converged first try)")
    if cap_exceeded_kept:
        print(
            f"  LIMITATION (client-approved, see CAP_EXCEEDED_KEEP_PARTIAL): "
            f"{len(cap_exceeded_kept)} replication(s) never converged after "
            f"{DEFAULT_ROUND_CAP} rounds and were kept as documented partial data, "
            "not excluded:"
        )
        for r in sorted(cap_exceeded_kept, key=lambda r: (r["governance"], r["scenario"], r["replication"])):
            print(f"    {r['run_id']} (governance={r['governance']}, scenario={r['scenario']})")
    if max_timesteps < MIN_TIMESTEPS_FOR_CONVERGENCE_CHECK:
        print(
            f"  NOTE: --max_timesteps={max_timesteps} is below the "
            f"{MIN_TIMESTEPS_FOR_CONVERGENCE_CHECK}-step (50-epoch) convergence "
            "minimum -- every cell is expected to read CONVERGING (not "
            "NON_CONVERGED). This is not a real convergence failure, just a "
            "run shorter than the protocol's measurement window."
        )
    print("  Replacement attempts needed, per cell:")
    for row in _build_cell_attempt_rollup(results):
        limitation_note = (
            f", {row['cap_exceeded_kept_count']} kept short as a documented limitation"
            if row["cap_exceeded_kept_count"]
            else ""
        )
        print(
            f"    {row['governance']}-{row['scenario']}: {row['total_replications']} reps, "
            f"{row['total_replacement_attempts_across_cell']} replacement attempt(s) needed "
            f"(max {row['max_attempts_for_a_single_replication']} for a single replication, "
            f"mean {row['mean_attempts_per_replication']}{limitation_note})"
        )
    print(f"{'=' * 60}")


def _run_convergence_rounds(
    cells: list[dict[str, Any]],
    common: dict[str, Any],
    seed_offset: int,
    n_jobs: int,
    round_cap: int = DEFAULT_ROUND_CAP,
) -> list[dict[str, Any]]:
    """Round-based exclude-and-replace-until-valid loop (client-directed
    reversal of the earlier capped two-retry protocol -- see DECISIONS.md
    "Single-criterion exclude-and-replace-until-valid -- client-directed
    reversal"). Round 1 dispatches every cell; each subsequent round only
    re-dispatches whichever (governance, scenario, replication) slots
    still need replacement (see _needs_replacement), at
    ``seed = compute_seed(...) + (round-1)*RETRY_SEED_DELTA`` -- preserving
    joblib-parallel batch efficiency (only the shrinking failing subset is
    ever re-dispatched) rather than looping one replication at a time.

    Continues until every slot reads CONVERGED or CONVERGING (both valid,
    done -- CONVERGING/early-collapse is never replaced, per the
    unchanged collapse-handling fix), or ``round_cap`` is exceeded, in
    which case ``ConvergenceBudgetExceededError`` is raised naming exactly
    which replications never resolved -- never a silent "kept, flagged
    unconverged" result.

    **Exception** (client-directed, see ``CAP_EXCEEDED_KEEP_PARTIAL``): if
    every replication still unresolved when the cap is hit belongs to one
    of the two pre-approved (governance, scenario) cells, they are kept as
    ``"NON_CONVERGED_CAP_EXCEEDED_KEPT"`` instead of raising -- a
    deliberate, documented, narrow exception, not a general softening of
    the abort mechanism. Any unresolved replication outside that set still
    raises exactly as before.
    """
    jobs = [{**common, **c} for c in cells]
    results = _run_batch(jobs, n_jobs, desc="Round 1")
    for r in results:
        r["attempt"] = 1

    round_number = 1
    while any(_needs_replacement(r) for r in results):
        round_number += 1
        if round_number > round_cap:
            unresolved = [r for r in results if _needs_replacement(r)]
            non_exempt = [
                r for r in unresolved
                if (r["governance"], r["scenario"]) not in CAP_EXCEEDED_KEEP_PARTIAL
            ]
            if non_exempt:
                raise ConvergenceBudgetExceededError(
                    f"{len(non_exempt)} replication(s) still not converged after "
                    f"{round_cap} rounds (cap reached), outside the pre-approved "
                    "partial-data exception: "
                    + ", ".join(
                        f"({r['governance']},{r['scenario']},{r['replication']})" for r in non_exempt
                    )
                )
            # Every unresolved replication belongs to the client's
            # pre-approved exception (see CAP_EXCEEDED_KEEP_PARTIAL) --
            # keep it as documented, partial data rather than aborting.
            for r in results:
                if _needs_replacement(r):
                    r["status"] = "NON_CONVERGED_CAP_EXCEEDED_KEPT"
            break
        to_replace = [r for r in results if _needs_replacement(r)]
        jobs = [
            {
                **common,
                "governance": r["governance"],
                "scenario": r["scenario"],
                "replication": r["replication"],
                "seed_override": compute_seed(r["governance"], r["scenario"], r["replication"], seed_offset)
                + (round_number - 1) * RETRY_SEED_DELTA,
            }
            for r in to_replace
        ]
        retries = _run_batch(jobs, n_jobs, desc=f"Round {round_number}")
        for r in retries:
            r["attempt"] = round_number
        results = _merge_retries(results, retries)

    return results


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    cells = build_cells(args.governance, args.scenario, args.n_reps)
    total = len(cells)

    if args.n_jobs != 1:
        # Measured directly (Sprint 3.4): with multiple worker processes,
        # each also spawning its own multi-threaded BLAS/torch thread pool
        # oversubscribes the CPU. setdefault() so an operator's own
        # explicit env var choice is respected, not silently overridden.
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ.setdefault(var, "1")

    common = dict(
        n_agents=args.n_agents,
        max_timesteps=args.max_timesteps,
        seed_offset=args.seed_offset,
        output_dir=str(output_dir),
        track_convergence=True,
        torch_threads=1 if args.n_jobs != 1 else None,
    )

    print(f"Running {total} replication(s), n_jobs={args.n_jobs}, round cap={DEFAULT_ROUND_CAP}")
    try:
        results = _run_convergence_rounds(cells, common, args.seed_offset, args.n_jobs)
    except ConvergenceBudgetExceededError as exc:
        print(f"\n{'=' * 60}")
        print("ABORTED -- convergence budget exceeded, no dataset artifacts written:")
        print(f"  {exc}")
        print(f"{'=' * 60}")
        return 1

    _write_attempts_summary(results, output_dir / "attempts_summary.csv")
    _write_cell_attempt_summary(results, output_dir / "cell_attempt_summary.csv")
    _write_loss_diagnostics(results, output_dir / "value_loss_diagnostics.csv")
    _print_summary(results, total, args.max_timesteps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
