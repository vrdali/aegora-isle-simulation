"""Unit tests for aegora_isle.experiments.run_experiment.

Covers M3 data-generation slice: the seed formula (spec §6.2), cell
expansion, a full run_single_cell smoke pass, determinism, and the
exclude-and-replace-until-valid mechanics (spec §6.2's "excluded and
replaced" protocol) -- the piece the client explicitly gated the real
360-run confirmatory batch on. Per the client's later reversal (see
DECISIONS.md "Single-criterion exclude-and-replace-until-valid --
client-directed reversal"), the real launch uses an unbounded round-based
retry loop (capped only as a safety valve). Convergence is decided by a
single criterion for every agent type in every regime: policy-loss
variance vs the 0.01 threshold (spec §6.3). The per-regime
harvested-amount override was retired by client decision -- see
DECISIONS.md "Convergence measurement -- two defects corrected".
"""

from __future__ import annotations

import numpy as np
import pytest

from aegora_isle.experiments.run_experiment import (
    DEFAULT_ROUND_CAP,
    GOVERNANCE_ORDER,
    RETRY_SEED_DELTA,
    SCENARIO_ORDER,
    ConvergenceBudgetExceededError,
    build_cells,
    compute_seed,
    run_single_cell,
    _build_cell_attempt_rollup,
    _classify_convergence,
    _merge_retries,
    _needs_replacement,
    _write_attempts_summary,
    _write_cell_attempt_summary,
    _write_loss_diagnostics,
)
from aegora_isle.kpi.convergence import ConvergenceMonitor, ConvergenceStatus
from aegora_isle.kpi.kpi_logger import CSV_COLUMNS


# --------------------------------------------------------------- compute_seed


@pytest.mark.parametrize(
    "governance,scenario,replication,seed_offset,expected",
    [
        ("G1", "A", 0, 0, 0),
        ("G1", "A", 1, 0, 1),
        ("G2", "A", 0, 0, 1000),
        ("G1", "B", 0, 0, 100),
        ("G1", "C", 0, 0, 200),
        ("G1", "D", 0, 0, 300),
        ("G3", "D", 5, 0, 2305),
        ("G1", "A", 0, 50, 50),
    ],
)
def test_compute_seed_matches_formula(governance, scenario, replication, seed_offset, expected) -> None:
    assert compute_seed(governance, scenario, replication, seed_offset) == expected


def test_compute_seed_no_collision_across_full_replication_range() -> None:
    """Regression guard for the pre-launch seed-collision fix (see
    DECISIONS.md "Seed-collision bug fixed before the real 360-run
    launch"): the old formula collided compute_seed("G1","A",11,0) with
    compute_seed("G1","B",1,0) (and 58 other pairs per governance
    condition) once replication >= 10. Every (governance, scenario,
    replication) combination the real n_reps=30 batch will actually use
    must now produce a distinct seed."""
    seeds = {}
    for governance in GOVERNANCE_ORDER:
        for scenario in SCENARIO_ORDER:
            for replication in range(1, 31):  # matches the real launch's n_reps=30
                seed = compute_seed(governance, scenario, replication, seed_offset=0)
                key = (governance, scenario, replication)
                assert seed not in seeds, f"{key} collides with {seeds.get(seed)} at seed={seed}"
                seeds[seed] = key


def test_compute_seed_old_formula_collision_no_longer_reproduced() -> None:
    """The exact collision the bug produced must no longer occur."""
    assert compute_seed("G1", "B", 1, 0) != compute_seed("G1", "A", 11, 0)


def test_compute_seed_deterministic() -> None:
    a = compute_seed("G2", "C", 7, 3)
    b = compute_seed("G2", "C", 7, 3)
    assert a == b


# ----------------------------------------------------------------- build_cells


def test_build_cells_all_all_count_and_order() -> None:
    cells = build_cells("all", "all", n_reps=2)
    assert len(cells) == len(GOVERNANCE_ORDER) * len(SCENARIO_ORDER) * 2
    # governance-major, then scenario, then replication (1-indexed)
    assert cells[0] == {"governance": "G1", "scenario": "A", "replication": 1}
    assert cells[1] == {"governance": "G1", "scenario": "A", "replication": 2}
    assert cells[2] == {"governance": "G1", "scenario": "B", "replication": 1}
    assert cells[2 * 4] == {"governance": "G2", "scenario": "A", "replication": 1}


def test_build_cells_single_governance_all_scenarios() -> None:
    cells = build_cells("G2", "all", n_reps=3)
    assert len(cells) == len(SCENARIO_ORDER) * 3
    assert all(c["governance"] == "G2" for c in cells)


def test_build_cells_single_scenario_all_governance() -> None:
    cells = build_cells("all", "D", n_reps=1)
    assert len(cells) == len(GOVERNANCE_ORDER)
    assert all(c["scenario"] == "D" for c in cells)


def test_build_cells_single_cell() -> None:
    cells = build_cells("G3", "C", n_reps=5)
    assert len(cells) == 5
    assert [c["replication"] for c in cells] == [1, 2, 3, 4, 5]


# -------------------------------------------------------------- run_single_cell


def test_run_single_cell_smoke(tmp_path) -> None:
    result = run_single_cell(
        governance="G1",
        scenario="A",
        replication=1,
        n_agents=20,
        max_timesteps=300,
        output_dir=tmp_path,
        track_convergence=False,
    )
    assert result["status"] == "SKIPPED_CONVERGENCE_CHECK"
    assert result["error"] is None
    assert result["is_retry"] is False

    csv_path = tmp_path / "runs" / "G1" / "G1_A_001.csv"
    assert csv_path.exists()
    import pandas as pd

    df = pd.read_csv(csv_path)
    assert list(df.columns) == CSV_COLUMNS

    json_path = csv_path.with_suffix(".json")
    assert json_path.exists()
    assert result["json_path"] == str(json_path)


def test_run_single_cell_seed_override_used_verbatim(tmp_path) -> None:
    result = run_single_cell(
        governance="G1",
        scenario="A",
        replication=1,
        n_agents=20,
        max_timesteps=100,
        output_dir=tmp_path,
        track_convergence=False,
        seed_override=999,
    )
    assert result["seed"] == 999
    assert result["is_retry"] is True


def test_run_single_cell_torch_threads_pins_and_does_not_crash(tmp_path) -> None:
    """Sprint 3.4 fix: --n_jobs > 1 pins each worker to 1 torch thread to
    avoid CPU oversubscription across concurrent worker processes."""
    import torch

    original = torch.get_num_threads()
    try:
        result = run_single_cell(
            governance="G1",
            scenario="A",
            replication=1,
            n_agents=20,
            max_timesteps=100,
            output_dir=tmp_path,
            track_convergence=False,
            torch_threads=1,
        )
        assert result["status"] == "SKIPPED_CONVERGENCE_CHECK"
        assert result["error"] is None
        assert torch.get_num_threads() == 1
    finally:
        torch.set_num_threads(original)


def test_run_single_cell_determinism(tmp_path) -> None:
    kwargs = dict(
        governance="G2",
        scenario="A",
        replication=1,
        n_agents=20,
        max_timesteps=200,
        track_convergence=False,
    )
    r1 = run_single_cell(output_dir=tmp_path / "run1", **kwargs)
    r2 = run_single_cell(output_dir=tmp_path / "run2", **kwargs)

    assert r1["status"] == "SKIPPED_CONVERGENCE_CHECK"
    assert r2["status"] == "SKIPPED_CONVERGENCE_CHECK"

    bytes1 = (tmp_path / "run1" / "runs" / "G2" / "G2_A_001.csv").read_bytes()
    bytes2 = (tmp_path / "run2" / "runs" / "G2" / "G2_A_001.csv").read_bytes()
    assert bytes1 == bytes2


def test_run_single_cell_writes_real_agent_type_distribution(tmp_path) -> None:
    import json

    run_single_cell(
        governance="G1",
        scenario="A",
        replication=1,
        n_agents=20,
        max_timesteps=100,
        output_dir=tmp_path,
        track_convergence=False,
    )
    json_path = tmp_path / "runs" / "G1" / "G1_A_001.json"
    state = json.loads(json_path.read_text())
    dist = state["agent_type_distribution"]
    assert set(dist.keys()) == {"extractor", "producer", "trader", "governor"}
    assert pytest.approx(sum(dist.values()), abs=1e-9) == 1.0


def test_run_single_cell_error_does_not_raise(tmp_path) -> None:
    """An invalid n_agents (below the env's minimum) must surface as an
    ERROR result, not propagate an exception -- one crashed cell must not
    be able to abort a joblib.Parallel batch."""
    result = run_single_cell(
        governance="G1",
        scenario="A",
        replication=1,
        n_agents=1,  # AegoraEnv requires n_agents >= 4
        max_timesteps=100,
        output_dir=tmp_path,
        track_convergence=False,
    )
    assert result["status"] == "ERROR"
    assert result["error"] is not None


# -------------------------------------------------------------------- retry path


def test_convergence_monitor_forces_non_converged() -> None:
    """Isolated retry-trigger check: a noisy loss trajectory classifies as
    NON_CONVERGED, mirroring test_convergence.py's own pattern -- no full
    env loop needed to test the classification logic run_single_cell uses."""
    monitor = ConvergenceMonitor(agent_types=("extractor", "producer", "trader"))
    rng = np.random.default_rng(1)
    for _ in range(50):
        monitor.record_epoch_loss("extractor", rng.uniform(0, 10))
        monitor.record_epoch_loss("producer", 0.5)
        monitor.record_epoch_loss("trader", 0.5)

    non_converged = [
        t
        for t in ("extractor", "producer", "trader")
        if monitor.check_convergence(t) is not ConvergenceStatus.CONVERGED
    ]
    assert non_converged == ["extractor"]


def test_run_single_cell_classification_treats_early_end_as_converging(tmp_path) -> None:
    """Regression test for the Sprint 3.3 pilot bug: a run that ends before
    the 50-epoch convergence-check minimum (e.g. an early resource-pool
    collapse in Scenario C/D) must classify as CONVERGING, not
    NON_CONVERGED -- it's a valid, keepable data point, not a training
    failure, and must not be routed into the exclude-and-retry loop.
    max_timesteps=300 (3 epochs) guarantees every agent type stays well
    under MIN_EPOCHS_BEFORE_CHECK=50 regardless of training dynamics."""
    result = run_single_cell(
        governance="G1",
        scenario="D",  # low abundance, high events -- collapses fast
        replication=1,
        n_agents=20,
        max_timesteps=300,
        output_dir=tmp_path,
        track_convergence=True,
    )
    assert result["status"] in ("CONVERGING", "ERROR")
    if result["status"] == "CONVERGING":
        assert result["non_converged_types"] == []
        assert result["converged"] is False


def test_convergence_status_filtering_excludes_converging_from_retry() -> None:
    """CONVERGING results must never be selected for Phase B retry --
    main() filters on status == 'NON_CONVERGED' specifically, which the
    classification fix ensures CONVERGING can never satisfy."""
    results = [
        {"governance": "G1", "scenario": "D", "replication": 1, "status": "CONVERGING"},
        {"governance": "G1", "scenario": "A", "replication": 1, "status": "NON_CONVERGED"},
        {"governance": "G1", "scenario": "A", "replication": 2, "status": "CONVERGED"},
    ]
    to_retry = [r for r in results if r["status"] == "NON_CONVERGED"]
    assert len(to_retry) == 1
    assert to_retry[0]["scenario"] == "A"
    assert to_retry[0]["replication"] == 1


def test_merge_retries_replaces_non_converged_with_retry_result() -> None:
    original_seed = compute_seed("G2", "A", 1, 0)
    results = [
        {"governance": "G2", "scenario": "A", "replication": 1, "status": "NON_CONVERGED", "seed": original_seed},
        {"governance": "G2", "scenario": "B", "replication": 1, "status": "CONVERGED", "seed": 110},
    ]
    retries = [
        {
            "governance": "G2",
            "scenario": "A",
            "replication": 1,
            "status": "CONVERGED",
            "seed": original_seed + RETRY_SEED_DELTA,
        }
    ]

    merged = _merge_retries(results, retries)

    assert len(merged) == 2
    a_result = next(r for r in merged if r["scenario"] == "A")
    b_result = next(r for r in merged if r["scenario"] == "B")
    assert a_result["status"] == "CONVERGED"
    assert a_result["seed"] == original_seed + RETRY_SEED_DELTA
    assert b_result is results[1]  # untouched, not part of the retry set


def test_merge_retries_no_retries_is_noop() -> None:
    results = [{"governance": "G1", "scenario": "A", "replication": 1, "status": "CONVERGED"}]
    assert _merge_retries(results, []) == results


# --------------------------------------- unbounded round-based retry (client reversal)


def test_round_seed_formula_generalizes_across_n_rounds() -> None:
    """seed(round=k) == compute_seed(...) + (k-1)*RETRY_SEED_DELTA for any
    k, generalizing the old fixed two-retry deltas to unbounded rounds."""
    base = compute_seed("G2", "B", 3, 0)
    for k in (1, 2, 3, 5, 10):
        seed_for_round_k = base + (k - 1) * RETRY_SEED_DELTA
        if k == 1:
            assert seed_for_round_k == base  # round 1 uses the base seed unmodified
        else:
            assert seed_for_round_k != base
    # distinct across rounds
    seeds = {base + (k - 1) * RETRY_SEED_DELTA for k in range(1, 11)}
    assert len(seeds) == 10


def test_merge_retries_preserves_extra_fields_like_attempt() -> None:
    """_merge_retries does whole-dict key-matched replacement -- an
    'attempt' field stamped before merging must survive untouched."""
    results = [{"governance": "G1", "scenario": "A", "replication": 1, "status": "NON_CONVERGED", "attempt": 1}]
    retries = [{"governance": "G1", "scenario": "A", "replication": 1, "status": "CONVERGED", "attempt": 2}]
    merged = _merge_retries(results, retries)
    assert merged[0]["attempt"] == 2
    assert merged[0]["status"] == "CONVERGED"


def test_needs_replacement_covers_non_converged_and_error() -> None:
    assert _needs_replacement({"status": "NON_CONVERGED"}) is True
    assert _needs_replacement({"status": "ERROR"}) is True
    assert _needs_replacement({"status": "CONVERGED"}) is False
    assert _needs_replacement({"status": "CONVERGING"}) is False


def test_build_cell_attempt_rollup_schema_and_aggregation() -> None:
    results = [
        {"governance": "G1", "scenario": "A", "attempt": 1, "status": "CONVERGED"},
        {"governance": "G1", "scenario": "A", "attempt": 1, "status": "CONVERGED"},
        {"governance": "G2", "scenario": "B", "attempt": 1, "status": "CONVERGED"},
        {"governance": "G2", "scenario": "B", "attempt": 4, "status": "CONVERGED"},
    ]
    rollup = {(row["governance"], row["scenario"]): row for row in _build_cell_attempt_rollup(results)}

    g1a = rollup[("G1", "A")]
    assert g1a["total_replications"] == 2
    assert g1a["total_attempts_across_cell"] == 2
    assert g1a["total_replacement_attempts_across_cell"] == 0
    assert g1a["max_attempts_for_a_single_replication"] == 1
    assert g1a["cap_exceeded_kept_count"] == 0

    g2b = rollup[("G2", "B")]
    assert g2b["total_replications"] == 2
    assert g2b["total_attempts_across_cell"] == 5
    assert g2b["total_replacement_attempts_across_cell"] == 3  # 5 attempts - 2 replications
    assert g2b["max_attempts_for_a_single_replication"] == 4
    assert g2b["mean_attempts_per_replication"] == pytest.approx(2.5)
    assert g2b["cap_exceeded_kept_count"] == 0


def test_build_cell_attempt_rollup_reports_cap_exceeded_kept_count() -> None:
    results = [
        {"governance": "G3", "scenario": "A", "attempt": 10, "status": "NON_CONVERGED_CAP_EXCEEDED_KEPT"},
        {"governance": "G3", "scenario": "A", "attempt": 3, "status": "CONVERGED"},
        {"governance": "G3", "scenario": "B", "attempt": 1, "status": "CONVERGED"},
    ]
    rollup = {(row["governance"], row["scenario"]): row for row in _build_cell_attempt_rollup(results)}
    assert rollup[("G3", "A")]["cap_exceeded_kept_count"] == 1
    assert rollup[("G3", "B")]["cap_exceeded_kept_count"] == 0


def test_write_attempts_summary_schema_and_row_count(tmp_path) -> None:
    results = [
        {"run_id": "G1_A_001", "governance": "G1", "scenario": "A", "replication": 1,
         "seed": 1, "attempt": 1, "status": "CONVERGED"},
        {"run_id": "G2_A_001", "governance": "G2", "scenario": "A", "replication": 1,
         "seed": 2101, "attempt": 7, "status": "CONVERGED"},
    ]
    path = _write_attempts_summary(results, tmp_path / "attempts_summary.csv")
    import csv

    with path.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2  # one row per replication, not per attempt
    assert set(rows[0].keys()) == {
        "run_id", "governance", "scenario", "replication", "seed", "attempts_used", "final_status",
    }
    row_by_id = {r["run_id"]: r for r in rows}
    assert row_by_id["G2_A_001"]["attempts_used"] == "7"
    # Under the reversed design, final_status is only ever CONVERGED/CONVERGING --
    # NON_CONVERGED never appears in the written dataset.
    assert row_by_id["G2_A_001"]["final_status"] == "CONVERGED"


def test_write_cell_attempt_summary_schema(tmp_path) -> None:
    results = [{"governance": "G1", "scenario": "A", "attempt": 1, "status": "CONVERGED"}]
    path = _write_cell_attempt_summary(results, tmp_path / "cell_attempt_summary.csv")
    import csv

    with path.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert set(rows[0].keys()) == {
        "governance", "scenario", "total_replications", "total_attempts_across_cell",
        "total_replacement_attempts_across_cell", "max_attempts_for_a_single_replication",
        "mean_attempts_per_replication", "cap_exceeded_kept_count",
    }


def test_run_convergence_rounds_resolves_within_a_few_rounds(monkeypatch) -> None:
    """Monkeypatched _run_batch: a cell that fails its first 2 attempts
    then succeeds on the 3rd must resolve without hitting the cap, and
    every attempt number must be stamped correctly."""
    import aegora_isle.experiments.run_experiment as run_experiment_module

    call_count = {"n": 0}

    def fake_run_batch(jobs, n_jobs, desc):
        call_count["n"] += 1
        results = []
        for job in jobs:
            status = "CONVERGED" if call_count["n"] >= 3 else "NON_CONVERGED"
            results.append({**job, "status": status, "non_converged_types": [] if status == "CONVERGED" else ["extractor"]})
        return results

    monkeypatch.setattr(run_experiment_module, "_run_batch", fake_run_batch)
    cells = [{"governance": "G2", "scenario": "A", "replication": 1}]
    results = run_experiment_module._run_convergence_rounds(
        cells, common={"seed": None}, seed_offset=0, n_jobs=1, round_cap=DEFAULT_ROUND_CAP
    )
    assert len(results) == 1
    assert results[0]["status"] == "CONVERGED"
    assert results[0]["attempt"] == 3


def test_run_convergence_rounds_raises_when_cap_exceeded(monkeypatch) -> None:
    """Non-exempt cells (not G3-A/G3-B) still abort loudly, exactly as
    before the client's partial-data fallback."""
    import aegora_isle.experiments.run_experiment as run_experiment_module

    def always_fails(jobs, n_jobs, desc):
        return [{**job, "status": "NON_CONVERGED", "non_converged_types": ["extractor"]} for job in jobs]

    monkeypatch.setattr(run_experiment_module, "_run_batch", always_fails)
    cells = [
        {"governance": "G2", "scenario": "A", "replication": 1},
        {"governance": "G1", "scenario": "B", "replication": 2},
    ]
    with pytest.raises(ConvergenceBudgetExceededError) as excinfo:
        run_experiment_module._run_convergence_rounds(
            cells, common={}, seed_offset=0, n_jobs=1, round_cap=2
        )
    message = str(excinfo.value)
    assert "(G2,A,1)" in message
    assert "(G1,B,2)" in message
    assert "2 rounds" in message


def test_run_convergence_rounds_g3_all_scenarios_kept_as_documented_limitation_on_cap_exceeded(monkeypatch) -> None:
    """The client's pre-decided fallback, originally G3-A/G3-B (2026-08-02
    email), extended to all four Learning-Based scenarios after the real
    batch showed the same instability in G3-C/G3-D (client follow-up
    after the abort): if any Learning-Based cell exhausts the round cap,
    keep the partial data instead of aborting."""
    import aegora_isle.experiments.run_experiment as run_experiment_module

    def always_fails(jobs, n_jobs, desc):
        return [{**job, "status": "NON_CONVERGED", "non_converged_types": ["extractor"]} for job in jobs]

    monkeypatch.setattr(run_experiment_module, "_run_batch", always_fails)
    cells = [
        {"governance": "G3", "scenario": "A", "replication": 1},
        {"governance": "G3", "scenario": "B", "replication": 2},
        {"governance": "G3", "scenario": "C", "replication": 3},
        {"governance": "G3", "scenario": "D", "replication": 4},
    ]
    results = run_experiment_module._run_convergence_rounds(
        cells, common={}, seed_offset=0, n_jobs=1, round_cap=2
    )
    assert len(results) == 4
    for r in results:
        assert r["status"] == "NON_CONVERGED_CAP_EXCEEDED_KEPT"
        assert r["attempt"] == 2  # exhausted exactly round_cap rounds


def test_run_convergence_rounds_mixed_exempt_and_non_exempt_still_raises(monkeypatch) -> None:
    """A non-exempt cell stuck alongside an exempt one must still abort
    the whole batch -- the exception doesn't get diluted into leniency for
    everything else."""
    import aegora_isle.experiments.run_experiment as run_experiment_module

    def always_fails(jobs, n_jobs, desc):
        return [{**job, "status": "NON_CONVERGED", "non_converged_types": ["extractor"]} for job in jobs]

    monkeypatch.setattr(run_experiment_module, "_run_batch", always_fails)
    cells = [
        {"governance": "G3", "scenario": "A", "replication": 1},  # exempt
        {"governance": "G2", "scenario": "A", "replication": 1},  # not exempt
    ]
    with pytest.raises(ConvergenceBudgetExceededError) as excinfo:
        run_experiment_module._run_convergence_rounds(
            cells, common={}, seed_offset=0, n_jobs=1, round_cap=2
        )
    message = str(excinfo.value)
    assert "(G2,A,1)" in message
    assert "(G3,A,1)" not in message  # exempt cell isn't named in the abort


def test_run_convergence_rounds_never_returns_non_converged_final_status(monkeypatch) -> None:
    """Documents the invariant the client is relying on for non-exempt
    cells: a batch that finishes (doesn't raise) must never contain a bare
    NON_CONVERGED result. (The one documented exception,
    NON_CONVERGED_CAP_EXCEEDED_KEPT for the pre-approved G3-A/G3-B
    fallback, is covered by its own dedicated test above -- this test uses
    G1, which is never exempt.)"""
    import aegora_isle.experiments.run_experiment as run_experiment_module

    call_count = {"n": 0}

    def fake_run_batch(jobs, n_jobs, desc):
        call_count["n"] += 1
        results = []
        for job in jobs:
            status = "CONVERGED" if call_count["n"] >= 2 else "NON_CONVERGED"
            results.append({**job, "status": status, "non_converged_types": []})
        return results

    monkeypatch.setattr(run_experiment_module, "_run_batch", fake_run_batch)
    cells = [{"governance": "G1", "scenario": "A", "replication": r} for r in range(1, 4)]
    results = run_experiment_module._run_convergence_rounds(
        cells, common={}, seed_offset=0, n_jobs=1, round_cap=DEFAULT_ROUND_CAP
    )
    assert all(r["status"] in ("CONVERGED", "CONVERGING") for r in results)


# ------------------------------------- authoritative-per-regime convergence check


def _flat_monitor(agent_types: tuple[str, ...], value: float = 0.5) -> ConvergenceMonitor:
    mon = ConvergenceMonitor(agent_types=agent_types)
    for _ in range(50):
        for t in agent_types:
            mon.record_epoch_loss(t, value)
    return mon


def _noisy_monitor(agent_types: tuple[str, ...], seed: int = 0) -> ConvergenceMonitor:
    mon = ConvergenceMonitor(agent_types=agent_types)
    rng = np.random.default_rng(seed)
    for _ in range(50):
        for t in agent_types:
            mon.record_epoch_loss(t, float(rng.uniform(0, 10)))
    return mon




# ------------------------------- single-criterion convergence (post-retirement)


def test_classify_convergence_single_criterion_all_types_flat_is_converged() -> None:
    """One criterion for every agent type in every regime: policy-loss
    variance. See DECISIONS.md "Convergence measurement -- two defects
    corrected"."""
    monitor = _flat_monitor(("extractor", "producer", "trader"), value=1.0)
    status, non_converged = _classify_convergence(monitor, ("extractor", "producer", "trader"))
    assert status == "CONVERGED"
    assert non_converged == []


def test_classify_convergence_flags_every_noisy_type_not_just_extractor() -> None:
    """No agent type is privileged or substituted any more -- whichever
    types are noisy are the ones reported."""
    monitor = _noisy_monitor(("extractor", "producer", "trader"), seed=3)
    status, non_converged = _classify_convergence(monitor, ("extractor", "producer", "trader"))
    assert status == "NON_CONVERGED"
    assert set(non_converged) == {"extractor", "producer", "trader"}


def test_classify_convergence_never_reports_a_kpi_pseudo_type() -> None:
    """Regression guard for the retired harvested-amount override: the
    synthetic 'extractor_kpi' label must never appear in a result again,
    under any regime."""
    monitor = _noisy_monitor(("extractor", "producer", "trader"), seed=4)
    _status, non_converged = _classify_convergence(monitor, ("extractor", "producer", "trader"))
    assert all(not t.endswith("_kpi") for t in non_converged)


def test_classify_convergence_takes_exactly_two_arguments() -> None:
    """The retired signature accepted (monitor, kpi_monitor, agent_types,
    extractor_signal). Locking the new 2-arg contract so a stale 4-arg
    call site fails loudly rather than silently mis-binding."""
    import inspect

    params = list(inspect.signature(_classify_convergence).parameters)
    assert params == ["monitor", "agent_types"]


def test_value_loss_diagnostic_never_changes_convergence_status(tmp_path) -> None:
    """The value-loss series is observational only. A run whose value loss
    is wildly unstable must still be judged solely on policy loss."""
    result = run_single_cell(
        governance="G3",
        scenario="A",
        replication=1,
        n_agents=20,
        max_timesteps=6000,
        output_dir=tmp_path,
        track_convergence=True,
    )
    assert result["error"] is None
    # Value loss is recorded...
    assert result["value_loss_variance"]
    assert all(k.endswith("_value") for k in result["value_loss_variance"])
    # ...but never appears in the gating decision.
    assert all(not t.endswith("_value") for t in result["non_converged_types"])


def test_write_loss_diagnostics_emits_a_row_per_agent_type(tmp_path) -> None:
    """Written for every run, not just failures -- a run passing
    comfortably vs by a hair is exactly what this artifact exposes."""
    import csv

    results = [
        {
            "run_id": "G1_A_001", "governance": "G1", "scenario": "A",
            "replication": 1, "status": "CONVERGED",
            "policy_loss_variance": {"extractor": 1e-5, "producer": 2e-5, "trader": 3e-5},
            "value_loss_variance": {
                "extractor_value": 678.6, "producer_value": 7.9, "trader_value": 0.0
            },
        }
    ]
    path = _write_loss_diagnostics(results, tmp_path / "value_loss_diagnostics.csv")
    with path.open() as f:
        rows = list(csv.DictReader(f))
    assert [r["agent_type"] for r in rows] == ["extractor", "producer", "trader"]
    assert float(rows[0]["policy_loss_variance"]) == pytest.approx(1e-5)
    assert float(rows[0]["value_loss_variance"]) == pytest.approx(678.6)
    assert all(r["final_status"] == "CONVERGED" for r in rows)
