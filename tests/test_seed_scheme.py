"""Seed scheme: no base or retry seed may collide with any other."""
from aegora_isle.experiments import run_experiment as rx


def test_no_seed_collisions_across_batch_and_retry_rounds():
    seen: dict[int, tuple] = {}
    for gov in rx.GOVERNANCE_ORDER:
        for sc in rx.SCENARIO_ORDER:
            for rep in range(1, 31):
                base = rx.compute_seed(gov, sc, rep, 0)
                for rnd in range(1, rx.DEFAULT_ROUND_CAP + 1):
                    seed = base + (rnd - 1) * rx.RETRY_SEED_DELTA
                    key = (gov, sc, rep, rnd)
                    assert seed not in seen, f"{key} collides with {seen[seed]}"
                    seen[seed] = key
    assert len(seen) == 3 * 4 * 30 * rx.DEFAULT_ROUND_CAP
