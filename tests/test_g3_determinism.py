"""Two G3 runs from the same seed must be byte-identical, across SAC
warm-up (learning_starts steps) and the first learning steps. Before the
seed-path fix, SAC's warm-up draws and torch state were unseeded."""
import hashlib

from aegora_isle.experiments.run_experiment import run_single_cell


def _digest(out_dir):
    r = run_single_cell("G3", "A", 1, max_timesteps=1100, output_dir=str(out_dir),
                        track_convergence=False)
    assert r["status"] != "ERROR", r["error"]
    return hashlib.sha256(open(r["csv_path"], "rb").read()).hexdigest()


def test_g3_repeat_run_is_identical(tmp_path):
    assert _digest(tmp_path / "a") == _digest(tmp_path / "b")
