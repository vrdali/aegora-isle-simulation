"""Runs the v2 confirmatory batch (or a sample of it) and verifies the output.

One command, any platform. It records the platform and library versions next
to the data, runs the experiment with the review-driven settings, then runs
the provenance verifier on the result.

Full batch (360 runs)::

    python scripts/run_v2_batch.py --out data_v2 --n_jobs 6

Cross-platform sample (same seeds on a second machine)::

    python scripts/run_v2_batch.py --out data_v2_sample --n_reps 5 --n_jobs 4

Settings: Governor slots excluded from every population indicator
(``--indicator_scope economic``), collapse burn-in (``--burn_in``), and
per-agent wealth logging every 100 steps.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True)
        return out.stdout.strip() + (" (uncommitted changes)" if dirty.stdout.strip() else "")
    except OSError:
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data_v2")
    ap.add_argument("--n_reps", type=int, default=30)
    ap.add_argument("--n_jobs", type=int, default=4)
    ap.add_argument("--indicator_scope", default="economic",
                    choices=["all", "economic", "economic_ex_trader"])
    ap.add_argument("--burn_in", type=int, default=50,
                    help="Equals the 50-step KPI rolling window; the no-burn-in timing is also saved.")
    ap.add_argument("--wealth_log_interval", type=int, default=100)
    ap.add_argument("--g1_placeholders", action="store_true",
                    help="G1 uses the G2/G3 population (12 inert Governor slots).")
    ap.add_argument("--governance", default="all")
    ap.add_argument("--scenario", default="all")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import numpy, pandas, torch, stable_baselines3  # noqa: E401
    env_info = {
        "python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
        "processor": platform.processor(), "numpy": numpy.__version__, "pandas": pandas.__version__,
        "torch": torch.__version__, "stable_baselines3": stable_baselines3.__version__,
        "git_commit": git_commit(), "settings": vars(args),
    }
    (out / "environment.json").write_text(json.dumps(env_info, indent=2))
    print(json.dumps(env_info, indent=2))

    sys.path.insert(0, str(ROOT))
    from aegora_isle.experiments import run_experiment as rx
    code = rx.main([
        "--governance", args.governance, "--scenario", args.scenario,
        "--n_reps", str(args.n_reps), "--output_dir", str(out), "--n_jobs", str(args.n_jobs),
        "--indicator_scope", args.indicator_scope, "--burn_in", str(args.burn_in),
        "--wealth_log_interval", str(args.wealth_log_interval),
    ] + (["--g1_placeholders"] if args.g1_placeholders else []))
    if code != 0:
        return code

    subprocess.run([sys.executable, str(ROOT / "scripts" / "package_dataset.py"),
                    "--data_dir", str(out), "--out", str(out / "dist"), "--skip-archive"], cwd=ROOT)
    manifest = out / "dist" / "dataset_manifest.csv"
    if manifest.exists():
        return subprocess.run([sys.executable, str(ROOT / "scripts" / "verify_dataset.py"),
                               "--data_dir", str(out), "--manifest", str(manifest)], cwd=ROOT).returncode
    print("No manifest produced; verification could not run.")
    return 3


if __name__ == "__main__":
    sys.exit(main())
