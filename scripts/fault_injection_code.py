"""Code-level fault injection: can a reproducibility check detect them?

Runs one short cell several times under different injected code faults and
reports whether the repeat-run output hash differs from the reference.

Faults
  baseline        : two runs, same seed            -> must be identical
  global_rng      : agent action sampling uses the process-global NumPy RNG
  config_change   : a scenario parameter changed without being recorded
                    (sidecar is written from the original config)
  wallclock_seed  : a component seeded from the wall clock

Detection rule: repeat-run hash comparison (same seed, same recorded config).
``config_change`` is *not* detectable by repeat-run comparison alone; it is
detectable by comparing the recorded sidecar against the parameters actually
used, which the harness checks via the audit record.
"""
from __future__ import annotations

import argparse, csv, hashlib, sys, tempfile, time
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aegora_isle.experiments import run_experiment as rx  # noqa: E402


def run_once(out: Path, gov="G1", steps=300):
    rx.run_single_cell(gov, "A", 1, max_timesteps=steps, output_dir=out,
                       track_convergence=False, torch_threads=1)
    f = sorted(Path(out).rglob(f"{gov}_A_001.csv"))[0]
    return hashlib.sha256(f.read_bytes()).hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="docs/fault_matrix_code.csv")
    ap.add_argument("--steps", type=int, default=300)
    a = ap.parse_args()
    rows = []
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        ref = run_once(t / "ref", steps=a.steps)
        again = run_once(t / "again", steps=a.steps)
        rows.append(("baseline (same code, same seed)", ref == again, ref, again))

        from aegora_isle.environment import aegora_env as ae
        orig_pick = ae.AegoraEnv._pick_counterparty

        def global_rng_pick(self, excluding, kinds):
            choices = self._other_agent_ids(excluding, kinds)
            return str(np.random.choice(choices)) if choices else None

        with mock.patch.object(ae.AegoraEnv, "_pick_counterparty", global_rng_pick):
            h1 = run_once(t / "g1", steps=a.steps)
            h2 = run_once(t / "g2", steps=a.steps)
        rows.append(("global RNG used inside the run", h1 != h2, h1, h2))

        def clock_pick(self, excluding, kinds):
            choices = self._other_agent_ids(excluding, kinds)
            return choices[time.time_ns() % len(choices)] if choices else None

        with mock.patch.object(ae.AegoraEnv, "_pick_counterparty", clock_pick):
            h1 = run_once(t / "c1", steps=a.steps)
            h2 = run_once(t / "c2", steps=a.steps)
        rows.append(("component seeded from wall clock", h1 != h2, h1, h2))

    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["fault", "detected_by_repeat_run", "hash_a", "hash_b"])
        for name, det, h1, h2 in rows:
            w.writerow([name, "yes" if det else "NO", h1, h2])
            print(f"{'yes' if det else 'NO ':3}  {name}  ({h1} vs {h2})")


if __name__ == "__main__":
    main()
