"""Run / Pause / Reset control for live experiment runs (spec §7.1).

Spec §7.1 requires the dashboard to drive ``run_experiment.py`` as a
subprocess. Pause and resume use ``psutil`` rather than raw POSIX signals
because the client intends to deploy on Windows: ``psutil.Process.suspend()``
maps to ``SIGSTOP`` on Unix and ``NtSuspendProcess`` on Windows, so one code
path covers both. See DECISIONS.md.

No Streamlit import here -- every function is pytest-testable with mocks.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psutil

# Mirrors run_experiment.build_arg_parser exactly. Kept as a tuple rather
# than hard-coded call sites so a flag rename surfaces here in one place
# instead of silently producing an invalid command line.
CLI_FLAGS: tuple[str, ...] = (
    "governance", "scenario", "n_reps", "n_agents",
    "max_timesteps", "seed_offset", "output_dir", "n_jobs",
)

TERMINATE_TIMEOUT_S: float = 5.0


@dataclass
class ProcessHandle:
    """A launched experiment subprocess."""

    pid: int
    argv: list[str] = field(default_factory=list)
    popen: Any | None = None


def build_command(args: dict[str, Any]) -> list[str]:
    """Builds the ``run_experiment`` argv from a dict of CLI options.

    Uses ``sys.executable -m`` rather than a bare script path so the
    subprocess inherits the same interpreter and virtualenv as the dashboard.
    Unknown keys are ignored rather than passed through, so a typo in the UI
    cannot produce an unparseable command.
    """
    cmd = [sys.executable, "-m", "aegora_isle.experiments.run_experiment"]
    for flag in CLI_FLAGS:
        if flag in args and args[flag] is not None:
            cmd.extend([f"--{flag}", str(args[flag])])
    return cmd


def launch_experiment(args: dict[str, Any], repo_root: str | Path) -> ProcessHandle:
    """Starts a batch in the background and returns its handle."""
    cmd = build_command(args)
    popen = subprocess.Popen(
        cmd, cwd=str(repo_root),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return ProcessHandle(pid=popen.pid, argv=cmd, popen=popen)


def _process(handle: ProcessHandle) -> psutil.Process | None:
    """The live ``psutil.Process``, or None if it has already exited.

    Every caller tolerates None: a user can press Pause after a run has
    finished on its own, and that should be a no-op rather than a crash.
    """
    try:
        return psutil.Process(handle.pid)
    except psutil.NoSuchProcess:
        return None


def is_running(handle: ProcessHandle | None) -> bool:
    if handle is None:
        return False
    proc = _process(handle)
    return bool(proc and proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE)


def pause(handle: ProcessHandle) -> bool:
    """Suspends the run and its workers. True if anything was suspended.

    Children are suspended first: ``run_experiment`` dispatches through
    joblib, so the workers do the actual computation. Suspending only the
    parent would leave them running and the "paused" state would be a lie.
    """
    proc = _process(handle)
    if proc is None:
        return False
    try:
        for child in proc.children(recursive=True):
            try:
                child.suspend()
            except psutil.NoSuchProcess:
                continue
        proc.suspend()
        return True
    except psutil.NoSuchProcess:
        return False


def resume(handle: ProcessHandle) -> bool:
    """Resumes the run and its workers. Parent first, mirroring pause."""
    proc = _process(handle)
    if proc is None:
        return False
    try:
        proc.resume()
        for child in proc.children(recursive=True):
            try:
                child.resume()
            except psutil.NoSuchProcess:
                continue
        return True
    except psutil.NoSuchProcess:
        return False


def reset(handle: ProcessHandle, timeout: float = TERMINATE_TIMEOUT_S) -> bool:
    """Stops the run entirely: terminate, then kill anything that survives.

    A suspended process cannot handle SIGTERM, so it is resumed first --
    otherwise pressing Pause then Reset would leave orphaned workers holding
    CPU. Children are terminated before the parent so joblib cannot respawn
    them.
    """
    proc = _process(handle)
    if proc is None:
        return False
    try:
        resume(handle)
        targets = proc.children(recursive=True) + [proc]
        for t in targets:
            try:
                t.terminate()
            except psutil.NoSuchProcess:
                continue
        _gone, alive = psutil.wait_procs(targets, timeout=timeout)
        for t in alive:
            try:
                t.kill()
            except psutil.NoSuchProcess:
                continue
        return True
    except psutil.NoSuchProcess:
        return False
