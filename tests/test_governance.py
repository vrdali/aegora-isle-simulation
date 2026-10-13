"""Unit tests for aegora_isle.governance.governance_module.GovernanceModule.

Covers M2 Sprints 2.1/2.5 acceptance criteria:
- G1 loads cleanly and behaves as an identity pass-through (matches the
  M1 GovernanceStub semantics exactly).
- G2 loads cleanly and constructs a real RuleBasedGovernor from YAML.
- G3 loads cleanly and constructs a real AdaptiveRLGovernor from YAML,
  including the train_freq list->tuple conversion SB3's constructor needs.
- Seeding contract: rng is required and must be a numpy.random.Generator.
- Config validation: missing/unknown regime raises.
- Single entry point (CLAUDE.md): only governance_module.py may import
  rule_based or adaptive_rl anywhere in the codebase.
"""

from __future__ import annotations

import pathlib
import subprocess

import numpy as np
import pytest

from aegora_isle.governance.adaptive_rl import AdaptiveRLGovernor
from aegora_isle.governance.governance_module import G1Handler, GovernanceModule
from aegora_isle.governance.rule_based import RuleBasedGovernor


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
CONFIGS_DIR = REPO_ROOT / "aegora_isle" / "governance" / "configs"


def _config(name: str) -> str:
    return str(CONFIGS_DIR / name)


# ------------------------------------------------------------------------ G1


def test_g1_regime_loads_and_reports_g1() -> None:
    gm = GovernanceModule(_config("g1_ungoverned.yaml"), np.random.default_rng(0))
    assert gm.regime == "G1"


def test_g1_on_action_is_identity_zero_delta() -> None:
    gm = GovernanceModule(_config("g1_ungoverned.yaml"), np.random.default_rng(0))
    action = {"action_type": 0, "amount": np.array([0.9], dtype=np.float32), "target_zone": 0}
    mod_action, delta = gm.on_action("extractor_0", action)
    assert mod_action == action
    assert delta == 0.0


def test_g1_on_step_is_noop() -> None:
    gm = GovernanceModule(_config("g1_ungoverned.yaml"), np.random.default_rng(0))
    assert gm.on_step({"timestep": 100, "agents_by_id": {}}) is None


def test_g1_state_dict() -> None:
    gm = GovernanceModule(_config("g1_ungoverned.yaml"), np.random.default_rng(0))
    assert gm.state_dict() == {"regime": "G1", "kind": "ungoverned"}


def test_g1handler_usable_standalone_without_rng() -> None:
    """G1Handler has no randomness; AegoraEnv's zero-config default constructs
    it directly without an rng (see aegora_env.py)."""
    handler = G1Handler()
    assert handler.regime == "G1"
    action = {"action_type": 1}
    assert handler.on_action("trader_2", action) == (action, 0.0)


# ------------------------------------------------------------------------ G2


def test_g2_regime_loads_and_builds_rule_based_governor() -> None:
    gm = GovernanceModule(_config("g2_rule_based.yaml"), np.random.default_rng(0))
    assert gm.regime == "G2"
    assert isinstance(gm._handler, RuleBasedGovernor)  # type: ignore[attr-defined]


def test_g2_state_dict_reflects_config() -> None:
    gm = GovernanceModule(_config("g2_rule_based.yaml"), np.random.default_rng(0))
    sd = gm.state_dict()
    assert sd["regime"] == "G2"
    assert sd["extraction_cap"] == pytest.approx(3.0)
    assert sd["redistribution_period"] == 100


# ------------------------------------------------------------------------ G3


def test_g3_regime_loads_and_builds_adaptive_rl_governor() -> None:
    gm = GovernanceModule(_config("g3_adaptive_rl.yaml"), np.random.default_rng(0))
    assert gm.regime == "G3"
    assert isinstance(gm._handler, AdaptiveRLGovernor)  # type: ignore[attr-defined]


def test_g3_state_dict_reflects_config() -> None:
    gm = GovernanceModule(_config("g3_adaptive_rl.yaml"), np.random.default_rng(0))
    sd = gm.state_dict()
    assert sd["regime"] == "G3"
    assert sd["weights"] == pytest.approx({"equality": 0.40, "sustainability": 0.35, "productivity": 0.25})
    assert sd["model_built"] is False  # lazy — only built on first on_step


def test_g3_config_reward_weights_sum_to_one() -> None:
    """The schema is locked (G3_PLAN.md); confirm it's still internally
    consistent now that construction actually happens."""
    import yaml

    with open(_config("g3_adaptive_rl.yaml")) as f:
        cfg = yaml.safe_load(f)
    assert cfg["regime"] == "G3"
    assert sum(cfg["reward_weights"].values()) == pytest.approx(1.0)


# ---------------------------------------- has_stochastic_extractor_sanction


def test_g1_has_no_stochastic_extractor_sanction() -> None:
    """G1 has no cap/sanction mechanism at all -- read by the M3 experiment
    runner to keep loss-variance authoritative for the Extractor (see
    DECISIONS.md "Authoritative-per-regime convergence check")."""
    gm = GovernanceModule(_config("g1_ungoverned.yaml"), np.random.default_rng(0))
    assert gm.has_stochastic_extractor_sanction is False


def test_g2_has_stochastic_extractor_sanction() -> None:
    """G2's over-cap sanction is a coin-flip (sanction_probability),
    independent of policy quality."""
    gm = GovernanceModule(_config("g2_rule_based.yaml"), np.random.default_rng(0))
    assert gm.has_stochastic_extractor_sanction is True


def test_g3_has_stochastic_extractor_sanction() -> None:
    """G3's sanction probability is SAC-learned (dynamic), but still a
    coin-flip at the point it's applied."""
    gm = GovernanceModule(_config("g3_adaptive_rl.yaml"), np.random.default_rng(0))
    assert gm.has_stochastic_extractor_sanction is True


# -------------------------------------------------------------- validation


def test_seeding_contract_rejects_non_generator() -> None:
    with pytest.raises(TypeError):
        GovernanceModule(_config("g1_ungoverned.yaml"), rng=42)  # type: ignore[arg-type]


def test_unknown_regime_raises(tmp_path: pathlib.Path) -> None:
    bad_cfg = tmp_path / "bad.yaml"
    bad_cfg.write_text("regime: G99\n")
    with pytest.raises(ValueError, match="unknown regime"):
        GovernanceModule(str(bad_cfg), np.random.default_rng(0))


def test_missing_regime_key_raises(tmp_path: pathlib.Path) -> None:
    bad_cfg = tmp_path / "bad.yaml"
    bad_cfg.write_text("extraction_cap: 3.0\n")
    with pytest.raises(ValueError, match="missing required 'regime' key"):
        GovernanceModule(str(bad_cfg), np.random.default_rng(0))


# ----------------------------------------------------- single entry point


def test_single_entry_point_for_rule_based_and_adaptive_rl() -> None:
    """CLAUDE.md: only governance_module.py may import rule_based or
    adaptive_rl. Everything else (env, agents, other governance helpers)
    must go through GovernanceModule.
    """
    result = subprocess.run(
        [
            "grep",
            "-rnE",
            "--include=*.py",
            r"^\s*(from|import)\s+aegora_isle\.governance\.(rule_based|adaptive_rl)",
            str(REPO_ROOT / "aegora_isle"),
        ],
        capture_output=True,
        text=True,
    )
    offenders = [
        line
        for line in result.stdout.splitlines()
        if "governance_module.py" not in line
    ]
    assert offenders == [], "Single-entry-point violation:\n" + "\n".join(offenders)
