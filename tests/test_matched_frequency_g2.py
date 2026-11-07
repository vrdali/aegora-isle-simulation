"""Matched-frequency G2: period/rate overrides reach the rule-based governor."""
import numpy as np
from pathlib import Path
from aegora_isle.governance.governance_module import GovernanceModule

CFG = Path("aegora_isle/governance/configs/g2_rule_based.yaml")


def test_period_and_rate_override():
    gm = GovernanceModule(str(CFG), np.random.default_rng(1),
                          {"redistribution_period": 1, "redistribution_rate": 0.001053})
    h = gm._handler
    assert h.redistribution_period == 1
    assert abs(h.redistribution_rate - 0.001053) < 1e-12


def test_default_unchanged():
    gm = GovernanceModule(str(CFG), np.random.default_rng(1))
    assert gm._handler.redistribution_period == 100
    assert gm._handler.redistribution_rate == 0.10
