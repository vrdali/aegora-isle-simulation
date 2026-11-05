"""Indicator scope (Governor slots) and collapse burn-in."""
import numpy as np
import pytest

from aegora_isle.environment.aegora_env import AegoraEnv
from aegora_isle.experiments.run_experiment import CONFIGS_DIR, GOVERNANCE_CONFIG_FILES
from aegora_isle.governance.governance_module import GovernanceModule
from aegora_isle.kpi import kpi_logger as kl


def _env(scope: str) -> AegoraEnv:
    gm = GovernanceModule(str(CONFIGS_DIR / GOVERNANCE_CONFIG_FILES["G2"]), np.random.default_rng(1))
    env = AegoraEnv(n_agents=50, governance_module=gm, seed=1, indicator_scope=scope)
    env.reset()
    return env


def _prices(env):
    return {k: env.market.get_price(k) for k in env.market.base_prices}


def test_scope_all_keeps_governors_economic_drops_them():
    allw = _env("all").indicator_wealths(_prices(_env("all")))
    e = _env("economic")
    eco = e.indicator_wealths(_prices(e))
    n_gov = sum(1 for a in e._agents_by_id if a.startswith("governor"))
    assert n_gov == 12 and len(allw) == 50 and len(eco) == 38


def test_scope_excluding_traders_is_smaller():
    e = _env("economic_ex_trader")
    n = len(e.indicator_wealths(_prices(e)))
    traders = sum(1 for a in e._agents_by_id if a.startswith("trader"))
    assert n == 50 - 12 - traders


def test_invalid_scope_rejected():
    with pytest.raises(ValueError):
        AegoraEnv(n_agents=50, indicator_scope="nope")


def test_burn_in_suppresses_early_collapse_but_keeps_raw_timing():
    e = _env("all")
    lg = kl.KPILogger(e, run_id="G2_A_001", scenario="A", burn_in=5)
    for t in (3, 6):
        e._timestep = t
        lg._was_collapsed_last_row = False
        lg._initial_mean_wealth = 10**9   # forces the subsistence flag on
        row = lg.log_step()
        assert row["collapse_event"] is (t > 5)
    rec = lg.audit_record()
    assert rec["collapse_timestep_without_burn_in"] == 3 and rec["collapse_timestep"] == 6
