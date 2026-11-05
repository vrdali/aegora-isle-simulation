"""G1 can be built with the G2/G3 population (inert Governor slots)."""
import numpy as np
from aegora_isle.environment.aegora_env import AegoraEnv
from aegora_isle.experiments.run_experiment import CONFIGS_DIR, GOVERNANCE_CONFIG_FILES
from aegora_isle.governance.governance_module import GovernanceModule


def _counts(regime, placeholders):
    gm = GovernanceModule(str(CONFIGS_DIR / GOVERNANCE_CONFIG_FILES[regime]), np.random.default_rng(1))
    env = AegoraEnv(n_agents=50, governance_module=gm, seed=1, inert_governor_slots=placeholders)
    env.reset()
    return env._agent_counts()


def test_g1_default_is_unchanged():
    assert _counts("G1", False) == {"extractor": 17, "producer": 17, "trader": 16, "governor": 0}


def test_g1_placeholders_match_g2_g3_population():
    assert _counts("G1", True) == _counts("G2", False) == _counts("G3", False)
    assert _counts("G1", True)["governor"] == 12
