"""Donor-set ablation: default behaviour unchanged; 'none' disables the levy."""
import numpy as np
from aegora_isle.governance.rule_based import RuleBasedGovernor


class _A:
    def __init__(self, c): self.cash = c


def _gov(donors=None):
    kw = {} if donors is None else {"redistribution_donors": donors}
    return RuleBasedGovernor(extraction_cap=5, sanction_probability=0.5, sanction_magnitude=-1,
                             redistribution_period=1, redistribution_rate=0.1,
                             rng=np.random.default_rng(0), **kw)


def _agents():
    return {"extractor_0": _A(0.0), "producer_0": _A(10.0), "trader_0": _A(10.0)}


def test_default_taxes_traders():
    g, a = _gov(), _agents(); g._redistribute({"agents_by_id": a})
    assert a["trader_0"].cash == 9.0 and a["extractor_0"].cash == 2.0


def test_traders_exempt():
    g, a = _gov(("producer",)), _agents(); g._redistribute({"agents_by_id": a})
    assert a["trader_0"].cash == 10.0 and a["extractor_0"].cash == 1.0


def test_none_disables_levy():
    g, a = _gov(()), _agents(); g._redistribute({"agents_by_id": a})
    assert a["trader_0"].cash == 10.0 and a["producer_0"].cash == 10.0 and a["extractor_0"].cash == 0.0
