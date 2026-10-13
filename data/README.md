# M3 Confirmatory Dataset — Data Dictionary

Covers `runs/{G1,G2,G3}/{run_id}.csv`, `runs/{G1,G2,G3}/{run_id}.json`,
`attempts_summary.csv`, `cell_attempt_summary.csv`, and
`convergence_failures.csv` from the 360-replication confirmatory batch
(3 governance conditions x 4 scenarios x 30 replications, 50 agents,
10,000 timesteps each).

## KNOWN LIMITATION — read before using `gdp_equivalent` or `hdi_proxy`

**`gdp_equivalent` and `hdi_proxy` are only meaningful for roughly the
first 1% of each run's timesteps (the first ~50-110 of 10,000 in the
replications checked).** Every agent starts with a fixed cash balance;
Extractor agents sell raw material for cash but have no mechanism to
spend cash back into the economy. By ~timestep 100, 100% of the
economy's cash has migrated to Extractors and stays there for the rest
of the run, in every replication and all three governance regimes. Once
Producers and Traders are out of cash they can never trade again, so
production and trade value (`gdp_raw`, and therefore `gdp_equivalent`)
drop to ~0 for the remaining ~99% of every run, and `hdi_proxy`'s
production sub-term (`capability_utilisation`) does the same, leaving
`hdi_proxy` pinned at ~0.6 almost everywhere.

This dataset (the one this file ships alongside) was collected **before**
this was found and fixed. A corrected re-run (fix: Extractors
automatically spend cash on food/tool, closing the loop — no action-space
or model changes) is in progress; a superseding dataset will be issued
once it completes and is verified. See
`client_correspondence/Client_Update_M3_Cash_Sink_Finding_v2.md` for the
full writeup and `DECISIONS.md` for the logged decision.

**Fields unaffected by this issue** (safe to use from this dataset as-is):
`gini_coefficient`, `gini_raw`, `resource_stock`, `depletion_rate`,
`collapse_event`, `timestep`, `governance_condition`, `scenario`,
`run_id`, `random_seed`, `n_agents`. These depend on wealth distribution
and extraction, not on trade continuing past the early window.

## CSV columns (`runs/{gov}/{run_id}.csv`)

| column | meaning |
|---|---|
| `timestep` | Simulation step, 1-10000 |
| `gini_coefficient` | 50-step rolling mean of `gini_raw` |
| `hdi_proxy` | Composite HDI proxy (0.4 resource_access + 0.4 capability_utilisation + 0.2 agent_longevity); logged every 100 steps, NaN elsewhere. **See limitation above.** |
| `gdp_equivalent` | `gdp_raw` normalised against the run's epoch-0 baseline. **See limitation above.** |
| `depletion_rate` | 50-step rolling mean of raw depletion rate |
| `resource_stock` | Current lagoon stock |
| `collapse_event` | True on the timestep subsistence-collapse is first detected, else False |
| `governance_condition` | G1 / G2 / G3 |
| `scenario` | A / B / C / D |
| `run_id` | `{governance}_{scenario}_{replication:03d}` |
| `random_seed` | Seed for this replication |
| `n_agents` | Agent count (50) |
| `gini_raw` | Unsmoothed per-step Gini coefficient |
| `gdp_raw` | Unsmoothed production + trade value this cycle |

## Other files

- `attempts_summary.csv` — one row per replication: seed, attempts used, final convergence status.
- `cell_attempt_summary.csv` — one row per (governance, scenario): replication counts, replacement-attempt stats, `cap_exceeded_kept_count`.
- `convergence_failures.csv` — audit log of every non-converging attempt, including replaced ones.
- `runs/{gov}/{run_id}.json` — per-replication metadata (seed, governance/scenario params, agent-type distribution, final KPI means, CSV checksum).
