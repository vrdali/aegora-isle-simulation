# Aegora Isle — Architecture and Developer Documentation

**Version 1.0 — 2026-08-12**

A multi-agent reinforcement-learning simulation comparing three governance
regimes for managing a shared, depletable resource. This document describes
how the system is built, what every file contains, how each agent and regime
behaves, every parameter that can be changed, and the design decisions and
limitations a maintainer needs to know about.

The authoritative behavioural record is the code. Where this document and
the code disagree, the code is correct and this document is a bug.

---

# 1. System architecture

## 1.1 Design principles

Four constraints shaped the structure, and most of the non-obvious choices
follow from them.

**The codebase is a scientific instrument, not a product.** Reproducibility
and schema fidelity outrank elegance. A run must be reproducible from its
recorded seed, byte for byte.

**Regime-dependent behaviour lives in exactly one place.** No module outside
`aegora_isle/governance/` branches on whether the regime is G1, G2 or G3. The
same environment code runs all three unmodified. This is what makes the
comparison valid: nothing but the governance layer differs between arms.

**No module creates its own randomness.** Every stochastic component receives
a `numpy.random.Generator` at construction. A module that called
`np.random.default_rng()` internally would silently break reproducibility.

**Configuration is data, not code.** Scenario and governance parameters come
from YAML. The environment contains structural constants only.

## 1.2 Layers

```
                    run_experiment.py
                 (factorial batch runner)
                            |
              +-------------+-------------+
              |                           |
        train_agents.py              KPILogger
     (SB3 rollout collector)     (KPIs -> CSV + JSON)
              |
          AegoraEnv  ................  GovernanceModule
       (PettingZoo AEC)                (G1 / G2 / G3)
              |
      +-------+--------+
      |       |        |
 ResourcePool Market  Agents
                    (Extractor, Producer,
                     Trader, Governor)

        data/runs/*.csv + *.json
                    |
              analysis/  ---->  tables, figures, PDF report
              dashboard/ ---->  live Streamlit UI
```

## 1.3 Modules

| Module | Responsibility |
|---|---|
| `environment/aegora_env.py` | The PettingZoo AEC environment. Owns the agent registry, the turn cycle, action application, and KPI state. The largest module and the system's hub. |
| `environment/resource_pool.py` | The shared lagoon: stock, logistic regeneration, and the three disruption event types. |
| `environment/market.py` | Scarcity-priced peer-to-peer exchange of produced goods (`food`, `tool`) and of raw material. |
| `agents/base.py` | `AgentBase`: inventory, cash, age, wealth, and the abstract observation/action/reward interface. |
| `agents/{extractor,producer,trader,governor}.py` | One class per agent type; each defines its own spaces, observation, and reward. |
| `agents/action_codec.py` | Flattens the agents' `Dict` action spaces into the flat `Box` that Stable-Baselines3 requires, and decodes back. |
| `governance/governance_module.py` | The single entry point for regime behaviour. Loads YAML, selects a handler, dispatches hooks. |
| `governance/rule_based.py` | G2: extraction caps, probabilistic sanctions, scheduled redistribution. |
| `governance/adaptive_rl.py` | G3: a SAC-trained Governor that sets the cap and redistribution rate each step. |
| `kpi/kpi_logger.py` | All five KPI formulas, the rolling-average pass, and the CSV/JSON writers. |
| `kpi/convergence.py` | The pre-registered convergence criterion. |
| `experiments/train_agents.py` | The manual SB3 rollout collector that bridges PettingZoo AEC to PPO/SAC. |
| `experiments/run_experiment.py` | The factorial batch runner: seeding, parallel dispatch, exclusion/replacement protocol, summary artifacts. |
| `experiments/normalisation.py` | Running mean/standard-deviation normalisation (Welford) for the G3 reward components. |
| `analysis/` | The statistical pipeline: run index, survival encoding, tests, figures, PDF report. |
| `dashboard/` | The Streamlit observability UI. |
| `scripts/` | Dataset packaging, download verification, and Markdown-to-PDF rendering. |

## 1.4 The turn cycle

The environment implements the PettingZoo **AEC** (Agent–Environment–Cycle)
API, in which agents act one at a time in a fixed order, not simultaneously.
This is a specification requirement, not a preference; the Parallel API is
never used.

One full cycle — every agent taking one turn — advances the simulation by one
timestep:

1. Each agent in turn receives an observation and returns an action.
2. `GovernanceModule.on_action` may modify that action and return a reward
   adjustment. This is where G2's cap clipping and sanctions apply.
3. The environment applies the (possibly modified) action.
4. When the last agent has acted, the environment ticks the resource pool
   (regeneration, then events), updates live KPIs, and calls
   `GovernanceModule.on_step`, which is where redistribution happens.

## 1.5 The market mechanism

Prices are global and scarcity-linked. There is no order book and no
per-agent price discovery:

```
price(item) = base_price[item]
              * (1 + alpha * (1 - stock / initial_stock))
              * volatility_multiplier
```

The middle term rises from 1.0 with a full lagoon to `1 + alpha` with an
empty one, so scarcity in the raw resource propagates into the price of
manufactured goods. The volatility multiplier is 1.0 except during a market
volatility event.

`Market.execute_trade` validates both sides — sufficient cash, sufficient
inventory, positive quantity — and returns `None` for any invalid trade
rather than raising. Callers treat `None` as "no trade occurred". This is
why the `_apply_*` methods contain no defensive error handling of their own.

**The single global price is the origin of the Trader wealth-neutrality
limitation** described in section 5.1.

## 1.6 The analysis pipeline

`analysis/` is strictly read-only with respect to `data/`.

1. `data_loading.load_run_index` streams all 360 replications and returns one
   row per run: governance, scenario, volatility, seed, per-KPI means, and
   the survival fields `duration` and `event_observed`. It never holds the
   full 3.6-million-row dataset in memory.
2. `statistical_analysis.py` produces the specification's required outputs:
   descriptive statistics, one-way ANOVA per KPI with Levene's test and
   automatic Welch substitution, Tukey HSD, two-way ANOVA with interaction,
   Kaplan–Meier with multivariate log-rank, Cox proportional hazards, and
   OLS regression.
3. `plots.py` renders every figure at 300 DPI.
4. `report.py` assembles the paper-ready PDF.

---

# 2. Data schema

## 2.1 Input files

### `aegora_isle/experiments/scenarios.yaml`

Four scenarios in a 2x2 design: abundance (high in A/B, low in C/D) crossed
with event frequency (low in A/C, high in B/D).

| Key | Meaning |
|---|---|
| `initial_stock` | Starting resource units in the lagoon |
| `regeneration_rate` | Logistic regrowth rate per timestep |
| `event_rate` | Disruption events per 100 timesteps |
| `critical_min` | Stock below which the pool counts as collapsed |
| `scarcity_alpha` | Strength of the scarcity term in the price formula |
| `base_prices` | Base prices for `food`, `tool`, and `raw` |

### `aegora_isle/governance/configs/*.yaml`

One file per regime. `g1_ungoverned.yaml` carries only `regime: G1` — it
takes no parameters and exists so that G1 flows through the same dispatcher
as the others rather than being special-cased. The G2 and G3 files are
listed in full in section 4.

## 2.2 Output: the run CSV

`data/runs/{G1,G2,G3}/{run_id}.csv` — one row per timestep, up to 10,000
rows. **This schema is frozen.** The analysis scripts depend on these exact
names; renaming or reordering a column breaks them.

| Column | Type | Meaning |
|---|---|---|
| `timestep` | int | 1-indexed simulation step |
| `gini_coefficient` | float | Gini over agent wealth, 50-step rolling average |
| `hdi_proxy` | float | Composite development index, 0–1 |
| `gdp_equivalent` | float | `gdp_raw` indexed to the epoch-0 mean |
| `depletion_rate` | float | Per-step percentage depletion, rolling average |
| `resource_stock` | float | Lagoon stock at this step |
| `collapse_event` | bool | Subsistence-collapse flag (see section 5.3) |
| `governance_condition` | str | `G1`, `G2`, or `G3` |
| `scenario` | str | `A`, `B`, `C`, or `D` |
| `run_id` | str | `{governance}_{scenario}_{replication:03d}` |
| `random_seed` | int | The seed this run was generated from |
| `n_agents` | int | Agent count |
| `gini_raw` | float | Gini before rolling-average smoothing |
| `gdp_raw` | float | Trade + production value this cycle, unindexed |

Two columns ship in both raw and smoothed form because the smoothing is a
display convenience; the raw series is what any re-analysis should use.

### KPI formulas

**Gini** — Lorenz-curve area over agent wealth:

```
gini = (n + 1 - 2 * sum(cumsum(sorted_wealth)) / sum(wealth)) / n
```

Returns 0.0 for an empty population, a single agent, or all-zero wealth.

**Depletion rate** — percentage, positive means depletion:

```
depletion = (stock_{t-1} - stock_t) / stock_{t-1} * 100
```

**HDI proxy** — linear composite with specification weights:

```
hdi = 0.4 * resource_access
    + 0.4 * capability_utilisation
    + 0.2 * agent_longevity
```

where `resource_access` is mean wealth relative to the run's starting mean,
`capability_utilisation` is actual production value against the theoretical
maximum for the cycle, and `agent_longevity` is mean agent age over elapsed
time — all clipped to `[0, 1]`. The specification names these three terms
but does not define them formulaically; how each was operationalised is
recorded in `DECISIONS.md`.

**GDP-equivalent** — `gdp_raw` divided by the mean `gdp_raw` over epoch 0,
so every run is indexed to its own opening output and runs are comparable
across scenarios with different price levels.

## 2.3 Output: the JSON sidecar

`data/runs/{G1,G2,G3}/{run_id}.json` — the reproducibility record. A CSV
without its sidecar is treated as a half-written run and is ignored by both
the dashboard and the analysis.

| Key | Contents |
|---|---|
| `run_id`, `governance_condition`, `scenario`, `n_agents` | Run identity |
| `random_seed` | The seed — everything else follows from it |
| `governance_params` | The resolved governance configuration |
| `scenario_params` | The resolved scenario configuration |
| `agent_type_distribution` | Proportions by agent type |
| `rl_hyperparams` | The RL settings actually used |
| `collapse_occurred`, `collapse_timestep` | Subsistence-collapse record |
| `final_kpi_means` | End-of-run KPI summary |
| `csv_checksum` | SHA-256 of the paired CSV |

## 2.4 Output: batch-level files

Written at the root of the output directory, one set per batch.

| File | Contents |
|---|---|
| `attempts_summary.csv` | One row per replication: `run_id`, governance, scenario, replication, seed, `attempts_used`, `final_status` |
| `cell_attempt_summary.csv` | Per-cell rollup of replacement attempts |
| `convergence_failures.csv` | Audit log of every non-converging attempt, including replaced ones |
| `value_loss_diagnostics.csv` | Per-run value-function loss variance. **Observational only** — it never gated any decision |
| `dataset_manifest.csv` | One row per replication: SHA-256 of both files, row count, seed, collapse timestep. Makes a download verifiable without unpacking it |

## 2.5 Analysis outputs

Everything under `analysis/outputs/` is regenerable from `data/` and is
therefore not tracked in version control: eight tables as CSV, thirteen
figures at 300 DPI, and the report PDF.

---

# 3. Simulation logic

## 3.1 Common agent state

Every agent carries an inventory (`raw`, `food`, `tool`), a cash balance, and
an age. Wealth — the quantity the Gini coefficient is computed over — is cash
plus inventory valued at current prices, with raw material valued at a flat
internal price of 1.0.

Agents do not die. There is no mortality mechanic, which is the source of the
`agent_longevity` limitation in section 5.6.

## 3.2 Extractor

**Role.** Harvests raw resource from the lagoon.

**Actions.** `action_type` (extract / idle / move_to_zone), `amount` as a
fraction of `MAX_EXTRACT_AMOUNT`, and `target_zone`.

**Observation** (6 values). Own raw, food, tool, cash; normalised zone stock;
last depletion penalty.

**Reward.** Raw units actually extracted this step — after any governance cap
has been applied, so a clipped extraction earns the clipped amount.

**Passive behaviour.** Each cycle an Extractor spends
`EXTRACTOR_SPEND_FRACTION` (15%) of its cash on food or tool from a randomly
chosen Producer or Trader. This is not a learned action and does not appear
in the action space; it exists to recirculate cash. Without it, cash flows
one-way from Producers to Extractors and the economy freezes within roughly
100 steps (see section 5.5).

## 3.3 Producer

**Role.** Transforms raw material into finished goods.

**Actions.** `action_type` (produce / store / trade_offer), `produce_type`,
`offer_item`, `offer_qty`.

**Observation** (7 values). Own raw, food, tool, cash; food and tool prices;
remaining production capacity.

**Reward.** Total value of goods manufactured this step at current prices.

**Passive behaviour.** A Producer that lacks raw material automatically buys
it from a randomly chosen Extractor at the scarcity-adjusted raw price, with
partial fills allowed. Like the Extractor's spending, this is an
environment-level mechanic with no action-space representation.

## 3.4 Trader

**Role.** Buys and sells produced goods.

**Actions.** `action_type` (buy / sell / idle), `item`, `qty` as a fraction of
capital when buying or of inventory when selling.

**Observation** (10 values). Own food, tool, cash; food and tool prices; the
last five trade volumes.

**Reward.** Net trade profit this step.

**Known limitation.** Under a single global price, buying and selling are
wealth-neutral, so this reward is structurally near zero. See section 5.1.

## 3.5 Governor

**Role.** Present only under G3. Observes the whole economy and sets
governance parameters each step.

**Observation.** Aggregate economic state including a 10-bin wealth
histogram.

**Reward.** The composite defined in section 3.8. Under G1 and G2 no Governor
agent is instantiated.

## 3.6 G1 — No governance

The baseline. `G1Handler` is the identity function on actions and a no-op on
world ticks: no parameters, no randomness, no Governor agent.

It exists as a handler rather than as an `if` statement in the environment so
that all three regimes traverse the same dispatch path. The alternative —
special-casing G1 inside `AegoraEnv` — would mean the ungoverned arm ran
through different code from the governed arms, which would undermine the
comparison.

## 3.7 G2 — Rule-based governance

Three fixed mechanisms, all parameterised from `g2_rule_based.yaml`.

**Extraction cap.** On each Extractor extract action, the requested amount is
compared against `extraction_cap`. If it exceeds the cap, the action is
clipped to the cap. Non-Extractor agents and non-extract actions pass through
untouched.

**Probabilistic sanction.** When and only when the cap is exceeded, a sanction
of `sanction_magnitude` is applied to the agent's reward with probability
`sanction_probability`. The sanction is flat: it does not scale with the size
of the violation. This was verified against the specification, which defines
no scaling formula.

**Scheduled redistribution.** Every `redistribution_period` timesteps, each
Producer and Trader contributes `redistribution_rate` of its cash to a pool,
which is split equally among all Extractors. This is a cash transfer between
agents — see section 5.2.

## 3.8 G3 — Adaptive RL governance

A Soft Actor-Critic agent learns the governance policy during the run.

**Action space.** A 3-dimensional `Box(0, 1)`, mapped linearly onto the
configured ranges: extraction cap into `extraction_cap_range`,
redistribution rate into `redistribution_rate_range`, and sanction
magnitude. A wider action space that flattened the Governor's full `Dict`
was implemented and then reverted; the reasoning is in `DECISIONS.md`.

**Composite reward** (specification weights, verbatim):

```
reward = 0.40 * equality
       + 0.35 * sustainability
       + 0.25 * productivity
```

with

```
equality       = 1 - gini
sustainability = stock_norm * max(0, 1 - depletion_rate)
productivity   = log1p(gdp_indexed)
```

Each component is normalised by a running mean and standard deviation
(Welford) and clipped to `[-5, +5]` before weighting, so that components on
very different scales contribute comparably.

The `sustainability` formula is a deliberate, client-approved departure from
the specification's literal `- depletion_rate`; the multiplicative form keeps
the term bounded and non-negative. This is recorded in `DECISIONS.md` as
confirmed final.

Mechanically, G3 applies the same cap-and-redistribute machinery as G2 — the
difference is that the parameters move every step under policy control
rather than staying fixed.

## 3.9 Disruption events

Three event types, each rolled independently every timestep at
`event_rate / 100`:

| Event | Effect |
|---|---|
| Resource shock | An immediate reduction in lagoon stock |
| Market volatility | A price multiplier drawn from `[0.7, 1.5]`, lasting 10–50 steps |
| Agent withdrawal | An agent is removed from play for 10–50 steps. Governors are never targeted |

---

# 4. Configurable parameters

## 4.1 Scenario parameters

Identical across scenarios except where shown. All in `scenarios.yaml`.

| Parameter | A | B | C | D |
|---|---|---|---|---|
| `initial_stock` | 10000.0 | 10000.0 | 8000.0 | 8000.0 |
| `regeneration_rate` | 0.05 | 0.05 | 0.05 | 0.05 |
| `event_rate` | 0.5 | 3.0 | 0.5 | 3.0 |
| `critical_min` | 100.0 | 100.0 | 50.0 | 50.0 |
| `scarcity_alpha` | 0.5 | 0.5 | 0.5 | 0.5 |
| `base_prices.food` | 10.0 | 10.0 | 10.0 | 10.0 |
| `base_prices.tool` | 20.0 | 20.0 | 20.0 | 20.0 |
| `base_prices.raw` | 1.0 | 1.0 | 1.0 | 1.0 |

C and D were recalibrated before the confirmatory batch — at their original
values every regime collapsed almost immediately, which removed the
scenarios' ability to discriminate between them.

## 4.2 G2 governance parameters

| Parameter | Value | Meaning |
|---|---|---|
| `extraction_cap` | 3.0 | Maximum raw units per Extractor per step |
| `sanction_probability` | 0.85 | Probability a sanction fires on a cap violation |
| `sanction_magnitude` | -2.0 | Reward penalty when it fires |
| `redistribution_period` | 100 | Redistribute every N timesteps |
| `redistribution_rate` | 0.10 | Fraction of donor cash redistributed |

`extraction_cap` is 3.0 rather than the specification's default of 5.0
because 5.0 equals `MAX_EXTRACT_AMOUNT` and would never bind — the cap would
have no effect. Confirmed final by the client.

## 4.3 G3 governance parameters

| Parameter | Value |
|---|---|
| `reward_weights.equality` | 0.40 |
| `reward_weights.sustainability` | 0.35 |
| `reward_weights.productivity` | 0.25 |
| `reward_normalisation` | `running` |
| `reward_clip` | 5.0 |
| `extraction_cap_range` | `[0.5, 5.0]` |
| `redistribution_rate_range` | `[0.0, 0.3]` |
| `sanction_default_magnitude` | -2.0 |

SAC hyperparameters: learning rate `3.0e-4`, buffer 100000, batch 256,
gamma 0.99, tau 0.005, `ent_coef` auto, `train_freq` `[4, step]`, gradient
steps 1, `learning_starts` 1000, target update interval 1, network
`[256, 256]`.

## 4.4 Structural constants

These are simulation mechanics rather than experimental variables, so they
live in code, not YAML. Changing one changes the model itself.

| Constant | Value | Location |
|---|---|---|
| `MAX_EXTRACT_AMOUNT` | 5.0 | `environment/aegora_env.py` |
| `MAX_PRODUCE_AMOUNT` | 3.0 | `environment/aegora_env.py` |
| `EXTRACTOR_SPEND_FRACTION` | 0.15 | `environment/aegora_env.py` |
| `TRADER_SPREAD` | 0.0 | `environment/market.py` |
| `MAX_AGENTS` | 500 | `agents/base.py` |
| `N_ZONES` | 4 | `agents/base.py` |
| `ITEMS` | `("food", "tool")` | `agents/base.py` |
| `WEALTH_BINS` | 10 | `agents/governor.py` |
| `MARKET_VOLATILITY_MULTIPLIER_RANGE` | `(0.7, 1.5)` | `environment/resource_pool.py` |
| `MARKET_VOLATILITY_DURATION_RANGE` | `(10, 50)` | `environment/resource_pool.py` |
| `AGENT_WITHDRAWAL_DURATION_RANGE` | `(10, 50)` | `environment/resource_pool.py` |
| `ROLLING_WINDOW` | 50 | `kpi/kpi_logger.py` |
| `EPOCH_LENGTH` | 100 | `kpi/kpi_logger.py` |
| `DEFAULT_SUBSISTENCE_FRACTION` | 0.5 | `kpi/kpi_logger.py` |
| `DEFAULT_BOTTOM_FRACTION` | 0.25 | `kpi/kpi_logger.py` |
| `PPO_N_STEPS` | 128 | `experiments/run_experiment.py` |
| `RETRY_SEED_DELTA` | 1000 | `experiments/run_experiment.py` |
| `DEFAULT_ROUND_CAP` | 10 | `experiments/run_experiment.py` |
| `MIN_TIMESTEPS_FOR_CONVERGENCE_CHECK` | 5000 | `experiments/run_experiment.py` |

## 4.5 Convergence constants

| Constant | Value | Status |
|---|---|---|
| `CONVERGENCE_THRESHOLD` | 2.5e-4 | Recalibrated — see section 5.4 |
| `CONVERGENCE_WINDOW` | 20 | Pre-registered, unchanged |
| `MIN_EPOCHS_BEFORE_CHECK` | 50 | Pre-registered, unchanged |

A test (`tests/test_convergence.py::test_pre_registered_constants_locked`)
asserts these values, so an unannounced edit fails the suite.

## 4.6 Batch runner options

| Flag | Default | Meaning |
|---|---|---|
| `--governance` | `all` | `G1`, `G2`, `G3`, or `all` |
| `--scenario` | `all` | `A`, `B`, `C`, `D`, or `all` |
| `--n_reps` | 30 | Replications per cell |
| `--n_agents` | 50 | Agents per run |
| `--max_timesteps` | 10000 | Steps per replication; also the survival censoring point |
| `--seed_offset` | 0 | Shifts every seed, for an independent batch |
| `--output_dir` | `data/` | Where runs and summaries are written |
| `--n_jobs` | 1 | Parallel workers |

## 4.7 The seed formula

```
seed = seed_offset
     + governance_index * 1000
     + scenario_index   * 100
     + replication
```

The specification's literal formula used multipliers of 100 and 10, which
treats the replication as a single decimal digit. At `n_reps = 30` that
overflows into the scenario column: `compute_seed("G1", "A", 11, 0)` and
`compute_seed("G1", "B", 1, 0)` both give the same seed. This affected 60 of
120 (scenario, replication) pairs per regime — half the dataset would have
been seeded identically to a different, supposedly independent replication.
The multipliers were widened before the confirmatory batch, preserving the
formula's place-value structure with enough headroom for the real design.

---

# 5. Known limitations and design decisions

Consolidated here because each affects how results should be read. Full
reasoning for every item is in `DECISIONS.md`.

## 5.1 Traders cannot accumulate wealth

**What.** Under a single global price, a Trader buys and sells at the same
price, so intermediation is wealth-neutral by construction. Trader wealth is
pinned near its starting endowment and decays under volatility — measured at
10.00 falling to 3.41 over 1500 steps, while Producers grew by a factor of
2882 in the same period.

**Why it matters.** Roughly a quarter of the population holds almost nothing,
which inflates every reported Gini figure. Excluding Traders, Gini falls to
approximately 0.387 / 0.605 / 0.600 across the three regimes. **The ordering
is unchanged**, so the inequality conclusion is robust, but the absolute
values are not a clean measure of inequality among earning agents.

**Root cause.** The specification defines the Trader as "Market Exchange"
with reward "net trade profit" but never defines a mechanism by which
intermediation could be profitable.

**Status.** A standard market-maker bid-ask spread is implemented and fully
tested, controlled by `TRADER_SPREAD`, and **disabled by default (0.0)**.
Validated on a 144-run pilot at 0.05: Trader wealth rose 74-fold and
convergence was unaffected — but it also **reversed the sign of the H3
effect**, from +0.179 to -0.098 on the governance difference in GDP-
equivalent. A non-specification mechanism that flips a pre-registered
hypothesis is not one to adopt silently; that would be engineering the answer
rather than measuring it. It is left available behind one constant should the
Trader role need to be made economically real.

## 5.2 Redistribution is a cash transfer, not a resource replenishment

**What.** Under both G2 and G3, redistribution moves cash from Producers and
Traders to Extractors, split equally. It does not return anything to the
resource pool.

**Why.** One passage of the specification describes value being "transferred
to resource pool", while the client's design record and the confirmed
sign-off both specify agent-to-agent. This is an internal inconsistency in
the specification, resolved by explicit client decision in favour of the
cash transfer.

**Consequence.** Redistribution affects the wealth distribution — and hence
Gini — but never the sustainability of the lagoon. A pool-directed variant
would be a different experiment and was offered separately.

## 5.3 Two collapse definitions, which disagree

This is the most consequential item in this section.

The delivered CSVs carry a `collapse_event` column encoding a **subsistence**
definition: the bottom fraction of agents falling below a subsistence share
of initial mean wealth. The specification's Table 4 defines a different
**MTSC** event: resource stock below `critical_min`, **or** GDP at zero for
three consecutive steps, censored at 10,000 steps.

They point in opposite directions:

| Regime | Spec Table 4 (MTSC) | Subsistence column |
|---|---|---|
| G1 No-Governance | 70 of 120 (58.3%) | 20 |
| G2 Rule-Based | 5 of 120 (4.2%) | 118 |
| G3 Learning-Based | 6 of 120 (5.0%) | 120 |

Both are real and mutually consistent: **governance protects the resource
while concentrating wealth away from the poorest agents.** That single
mechanism explains the divergence, and it is the same mechanism behind the
inequality finding.

Note that the governed regimes are not at exactly zero under Table 4. Their
handful of events come from the *second* clause — GDP at zero for three
consecutive steps — not from pool death, which never occurs under
governance. Earlier interim summaries that reported "0%" had applied only the
pool-death clause.

**Which is used where.** The statistical analysis uses the specification's
Table 4 definition exclusively; no delivered table or figure is driven by the
subsistence column. The dashboard's Collapse Events view uses the subsistence
column, because that is what the run itself recorded, and states so on
screen. The subsistence column remains in the delivered data so the contrast
stays inspectable.

## 5.4 The convergence threshold was recalibrated

**What.** The criterion is a rolling variance of policy loss below a
threshold, over a 20-epoch window, after a minimum of 50 epochs. The
threshold was changed from the pre-registered 0.01 to 2.5e-4.

**Why.** Two measurement defects were found and corrected. The first recorded
Stable-Baselines3's *combined* loss (policy plus value plus entropy) where
the specification says *policy loss*; the value term is dominated by reward
unpredictability, which for Producers is irreducible. The second applied a
harvested-amount override that was structurally biased against G3, because
harvested amount is recorded post-cap and therefore tracks the Governor's
cap, which moves by design. With both corrected, the observed variances
cluster so tightly that 0.01 no longer discriminates between anything.

**Why it cannot have biased the results.** Convergence status is **never used
as a filter**. All 360 replications enter every analysis regardless of
status. The 65 runs recorded as `CONVERGING` are predominantly early
resource-pool collapses that ended before the 50-epoch minimum could be
evaluated; excluding them would have deleted precisely the events the
survival analysis exists to measure. Status is carried in
`attempts_summary.csv` for transparency, not applied as a selection rule.

**Final composition.** 295 `CONVERGED`, 65 `CONVERGING`, across 364 attempts
for 360 replications.

## 5.5 The Extractor cash sink

**What.** Extractors sell raw material to Producers but originally had no way
to spend the proceeds. Total system cash is conserved, but all of it migrated
to Extractors by roughly timestep 100 of 10,000, in every replication and
every regime. Once Producers and Traders reached zero cash they could never
trade again.

**Effect on the earlier dataset.** GDP-equivalent and HDI collapsed to near
zero for about 99% of every run. Gini and the survival measures were
unaffected.

**Fix.** `EXTRACTOR_SPEND_FRACTION`, a passive per-cycle purchase of food or
tool. The confirmatory dataset was regenerated after this fix; the earlier
dataset is superseded.

## 5.6 Agent longevity is near-constant

There is no agent-death mechanic, so every agent persists from `reset()`.
The `agent_longevity` term of the HDI proxy is therefore approximately 1.0
for nearly the entire run, and contributes little variance. This is a
documented limitation of the HDI proxy, not a defect.

## 5.7 Effect sizes are not Welch-adjusted

Levene's test fails for Gini, HDI proxy, and depletion rate, so Welch's ANOVA
is substituted for those. Partial eta-squared is computed from the
conventional sums of squares and is therefore **unaffected** by that
substitution — the Welch correction applies to the F-statistic, the
denominator degrees of freedom, and the p-value only. For the Gini analysis
the standard and Welch F-statistics differ by nearly a factor of two while
partial eta-squared is identical to six decimal places. This is standard
practice and is stated in the report.

## 5.8 Provenance verification

A replaced replication is written in place under the same `run_id`, so a
stale file is indistinguishable from a valid one by checksum alone — it is
internally consistent with itself. `scripts/verify_dataset.py` therefore
cross-references each run's recorded seed against `attempts_summary.csv` in
addition to verifying checksums. This caught exactly one such mismatch in the
confirmatory dataset, which was corrected by re-running that replication at
its recorded replacement seed.

---

# 6. Dependencies and environment

## 6.1 Runtime requirement

Python **3.10 – 3.12** (`>=3.10,<3.13`). Python 3.13 is excluded because not
all pinned scientific dependencies support it.

## 6.2 Declared dependencies

| Package | Minimum | Purpose |
|---|---|---|
| `pettingzoo` | 1.24.0 | Multi-agent environment API (AEC only) |
| `stable-baselines3` | 2.0.0 | PPO and SAC |
| `gymnasium` | 0.29.0 | Spaces |
| `numpy` | 1.24.0 | Numerics, seeded generators |
| `pandas` | 2.3.3 | Tabular data |
| `pyyaml` | 6.0 | Configuration |
| `matplotlib` | 3.7.0 | Figures |
| `seaborn` | 0.12.0 | Statistical plots |
| `joblib` | 1.5.3 | Parallel batch dispatch |
| `tqdm` | 4.69.0 | Progress reporting |
| `scipy` | 1.11.0 | Statistical primitives |
| `pingouin` | 0.5.4 | ANOVA, Levene, Welch, Tukey |
| `lifelines` | 0.27.0 | Kaplan–Meier, Cox PH |
| `statsmodels` | 0.14.0 | Two-way ANOVA, OLS |
| `reportlab` | 4.0.0 | PDF generation |
| `streamlit` | 1.30.0 | Dashboard |
| `psutil` | 5.9.0 | Cross-platform process control |

Development extra: `pytest`, `pytest-cov`, `ruff`.

## 6.3 Versions the confirmatory dataset was generated under

| Package | Version |
|---|---|
| Python | 3.10.19 (arm64, Darwin) |
| pettingzoo | 1.26.1 |
| stable-baselines3 | 2.9.0 |
| gymnasium | 1.3.0 |
| torch | 2.12.0 |
| numpy | 2.2.6 |
| pandas | 2.3.3 |
| scipy | 1.15.3 |
| pingouin | 0.6.1 |
| lifelines | 0.30.0 |
| statsmodels | 0.14.6 |
| seaborn | 0.13.2 |
| matplotlib | 3.10.9 |
| streamlit | 1.61.1 |

## 6.4 Environment assumptions

**Reproducibility is within-platform.** A run reproduces byte-identically on
the same platform with the same library versions. Across platforms or across
PyTorch versions, floating-point differences can accumulate; this is inherent
to neural-network training and not specific to this codebase. The recorded
seed plus the versions above are what make a run reproducible.

**Deployment target includes Windows.** Dashboard process control uses
`psutil` rather than POSIX signals for exactly this reason:
`Process.suspend()` maps to `SIGSTOP` on Unix and `NtSuspendProcess` on
Windows, so one code path covers both.

**Hardware.** A full 360-run batch wants 6 or more cores and 16 GB RAM to
finish within a working day. The dashboard and analysis run comfortably on
any modern laptop. No GPU is required or used.

**Linting is pinned to an explicit rule set** (`E4`, `E7`, `E9`, `F`). The dev
extra specifies `ruff>=0.1.0`, so a fresh install picks up whatever is
current, and ruff's default rule set has widened across releases — the same
unchanged codebase went from clean to several hundred findings on a version
bump alone. Stating the rules keeps `ruff check` reproducible.

## 6.5 Testing

335 tests. Every KPI has a known-input/known-output test. The reproducibility
test loads a JSON sidecar, re-runs, and requires a byte-identical CSV.

```
pytest
```

---

# Appendix — Where to look first

| To understand | Read |
|---|---|
| The turn cycle and action application | `environment/aegora_env.py` |
| How a regime changes behaviour | `governance/governance_module.py`, then the handler |
| A KPI formula | `kpi/kpi_logger.py` |
| Why something is the way it is | `DECISIONS.md` |
| Project rules and constraints | `CLAUDE.md` |
| How to run any of it | `README.md` |
| How to load the dataset | `DATA_SETUP.md` |
