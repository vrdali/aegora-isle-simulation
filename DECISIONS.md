# DECISIONS.md — Aegora Isle Engineering Decisions

Living record of design choices that interpret or deviate from the
spec (`Aegora_Isle_Freelancer_Specification_1.docx` v1.0). The professor /
client should see this during M1 review.

## Resources — multiple goods (spec §2.1 inference)

The spec defines `produce(type)` and `buy/sell(item, qty)` action signatures
implying multiple produced goods, while `resource_stock` is singular. Locked:
**one raw resource pool** + **two produced goods** (`food`, `tool`).

Per-agent wealth used by Gini / GDP / HDI (M2):

```
wealth = raw_units
       + food_qty  * food_price
       + tool_qty  * tool_price
       + cash
```

## Agent counts — n_agents=50 rounding

Spec §2.1 mandates 25% per type; 50 / 4 = 12.5 doesn't divide evenly.

- **With Governor (G2, G3):** 12 Extractors / 13 Producers / 13 Traders / 12 Governors.
- **Without Governor (G1, spec §2.2):** redistribute the Governor share →
  17 Extractors / 17 Producers / 16 Traders.

Rounding rule encoded explicitly in `AegoraEnv._instantiate_agents()`.

## Action-space encoding — Dict spaces matching spec §2.1 literally

- **Extractor:** `Dict{action_type ∈ {extract, idle, move_to_zone}, amount, target_zone}`
- **Producer:**  `Dict{action_type ∈ {produce, store, trade_offer}, produce_type, offer_item, offer_qty}`
- **Trader:**    `Dict{action_type ∈ {buy, sell, idle}, item, qty}`
- **Governor:**  `Dict{action_type ∈ {set_cap, set_redistribution, sanction, idle}, value, target_agent_idx}`

Irrelevant parameter fields per `action_type` are ignored by the env.
`target_agent_idx` domain is `MAX_AGENTS = 500` to satisfy spec §11.2
(forward-compat to 500-agent runs).

**Update (M2 Sprint 2.1 research pass):** SuperSuit is **not** used for SB3
training. The locked architecture (confirmed against installed
`stable-baselines3==2.9.0`/`gymnasium==1.3.0`/`pettingzoo==1.26.1`, no
`supersuit` dependency) is one PPO/SAC model per agent type with a manual
rollout collector walking `env.agent_iter()` directly, plus a custom,
type-agnostic `aegora_isle/agents/action_codec.py` (Sprint 2.5) that
flattens each agent type's `Dict` action space to a `Box` for SB3 and
decodes back via `argmax` per `Discrete` sub-slot — deliberately not via
`gymnasium.spaces.utils.unflatten`, which was verified against gymnasium
1.3.0's actual source to take the *first nonzero index* rather than the
largest, silently wrong when fed a continuous policy's raw output.

## Zones — N_ZONES = 4

Spec §2.1 Extractor action `move_to_zone` implies a zoning of the island but
gives no count. Locked: **4 zones**, one nominal correspondence with the four
scenarios' spatial layout. **Asked client to confirm.**

## Collapse threshold — env termination vs. CSV `collapse_event` (M2 Sprint 2.3)

Two different, deliberately separate notions of "collapse" coexist:

1. **Env-level termination** (unchanged since M1): `ResourcePool.is_collapsed()`
   uses the spec's static `stock < critical_min` placeholder, purely to decide
   when `AegoraEnv` truncates an episode. Spec §3's full MTSC definition is
   `stock < critical_min OR GDP=0 for 3 consecutive steps` (verbatim-confirmed
   from the spec docx) — the env still uses only the first clause; the second
   clause was never load-bearing for termination and isn't needed now that
   `collapse_event` (below) is the actual MTSC signal feeding M4's
   Kaplan-Meier survival analysis.

2. **The CSV's `collapse_event` column** (`kpi_logger.compute_subsistence_collapse`):
   implements the **professor's subsistence definition** (clarification dated
   2026-06-12), not the spec's literal dual-clause, per explicit sign-off:
   `True` when the bottom 25% of agents by wealth (`bottom_fraction=0.25`) have
   mean wealth below `subsistence_fraction × initial_mean_wealth`
   (`subsistence_fraction=0.5`, both new tunable constants). Fires as a
   **rising-edge pulse** — `True` only on the timestep collapse is first
   detected, `False` on every other row including after collapse — per the
   spec's own phrasing for the column ("True on the timestep... False
   otherwise").

**Known consequence, accepted knowingly:** M4's Kaplan-Meier survival
analysis will treat the professor's subsistence definition as "the event,"
not the spec's literal dual-clause that §3's own MTSC row is written
against. If a future reviewer expects MTSC to mean the spec's literal
definition, this is the place that explains why it doesn't.

## KPI formulas (M2 Sprint 2.3) — sign-off items

Verbatim-extracted from the spec docx §3/§3.1 (not paraphrased — see
`M2_PLAN.md` Sprint 2.3 for the full derivation). Three places where the
spec names a concept but doesn't give an exact sub-formula, each resolved
and documented here rather than guessed silently:

- **HDI proxy** is a *linear* composite (not the geometric mean an earlier
  draft of this plan assumed): `0.4·resource_access + 0.4·capability_utilisation
  + 0.2·agent_longevity`. The spec names these three terms but not their
  formulas; locked as:
  - `resource_access = clip(mean_wealth / initial_mean_wealth, 0, 1)`
  - `capability_utilisation = clip(cycle_production_value / (n_producers × MAX_PRODUCE_AMOUNT × mean_price), 0, 1)`
  - `agent_longevity = clip(mean_agent_age / current_timestep, 0, 1)` — **honest
    caveat**: with no agent-death mechanic implemented, this term is ≈1.0 for
    nearly the entire run (every agent persists from `reset()`). Documented
    limitation, not a bug.
- **Depletion-rate sign**: implemented as `(stock_{t-1}-stock_t)/stock_{t-1}×100`
  (positive = depletion), matching the CSV schema's explicit Notes-column
  wording rather than the spec's bare `Δ(stock_t-stock_{t-1})` shorthand,
  which reads ambiguously against it.
- **GDP-equivalent** sums *production* value (Producer output, valued at
  scarcity price at production time) **plus** trade value — not trade value
  alone, which an earlier draft of `M2_PLAN.md` had wrong. Indexed to the
  mean of epoch 0 (steps 0–99), not "timestep 100" as that earlier draft
  also said.
- **Live vs. CSV KPI values**: `AegoraEnv._snapshot_state()`'s `kpis` dict
  (read by every `Governor.get_observation` call, and by G2/G3 governance
  handlers) intentionally differs from the CSV's columns: `gini`/`depletion_rate`
  are instantaneous, not rolling-mean smoothed; `hdi_proxy` is recomputed
  every step (no 100-step NaN-gating, so it always satisfies
  `Governor.observation_space.contains(...)`); `gdp_indexed` is the raw
  per-cycle value (not epoch-0-indexed); and `depletion_rate` is a
  **fraction**, not the CSV's percentage, specifically to fit
  `Governor.observation_space`'s declared `[-1, 1]` bound. All differences
  are commented inline at the `_snapshot_state` call site.
- **JSON state field count correction**: an earlier research pass (M2
  Sprint 2.1 planning) mis-tallied spec §3.2's verbatim bulleted field list
  as "12 fields"; recounting the same verbatim list gives **13** fields
  (`run_id, random_seed, governance_condition, scenario, governance_params,
  scenario_params, n_agents, agent_type_distribution, rl_hyperparams,
  collapse_occurred, collapse_timestep, final_kpi_means, csv_checksum`).
  `kpi_logger.JSON_FIELDS` implements the correct 13; only the planning
  doc's summary number was wrong.

## Producer raw-material acquisition — resolved (post-M2 fix)

**Originally found during M2 Sprint 2.3, resolved post-M2 sign-off.**
`Extractor._apply_extractor` places extracted raw into the **Extractor's
own** inventory; `Market` originally only traded `food`/`tool`. There was
no mechanism, in any regime, for a Producer to receive raw material from
an Extractor — `Producer._apply_producer`'s `produce` branch was
consequently a no-op in practice, and `gdp_raw`'s production component
(plus HDI's `capability_utilisation` sub-term) was structurally zero
regardless of governance regime or agent policy.

Two candidate fixes were considered and rejected before landing on the
actual fix:
- **Producers extracting directly** — rejected: this would make
  Extractors redundant, undermining the spec's four-distinct-agent-role
  design. Confirmed the spec is genuinely silent on the transfer
  mechanism (verbatim-extracted the full docx and searched for "raw
  material"/"supply chain"/"feedstock"/etc. — zero hits anywhere), so
  this was legitimately open engineering judgment, not a spec deviation.
- **A new explicit "buy_raw"/"sell_raw" action_type** on Producer/Extractor's
  action spaces — rejected: would widen both `Discrete(3)` action spaces
  to `Discrete(4)`, rippling into `action_codec.py`'s flatten/encode/decode
  dimensions, every space-shape test, and invalidating any already-trained
  PPO checkpoint outright.

**Fix implemented**: `_apply_producer`'s `produce` branch now automatically
buys any raw shortfall from a randomly-picked Extractor counterparty via
the existing `_pick_counterparty` + `Market.execute_trade` pattern —
identical to how `_apply_trader`'s buy branch and `_apply_producer`'s own
`trade_offer` branch already work. No new action_type on either agent;
zero action/observation space changes. `Market.get_price`/`execute_trade`
already had no hardcoded item lists (confirmed against source before
building this), so adding `"raw"` to every scenario's `base_prices` "just
works."

**Pricing**: raw is now scarcity-priced via `Market`'s existing
`_scarcity_multiplier()` formula (same as food/tool), with
`base_prices["raw"] = 1.0` in every scenario (`scenarios.yaml`) and in
`aegora_env.py`'s `DEFAULT_SCENARIO_PARAMS`. **This coexists deliberately
with `AgentBase.wealth()`'s separate, unchanged, flat `1.0` internal price
for raw** — `wealth()` was explicitly left untouched (raw's contribution
to Gini/HDI-`resource_access`/subsistence-collapse stays a flat `1.0` per
unit, decoupled from the market price) to keep this fix's blast radius
contained to "Producers can now get raw material," not "how Gini is
computed changes." If a future reader is confused why there are two
different "raw price" numbers in this codebase, this is why.

**Partial-fill semantics**: if an Extractor doesn't have enough raw, or
the Producer can't afford the full shortfall, `execute_trade`'s existing
clipping (`min(qty, seller_qty)`, then further clipped to what the buyer
can afford) means the Producer buys whatever's actually available/affordable
and produces that much — never all-or-nothing, never a crash. Confirmed via
`tests/test_producer_raw_sourcing.py`'s partial-fill and zero-supply test
cases.

**GDP accounting**: the raw purchase's `value` counts once toward
`_cycle_trade_value` (intermediate-goods trade), and the subsequent
production step separately counts toward `_cycle_production_value` at the
finished good's price (value-added on top) — **not** deduplicated. This
matches the existing precedent already in this codebase: a finished good
already gets counted once as production (when manufactured) and again as
trade (when later sold to a Trader) — `gdp_raw` here is a naive sum of all
market activity, not a value-added chain requiring deduplication. Stated
explicitly so it isn't mistaken for an oversight later.

**HDI side-effect fixed alongside**: once raw became a market item, the
epoch-boundary `mean_price` calculation feeding `capability_utilisation`'s
theoretical-max denominator (in both `kpi_logger.py` and `aegora_env.py`'s
live KPI injection) would have silently pulled raw's (input) price into a
term meant to represent finished-goods (output) pricing. Fixed by
computing `mean_price` over `food`/`tool` only, explicitly — everywhere
else that iterates `market.base_prices` was confirmed unaffected (named
`.get("food")`/`.get("tool")` lookups, not full-dict averages).

Verified: 200/200 tests passing (9 new: `tests/test_producer_raw_sourcing.py`
×7, `tests/test_market.py::test_execute_trade_supports_raw_item`,
`tests/test_kpi.py::test_hdi_mean_price_excludes_raw`), AC01/AC03/AC06
re-verified unaffected, total cash exactly conserved across the new trade
type (confirmed both via unit test and a full-run manual check).

## G3 SAC action space — 3-dim, not a 505-dim Dict flatten (M2 Sprint 2.5)

A design review (against the actual installed `stable-baselines3==2.9.0`
source) caught a regression in an earlier sketch: flattening the
Governor agent's full `Dict{action_type: Discrete(4), value: Box(1,),
target_agent_idx: Discrete(500)}` for SAC would produce a 505-dim action
(once `Discrete` fields are one-hot-expanded per `action_codec.py`'s
scheme), of which ~500 dims (`target_agent_idx`) are relevant only on the
rare `sanction` branch. SAC's actor/critic would have to learn near-total
insensitivity to 500 of 505 dims, diluting the 3 dims that actually
matter (cap, redistribution rate, sanction likelihood) into 0.8% of the
action vector — a textbook "low effective dimensionality buried in high
nominal dimensionality" setup that slows critic convergence without
adding any real capability.

**Locked**: `AdaptiveRLGovernor`'s SAC operates on a **3-dim**
`Box(0,1,(3,)) = [cap_norm, redist_norm, sanction_probability]` — exactly
what `G3_PLAN.md`'s own "Action (SAC output)" table specified before the
regression. "Which agent to sanction" is **not** SAC-learned targeting;
it's handled the same way G2 handles cap violations — reactively,
per-Extractor, inside `on_action`, comparing that Extractor's own
requested amount against `self.current_cap`. The Governor *agent's* own
`Dict` action space in `governor.py` is completely unaffected — this
simplification is internal to `adaptive_rl.py` only, isolated by the
existing `on_action`/`on_step` interface boundary. A consequence, stated
plainly: individual Governor agents' own emitted actions are not
consumed by `AdaptiveRLGovernor` at all under G3 — SAC operates directly
on the macro KPI/wealth-histogram state (reusing a live Governor agent's
own `get_observation`/`observation_space`, not duplicating that
construction), not on any aggregation of the population's Governor
agents' individual Dict actions.

## SAC replay-buffer fix (M2 Sprint 2.5)

The original `G3_PLAN.md` sketch for `AdaptiveRLGovernor.on_step()` called
`self.model.train(...)` without ever calling
`self.model.replay_buffer.add(...)` first — confirmed against SB3's
actual `SAC.train()` source, which only ever samples from the replay
buffer, never fills it. Left as originally sketched, every `on_step()`
call would have trained on an empty (all-zero) buffer: no crash, a
real-looking loss number logged, zero actual learning — exactly the
"looks like it's training, isn't" failure class this project is most at
risk of. Fixed in the implementation: `on_step()` now keeps the previous
cycle's `(obs, sac_action)` pair and pushes a complete
`(obs, next_obs, action, reward, done, infos)` transition via
`replay_buffer.add(...)` *before* calling `train()`, mirroring SB3's own
`collect_rollouts()` → `train()` ordering.

**SAC done-flag scope**: `env_state` (the snapshot dict `on_step`
receives) doesn't carry the env's termination/truncation flags — those
are set by `AegoraEnv.step()` *after* `governance_module.on_step()`
returns. `AdaptiveRLGovernor._is_terminal()` uses the best available
proxy from `env_state` alone: `stock < critical_min` (the env's own
collapse check). It cannot detect horizon-truncation (`max_timesteps`
isn't in the snapshot) — but treating a horizon cutoff as non-terminal
(`done=False`, always bootstrap) is the conventionally correct treatment
of censored data anyway, not merely a simplification papering over a gap.

## Manual rollout collector corrections (M2 Sprint 2.5)

A design review against the actual installed SB3 2.9.0 source
(`buffers.py`, `on_policy_algorithm.py`, `policies.py`) found three
implicit state-machine requirements that `.learn()` normally manages
invisibly, which a hand-rolled collector must replicate explicitly:

1. **Action/value/log-prob must come from `model.policy(obs_tensor)`**
   (returns `(actions, values, log_probs)` as tensors), **not**
   `model.predict()` — confirmed `BasePolicy.predict()` returns only
   `(actions, state)`, with no path to a value or log-prob at all.
2. **`model.num_timesteps` must be incremented manually** per transition
   pushed — `collect_rollouts()` does this as a side effect with no
   public setter equivalent.
3. **`episode_start` must be tracked correctly** for `RolloutBuffer.add()`
   and `compute_returns_and_advantage()` — wrong values silently corrupt
   GAE bootstrapping without raising. `train_agents.py`'s scope for this
   sprint (a short smoke-validation pilot, not a tuned run) uses a single
   continuous episode per `run()` call: `episode_start=True` only for
   each agent type's very first pushed transition, `False` thereafter.
   Multi-episode (mid-collection `env.reset()`) handling is deferred to
   M3's real training runs.
4. **`model._current_progress_remaining` is held constant** (LR/clip-range
   schedules don't anneal under the manual driver) — an accepted,
   documented M2 simplification. `.learn()` would normally update this
   every step; nothing in this sprint's scope depends on schedule decay
   actually happening.
5. **SB3's internal logger must be configured manually** via
   `model.set_logger(configure(folder=None, format_strings=[]))` before
   any `train()` call — `.learn()` normally does this in its setup phase;
   skipping it crashes the first `train()` call with
   `AttributeError: 'SAC'/'PPO' object has no attribute '_logger'`. The
   empty `format_strings` makes it a no-op logger (nothing is actually
   written), it exists only to satisfy internal `self.logger.record(...)`
   calls.
6. **SB3 requires a real `gymnasium.Env` subclass**, not a duck-typed
   object — `SB3's `_patch_env` rejects anything that merely matches the
   interface structurally. Both `adaptive_rl.py`'s `_DummyGovernorEnv` and
   `train_agents.py`'s `_DummyTypeEnv` subclass `gym.Env` for this reason.
7. **YAML's `train_freq: [4, step]` loads as a Python `list`**, but SB3's
   `_convert_train_freq` requires a `tuple` (`isinstance(train_freq,
   tuple)`) or it misinterprets the whole list as the frequency integer
   and raises `ValueError`. `AdaptiveRLGovernor.__init__` converts this
   explicitly; `train_agents.py`'s PPO construction doesn't take
   `train_freq` at all (irrelevant to on-policy algorithms).

## Seeding contract — non-negotiable

Every module accepts `numpy.random.Generator` at construction; nothing creates
its own. Top-level seed flows:

```
run_experiment.py
    → AegoraEnv(seed)
        → ResourcePool(rng)
        → Market(rng)
        → GovernanceModule(rng)
        → each Agent(rng)
        → each SB3 model (seed)
```

## Governor reward — placeholder stays, composite reward lands elsewhere (M2 Sprint 2.5)

Spec §2.2.2's composite reward (`w_equality=0.40, w_sustainability=0.35,
w_productivity=0.25`, verbatim-confirmed against the spec docx) is now
implemented as `AdaptiveRLGovernor.compute_composite_reward()` — but
`Governor.compute_reward()` (the **agent** method, in `governor.py`)
**still returns the M1 `0.0` placeholder and is left unchanged**. This is
deliberate, not an oversight: nothing in the architecture reads
`Governor.compute_reward()`'s return value for actual learning under any
regime — `aegora_env.py` calls it and assigns the result to
`self.rewards[governor_agent_id]`, but G2's `RuleBasedGovernor` and G3's
`AdaptiveRLGovernor` both operate entirely through the `on_action`/
`on_step` governance hooks and the live agent registry, never through any
individual agent's own reward method. SAC's actual training signal comes
from `compute_composite_reward(env_state)`, computed independently inside
the governance handler from macro KPI state. Changing
`Governor.compute_reward()` would be a no-op change with no functional
effect — left alone to avoid touching M1-locked agent code for no reason.

## Smoke test scenario — G1-D (spec §10 AC03)

`MASTER_PLAN.md` Sprint 1.5 names Scenario C; `M1_PLAN.md` Sprint 1.5 names
Scenario D. The spec (§10) is the tiebreaker — **D** (low abundance + high
events) is the harshest cell and is the named AC03 scenario. Locked: smoke
test runs G1-D.

## G3 sustainability reward — confirmed final (client sign-off, pre-M3)

`AdaptiveRLGovernor._component_sustainability()` computes
`stock_norm * max(0, 1 - depletion_rate)` — the product of the pool's
current fullness and how fast it's currently depleting. Spec §2.2.2's
literal text names the sustainability weight as applying to a
`(-depletion_rate)` component alone, with no stock-level factor.

This was originally an undocumented implementation-time deviation — no
`DECISIONS.md` entry existed for it prior to this reconciliation. The
rationale, surfaced to and confirmed by the client: a bare
`-depletion_rate` term only measures the *instantaneous* rate of change, so
a Governor could deplete the pool aggressively for many cycles, then coast
at ~0% depletion for a single cycle and receive full sustainability credit
despite the pool being nearly empty. Multiplying by `stock_norm` closes
that loophole — a high sustainability score now requires the pool to be
both genuinely healthy *and* not currently being drained quickly.

**Client explicitly confirmed this is the final, intentional design** (see
`Client_Response_Governance_Confirmation.md` and the client's follow-up
reply) — kept as-is going into Milestone 3, not reverted to the literal
spec formula.

## G2 `extraction_cap = 3.0` — confirmed final (client sign-off, pre-M3)

`g2_rule_based.yaml`'s `extraction_cap: 3.0` deviates from spec §2.2.1's
literal stated default of `5`. This was chosen during `G2_PLAN.md`'s
initial design (comment: "MAX_EXTRACT_AMOUNT is 5.0 in env, so cap = 60% of
free-market max") but was never explicitly flagged to the client as a
spec-default deviation at the time.

Rationale: `MAX_EXTRACT_AMOUNT` (the environment's hard physical ceiling on
any single extraction) is also `5.0`. A rule-based cap set equal to the
hard ceiling can never actually bind — no Extractor action can ever exceed
it in the first place — so G2 would behave identically to G1 (ungoverned)
until a future tuning pass lowered it. `3.0` was chosen so the cap is an
active, real constraint from Sprint 2.2 onward.

**Client has explicitly signed off on `3.0` as frozen** going into the
Milestone 3 confirmatory dataset — not the spec's literal default of `5`.
No further tuning of this value without flagging the client first
(CLAUDE.md "ask before changing YAML configs").

## Redistribution mechanism — confirmed final (client sign-off, pre-M3)

Both `RuleBasedGovernor._redistribute()` (G2) and
`AdaptiveRLGovernor._redistribute()` (G3) implement an **agent-to-agent
equal-split cash transfer**: a configurable fraction of Producer + Trader
cash is pooled, then divided equally across every Extractor, regardless of
individual need — not a needs-based/targeted transfer, and not a
replenishment of `ResourcePool.stock` (the shared physical resource).

Spec §2.2.1's literal text describes `redistribution_rate` as a fraction
"transferred to **resource pool** per epoch" — read literally, and
consistent with how "resource pool" is used everywhere else in the spec
document, this would mean topping up the shared physical stock, not moving
cash between agent classes. This ambiguity was surfaced explicitly to the
client during the pre-M3 reconciliation round.

**Client has explicitly confirmed the implemented agent-to-agent cash
transfer is correct and final** — not a replenishment of the physical
resource pool. Locked going into Milestone 3; documented here so the
distinction is on record rather than read as an unexplained deviation from
the spec's literal wording.

## Convergence-retry protocol only applies to true NON_CONVERGED, not early collapse (M3 Sprint 3.3)

`run_experiment.py`'s `run_single_cell` originally classified any agent
type not in `ConvergenceStatus.CONVERGED` as needing the exclude-and-retry-
at-`seed+1000` treatment (`if check_convergence(t) is not CONVERGED`). This
wrongly conflated two different states:

- `CONVERGING` — fewer than `MIN_EPOCHS_BEFORE_CHECK=50` epochs recorded.
  This includes a run that legitimately ended early via a resource-pool
  collapse (the normal, expected outcome for Scenario C/D's "chronic
  scarcity" design) — a valid, keepable data point (`collapse_event=True`
  in the CSV; exactly what the spec's MTSC/survival analysis is designed to
  measure), not a training failure.
- `NON_CONVERGED` — at least 50 epochs recorded, and the trailing-20-epoch
  loss variance stayed at or above `CONVERGENCE_THRESHOLD=0.01`. This is
  the only case the spec's §6.2 "excluded and replaced" protocol actually
  describes.

Surfaced by the Sprint 3.3 pilot run (3 reps × 12 cells, real scale):
`0/36` cells converged even after retry, but investigation showed 21 of the
36 never reached 50 epochs at all (mostly Scenario C/D collapsing in 1–4
epochs, as designed) — these were being needlessly retried at `seed+1000`
even though the *same* scenario parameters would very likely collapse the
retry too, wasting compute and mischaracterizing a valid outcome as a
failure.

**Fix**: classification now checks `is ConvergenceStatus.NON_CONVERGED`
specifically (not `is not CONVERGED`) for the retry-eligible set, and adds
a distinct `"CONVERGING"` result status for runs that ended before the
50-epoch minimum — reported separately in `_print_summary`, never routed
to Phase B. `ConvergenceMonitor`/`CONVERGENCE_THRESHOLD`/`CONVERGENCE_WINDOW`
(pre-registered, spec §6.3) are unchanged — this is a classification-logic
fix in `run_experiment.py`, not a re-tuning of the locked convergence
protocol.

## PPO observation/reward normalisation — confirmed gap, fixed (M3 Sprint 3.3)

The same pilot run also surfaced a second, unrelated problem: for cells
that *did* reach 50+ epochs (Scenario A across all regimes, Scenario B for
G2/G3), the Extractor agent type's logged PPO loss variance reached the
**trillions** in at least two confirmed cells (G1_B_001, G1_B_002) — a
genuine numerical blow-up, not sampling noise.

**Root cause, confirmed by reading the actual code** (not assumed):
`aegora_isle/experiments/train_agents.py`'s `ManualPPOCollector` had **no
observation or reward normalisation anywhere** — raw values from
`env.last()` went straight into `obs_as_tensor(...) → model.policy(...)`
and `rollout_buffer.add(...)`, with no `VecNormalize`-equivalent wrapping
(the collector's dummy env, used only for space introspection, is never
wrapped in one, unlike what SB3's own `.learn()` would normally provide).
Extractor/Producer/Trader's `own_cash`/inventory observation dimensions are
declared with bounds up to `1e6`, and **no cap exists on cumulative wealth
accumulation under G1** (`G1Handler.on_action`/`on_step` are literal
no-ops; extraction caps and redistribution only exist under G2/G3). Over a
real 10,000-step G1 run, cash/inventory can grow unboundedly, feeding
raw large-magnitude values into a freshly-initialized default-size (64×64)
MLP — a well-known PPO instability trigger. This matches the diagnostic
evidence exactly: Scenario B (highest abundance/regeneration/event-rate)
is exactly where the blow-up was observed.

Checked `DECISIONS.md`/`M2_PLAN.md`/`G3_PLAN.md`/`MASTER_PLAN.md` for any
prior documented, signed-off decision to omit PPO-side normalisation —
**none found**. The only normalisation anywhere in the codebase prior to
this fix was `governance/adaptive_rl.py`'s `_RunningNormaliser`, scoped
exclusively to G3's Governor SAC composite reward (unaffected by this fix,
untouched). This was a genuine gap, not an accepted M2 simplification.

**Fix**: new `aegora_isle/experiments/normalisation.py::RunningNormaliser`
— a per-dimension Welford running mean/std with clipping, a vector
generalisation of `_RunningNormaliser`'s exact algorithm. Deliberately
**not** shared/imported between the two modules (`CLAUDE.md`'s
single-entry-point rule forbids `experiments/` from importing
`governance/adaptive_rl.py`, and G3/SAC training stays out of scope for
this fix). `ManualPPOCollector` gains one normaliser instance per agent
*type* (matching its existing parameter-sharing design — one shared policy
per type), applied to both observations (before every policy forward pass,
including the rollout-boundary bootstrap value estimate) and rewards
(before `rollout_buffer.add`), on by default
(`normalise_observations=True`, `normalise_rewards=True`). Clip bounds:
`obs_clip=10.0` (matches SB3's own `VecNormalize` default — this fix
reinstates what `VecNormalize` would have supplied), `reward_clip=5.0`
(matches the existing `_RunningNormaliser` convention already used for
G3). Verified directly (not just asserted): under a synthetic large-and-
growing-cash sequence, loss variance dropped ~150× (1278 → 8.5) and max
loss magnitude dropped from 240 to 14 with normalisation enabled vs.
disabled, over the same input.

Deterministic by construction (pure numpy arithmetic on an
already-seed-determined observation/reward sequence — same exemption
already accepted for `_RunningNormaliser`: "no `numpy.random.Generator`
needed") — confirmed `tests/test_run_experiment.py::
test_run_single_cell_determinism` (same-seed-twice → byte-identical CSV)
still passes with the fix in place.

## Two-retry protocol — client-directed follow-up (M3 Sprint 3.3)

After reviewing `Client_Update_M3_Pilot_Findings.md`/`.pdf` (the pilot
finding that Extractor takes longer to reach stable training under active
governance, since it's the only agent type whose reward is shaped by a
governance sanction), the client made four explicit decisions for the
confirmatory batch:

1. **Accept the finding** — reported as a genuine result, not something to
   eliminate.
2. **One additional automatic retry** (2 retries total, 3 attempts per
   run) — `run_experiment.py` gained `SECOND_RETRY_SEED_DELTA = 2 *
   RETRY_SEED_DELTA = 2000` and a Phase C, structurally identical to
   Phase B, dispatched on cells still `NON_CONVERGED` after Phase B. Seed
   recomputed fresh via `compute_seed(...) + SECOND_RETRY_SEED_DELTA`
   (algebraically identical to chaining deltas, no need to thread an
   "original seed" field through the merge chain). **Explicitly declined**
   for this batch: extending `max_timesteps` past 10,000, and softening
   the sanction penalty's abruptness — both stay untouched.
3. **Attempt count is logged** — `main()` stamps `result["attempt"] =
   1/2/3` on each phase's own results before merging (an
   orchestration-level concern; `run_single_cell` itself still only knows
   `is_retry: bool`). New `attempts_summary.csv` (one row per cell, final
   outcome only) — a separate artifact, never touching `kpi_logger.py`'s
   locked 14-column CSV / 13-field JSON.
4. **A second, independent convergence check** was added: flatness of the
   Extractor's own KPI trajectory (see next section for exactly what this
   tracks and why) over the same 20-epoch window as the existing
   loss-variance check. A run is `CONVERGED` only if **both** checks
   independently agree (AND-logic, `_classify_convergence` in
   `run_experiment.py`) — either failing routes the run to retry (if
   attempts remain) or keeps it flagged (if attempts are exhausted).
5. **Exclusion/retry rate reported per governance condition** — new
   `retry_rate_by_governance.csv` (columns: `governance, total_runs,
   converged_1st_try, converged_2nd_try, converged_3rd_try,
   still_unconverged_after_2_retries, exclusion_rate_pct`), plus a console
   summary in `_print_summary`. The client explicitly wants this treated
   as a first-class result (informative for H3), not just a housekeeping
   stat — a higher exclusion rate under G3 is itself evidence for the same
   governance-difficulty finding.

Runs still unconverged after both retries stay in the dataset (the CSV/JSON
were always written unconditionally, retries just overwrite in place — no
behavior change there), flagged via `attempts_summary.csv`'s
`final_status` column. No further, third retry — decided now, per the
client's explicit instruction, not deferred until the real batch's numbers
come in.

## Extractor KPI-flatness check — pre-sanction extraction amount, not reward (M3 Sprint 3.3)

A design review caught a real ambiguity in how the client described the
new check: *"the Extractor's own reward signal, which literally equals
amount extracted."* That's only true under G1. Under G2/G3, `aegora_env.py`'s
`step()` adds the governance sanction delta on top:

```python
mod_action, reward_delta = self.governance_module.on_action(agent_id, action)
base_reward = float(agent.compute_reward(snap, mod_action))
self.rewards[agent_id] = base_reward + float(reward_delta)
```

Tracking `env.last()`'s actual reward (sanction-inclusive) would make the
new check largely redundant with the existing loss-variance check — both
driven by the same governance-induced noise — and wouldn't be bounded to
`[0, MAX_EXTRACT_AMOUNT]` (a `-2.0` sanction can make it negative).

**Confirmed with the user (and reflected in the client-facing framing):
the check tracks the pure, pre-sanction extraction amount** —
`env._step_log[agent_id]["extracted"]`, set inside `_apply_extractor`
before governance ever touches the action/reward, bounded `[0,
MAX_EXTRACT_AMOUNT=5.0]`. This is a genuinely independent second signal
(measures whether Extractor's actual harvesting *behavior* has stabilized,
decoupled from reward-shaping noise) and matches the client's literal
"amount extracted" framing exactly — identical to the real training reward
under G1, diverging from it only under G2/G3, precisely where the check
needs to add information beyond the loss-variance check alone.

Implementation: `ManualPPOCollector._record_extractor_extraction`/
`_maybe_record_epoch_kpi` (mirrors the existing `_maybe_record_epoch_loss`
hook exactly, same `EPOCH_LENGTH=100` cadence, same file — kept here rather
than in `run_experiment.py` for symmetry, avoiding two independent
epoch-boundary bookkeeping sites that could drift out of sync). Fed into a
**second, separate `ConvergenceMonitor(agent_types=("extractor_kpi",))`
instance** — the class is reused completely unmodified (it's already
generic over the `agent_type` string label); `convergence.py` itself is
untouched, and `"extractor_kpi"` is simply a new, valid value in
`convergence_failures.csv`'s existing `agent_type` column.

**Threshold: reuses `CONVERGENCE_THRESHOLD = 0.01` literally — no new
constant introduced.** Justified quantitatively, not just asserted: at the
real batch's `n_agents=50` (~12 Extractors × 100 steps ≈ 1,200 raw
extraction draws per epoch mean), the epoch mean's sampling-noise floor is
roughly 0.001–0.003 — comfortably under 0.01, so a `NON_CONVERGED` read is
very likely picking up genuine behavioral drift, not finite-population
noise. **Caveat, flagged explicitly rather than silently absorbed**: 0.01
was pre-registered for PPO's unitless loss, not independently calibrated
for this physically-bounded `[0,5]` quantity. If the real batch's G1 cells
(which have no sanction noise to explain instability) show a nontrivial
KPI-check-driven exclusion rate too, that's a signal 0.01 is miscalibrated
for this quantity and worth revisiting with the client — not something to
quietly patch mid-batch.

## Single-criterion exclude-and-replace-until-valid — client-directed reversal (M3, pre-Sprint-3.5)

Before the real 360-run launch, the client reversed part of the two-retry/
dual-check follow-up above. Two things are explicitly confirmed unchanged
by the client and verified directly against git history before responding
(not just asserted): `CONVERGENCE_THRESHOLD=0.01` appears exactly once in
the entire project history, never modified, including while the KPI check
was active; `aegora_isle/experiments/normalisation.py` was created and
committed once, never touched since, with its original defaults
(`normalise_observations=True`, `normalise_rewards=True`, `obs_clip=10.0`,
`reward_clip=5.0`) intact. Confirmed reversed:

1. **Convergence decided on the PPO loss-variance check alone by default.**
   The Extractor KPI-flatness check is parked, not deleted —
   `run_single_cell` gains `track_kpi_convergence: bool = False`;
   `_classify_convergence` takes `kpi_monitor: ConvergenceMonitor | None`
   and degrades to loss-variance-only classification when `None` (a
   strict generalisation, byte-identical to the prior AND-logic whenever
   a real `kpi_monitor` is supplied). `--track_kpi_convergence` (CLI,
   `store_true`, default off) is the opt-in; the real launch never passes
   it.
2. **Unbounded round-based exclude-and-replace, replacing the capped
   two-retry protocol.** `_run_convergence_rounds` dispatches Round 1 for
   every cell, then re-dispatches only the still-`NON_CONVERGED`/`ERROR`
   subset each subsequent round at `seed = compute_seed(...) +
   (round-1)*RETRY_SEED_DELTA`, until every replication reads `CONVERGED`
   or `CONVERGING` (early-collapse handling is unchanged — `CONVERGING`
   is never replaced) or `DEFAULT_ROUND_CAP=10` is exceeded.
   `SECOND_RETRY_SEED_DELTA` (the old fixed two-round constant) is removed
   — superseded by the generalised per-round formula.
3. **`ERROR` folded into the same replace-and-retry bucket as
   `NON_CONVERGED`** (confirmed) — an errored replication gets a fresh
   seed and another attempt rather than permanently leaving its cell short
   of `n_reps` valid replications.
4. **Safety cap = 10 total attempts per replication, confirmed.** Sizing
   math (client's own 50%-per-attempt framing): `(1-0.5^10)^30 ≈ 97%`
   chance a single cell clears within 10 rounds; `0.971^4 ≈ 89%` chance
   all 4 at-risk cells (G2-A, G2-B, G3-A, G3-B) clear together. Hitting
   the cap raises `ConvergenceBudgetExceededError`, naming the exact
   unresolved `(governance, scenario, replication)` triples and the round
   count — `main()` does not write `attempts_summary.csv`/
   `cell_attempt_summary.csv` on this path, since an aborted batch must
   not produce artifacts that look like a completed, valid dataset. No
   run is ever kept in the final dataset flagged "unconverged."
5. **Per-cell attempt logging**: `attempts_summary.csv` (existing,
   unchanged code — `final_status` is now always `CONVERGED`/`CONVERGING`,
   never `NON_CONVERGED`) stays the per-replication detail log.
   `retry_rate_by_governance.csv` (per-governance, from the now-retired
   two-retry design) is replaced by `cell_attempt_summary.csv`
   (`_build_cell_attempt_rollup`/`_write_cell_attempt_summary`) — one row
   per `(governance, scenario)` cell: total replications, total attempts,
   total *replacement* attempts, max attempts for a single replication,
   mean attempts per replication. `convergence_failures.csv` (locked
   schema) needed no changes — it already logs a row on every failed
   attempt regardless of how many rounds there are.

**Empirical finding, surfaced to the client rather than silently absorbed**:
before this was built, `data/pilot_v3/convergence_failures.csv` was
re-analysed and one cell (`G3_A_001`) was re-run fresh at its original
seed (201) to confirm directly (not just inferred) that disabling the KPI
check would not resolve the underlying instability — every one of the 12
G2/G3-Scenario-A/B replications that were "still unconverged" in the prior
pilot already failed the loss-variance check alone (`extractor` variance
1–260, one to four orders of magnitude above 0.01); the fresh re-run
reproduced this exactly (`NON_CONVERGED`, variance 156.37). Confirmed
further, mechanically, once the redesign landed: a live smoke test on
`G2-A` (6,000 steps) ran all 10 rounds and hit
`ConvergenceBudgetExceededError` exactly as anticipated — the cap fired
for real, on the first cell tested, not just in theory. This points to a
genuine, still-unresolved PPO training-instability issue for Extractor
specifically under active/adaptive governance, independent of both the
KPI-check reversal and the retry-cap redesign — flagged to the client in
`Client_Update_M3_Confirmation_Before_Launch.md`, not fixed here (per
CLAUDE.md: surface non-convergence, don't silently patch around it).

## Scenario C/D resource-pool recalibration — client-directed, pilot-validated (M3, pre-Sprint-3.5)

Client-directed diagnostic and fix, ahead of the confirmatory 360-run
launch: all 36 practice-batch replications under Scenarios C/D collapsed
within 102–345 timesteps, far short of the 5,000-timestep (50-epoch)
convergence-evaluation gate, across all three governance conditions.

**Root cause, confirmed analytically and empirically**: `ResourcePool`'s
maximum possible logistic regeneration rate
(`regeneration_rate * initial_stock / 4`, the peak of the logistic growth
curve) was below the fixed 50-agent population's extraction demand
(measured at ~9–22 units/timestep depending on governance) at the original
`initial_stock=3000, regeneration_rate=0.01`. Collapse followed directly
from the parameter values, not from training outcome. Per spec §2.3,
scenario parameters are implementation-level (`scenarios.yaml`), not fixed
by the spec text — this is a calibration fix, not a scenario redesign.

**Two rounds of real-training validation** (not just analytical estimate):
candidate 1 (`initial_stock=4000, regeneration_rate=0.04`) reached the
50-epoch checkpoint in only 2 of 12 replications — trajectories revealed a
scarcity-price feedback effect (falling stock raises the scarcity-adjusted
extraction price, increasing the incentive to extract, accelerating
depletion further) not captured by a constant-extraction-rate estimate.
Candidate 2 (`initial_stock=8000, regeneration_rate=0.05`) reached the full
10,000-timestep horizon in 10 of 12 replications; the two exceptions were
both No-Governance/Scenario-D replications (collapse at t=794, t=469) —
both governed conditions (Rule-Based, Learning-Based) survived the full
horizon in both scenarios and both replications tested.

**Deviation adopted, confirmed by client and committed to `scenarios.yaml`**:
`initial_stock` and `regeneration_rate` for Scenarios C and D revised from
(3000, 0.01) to (8000, 0.05). `critical_min` and `event_rate` — the
parameters defining each scenario's intended scarcity/volatility
character — are unchanged.

**Outstanding item**: No-Governance/Scenario-D still collapses at these
parameters. Per the existing early-collapse rule (unchanged, see
"Convergence-retry protocol only applies to true NON_CONVERGED, not early
collapse" above), this reads as `CONVERGING` and is kept as valid,
short-run data, never excluded/replaced — confirmed correct by direct code
inspection (`_needs_replacement` only flags `NON_CONVERGED`/`ERROR`) and by
empirical observation across this diagnostic sequence. Whether to instead
push Scenario D's parameters further to force full survival (at the cost
of narrowing its abundance contrast with Scenario B) is the client's call,
not yet made.

## Authoritative-per-regime convergence check — wired into the real tool (M3, pre-Sprint-3.5)

Client-directed follow-up: the harvested-amount-authoritative design
(adopted by the client after the diagnostic sequence documented above) had
only ever been computed manually inside one-off diagnostic scripts. The
client checked the actual batch tool directly and confirmed it still ran
the old AND-combined logic gated behind a manual, off-by-default
`track_kpi_convergence` flag — the real 360-run launch would have silently
defaulted to loss-variance-only. This closes that gap.

**Design constraint respected**: CLAUDE.md forbids `if governance == "G2"`
(or any regime-literal branching) outside `governance/`. Which convergence
signal is authoritative is exposed as a new capability,
`has_stochastic_extractor_sanction: bool`, on `G1Handler` (`False`),
`RuleBasedGovernor` (`True`), `AdaptiveRLGovernor` (`True`) — mirroring the
existing `regime: str` class-attribute pattern exactly — and delegated
through `GovernanceModule.has_stochastic_extractor_sanction`, same pattern
as the existing `regime` property. `run_single_cell` reads this off the
already-constructed `gm` (`extractor_signal = "kpi" if gm.has_stochastic_
extractor_sanction else "loss"`) and passes it to `_classify_convergence` —
a capability query, never a regime-identity check in `run_experiment.py`.

**`_classify_convergence`** gained a 4th parameter, `extractor_signal: str
= "loss"` (`"loss" | "kpi"`). It *substitutes* (not adds) which check
governs the `"extractor"` slot specifically — when `"kpi"` and a
`kpi_monitor` is supplied, the Extractor's slot is judged by
`kpi_monitor.check_convergence("extractor_kpi")` instead of
`monitor.check_convergence("extractor")`; the checked key becomes
`"extractor_kpi"`, letting the existing `convergence_failures.csv` logging
dispatch (`active_monitor = kpi_monitor if agent_type == "extractor_kpi"
else monitor`) route correctly with zero changes. Producer/Trader are never
affected by `extractor_signal` — they were never implicated in any
diagnostic evidence, only the Extractor's sanction-contaminated signal was.
Falls back to loss-variance for `"extractor"` defensively if `kpi_monitor`
is `None` even when `extractor_signal == "kpi"`.

**`track_kpi_convergence` retired, not kept alongside**: `kpi_monitor` is
now constructed unconditionally whenever `track_convergence` is on (cheap,
passive observer, no training effect — same property that made the whole
diagnostic sequence possible). There is nothing left to opt into: tracking
is automatic, authority is capability-driven. `--track_kpi_convergence`
removed from the CLI; the parameter removed from `run_single_cell`'s
signature entirely, rather than left as a now-meaningless manual toggle.

Verified: 5 new `_classify_convergence` unit tests (substitution overrides
noisy loss when authoritative; still fails on genuine KPI instability;
defaults to loss when not authoritative; falls back to loss if
`kpi_monitor` missing; Producer/Trader never affected), 2 new
`run_single_cell` tests (G1 never flags `extractor_kpi`; G2 can, with
`"extractor"` never appearing since it's substituted not added), 3 new
`GovernanceModule.has_stochastic_extractor_sanction` tests (one per
regime). 257/257 tests passing, ruff clean. Live smoke-tested end-to-end on
short G1 and G2 cells: G1's `convergence_failures.csv` logs failures as
`extractor` (loss-based) only; G2 is wired to use `extractor_kpi` when
its own check fires.

## Cap-exceeded partial-data fallback for G3-A/G3-B — client-directed, pre-decided (M3, pre-Sprint-3.5)

Client's 2026-08-02 sign-off email ("M3 Addendum — Confirmed, one
scarcity-contrast check, cleared to launch") approved the diagnostic
sequence's Addenda 1–3 and authorised the real 360-run launch, contingent
on a pre-decided fallback: the harvested-amount fix showed 0-of-6
improvement for Learning-Based non-convergence (vs. 4-of-6 for
Rule-Based), so the client anticipated Learning-Based Scenarios A/B (G3-A,
G3-B) may still exhaust the 10-attempt round cap in the real run. Rather
than deciding after seeing the result, they pre-decided now: if it's
specifically those two cells, keep whatever valid replications they
achieved instead of aborting the entire batch, documented clearly as a
limitation — the same honest treatment already given to No-Governance/
Scenario-D collapse.

**Scope, confirmed narrow**: this is a named exception for exactly
`{("G3","A"), ("G3","B")}` — a scenario-design decision by the client, not
a property of G3 as a regime. Any other cell exhausting the round cap
(including G3-C/G3-D, or any cell under any other governance condition)
still raises `ConvergenceBudgetExceededError` and aborts the whole batch
with zero artifacts, exactly as before. Implemented as an explicit,
documented lookup constant, `CAP_EXCEEDED_KEEP_PARTIAL: frozenset[tuple[str,
str]]`, matching the existing `GOVERNANCE_CONFIG_FILES`-style "data
selection, not behaviour branching" pattern already used in
`run_experiment.py` — not a literal `if governance == "G3"` conditional
(CLAUDE.md's regime-branching rule).

**Mechanism**: `_run_convergence_rounds`'s cap-exceeded branch now
partitions unresolved replications into exempt vs. non-exempt by
`(governance, scenario)`. If any non-exempt replication remains unresolved,
the abort is unchanged (raises, names only the non-exempt ones). If every
unresolved replication is exempt, they're relabelled
`"NON_CONVERGED_CAP_EXCEEDED_KEPT"` (a new, documented `final_status`
value — `attempts_summary.csv`'s schema is unchanged, just one more
possible value) and the function returns normally instead of raising.
`_needs_replacement` already only matches `"NON_CONVERGED"`/`"ERROR"`
exactly, so the relabelled status naturally stops the replace loop with no
change needed there.

`_build_cell_attempt_rollup` gained one new column,
`cap_exceeded_kept_count` (0 for every cell except one that actually hit
this path), so `cell_attempt_summary.csv` surfaces the limitation directly
without cross-referencing the per-replication file. `_print_summary` gained
an explicit call-out block naming the affected replications by run_id
whenever any exist, plus an inline note on the per-cell line — "documented
clearly," not just implicitly present in a count.

Verified: 3 new tests (`test_run_convergence_rounds_g3_a_b_kept_as_
documented_limitation_on_cap_exceeded`, `test_run_convergence_rounds_mixed_
exempt_and_non_exempt_still_raises` — confirms a non-exempt cell stuck
alongside an exempt one still aborts, not diluted into blanket leniency —
and `test_build_cell_attempt_rollup_reports_cap_exceeded_kept_count`), plus
the existing `test_run_convergence_rounds_raises_when_cap_exceeded` updated
(its original fixture happened to use `("G3","B",2)` as one of two failing
cells, which is now exempt — moved to a genuinely non-exempt pair so the
test still verifies the abort path it's meant to). 260/260 tests passing,
ruff clean.

## Seed-collision bug fixed before the real 360-run launch (M3, pre-Sprint-3.5)

Flagged as an open question early in the project (CLAUDE.md's "Open
questions for the human": "Seed formula's replication≥10 collision quirk
... worth a client heads-up before the real 360-run launch, since n_reps=30
guarantees this triggers") but never actually raised to the client, and
unrelated to anything in the diagnostic sequence documented above. Caught
and fixed just before launch, prompted by re-examining every open item
before actually starting the multi-hour production run.

**The bug**: spec §6.2's literal formula, `seed = seed_offset +
governance_index*100 + scenario_index*10 + replication`, encodes each
component as a decimal digit place — but `replication` is added as a raw
integer with no bound, so it only stays within its own "digit column"
while `replication < 10`. Once `replication >= 10`, it carries into the
tens column reserved for `scenario_index`, producing collisions: e.g. the
old formula gave `compute_seed("G1","A",11,0) == compute_seed("G1","B",1,0)
== 11`. With the real launch's `n_reps=30`, this affected **60 of the 120
(scenario, replication) pairs per governance condition — half the entire
360-run confirmatory dataset** would have been seeded identically to a
different, supposedly-independent replication, undermining the statistical
independence the confirmatory analysis depends on. Confirmed directly by
enumerating all collisions before deciding to fix it, not assumed from the
docstring alone.

**Fix**: widened each digit block by one order of magnitude —
`seed = seed_offset + governance_index*1000 + scenario_index*100 +
replication` — preserving the spec formula's additive, place-value
structure exactly, just with enough headroom for replication up to 999
(the real batch only needs 30) without colliding into scenario's column,
and scenario up to 9 without colliding into governance's. This is a
correction to a bug the code's own prior docstring explicitly flagged as
"known, not silently patched" — not a redesign, and not something touched
without the client's/user's explicit direction to fix it before launch.

**Consequence for prior data**: every seed used by any pilot or diagnostic
run so far (all practice-batch work documented above) used the old,
buggy formula — this does not retroactively invalidate any of that work,
since none of it was the confirmatory dataset itself, and every finding
from the diagnostic sequence stands on its own regardless of which seed
formula produced the seeds involved. The real 360-run confirmatory batch
will be the first to use the corrected formula.

Verified: 2 new tests (`test_compute_seed_no_collision_across_full_
replication_range` — enumerates all 360 (governance, scenario, replication)
combinations the real launch will use and asserts zero collisions;
`test_compute_seed_old_formula_collision_no_longer_reproduced` — the exact
collision pair the bug produced no longer matches). Existing
`test_compute_seed_matches_formula`'s parametrized expected values updated
for the new formula. 261/261 tests passing, ruff clean.

## Cap-exceeded fallback extended to G3-C/D — client-directed follow-up (M3, pre-Sprint-3.5)

The real 360-run confirmatory batch launch aborted after 6h26m (all 10
rounds completed): 34 replications never converged, all Learning-Based
Scenarios C and D (14 in C, 20 in D) — cells that were never part of the
original `CAP_EXCEEDED_KEEP_PARTIAL` exception (which covered only
Learning-Based A/B, per the earlier client sign-off).

**Root cause, consistent with everything already found**: the diagnostic
evidence for Learning-Based instability (the Governor's own cap/sanction-
probability settings never settling, even at a sixfold-lower learning
rate) was only ever gathered against Scenarios A/B, because Scenario C/D
couldn't survive long enough to be measured before the resource-pool fix.
Once that fix let C/D run to full length, the same underlying Governor
non-stationarity showed up there too — not scenario-specific after all.

**Reporting gap on our side, surfaced honestly to the client**: the
earlier Scenario C/D pool-fix validation checked whether each run
*survived* to 10,000 steps, not whether it *converged* — two Learning-
Based Scenario C validation replications and two Scenario D replications
had already come back `NON_CONVERGED` at that time, which wasn't clearly
flagged as a separate risk before the real launch.

**Client's decision** (email, "Decision: Option 1..."): extend the same
pre-approved partial-data fallback to Learning-Based Scenarios C and D —
identical treatment to A/B, same underlying reason. Declined to raise the
round cap (client's own words: "the existing evidence already tells us it
won't help" — consistent with the reduced-learning-rate experiment finding
no improvement). Separately requested (explicitly non-blocking): the same
Governor cap/sanction-probability trace evidence gathered for A/B, now for
C/D, to confirm the pattern directly rather than by inference.

**Implementation**: `CAP_EXCEEDED_KEEP_PARTIAL` extended from
`{("G3","A"), ("G3","B")}` to all four Learning-Based scenarios. No other
code changes needed — `_run_convergence_rounds`'s partition logic,
`_build_cell_attempt_rollup`'s `cap_exceeded_kept_count` column, and
`_print_summary`'s limitation call-out were all already generic over
whatever's in this constant, which is exactly why it was built as a single
lookup rather than inline conditionals. Test extended to cover all four
scenarios (was G3-A/G3-B only). 261/261 tests passing, ruff clean.

## Extractor cash-sink fix — Option 1 chosen (client decision, 2026-08-07)

**Found post-delivery**, while pulling M4 dashboard preview charts against
the real 360-run confirmatory dataset: Extractors sell raw material for
cash via `Market.execute_trade` but have no mechanism to spend cash back
into the economy. Verified via live instrumentation of a real replication
(`G1_A_001`, seed 1): total system cash is conserved (500.0 throughout)
but 100% of it migrates to Extractors by timestep ~100 of 10,000, in every
replication checked, across all three governance regimes. Once Producers/
Traders hit zero cash they can never trade again for the rest of the run.
Confirmed impact: `gdp_raw`/`gdp_equivalent` collapse to ~0 after the
first ~1% of every run; `hdi_proxy`'s `capability_utilisation` sub-term
does the same, leaving `hdi_proxy` pinned at ~0.6 almost everywhere. H1
(Gini) and MTSC (collapse/survival) are unaffected — both depend on
wealth distribution/extraction dynamics from the still-active early
window, not on trade continuing. H2 and H3 are not testable on the
dataset as collected. See
`client_correspondence/Client_Update_M3_Cash_Sink_Finding_v2.md` for the
full writeup, including the fix options presented.

**Three fix options presented to the client**:
1. Extractors automatically spend cash on food/tool — same pattern as the
   existing Producer raw-material auto-purchase fix (no action-space
   change, no retraining).
2. Periodic passive cash redistribution — simpler, but mechanically
   applies a redistribution mechanism even under G1 ("No-Governance"),
   which sits awkwardly against that condition's premise.
3. A new "spend" action_type on Extractor's action space — most literal/
   emergent, but widens the action space and invalidates every already-
   trained model, for the same reasons this was already rejected once for
   the symmetric Producer fix (see "Producer raw-material acquisition —
   resolved" above).

**Client's decision**: Option 1, explicitly reasoned as the project's
pre-registered commitment to testing all four hypotheses (H1/H2/H3/MTSC),
not just the two currently unaffected. Full 360-run batch to be
regenerated once implemented. Client also requested: (a) a caveat note on
the already-collected dataset — added at `data/README.md` and folded into
the existing compressed archive and `Zenodo_Dataset_Description.md`; (b)
re-verification of all four hypotheses' preview charts against the
corrected dataset before anything is built on top of it, the same way
this issue was originally caught; (c) this logged decision entry.

## KPILogger capability_utilisation read-after-reset fix (found during Option 1 smoke check, 2026-08-07)

**Found while smoke-checking the Extractor cash-sink fix** (above), before
committing to the real relaunch: fixing the cash sink restored `gdp_raw`/
`gdp_equivalent` (confirmed nonzero throughout a 2000-step smoke run), but
`hdi_proxy` stayed pinned at exactly ~0.6 regardless. Root cause is
unrelated to the cash sink: `AegoraEnv.step()` (`aegora_env.py:606-618`)
captures the cycle's production value into `_last_cycle_gdp_raw` and, in
the *same* call, resets `_cycle_production_value` to `0.0` for the next
cycle. `run_experiment.py`'s driving loop calls `logger.log_step()`
immediately after noticing the cycle rolled over -- i.e. after that reset
already happened. `KPILogger.log_step()` was reading `env._cycle_production_value`
directly for HDI's `capability_utilisation` term, so it always read the
just-reset `0.0` -- structurally, at every epoch boundary, in the entire
real dataset, independent of the cash-sink bug or its fix. `gdp_raw` never
had this problem, since it reads the separately-and-correctly-captured
`_last_cycle_gdp_raw` instead.

**Fix**: added `AegoraEnv._last_cycle_production_value`, captured
immediately before the reset (mirroring `_last_cycle_gdp_raw` exactly, in
both `reset()` and the end-of-cycle block in `step()`). `_cycle_production_value`
itself is left untouched as the live, in-progress accumulator (still read
directly by `_snapshot_state` for per-step Governor observations -- a
different, correct consumer). `KPILogger.log_step()` now reads
`_last_cycle_production_value` instead. Regression test:
`tests/test_kpi.py::test_capability_utilisation_ignores_already_reset_live_accumulator`.

Confirmed via smoke re-run (`G1_A_001`, seed 1, 2000 steps, both fixes
in place): `hdi_proxy` now ranges ~0.76-0.97 (mean ~0.88, std ~0.056)
instead of pinned at exactly 0.6. Both fixes ship together in the same
relaunch -- no separate batch needed for this one.

## PPO_N_STEPS=64 pilot -- no effect, reverted (2026-08-07)

Client-approved "Option A" test for the G2/G3 redistribution-reactivation
finding (see "Extractor cash-sink fix" and this file's redistribution
entries): halved `PPO_N_STEPS` from 128 to 64 to double Extractor/
Producer/Trader's gradient-update frequency within the same 10,000-step
episode (spec-compliant, no horizon change), on the hypothesis that more
frequent updates would help policies adapt to G2/G3's now-active
redistribution mechanism.

Tested directly: ran the exact 28 replications that were stuck
identically across both `EXTRACTOR_SPEND_FRACTION` pilots (0.05 and
0.15), single attempt each at their real seeds, via `_run_batch` (same
engine the CLI uses per round). Result: **0 of 28 converged** -- no
improvement at all, same complete failure as before. Reverted to 128
per the change's own pre-stated condition. Rules out update-frequency as
the lever; the difficulty looks more fundamental than either spend
magnitude or gradient-update cadence.

## max_timesteps=15000 pilot (Option B) -- also no effect (2026-08-07)

Same 28 stuck replications, single attempt each, real seeds, `PPO_N_STEPS`
back at its reverted default (128) -- isolating "more raw training time"
as the only variable, extending the episode from 10,000 to 15,000 steps
(diagnostic-only; the spec's episode length is fixed at 10,000, so this
could never be used for final delivery even if it had worked). Result:
**0 of 28 converged** -- identical complete failure to both the spend-
fraction test and the PPO_N_STEPS=64 test.

Three independent, targeted interventions now tested against the same 28
replications -- spend magnitude, update frequency, and training horizon
-- and all three show 0% improvement. This is a strong, consistent signal
that G2/G3's post-reactivation instability isn't resolvable by tuning
training budget or dynamics parameters in the ranges tested; see
DECISIONS.md's redistribution-reactivation entries for the underlying
mechanism.

## Convergence measurement — two defects corrected (2026-08-09, client-directed)

**Context.** The M3 corrected relaunch reported 289/360 non-converged and
0% convergence for G3 across all four scenarios. Three remediation attempts
(`EXTRACTOR_SPEND_FRACTION` 0.05 vs 0.15; `PPO_N_STEPS` 128 vs 64;
`max_timesteps` 10,000 vs 15,000) each returned **0/28 improvement** on the
stuck subset. We proposed accepting "structural non-convergence" as a
finding; the client declined and asked whether the criterion itself had ever
been calibrated against dynamics with redistribution genuinely active. That
challenge was correct and led directly to both defects below.

**Neither was a real convergence failure. No agent policy ever failed to
converge.**

### Defect 1 — combined loss recorded where the spec says policy loss

Spec §6.3: *"rolling mean **policy loss** variance."*
`train_agents.py` recorded SB3's combined `train/loss`
(`policy_gradient_loss + value_loss*vf_coef + entropy_loss*ent_coef`).

The value term measures how *predictable* an agent's reward is, not whether
its policy has settled. Producer's reward is irreducibly noisy (random
counterparty via `_pick_counterparty`, partial fills), so its value loss
stayed permanently high and swamped the measurement. Measured on a real
stuck replication (`G1_A_001`, seed 5001): producer combined-loss variance
**0.303** (FAIL, 30x) vs policy-loss variance **0.000072** (PASS, 139x
under threshold); value loss alone **1.245**.

This also explains the pre/post-cash-sink-fix discrepancy: before that fix
the economy froze at ~step 100, Producer's reward was a constant ~0, a
constant is perfectly predictable, value loss ~0, and it passed
automatically. **Producer was passing convergence because it was
economically dead** — 0 failures pre-fix, 877 after.

**Validation**: re-running all 28 stuck replications on
`train/policy_gradient_loss`, threshold and window unchanged:
**16/28 converged, up from 0/28**, with every PPO policy passing in every
run and regime (2.5e-5 … 2.3e-4 vs 0.01).

### Defect 2 — the harvested-amount override was biased against G3

All 12 residual failures were `extractor_kpi`, 11 of them G3, and all
*marginal* (1.1x–4.1x threshold vs producer's earlier 337x).

`_record_extractor_extraction` read `_step_log["extracted"]`, which
`_apply_extractor` writes **after** `governance_module.on_action` clips the
action to the cap. So the recorded amount is post-cap and tracks the cap:

| | |
|---|---|
| G3 Governor cap variance, last 20 epochs | **0.01819** |
| `extractor_kpi` failure variance, median | **0.01768** |

Same number. G2's cap is fixed at 3.0 (11/12 pass); G1 never applied the
check (4/4 pass); G3 sweeps `[0.5, 5.0]` by design (1/12 pass). The
criterion measured **Governor adaptivity, not Extractor convergence** —
penalising G3 for exactly what spec Table 2 defines it to be.

Second-order: the override's original justification (sanction noise
contaminating loss variance) was itself an artifact of Defect 1 — sanction
randomness enters via reward → value-function error → combined loss.
Measured on policy loss, the Extractor passes comfortably in every G2/G3
run. This **supersedes** the earlier "Authoritative-per-regime convergence
check" entry.

### Client decisions (email, 2026-08-09)

1. Fix Defect 1 — record pure policy-loss variance. Threshold (0.01) and
   window (20 epochs) untouched; this *ends* a spec deviation.
2. **Retire the harvested-amount override entirely** — explicitly *not*
   normalise it by the cap. Single original criterion, all regimes.
3. Do **not** pre-emptively tighten the threshold. Log value-function
   variance alongside as a secondary diagnostic and let the corrected
   re-run show whether tightening is warranted.

Client's two proposed tests (redistribution phase-in; entropy annealing) are
**on hold**, to be run only on whatever genuinely still fails. Note his
entropy test is not runnable as described: `g3_adaptive_rl.yaml` sets
`ent_coef: auto` (SAC dual-optimisation against a target entropy), so there
is no annealing schedule to slow — the adjacent lever is `target_entropy`.

### Implementation

- `train_agents.py`: records `train/policy_gradient_loss` into the gating
  monitor; `train/value_loss` into a new, purely observational
  `value_diagnostic_monitor` that never gates status. Removed
  `_record_extractor_extraction`, `_maybe_record_epoch_kpi` and their state.
- `run_experiment.py`: `_classify_convergence(monitor, agent_types)` — the
  `kpi_monitor`/`extractor_signal` parameters and substitution branch are
  gone. New `_write_loss_diagnostics` emits **`value_loss_diagnostics.csv`**
  (per run, per agent type, both variances) for *every* run, not just
  failures — a new artifact alongside `attempts_summary.csv`, leaving the
  spec-locked KPI CSV/JSON schemas and `CONVERGENCE_FAILURES_COLUMNS`
  untouched.
- `governance/`: **behaviourally untouched** by user decision.
  `has_stochastic_extractor_sanction` and its 3 tests are retained as a
  true regime capability; comment-only corrections were made in all three
  files because their docstrings claimed the runner reads it to select a
  convergence signal, which is no longer true.

### Correction to an earlier report

`client_correspondence/Client_Update_M3_Redistribution_Reactivation_v2.md`
attributed the non-convergence to G2/G3's redistribution mechanism
reactivating. **That causal claim was wrong and is withdrawn.** The
redistribution finding itself stands (G2 fired 100/100 events; G3 6,737 of
10,000 cycles) — it is simply not what caused the failures. The
under-weighted clue: Producer was failing in G1 too, and G1 has no
redistribution mechanism.

## Spec §2.3.1 events — the two missing types (2026-08-09)

Spec §2.3.1 requires **three** environmental event types. Only
``disruption`` was ever implemented; ``market_volatility`` (temporary price
multiplier on all trades for N steps) and ``agent_withdrawal`` (random agent
idle for M steps) were absent from the codebase entirely — no code, no
tests, no prior DECISIONS.md entry recording it as a deliberate omission.
Found while investigating why the Trader holds no wealth.

Consequence beyond tidiness: Scenarios B/D are the "high event frequency"
arms H3 depends on, and were firing one of three specified mechanisms —
notably not the one called "market volatility".

**Implementation.** `ResourcePool` rolls all three (the spec assigns event
injection to that module) but applies only `disruption`, since it owns the
stock; the other two are emitted via `pending_events()` and applied by
`AegoraEnv`, the only layer holding both the Market and the agent registry.
This preserves the dependency direction (Market imports ResourcePool, never
the reverse).

**Spec-ambiguity resolutions** (the spec gives magnitudes only for
disruption's 10-30% cut):
- Each type rolls **independently** at `event_rate/100`. The spec fixes the
  rate but not how it splits. Independent rolls keep disruption frequency
  identical to the previous single-event implementation, so the Scenario C/D
  tuning agreed with the client remains valid.
- `MARKET_VOLATILITY_MULTIPLIER_RANGE = (0.7, 1.5)`, durations `(10, 50)`,
  `AGENT_WITHDRAWAL_DURATION_RANGE = (10, 50)`.
- Governor agents are excluded from withdrawal targeting: withdrawing the
  policy-setter would silently disable the regime under test.

**Measured effect: near zero.** 144-run pilot with all three vs disruption
only — convergence 144/144 either way; Gini 0.574/0.697/0.694 →
0.568/0.696/0.695; Trader wealth share 0.0% → 0.0%. The events fire hard
(food price 8.36-18.58 against a base of 10.0, active 54% of cycles), but
cannot move these numbers: volatility without a spread is a *cost* to a
middleman, not an opportunity, and withdrawal idles 0.89 of 50 agents at
steady state (`p x mean duration = 0.03 x 30`), below the noise floor.
Counterintuitively the ungoverned commons survived slightly *longer*
(G1-D 503 → 554 steps) because withdrawn agents stop extracting.

**Kept** — not because it changes results, but because the methodology
describes three event types and the code must match.

## Trader bid-ask spread — implemented, disabled by default (2026-08-09)

The spec defines the Trader as "Market Exchange" with reward "net trade
profit" and an observation of market prices + trade history, but never
defines any mechanism making intermediation profitable. With a single global
price, buy and sell are wealth-neutral by construction: measured Trader
wealth fell 10.00 → 3.41 over 1500 steps while Producers grew 2882x
(13 → 38,484). Producers create 10-20x value per step (raw ~1.0 → food 10.0
/ tool 20.0) and compound; the Trader's capital is pinned at its starting
endowment.

**Implemented** the standard market-maker mechanism in
`Market._execution_price`: Trader as buyer pays the bid `mid*(1-s)`, as
seller receives the ask `mid*(1+s)`, trader-to-trader and non-trader pairs
transact at mid. Role read from the agent-id prefix, matching the existing
`_pick_counterparty` convention (role dispatch, not governance branching).

**Validated at 0.05 on a 144-run pilot: it works.** Trader wealth 7.88 →
582.90 at t=2000 (74x), compounding (81 → 190 → 583), convergence unchanged
at 144/144.

**But it flips H3.** G3-minus-G2 GDP-equivalent went from **+0.179**
(Cohen's d +0.40, p=0.057 — supports "adaptive beats rule-based") to
**-0.098** (d -0.24, p=0.251 — contradicts it). Neither is significant at
n=12, but the direction reverses and at the real n=30 it could resolve
either way — whichever is baked in.

**Decision: `TRADER_SPREAD = 0.0` (disabled).** A non-spec mechanism that
reverses the direction of a pre-registered hypothesis is not ours to adopt —
that would be engineering the answer rather than measuring it. The mechanism
is left fully implemented and tested (6 tests, including one asserting the
shipped default is 0.0) so the client can enable it with a one-constant
change if they decide the Trader must be economically real. H1 is unaffected
either way: inverted in all three pilot configurations, AC-08 holds 0/12
throughout.

## Convergence threshold recalibrated from data (2026-08-10, client-directed)

**Not a silent re-tune.** The client wrote: *"agreed, the current one is too
permissive to mean anything. Please derive a properly calibrated threshold
from the value-function diagnostic data you already logged... rather than
picking a number ad hoc."*

**Old value 0.01** — pre-registered but never calibrated against this
implementation. On the combined loss it ran ~337x too tight; on the spec's
actual quantity (policy loss) ~30x too loose. All 921 measurements in the
360-run dataset passed it; the maximum observed variance was 30x below it.

**New value 2.5e-4**, derived from that dataset's own distribution via
Tukey's far fence (Q3 + 3*IQR), the conventional definition of an extreme
outlier: Q3 = 1.143e-04, IQR = 4.53e-05 -> 2.509e-04.

Candidates considered, with runs failed (of 307 with a verdict):
Tukey inner (Q3+1.5*IQR) = 1.83e-4 -> 32; **Tukey far = 2.51e-4 -> 4**;
mean+3SD = 2.15e-4 -> 16; p99 = 2.31e-4 -> 10; p99.5 = 2.49e-4 -> 5.
Tukey far chosen: conventional, data-derived, and the only candidate that
does not exclude a double-digit share of runs behaving identically to the
rest.

**The calibration's own finding, recorded honestly:** the policy-loss
distribution is unimodal (peak 7.3e-5 - 9.9e-5), mildly skewed (+1.36), with
max/median = **3.7x**. There is no bimodality and no natural cut point. A
genuine converged/non-converged split separates by orders of magnitude --
which the defective combined-loss measurement did show. It does not appear
here. **Every run converges tightly; the criterion has nothing left to
detect.** The new threshold flags distributional outliers, not demonstrated
training failures. This belongs in the methodology as a property of the
instrument.

**Why not derived from the value-loss series**, as the client's wording
suggested: value-loss variance tracks reward *magnitude*, not policy
stability. Medians -- extractor 48.5, producer 22.5, **trader 1.5e-10**
(near zero because the Trader earns nothing, so its reward is trivially
predictable). A criterion built on it would fail 99-100% of Extractors and
Producers and pass 100% of Traders. Retained as a diagnostic only.

**Re-classification, not regeneration.** The 360-run batch had already
completed when the instruction arrived (sequencing disclosed to the client).
Because per-run variance is logged, the dataset was re-classified rather
than re-run: 4 replications changed status (G1_C_001, G2_B_022, G2_C_027,
G2_C_030), all between 1.00x and 1.33x the new threshold. Re-run at
seed+1000 per the replacement protocol; **all four converged first retry**.
Final: 295 CONVERGED, 65 CONVERGING, 364 total attempts for 360
replications. Every H1/H2/H3 conclusion is unchanged.

`CONVERGENCE_WINDOW` (20) and `MIN_EPOCHS_BEFORE_CHECK` (50) are untouched.
`tests/test_convergence.py::test_pre_registered_constants_locked` still
guards against further unannounced edits, now locking 2.5e-4.

## AC-08 — the "80%" originates in the specification itself (2026-08-10)

The client asked where the 80% heuristic came from, not recognising it. It
is in the spec document, §9.1 paragraph 180 (M3 payment release): *"G1 Gini
> G2 Gini > G3 Gini in at least 80% of Scenario A runs."*

The spec also contains a **second, different** version in Table 11 (AC-08):
*"Mean Gini G1 > G2 > G3 across >= 20 pilot runs in Scenario A."* A per-run
success rate and a comparison of means are not the same test. Client has
directed that H1 be reported against the pre-registered ANOVA/Tukey
framework instead. Both spec locations flagged to him for correction in the
methodology text.

Related: the "pool-directed" redistribution wording we referenced is also
genuinely in the spec (§2.2.1, paragraph 35), but the client's July design
record and this file's existing sign-off entry both specify agent-to-agent.
Spec-internal inconsistency, not an implementation error. Confirmatory
design unchanged; a pool-directed exploratory variant offered separately.

## Dashboard — two design forks resolved (2026-08-11)

Spec §7 requires a Streamlit observability app at the exact path
`aegora_isle/dashboard/app.py`. Two questions had no spec answer.

**1. Live mid-run KPI plotting — replication-level only.** `KPILogger`
writes each replication's CSV exactly once, when that replication finishes
(`kpi_logger.write_csv`). There is no incremental write, so mid-run there is
no partial file to read. The options were to add incremental flushing to
`KPILogger` — the single module that produced the entire confirmatory
dataset — or to report honest completion counts. **Chose the latter.** The
Experiment Progress view shows completed-vs-expected replications and plots
KPIs from replications already on disk. A synthesised in-flight curve would
have looked better and meant less; the caption in the UI says so explicitly
so a viewer cannot mistake it for a within-run trace.

**2. Run/Pause/Reset — `psutil`, not POSIX signals.** The client intends to
deploy on Windows, which has no `SIGSTOP`/`SIGCONT`. `psutil.Process.suspend()`
maps to `SIGSTOP` on Unix and `NtSuspendProcess` on Windows, so one code path
covers both targets and real pause/resume survives rather than being dropped
from the deliverable. Two details that are not incidental:

- **Children are suspended too.** `run_experiment` dispatches through
  joblib; suspending only the parent would leave the workers computing while
  the UI claimed "paused". Verified live, not just against mocks: parent and
  joblib child both reached state `T` at 0% CPU, and both returned to `R` on
  resume.
- **Reset resumes before terminating.** A suspended process cannot handle
  `SIGTERM`, so Pause-then-Reset would otherwise strand workers holding CPU.

`data_access.py` and `process_control.py` deliberately import no Streamlit,
so both are pytest-testable without a server (`tests/test_dashboard.py`, 24
tests). `app.py` is thin glue, verified in a browser across all seven views
against the real 360-run dataset, including a live Run/Pause/Reset cycle
against a throwaway `--output_dir` so `data/` was never written to.

**Collapse-marker definition.** The Collapse Events view marks the CSV's
`collapse_event` (professor's subsistence definition), not the spec Table 4
MTSC definition used by `analysis/`. The two disagree — 258/360 versus
much lower — and the view states this on screen. For an observability tool
the right signal is what the run itself recorded; the statistical analysis
keeps Table 4. See the entries above on the collapse-definition split.

## G1_C_001 provenance defect — found, fixed, guarded (2026-08-11)

Found while reproducing the H1 figures to answer a client question about
Cohen's f. Re-running the analysis from `data/` gave f = 6.85 where the
delivered report said 6.93.

**Cause.** One replication of 360, `G1_C_001`, was present on disk as its
*original* attempt (seed 201) rather than the converged replacement
(seed 1201) that `attempts_summary.csv` records. The original is the attempt
that failed the convergence criterion and was excluded under the replacement
protocol -- `convergence_failures.csv` has exactly one row, and it is this
run. The correct replacement was in place when the delivered analysis was
produced, and was reverted afterwards. The exact mechanism is not
recoverable: no backup directories survive, and `data/runs` mtimes show a
bulk touch across 00:44-01:07 that could not be attributed to any operation
still visible.

**The batch retry loop is not implicated.** Three independent reasons:
`run_id` derives only from (governance, scenario, replication) and never
from seed, so a replacement necessarily overwrites in place; rounds are
sequential, so no slow original can clobber a replacement; and three of the
four replaced runs (G2_B_022, G2_C_027, G2_C_030) carry their correct
seed+1000 values. Those three never appear in `convergence_failures.csv` at
all -- they were flagged by the post-hoc threshold recalibration, not by the
batch, so all four were re-run as a manual step afterwards. That manual step
is where the one file was lost.

**Scope, verified rather than assumed.** Every one of the 360 sidecar seeds
was compared against `attempts_summary.csv`: exactly one mismatch. Every CSV
matches the SHA-256 in its own sidecar, so nothing was corrupted -- it was
the wrong *version* of one file, not a damaged one. `G1_C_001`'s CSV, its
`random_seed` column, and its sidecar all agreed on 201, which is precisely
why no integrity check caught it.

**No delivered figure changed.** With the correct replication restored, all
eight analysis tables reproduce **byte-identically** against the versions
already sent to the client. Cohen's f = 6.93 is confirmed. The earlier 6.85
was an artifact of the drifted dataset, not a defect in the report.

**Fix.** Re-run at seed 1201 under current code (converged first attempt);
seed, schema, row count, and checksum verified before installing; manifest,
sample, archive, and analysis regenerated.

**Guard.** `scripts/verify_dataset.py` now cross-references each run's
sidecar seed against `attempts_summary.csv`. Checksums cannot catch this
class of defect, because the stale file is perfectly consistent with itself;
only the attempt log exposes it. This ships with the dataset so a reviewer
can verify provenance independently.

## Client-directed closeout changes (2026-08-12)

Two changes requested by the client on accepting the M4 milestone.

**1. Effect-size footnote added to the report.** The client asked whether
Cohen's f had been computed from the standard or the Welch F-statistic,
given that Levene's test failed for Gini, HDI and depletion rate and Welch
was substituted. It is computed from neither: `pingouin.welch_anova` derives
`np2` from the *unadjusted* sums of squares, `ss_bet / (ss_bet + ss_res)`,
while the Welch-adjusted term feeds only the F-statistic, ddof2 and the
p-value. Verified empirically -- standard and Welch ANOVA return partial
eta-squared identical to six decimal places (0.979613, f = 6.9318) despite
F-statistics of 8576.99 and 4535.09. `analysis/report.py` now states this in
the "How to read this document" section, so a reader does not have to infer
it.

**2. `subsistence_collapse` removed from the run-index builder.** The field
was computed in `summarise_run` and read by nothing -- it reached no table,
no figure, and no output file. Retained originally as a breadcrumb for the
collapse-definition disclosure note; removed at the client's request so the
codebase does not imply an ambiguity the analysis does not have. The
subsistence definition still exists where it belongs: in the delivered CSVs'
`collapse_event` column (written by `kpi_logger.compute_subsistence_collapse`,
which is untouched -- it produces the locked schema) and in the disclosure
note in `data_loading.py`'s module docstring.

All eight analysis tables reproduce unchanged after the removal, confirming
the field was genuinely dead.

## Batch-summary truncation and the verifier gap (2026-08-12)

A second truncation, found when the diffstat of the previous commit showed
unexplained deletions. `data/attempts_summary.csv` had been reduced to its
120 G1 rows, `cell_attempt_summary.csv` to 4 of 12 cells, and
`value_loss_diagnostics.csv` to G1 only. The loss propagated into
`data/sample/` and into the built archive, because `package_dataset.py`
copies these files verbatim.

**Effect on delivered work: none.** The report already in the client's hands
was generated before the truncation and states the correct 295 CONVERGED /
65 CONVERGING. But regenerating it afterwards would have read 55/65 --
`report._status_counts` reads this file by design, precisely so the caveat
tracks the dataset. A reader re-running the analysis would have got numbers
that contradicted the delivered document.

**Recovery.** These files are gitignored at their `data/` paths, so they have
no history there. They survive only because `data/sample/` *is* tracked and
carries verbatim copies; restoring from `92b29fc` recovered all 360 rows,
12 cells, and the G1_C_001 seed-1201 entry intact.

**The verifier gap this exposed.** `check_seed_consistency` compared each
run's sidecar seed against `attempts_summary.csv` but skipped runs with no
entry in it. With the summary truncated to G1, 240 runs were silently
unverified and the check still reported success. A provenance check that
passes because the reference file is missing two thirds of its rows is worse
than no check, since it produces false confidence. It now reports absent runs
as findings in their own right.

**Cause, still unknown.** As with the G1_C_001 revert, no backup survives and
the mtimes cannot be attributed to any visible operation. Both truncations
share a shape -- a full-dataset file reduced to its G1 subset -- which is
consistent with something writing a single-condition batch's summaries over
the full ones, but that is inference, not evidence.
