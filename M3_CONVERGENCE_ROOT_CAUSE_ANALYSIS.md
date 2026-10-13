# M3 Convergence Failure — Root Cause Analysis

**Date:** 2026-08-08
**Status:** **VALIDATED on all 28 stuck replications — 16/28 now converge
(from 0/28).** Two distinct measurement defects identified, both confirmed
quantitatively. Nothing implemented in the repo; nothing sent to the client
yet.

> **Summary for a returning reader:** the M3 "structural non-convergence"
> result was **not** structural. It was two independent measurement bugs
> stacked on top of each other. Defect 1 (combined-vs-policy loss) resolves
> 16 of 28. Defect 2 (`extractor_kpi` vs G3's moving cap) accounts for all
> 12 remaining, and is structurally biased against the G3 condition by
> construction. No agent policy ever failed to converge.

---

## The finding

Spec §6.3 states:

> "Compute rolling mean **policy loss** variance across the most recent 20
> epochs. If variance < 0.01: mark as converged."

`aegora_isle/experiments/train_agents.py:308` records
`model.logger.name_to_value.get("train/loss")` — SB3's **combined** loss
(policy_gradient_loss + value_loss x vf_coef + entropy_loss x ent_coef),
**not** policy loss. This is documented in that module's own docstring
(lines 40-41) as a deliberate choice, but it does not match the spec's
wording.

Measured all three scalars separately on a real stuck replication
(`G1_A_001`, seed 5001, 6000 steps, G1/Scenario A):

| agent | metric | var(last 20 epochs) | vs 0.01 threshold |
|---|---|---|---|
| producer | `train/loss` (what we record) | 0.303 | FAIL — 30x over |
| producer | `train/policy_gradient_loss` (what spec says) | 0.000072 | **PASS — 139x under** |
| producer | `train/value_loss` | 1.245 | FAIL |
| extractor | `train/loss` | 0.00074 | PASS |
| extractor | `train/policy_gradient_loss` | 0.00015 | PASS |
| extractor | `train/value_loss` | 1.37e-10 | PASS |
| trader | `train/loss` | 0.000034 | PASS |
| trader | `train/policy_gradient_loss` | 0.0000072 | PASS |
| trader | `train/value_loss` | 2.90e-13 | PASS |

That run was classified `NON_CONVERGED` on `producer`. **Its policy had
converged — 139x below threshold.** The failure is entirely in the
*value function* term, which dominates the combined scalar we measure.

---

## Why this explains the whole history

Producer's reward is genuinely stochastic: production depends on which
random Extractor `_pick_counterparty` selects and whether that extractor
happens to hold raw material (plus partial-fill semantics). The value
function structurally cannot predict that precisely, so value loss stays
high and fluctuating. **That is irreducible environmental randomness, not
policy instability.**

- **Pre-fix**: producer's reward was ~0 constant (no production possible
  once the cash sink drained Producers/Traders at ~step 100) -> value loss
  ~0 -> combined loss ~0 -> "CONVERGED". Producer was passing the check
  **by accident, because the economy was dead.** This is why `producer`
  has **0 failures** in the archived pre-fix dataset and **877** in the
  corrected relaunch.
- **Post-fix**: reward is real and noisy -> value loss real -> combined
  loss fails.

It also explains why all three remediation attempts returned exactly 0/28:
spend fraction (0.05 vs 0.15), update frequency (`PPO_N_STEPS` 128 vs 64),
and training horizon (10,000 vs 15,000 steps) **cannot tune away
irreducible reward noise inside a value function.**

### Supporting evidence already on file

- `producer` failures: **0 pre-fix, 877 post-fix**. 100% of the 295 stuck
  runs involve `producer`.
- `extractor` / `extractor_kpi` failure variances are essentially
  **unchanged** pre vs post (median 0.043 -> 0.036, and 0.018 -> 0.017) —
  those agents behave identically before and after the fix.
- `producer` median failure variance is **3.37 = 337x threshold**, versus
  `extractor` at 4x. Two orders of magnitude apart.

---

## CORRECTION to the redistribution-reactivation report

The document
`client_correspondence/Client_Update_M3_Redistribution_Reactivation_v2.md`
is **wrong on causation** and must not be sent as written.

The redistribution fire-rate evidence itself is real and verified (G2:
100/100 events fired; G3: 6,737/10,000 cycles). But it does **not** explain
the convergence failures. The clue that was under-weighted: `producer`
fails in **G1 as well**, and G1 has no redistribution mechanism at all.

The redistribution reactivation remains a genuine, documentable finding
about G2/G3 mechanics — it is simply not the cause of non-convergence.

---

## Why the policy-loss fix is the strongest option

It is a **bug fix toward the spec, not a deviation from it**:

- Threshold stays **0.01** — unchanged
- 20-epoch window — unchanged
- Exclusion/replacement protocol — unchanged
- Only the *measured scalar* changes, to the one the spec explicitly names

This means it does **not** trip CLAUDE.md's "never re-tune the convergence
threshold" rule, nor spec §12.2 ("Convergence threshold (0.01) and
exclusion protocol must not be changed"). We would be **ending** an
existing deviation rather than creating a new one.

### Honest caveat (must be disclosed to the client)

PPO's policy-gradient loss is naturally tiny (~1e-5 range), so a 0.01
threshold may rarely bind — the convergence check risks becoming close to a
rubber stamp. This is a real methodological concern, not a detail to gloss.

Two mitigations worth offering the client:
1. Report **value-loss variance alongside** as a secondary diagnostic, so
   value-function instability is still visible and logged.
2. Propose a **tighter threshold specifically for policy loss**, calibrated
   to its natural scale.

Either way this is the client's call — it touches pre-registered
methodology.

---

## Other ideas, ranked below the above

2. **Reduce reward stochasticity at source.** Let Producers source raw
   material from the best-available Extractor (or pool across extractors)
   instead of one random pick via `_pick_counterparty`. Cuts the
   irreducible noise driving value loss. Real environment-mechanics change
   (same category as the cash-sink fix) — needs client sign-off, and would
   need its own validation run.
3. **Tighten the reward-normalisation clip** (currently `reward_clip=5.0`
   in `ManualPPOCollector`). Cheap and quick to test, but likely just
   shifts numbers rather than addressing the mechanism.
4. **Deliver M3 on the archived pre-fix dataset**, with the corrected
   dataset attached as a robustness appendix. Pragmatic fallback if
   everything above is rejected. Note the tradeoff: the archived dataset
   satisfies AC-09 (360 CSVs present) and can be evaluated for AC-08
   (Scenario A Gini ordering), but carries the GDP/HDI defect that makes
   H2 and H3 untestable — which is precisely what the client chose to fix.

---

## VALIDATION RESULT (run 2026-08-08) — Defect 1 confirmed

Ran all 28 stuck replications, single attempt each at their real seeds,
feeding the existing `ConvergenceMonitor` `train/policy_gradient_loss`
instead of `train/loss`. Threshold (0.01) and window (20 epochs)
**unchanged**. No repo files modified — patch applied inside each joblib
worker (loky spawns fresh processes on macOS, so a parent-process
monkeypatch would not propagate).

Script: `scratchpad/validate_policy_loss_criterion.py`
Output: `data/validate_policy_loss/` (gitignored)

**Result: 16 of 28 CONVERGED, up from 0 of 28.**

| regime | converged | uses `extractor_kpi`? | cap behaviour |
|---|---|---|---|
| G1 | 4/4 | no | n/a |
| G2 | 11/12 | yes | **fixed at 3.0** |
| G3 | 1/12 | yes | **dynamic [0.5, 5.0]** |

**Every PPO policy loss passed, in every run, in every regime** —
extractor / producer / trader all in the range 2.5e-5 to 2.3e-4 against a
0.01 threshold. Zero policy-loss failures across 28 runs. The agent
policies were converging the entire time.

---

## DEFECT 2 — `extractor_kpi` is structurally biased against G3

All 12 remaining failures are on `extractor_kpi` (the harvested-amount
flatness check): 11 G3, 1 G2. They are **marginal** — 1.1x to 4.1x
threshold, median 1.8x — versus producer's 337x under the old metric.

### Quantitative smoking gun

| quantity | value |
|---|---|
| G3 Governor's cap variance, last 20 epochs | **0.01819** |
| `extractor_kpi` failure variance, median | **0.01768** |
| threshold | 0.01 |

These are the same number.

### Mechanism (verified in code, not inferred)

`ManualPPOCollector._record_extractor_extraction` reads
`self.env._step_log[agent_id]["extracted"]`. That log entry is written by
`AegoraEnv._apply_extractor`, which receives `mod_action` — the action
**after** `governance_module.on_action` has clipped it to the cap. So the
recorded harvested amount is **post-cap** and tracks the cap directly.

- **G3**: the Governor sweeps the cap across `[0.5, 5.0]` by design
  (observed epoch means 2.461 - 3.164). Harvested amount follows it.
  Harvested-amount variance therefore *is* cap variance.
- **G2**: `extraction_cap: 3.0`, fixed. No cap-driven variance component.
  11/12 pass.
- **G1**: no stochastic extractor sanction, so `extractor_kpi` is never
  authoritative. 4/4 pass.

**Conclusion: in G3, `extractor_kpi` measures the Governor's adaptivity,
not the Extractor's convergence.** It penalises G3 for doing exactly what
spec Table 2 defines G3 to be: *"Active RL learner (SAC) — adjusts
governance params in real time."* The criterion is structurally
incompatible with the condition it is judging.

### Second-order finding

The original justification for introducing the `extractor_kpi` override
(stochastic *sanction* noise contaminating loss variance) appears to have
been **itself an artifact of Defect 1**. Sanction randomness enters via the
reward -> value-function prediction error -> combined loss. Measured on
policy loss, extractor variance passes comfortably in *every* G2 and G3
run. The override may no longer be needed at all.

Note: this override was adopted with explicit client sign-off, so removing
or changing it requires going back to them — it is not ours to revert
unilaterally.

---

## Options for the remaining 12

1. **Drop the `extractor_kpi` override.** Its justification rested on the
   buggy metric. Requires client sign-off (they approved adopting it).
2. **Normalise harvested amount by the current cap** (`harvested / cap`).
   Removes the cap-driven component while retaining a genuine
   extractor-behaviour stability check. More conservative — **preferred**.
3. **Keep it as-is and document G3 as a limitation.** Weakest option now
   that the mechanism is understood: it would document a measurement
   artifact as a scientific finding.

---

## Bearing on the client's two proposed tests

The client proposed (a) phasing in redistribution as a curriculum, and
(b) slowing G3's entropy-coefficient annealing. Both target **learning
dynamics**. The remaining blocker is a **measurement criterion** that is
biased against G3 by construction.

Two concrete cautions:

- Both tests could "succeed" by making the Governor **less adaptive**
  (a flatter cap trajectory lowers harvested-amount variance and passes
  the check) — which would be scientifically the wrong outcome, and would
  bake a wrong causal story into the thesis.
- Test (b)'s premise does not hold as stated: `g3_adaptive_rl.yaml` sets
  `ent_coef: auto`, i.e. SB3 SAC's automatic entropy tuning via dual
  optimisation against a target entropy. **There is no annealing schedule
  to slow down.** The real adjacent lever is `target_entropy` (defaults to
  `-dim(A)` = -3 here), which can be raised to hold exploration higher for
  longer.

Recommendation: resolve Defect 2 first; run the client's tests only on
whatever fails afterwards.

---

## Reproduction commands

```bash
# Measure all three loss scalars on a single replication
uv run python -c "
import sys; sys.path.insert(0,'.')
import numpy as np
import aegora_isle.experiments.run_experiment as rem
from aegora_isle.experiments.train_agents import ManualPPOCollector
cap={}
orig = ManualPPOCollector._maybe_record_epoch_loss
def patched(self):
    for kind, model in self.models.items():
        nv = model.logger.name_to_value
        for key in ('train/loss','train/policy_gradient_loss','train/value_loss'):
            v = nv.get(key)
            if v is not None: cap.setdefault((kind,key),[]).append(float(v))
    return orig(self)
ManualPPOCollector._maybe_record_epoch_loss = patched
r = rem.run_single_cell(governance='G1', scenario='A', replication=1, n_agents=50,
    max_timesteps=6000, seed_override=5001, output_dir='/tmp/policyloss_probe',
    track_convergence=True)
ManualPPOCollector._maybe_record_epoch_loss = orig
print('status:', r['status'], r.get('non_converged_types'))
for (kind,key),v in sorted(cap.items()):
    a=np.array(v)
    if len(a)>=20: print(kind, key, float(np.var(a[-20:])))
"
```

---

# 144-RUN PILOT RESULTS (2026-08-09) — both fixes validated at scale

`data/pilot_corrected_criterion/` (gitignored). Commits `329fc5c` (both
defect fixes) + `14961f7` (EXTRACTOR_SPEND_FRACTION restored to 0.15).

**144/144 resolved in ONE round. Zero retries.** (Was: 289/360 unresolved
after 3 rounds.) G2 48/48 CONVERGED, G3 48/48 CONVERGED, all scenarios.
Both defects confirmed fixed at production scale.

## Permissiveness — confirmed, the check no longer discriminates

| agent | median policy-loss var | margin under 0.01 | worst case |
|---|---|---|---|
| extractor | 9.2e-5 | 108x | 39x |
| producer | 9.0e-5 | 112x | 45x |
| trader | 9.2e-5 | 109x | 38x |

**0 of 369 measurements within 38x of the threshold.** Note PPO's
policy-gradient loss is small *by construction* (clipped surrogate), so low
variance is partly an algorithm property, not run evidence. Neither
candidate quantity fits 0.01 — combined loss ran ~337x over, policy loss
~110x under — suggesting the pre-registered threshold was never calibrated
against this implementation.

Value-loss diagnostic: **63% of measurements exceed 0.01** (extractor
median 18.3, max 2,193). Real instability, visible, non-gating — as
specified.

## KPI findings

**H2 / MTSC — strongly supported.** Median steps survived:

| | A | B | C | D |
|---|---|---|---|---|
| G1 | 10000 | **1181** | 9054 | **503** |
| G2 | 10000 | 10000 | 10000 | 10000 |
| G3 | 10000 | 10000 | 10000 | 10000 |

Mean depletion: G1 +0.131 (net depletion) vs G2 -0.042, G3 -0.043 (net
regeneration). G1-B/D terminate when the pool drops below `critical_min`
(B: stock 70.8 vs min 100; D: 49.0 vs 50) — genuine ecological collapse,
*not* a training failure. Note `collapse_occurred` is False for these: the
JSON flag uses the subsistence-based KPI definition, which is a different
concept from pool-death (documented in DECISIONS.md "Collapse definition").

**H1 — INVERTED, and AC-08 cannot be met.** Final Gini:

| | A | B | C | D |
|---|---|---|---|---|
| G1 | **0.544** | 0.593 | 0.556 | 0.603 |
| G2 | 0.697 | 0.698 | 0.698 | 0.696 |
| G3 | 0.694 | 0.695 | 0.694 | 0.694 |

Scenario A only (no run-length confound, all reach 10,000 steps):
G1 0.544 < G2 0.697 ≈ G3 0.694. **One-way ANOVA F=1885.8, p=9.6e-35.**
AC-08 ("G1 > G2 > G3 in ≥80% of Scenario A runs") holds in **0/12**.

### Mechanism (measured, wealth by role at t=2000, Scenario A, seed 1)

| role | G1 share | G2 share |
|---|---|---|
| producer | 88.0% | **92.7%** |
| extractor | 12.0% | **7.3%** |
| trader | ~0% | ~0% |

**Correction to an assumption carried earlier in this investigation:**
Producers, not Extractors, hold the overwhelming majority of *wealth* —
the earlier cash-sink analysis measured **cash**, but wealth is dominated
by goods inventory at market prices. Two compounding effects:

1. The **extraction cap** (3.0 vs MAX_EXTRACT_AMOUNT 5.0) suppresses the
   poorer role's income, concentrating wealth further in Producers.
2. **Redistribution only moves cash**, but Producer wealth is mostly
   inventory — so it touches a small slice of the real distribution and
   cannot offset (1).

Relevant: spec §2.2.1's literal wording is redistribution *"transferred to
resource pool"*; the pre-M3 reconciliation confirmed an agent-to-agent
cash transfer to Extractors instead (DECISIONS.md "Redistribution
mechanism — confirmed final"). A pool-directed transfer would not have
this wealth-concentration side effect. **Client decision, frozen by
sign-off — not ours to change.**

Caveat on confidence: the aggregate Gini effect is huge and consistent
(12 reps x 4 scenarios, p=9.6e-35). The *mechanism* rests on a single-seed
wealth-composition probe — directionally clear, but worth a broader check
before it is asserted in the thesis.

**H3 — data available.** GDP-equivalent (mean/var), A/C = low volatility,
B/D = high: G1 1.717/0.080 and 2.226/0.105; G2 3.068/0.250 and
3.173/0.138; G3 3.198/0.276 and **3.400/0.148**. G3 highest in both, gap
over G2 wider under high volatility — directionally consistent with H3.
Both governed regimes show higher variance than ungoverned at low
volatility, which matters for interpreting the interaction term. Full
two-way ANOVA is M4 `statistical_analysis.py` scope.

## Open decisions before the full 360

1. **AC-08 cannot be satisfied as written** — restate the criterion, or
   revisit the redistribution target (a design change needing client
   direction).
2. **Convergence threshold** — tighten now or accept permissive + document.
   Regenerating later is another full batch.
3. **G1-B/D** — confirm these cells contributing collapse/MTSC evidence
   rather than convergence data is acceptable.

Client draft: `client_correspondence/Client_Update_M3_Corrected_Pilot_Results.md`
