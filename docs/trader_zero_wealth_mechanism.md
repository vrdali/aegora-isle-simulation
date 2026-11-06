# Why Traders end at zero wealth under G2/G3 (2026-11-06)

Evidence
- Redistribution (rule_based.py `_redistribute`, adaptive_rl.py `_redistribute`) removes `redistribution_rate` x cash from every Producer and Trader and pays it equally to Extractors.
- G2: rate 0.10 every 100 steps. G3: rate ~0.17-0.24, applied almost every step (9,913 events in G3_A_001).
- Per-step pilot (scenario A, seed 1, economic scope): mean Trader wealth G1-matched ~10 throughout; G2 10.0 -> 9.6 by step 200 (-> 0 by step 10,000: 10 x 0.9^99); G3 10 -> 7.6 at step 1, 0.8 at step 20, 0.08 at step 50, 0.0 from step 100.
- Traders hold only cash and a little inventory and earn only trade profit, which in G1 roughly breaks even (wealth stays ~8-10).

Interpretation (not yet tested)
- The zero-wealth outcome is a direct consequence of a proportional cash levy on agents with no income stream, not (necessarily) a learning failure. Under G3 the SAC governor learns high-frequency redistribution because the reward weights equality (0.4), and the Gini over the remaining population does not penalise the destitute Traders in an equality-improving direction.

Proposed test
- Ablation: G2 and G3 with redistribution_rate = 0 (or Traders exempt) vs unchanged; and G2 with period 1 and rate matched to G3. Prediction: Trader wealth stays ~10 and the collapse state rate falls to the G1-matched level if the levy is the cause.
