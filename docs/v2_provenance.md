# v2 re-run: provenance record

This file records which code produced which batch, the SHA-256 of each batch's
`dataset_manifest.csv`, and how each batch was verified. It was written on
7 November 2026, after batches A to G were complete; batch H was added the same day.

## Batches

| Batch | Contents | Runs | Code commit | SHA-256 of dist/dataset_manifest.csv |
|---|---|---|---|---|
| A_economic | G1, G2, G3; economic agents only | 360 | 13ee2ec | 7b003a25ce73a82cc8f3d62e1933e87001b918542511b45780b00b65c31707cb |
| B_all | G1, G2, G3; all agents (as in v1) | 360 | 13ee2ec | 1a86cd999e3410fee4f446dfadafd3a208988c4a6c177a8d6456f18632401b8c |
| C_g1_matched | G1 with the same inert Governor slots as G2/G3 | 120 | 9412ab0 | ee5d421750d18ccf1d3674c5fa4df1dcb2b6d0b0078ca625dcf5a4f5baaf1bfe |
| D_G2_traders_exempt | G2, Traders exempt from the levy | 120 | bae973d | 4b66925ec69e396468e59e59e969e41d314489710b97de12aeb4ea0414807bb8 |
| E_G3_traders_exempt | G3, Traders exempt from the levy | 120 | bae973d | 9c32ffd9e99f7498ec16ae7ae6fd53b8d4509ca3769505e98cb2ecc8a3456465 |
| F_G2_no_levy | G2, no redistribution levy | 120 | bae973d | e956e525abcee7fefbfddf4cd3ec60518d34dc7911feda114535d6fdb0c55237 |
| G_G3_no_levy | G3, no redistribution levy | 120 | bae973d | e5762d99487a63204ce9406f0724837555a59bc0ba8814907527d0e01d698d33 |
| H_G2_matched_frequency | G2 with the levy applied every step at 0.1053% (compounds to 10% per 100 steps), economic scope | 120 | a0a25bd (overrides `--redistribution_period 1 --redistribution_rate 0.0010530503`) | e2e220e42c4e36f1f320b4bad8097a14ac6a4ef69af1869c951ee666f26964fd |

All commits are ancestors of the tagged commit. Batches A and B were exported
without a `.git` directory, so the `git_commit` field in their
`environment.json` shows the parent checkout; this table is the authoritative
code identifier. Batches D to H were verified with `--expected_per_cell 30` and
`--manifest_sha256` set to the value in the table (the anchor is the batch's own
manifest hash and is not an independently published value).

## Environment (Dell production machine)

Windows 11, Python 3.12.10, NumPy 2.5.4, pandas 2.3.3, PyTorch 2.14.1 (CPU),
stable-baselines3 2.9.2. Settings: 30 replications per cell, 6 jobs, burn-in 50,
wealth logged every 100 steps.

## Known limitations of this record

- The v2 runner applied the post-hoc convergence threshold as its retry rule;
  7 (A), 8 (B), 3 (C) and 1 (G) runs have a second attempt. Every retained
  measurement is below the pre-specified 0.01 threshold. Original attempts are
  logged, but their files were overwritten.
- The collapse flag in each run's JSON sidecar combines stock collapse and
  subsistence events. `scripts/mtsc_stock.py` recomputes commons collapse from
  the stock and output series alone, and `scripts/state_analysis.py` computes the
  final-state subsistence indicator and Trader wealth from the wealth logs.
- Batch data are not in this repository.
