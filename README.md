# Aegora Isle

Multi-agent reinforcement-learning simulation of synthetic economies under
three governance regimes (G1 ungoverned, G2 rule-based, G3 adaptive-RL)
across four environmental scenarios (A/B/C/D). The codebase produces 360
reproducible CSV runs that feed doctoral-grade statistical analysis
(ANOVA, Kaplan-Meier survival, Cox PH).

This repository is the **scientific instrument** for the project; engineering
choices are documented in `DECISIONS.md` (deviations from the spec).

---

## Quick start

Three commands to a running dashboard, assuming you have the dataset:

```bash
pip install -e .
tar -xzf aegora_isle_dataset_v1.tar.gz && cp -r aegora_isle_dataset/runs data/
streamlit run aegora_isle/dashboard/app.py
```

**Do not have the dataset?** The repository ships a 17.9 MB downsampled copy
at `data/sample/`, so the dashboard works immediately on a fresh clone:

```bash
pip install -e .
streamlit run aegora_isle/dashboard/app.py
# then set the sidebar's "Data directory" to: data/sample
```

Full details, including what to do when it reports "0 replications
discovered", are in **[DATA_SETUP.md](DATA_SETUP.md)**.

---

## Install

Python 3.10-3.12 (locked in `pyproject.toml`). Either installer works:

```bash
# with uv (what the project uses)
uv venv --python 3.10
uv pip install -e ".[dev]"

# or with plain pip
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Verify:

```bash
uv run python -c "import pettingzoo, stable_baselines3, streamlit; print('OK')"
uv run pytest tests/ -q          # 325 tests
```

The install pulls PyTorch and Stable-Baselines3, so expect a few minutes.

---

## 1. The dashboard (frontend)

The observability UI required by spec §7. Launch it with:

```bash
streamlit run aegora_isle/dashboard/app.py
```

It opens at <http://localhost:8501>. The sidebar reports how many
replications it found — that number is your confirmation the data is wired
up correctly.

### The seven views

| View | What it does |
|---|---|
| **Run Selector** | Every discovered replication, filterable by governance, scenario, and convergence status |
| **KPI Time Series** | All five KPIs for one run, raw series behind a 50-step rolling average |
| **Multi-Run Overlay** | Several runs on one axis, for comparing regimes directly |
| **Collapse Events** | Marks the collapse timestep with a red dashed line |
| **Experiment Progress** | Launch, pause, resume, and stop a batch; progress bar and live chart |
| **CSV Export** | Download a run's raw CSV and its JSON reproducibility sidecar |

### Notes on Experiment Progress

- It is the **only view that works with no data at all** — the launch form
  is the escape hatch for a fresh clone.
- Progress is **replication-level**, not within-run. Each CSV is written once,
  when its replication finishes, so there is no partial file to plot
  mid-run; the live chart shows completed replications rather than a
  synthesised in-flight curve.
- **Pause suspends the worker processes too**, not just the parent. Pausing
  only the parent would leave joblib workers burning CPU while the UI
  claimed to be paused.

---

## 2. Running experiments

### The full confirmatory batch

This is the 360-replication run that produced the published dataset:

```bash
uv run python -m aegora_isle.experiments.run_experiment \
  --governance all --scenario all \
  --n_reps 30 --n_agents 50 --max_timesteps 10000 \
  --n_jobs 6 --output_dir data/
```

**This is a multi-hour job.** 3 regimes x 4 scenarios x 30 replications, each
training PPO agents (and SAC for G3) over 10,000 timesteps. Budget most of a
day on a laptop, and expect all cores at `--n_jobs 6` to be busy throughout.

**Back up `data/` first.** The runner writes into `data/runs/` and will
overwrite same-named replications.

### A quick smoke run

To confirm the pipeline works end to end without waiting hours:

```bash
uv run python -m aegora_isle.experiments.run_experiment \
  --governance G1 --scenario A \
  --n_reps 1 --n_agents 20 --max_timesteps 2000 \
  --n_jobs 1 --output_dir /tmp/smoke/
```

About a minute. Writes one CSV/JSON pair plus the summary tables.

### All options

| Flag | Default | Meaning |
|---|---|---|
| `--governance` | `all` | `G1`, `G2`, `G3`, or `all` |
| `--scenario` | `all` | `A`, `B`, `C`, `D`, or `all` |
| `--n_reps` | 30 | Replications per cell |
| `--n_agents` | 50 | Agents per run |
| `--max_timesteps` | 10000 | Steps per replication (also the survival censoring point) |
| `--seed_offset` | 0 | Shifts every seed; use for an independent batch |
| `--output_dir` | `data/` | Where runs and summaries are written |
| `--n_jobs` | 1 | Parallel workers |

### What a batch writes

```
<output_dir>/
├── runs/{G1,G2,G3}/<run_id>.csv     # the timestep KPI series
├── runs/{G1,G2,G3}/<run_id>.json    # seed + config + CSV checksum
├── attempts_summary.csv             # per-replication final status
├── cell_attempt_summary.csv         # per-cell convergence rates
└── value_loss_diagnostics.csv       # secondary stability diagnostic
```

Non-converging replications are retried once at `seed + 1000`. Both the
original and the retry are recorded, so the exclusion is auditable rather
than silent.

---

## 3. Statistical analysis

Reproduces every spec §8.1 output, plus a paper-ready PDF with each figure
and a plain-language explanation:

```bash
uv run python -m analysis.statistical_analysis --data_dir data/
```

Writes to `analysis/outputs/`: tables to `tables/*.csv`, 300-DPI figures to
`figures/*.png`, and the report PDF alongside them. Read-only with respect
to `data/`.

**Run this against the full dataset, never `data/sample/`.** The sample is
bin-averaged; variance and survival estimates computed from it are not the
pre-registered quantities.

---

## 4. Packaging the dataset for distribution

```bash
uv run python scripts/package_dataset.py --data_dir data/ --out dist/
```

Produces the archive for Zenodo, a per-replication manifest with SHA-256
checksums, and the downsampled sample. To check a downloaded copy:

```bash
uv run python scripts/verify_dataset.py --data_dir data/
```

To re-render a client note as PDF:

```bash
uv run python scripts/md_to_pdf.py some_note.md
```

---

## Repository structure

```
aegora_isle/
  environment/
    resource_pool.py    # seeded lagoon: logistic regen + disruption events
    market.py           # scarcity-priced peer-to-peer trades of food/tool
    aegora_env.py       # PettingZoo AECEnv wiring agents + pool + market
  agents/               # extractor, producer, trader, governor, base
  governance/
    governance_module.py  # single entry point: G1 / G2 / G3 dispatch
    rule_based.py         # G2: extraction caps, sanctions, redistribution
    adaptive_rl.py        # G3: SAC governor
    configs/              # per-regime YAML
  kpi/
    kpi_logger.py       # the frozen CSV schema, written once per replication
    convergence.py      # pre-registered convergence criterion
  experiments/
    run_experiment.py   # the factorial batch runner
    train_agents.py     # SB3 rollout collector
    scenarios.yaml      # A/B/C/D parameters
  dashboard/
    app.py              # Streamlit UI (spec §7)
    data_access.py      # discovery + loading, no Streamlit import
    process_control.py  # run/pause/reset via psutil
analysis/
  data_loading.py       # run index, survival encoding
  statistical_analysis.py
  plots.py / report.py
scripts/
  package_dataset.py    # archive + manifest + sample
  verify_dataset.py     # checksum a downloaded dataset
  md_to_pdf.py          # client-note rendering
tests/                  # 325 tests
```

Directory layout is fixed by spec §5; the analysis scripts depend on it.

---

## Data schema

The canonical run-CSV schema (spec §3.1) is **frozen**:

```
timestep, gini_coefficient, hdi_proxy, gdp_equivalent, depletion_rate,
resource_stock, collapse_event, governance_condition, scenario, run_id,
random_seed, n_agents, gini_raw, gdp_raw
```

Renaming any column breaks the doctoral analysis scripts; do not edit.
See `ARCHITECTURE.md` "Output: the run CSV" for the enforcement rule.

### A note on `collapse_event`

This column records the *subsistence* collapse definition. The statistical
analysis uses spec Table 4's MTSC definition instead, and **the two
disagree**. Both are documented in `DECISIONS.md`; the dashboard states
which one it is showing. This is a real methodological split, not a bug.

---

## Reproducibility

Every stochastic module accepts a `numpy.random.Generator` at construction
(see `DECISIONS.md` "Seeding contract"). The top-level seed flows:

```
run_experiment.py  ->  AegoraEnv(seed)
                         ->  ResourcePool(rng)
                         ->  Market(rng)
                         ->  GovernanceModule(rng)
                         ->  each Agent(rng)
                         ->  each SB3 model (seed)
```

Per-run JSON sidecars store the seed, config, and a SHA-256 of the CSV. The
reproduction test (`tests/test_reproducibility.py`) loads a JSON, re-runs,
and requires a byte-identical CSV.

---

## Hardware

Developed on macOS (Apple Silicon, Python 3.10). A full 360-run batch wants
6+ cores and 16 GB RAM to finish in a working day; it will complete on less,
more slowly. The dashboard and analysis scripts run comfortably on any modern
laptop.

---

## License

MIT. See LICENSE. Copyright (c) 2026 Mohamed Ali Hammami.

## Citation

```bibtex
@misc{aegora_isle,
  title  = {Aegora Isle: Multi-Agent RL for Synthetic-Economy Governance},
  author = {Hammami, Mohamed Ali},
  year   = {2026},
  note   = {Doctoral thesis, Doctorate in AI and Technology Management}
}
```

## Acknowledgement

Simulation implementation developed with an external engineering contractor.
