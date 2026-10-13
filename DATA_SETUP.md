# Getting the data into the dashboard

The code lives on GitHub. The **data does not** — 360 replications is 423 MB
of CSV, which does not belong in a git repository. This guide covers getting
the two back together.

Written for someone who has not used Python before. If a step assumes
something you do not have, say so and it will be corrected.

---

## What you should have

| Thing | Where it comes from | Size |
|---|---|---|
| The code | GitHub repository | ~5 MB |
| The dataset | Zenodo (DOI) or the Google Drive link | 165 MB download, 423 MB unpacked |

The dataset download is a single file: `aegora_isle_dataset_v1.tar.gz`.

---

## Step 1 — Install the code

You need **Python 3.10 or newer**. Check what you have:

```bash
python3 --version
```

If that prints `3.10` or higher, continue. If it errors or shows an older
version, install Python from <https://www.python.org/downloads/> first.

Download the code and install it:

```bash
git clone https://github.com/<OWNER>/aegora-isle.git
cd aegora-isle
pip install -e .
```

The last command takes a few minutes — it is fetching PyTorch and
Stable-Baselines3, which are large.

---

## Step 2 — Unpack the dataset

Download `aegora_isle_dataset_v1.tar.gz`, then unpack it **into the project
folder** you just cloned:

```bash
tar -xzf ~/Downloads/aegora_isle_dataset_v1.tar.gz -C .
```

On Windows, 7-Zip or WinRAR will open a `.tar.gz`; extract it into the
project folder.

You now have a folder called `aegora_isle_dataset/`. Move its contents into
`data/`:

```bash
cp -r aegora_isle_dataset/runs data/
cp aegora_isle_dataset/*.csv data/
```

### What the layout must look like

This is the part that matters. The dashboard looks for
`<data directory>/runs/<governance>/<run_id>.csv`:

```
aegora-isle/
├── data/
│   ├── runs/
│   │   ├── G1/
│   │   │   ├── G1_A_001.csv     <- the timestep-by-timestep KPIs
│   │   │   ├── G1_A_001.json    <- seed + config, needed for reproducibility
│   │   │   └── ... (120 runs)
│   │   ├── G2/  ... (120 runs)
│   │   └── G3/  ... (120 runs)
│   ├── attempts_summary.csv
│   └── dataset_manifest.csv
└── aegora_isle/
```

**Each CSV must sit next to its `.json` file.** A CSV on its own is ignored —
the JSON is what records the seed, so without it the run is not reproducible
and the dashboard will not offer it.

---

## Step 3 — Start the dashboard

```bash
streamlit run aegora_isle/dashboard/app.py
```

Your browser opens at `http://localhost:8501`. The sidebar should say:

> **360 replications discovered**

That is the confirmation that step 2 worked.

---

## If it says "0 replications discovered"

The app will not crash — it shows:

> No replications found under `data/runs/`.

Three things to check, in order:

1. **Are you in the right folder?** Run `ls data/runs` — you should see
   `G1  G2  G3`. If you get "No such file or directory", the archive was
   unpacked somewhere else.
2. **Did the folders nest twice?** A common slip is ending up with
   `data/runs/runs/G1/`. The path must be exactly `data/runs/G1/`.
3. **Point the app somewhere else instead.** The sidebar has a **Data
   directory** box. You can leave the dataset wherever it landed and type
   that path in — for example `/Users/you/Downloads/aegora_isle_dataset/`.
   Nothing has to be moved at all.

---

## Verifying the download is intact

The archive ships with `dataset_manifest.csv`: one row per replication with
a SHA-256 checksum, row count, seed, and collapse timestep. To confirm
nothing was corrupted in transit:

```bash
python scripts/verify_dataset.py --data_dir data/
```

This reports any file whose checksum does not match. A partial or truncated
download is the most common cause of odd-looking charts, and this catches it
in about a minute.

---

## What each view needs

| View | Needs |
|---|---|
| Run Selector | the dataset |
| KPI Time Series | the dataset |
| Multi-Run Overlay | the dataset |
| Collapse Events | the dataset |
| CSV Export | the dataset |
| **Experiment Progress** | **nothing** — this one runs without any data |

Experiment Progress is deliberately exempt. It launches *new* experiments, so
it works on a fresh clone with an empty `data/`. That is the escape hatch if
the dataset is unavailable: you can generate your own. Be aware that a full
360-run batch is a multi-hour job on a normal laptop.

---

## A note on the sample dataset

If the repository includes `data/sample/`, that is a **downsampled** copy —
every 25th timestep, 17 MB instead of 423 MB — so that a fresh clone has
something to display immediately.

It is for demonstration, not analysis:

- **The smoothed charts are exact, not approximate.** Each row is the
  *average* of its 25 timesteps rather than one sampled point, and the
  dashboard detects the spacing and scales its rolling window to match. A
  50-step mean over the sample and a 50-step mean over the full data are
  then the same arithmetic — measured at 0.00% deviation across all five
  KPIs, in both a G1 and a G3 run.
- Collapse markers are exact, because the collapse timestep is read from the
  JSON sidecar rather than recomputed from the series.
- **What is lost is within-bin detail.** Single-timestep spikes are averaged
  into their neighbours. Nothing in the dashboard displays un-smoothed
  series, so this is invisible there — but it is real, and it is why the
  next point holds.
- **Do not run the statistical analysis on it.** `analysis/` expects full
  resolution. Variance, collapse timing, and survival estimates computed
  from binned data are not the pre-registered quantities.
- **CSV Export from the sample exports binned rows**, not raw ones. Anyone
  exporting data to analyse elsewhere needs the full dataset.

Use the full dataset for anything that goes in the thesis.
