# ExDBN-DNN (CausalGPT runner)

This folder contains a runnable pipeline for:

- **EXDBN (MILP/Gurobi)** adjacency estimation
- **DNN refinement stage** (currently implemented as a tiny **nanoGPT-style** causal transformer in
	`CausalGPT.utils.dnn_constraints_utils.train_dnn_with_constraints`)
- Writing CaMML-style constraint files (`.anc`) and saving adjacency matrices (`.csv/.npy`)
- Optional plotting (heatmap + directed network)

The main runnable entrypoints live under `src/CausalGPT/`.

## Project structure

The project uses Cookiecutter-style layout and is based on a Machine Learning Operations template.

```txt
├── .dvc                      # Data Version Control
│   ├── cache
│   ├── tmp
│   ├── config
│   └── config.local
├── .github/                  # Github actions
│   └── workflows/
│       ├── evaluation.yaml
│       ├── linting.yaml
│       └── tests.yaml
├── .secrets/
│   └── gcp-key.json          # GCP service account key (to be added by user)
├── .venv/                    # Virtual environment (to be added by user)
├── configs/
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.dockerfile
│   ├── api_requirements.txt
│   ├── dvc.dockerfile
│   ├── streamlit.dockerfile
│   └── streamlit_requirements.txt
├── models/                   # Trained models
├── outputs/
├── reports/                  # Reports
│   └── figures/
├── scripts/                  # Helper scritpt for testing
├── src/                      # Source code
│   ├── mlo_group_project/
│   │   ├──config/            # Configuration files
│   │   ├──styles/            # Streamlit styles
│   │   ├──training/          # Training scripts
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── guardrails.py
│   │   ├── model.py
│   │   ├── streamlit_app.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── conftest.py
│   ├── sample_data.pt        # Sample data for tests (to be added automatically when tests are run)
│   ├── test_data.py
│   └── test_model.py
├── wandb/                    # Weights & Biases files
├── .dvcignore
├── .env                      # Environment variables (to be added by user)
├── .gcloudignore
├── .gitignore
├── .pre-commit-config.yaml
├── cloudbuild_api.yaml           # Google Cloud Build file
├── cloudbuild_stramlit_app.yaml  # Google Cloud Build file
├── docker-compose.yml            # Docker compose file
├── dvc.lock                      # DVC lock file
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Project development requirements
├── tasks.py                  # Project invoke tasks
└── uv.lock                   # uv lock file
```

## Quickstart (macOS)

### 1) Create an environment

Recommended: Python 3.11/3.12.

```bash
cd ExDBN-DNN

# Option A: use your own venv (recommended)
python3 -m venv .venv
source .venv/bin/activate

# editable install (so imports work)
python -m pip install -e .

# verify torch
python -c "import torch; print(torch.__version__, 'mps=', torch.backends.mps.is_available())"
```

Notes:

- Python 3.13 often breaks torch; this repo is set up to work with **Python 3.11/3.12**.
- You still need a working **Gurobi license** to run the EXDBN MILP stage.

### 2) Run CoDiet experiment (EXDBN → `.anc` → DNN)

```bash
cd ExDBN-DNN/src/CausalGPT
bash tests/Test_codiet.sh
```

This script drives `tests/ExDBN_perform.py` against `codiet_302_*.csv` replicates.

## 运行指南 (step-by-step)

Below is a more explicit “checklist-style” guide (useful if you want to run individual pieces).

### 0) Enter the runner folder

```bash
cd ExDBN-DNN/src/CausalGPT
```

### 1) Create + activate a venv

You can do this at repo root (`ExDBN-DNN/`) or inside `src/CausalGPT/`. A common pattern:

```bash
cd ExDBN-DNN
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Alternative (if you prefer requirements files):

```bash
cd ExDBN-DNN
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### 2) (Optional) Adjust EXDBN hyperparams in the external EXDBN repo

If you are using the external EXDBN implementation under this workspace (e.g. `Causal-Methods/bestdagsolverintheworld-main/`), you may want to adjust its YAML hyperparameters (for example, `max_features`, `mipgap`, `time_limit`, etc.).

Where that YAML lives depends on the EXDBN repo version you cloned. Find it inside the EXDBN repo under something like:

- `.../experiments_conf/exdbn_hyperparams.yaml`

### 3) Run EXDBN → `.anc` → DNN (limited features)

This is the recommended “fast-ish” run.

```bash
cd ExDBN-DNN/src/CausalGPT

# example knobs
export EXDBN_TIME_LIMIT=120
export EXDBN_TARGET_MIP_GAP=0.10
export DEGREE=1
export EXDBN_MAX_FEATURES=15

bash tests/Test_codiet.sh
```

### 4) Run EXDBN → `.anc` → DNN (all features)

This can be much slower and heavier.

```bash
cd ExDBN-DNN/src/CausalGPT
export EXDBN_MAX_FEATURES=
bash tests/Test_codiet_with_all_features.sh
```

### 5) (Optional) EXDBN → NanoChat pipeline smoke test

Use this when you want to validate the “EXDBN priors → NanoChat consumes priors → write a hard `.anc`” code path.

If you do not have real NanoChat artifacts yet, you can generate dummy ones:

```bash
cd ExDBN-DNN

# writes: reports/tmp_sanity/dummy_ckpt.pt + reports/tmp_sanity/vocab.txt
python -m CausalGPT.utils.make_vocab_gpt4style dummy-assets \
	--out_ckpt reports/tmp_sanity/dummy_ckpt.pt \
	--out_vocab reports/tmp_sanity/vocab.txt \
	--vocab_size 64
```

Then run the pipeline script:

```bash
cd ExDBN-DNN/src/CausalGPT

# simplest: run on one CSV directly
SINGLE_DATA_CSV=/Users/xiaoyuhe/Datasets/CoDiet/data/codiet_302_0.csv \
	bash tests/Run_codiet_exdbn_to_nanochat.sh
```

Notes:

- The script will automatically pick up `ExDBN-DNN/reports/tmp_sanity/{dummy_ckpt.pt,vocab.txt}` if present.
- For strict vocab sanity checks against `.anc` node tokens, set `NANOCHAT_SANITY_STRICT=1` (default for non-dummy runs).

### 6) (Optional) nanochat / constr-nanochat scripts

These scripts clone external repos into `~/.cache/nanochat/...` and create a `.venv` inside that checkout.

Smoke-check (default on macOS):

```bash
cd ExDBN-DNN/src/CausalGPT
bash tests/vanilla_sft.sh
bash tests/constr_sft.sh
bash tests/tok_pre_mid_d20.sh
```

Full training (can take a long time; requires proper torch/CUDA setup):

```bash
cd ExDBN-DNN/src/CausalGPT
RUN_FULL=1 bash tests/tok_pre_mid_d20.sh
```

To generate EXDBN `.anc` inside `tok_pre_mid_d20.sh` and pass it into constrained SFT:

```bash
cd ExDBN-DNN/src/CausalGPT

RUN_EXDBN_ANC=1 \
EXDBN_DATA_CSV=/Users/xiaoyuhe/Datasets/CoDiet/data/codiet_302_0.csv \
EXDBN_ANC_OUT=~/.cache/nanochat/exdbn_priors/ExDBN_LLM.anc \
RUN_CONSTR_SFT=1 \
RUN_FULL=1 \
	bash tests/tok_pre_mid_d20.sh
```

## What gets written (important outputs)

Runs write to a dataset-specific output directory under `ExDBN-DNN/reports/`.
Typical artifacts:

- `ExDBN_LLM.anc`: CaMML-style constraint file with **feature names** in `arcs{ ... }`
- `ExDBN_LLM_idx.anc`: same but **integer node ids** (for tooling that expects indices)
- `adj_exdbn_*.csv/.npy`: EXDBN predicted adjacency (binary)
- `adj_dnn_*.csv/.npy`: DNN (nanoGPT-style) refined adjacency
- `*.png` / `*.net.png` (optional): adjacency heatmap + directed network plots

### About `.anc` format

The `arcs{ ... }` file contains three kinds of constraints (as written by `CausalGPT.exdbn_ban_edges.write_camml_anc`):

- `A => B conf;` soft ancestor prior (encourage)
- `A => B reconf;` soft forbidden ancestor prior (discourage)
- `A -> B reconf;` hard forbidden edge (forbid A→B)

In the current DNN stage, `CausalGPT.utils.dnn_constraints_utils.parse_anc_file` treats lines with `->` as **forbidden edges**.

## nanochat integration (optional)

This repo does **not** vendor karpathy/nanochat (or constrained forks). Instead, the helper scripts under `src/CausalGPT/tests/` will:

- clone the relevant nanochat repo into `~/.cache/nanochat/...`
- create a `.venv` inside that checkout using `uv`
- run the requested `python -m nanochat.*` / `python -m scripts.*` entrypoints

Scripts:

- `src/CausalGPT/tests/vanilla_sft.sh`: bootstrap upstream nanochat and (by default on macOS) do a smoke-check. Set `RUN_FULL=1` to run training.
- `src/CausalGPT/tests/constr_sft.sh`: bootstrap `andrewklayk/constr-nanochat` and run constrained SFT (smoke-check on macOS unless `RUN_FULL=1`).
- `src/CausalGPT/tests/tok_pre_mid_d20.sh`: tokenizer + pretrain + mid-train “speedrun” (smoke-check on macOS unless `RUN_FULL=1`).

### Feeding EXDBN `.anc` into constrained SFT

`tok_pre_mid_d20.sh` can optionally generate an EXDBN priors file and then pass it into constrained SFT (requires a fork that provides `scripts.chat_sft_constr`).

Key env vars:

- `RUN_EXDBN_ANC=1`: enable EXDBN `.anc` generation
- `EXDBN_DATA_CSV=/path/to.csv`: input CSV for EXDBN
- `EXDBN_ANC_OUT=~/.cache/nanochat/exdbn_priors/ExDBN_LLM.anc`: output `.anc` path
- `RUN_CONSTR_SFT=1`: after mid-train, run constrained SFT and pass `--anc $EXDBN_ANC_OUT`

Note: The constrained fork does not (currently) have a standard `--anc` flag upstream; we carry a patch in `src/CausalGPT/tests/patches/` and the shell scripts attempt to apply it automatically inside the cloned checkout.

## Where nanochat outputs go

The nanochat scripts write under `NANOCHAT_BASE_DIR` (default in our scripts: `~/.cache/nanochat`). Common subdirectories:

- `~/.cache/nanochat/report/` (generated markdown sections + final `report.md`, also copied to the current working directory of the nanochat checkout)
- `~/.cache/nanochat/base_checkpoints/`, `~/.cache/nanochat/mid_checkpoints/`, `~/.cache/nanochat/chatsft_checkpoints/`
- `~/.cache/nanochat/tokenized_data/`

You can override with `export NANOCHAT_BASE_DIR=/some/path` before running the scripts.

## Repo map

- `src/CausalGPT/tests/ExDBN_perform.py`: main end-to-end runner (EXDBN → constraints → DNN → outputs)
- `src/CausalGPT/exdbn_ban_edges.py`: EXDBN call + constraint writer utilities
- `src/CausalGPT/utils/dnn_constraints_utils.py`: DNN stage (nanoGPT-style transformer) + `.anc` parser
- `src/CausalGPT/run_exdbn_to_nanochat.py`: helper to run EXDBN priors into a constrained nanochat generator and write a hard `.anc`
