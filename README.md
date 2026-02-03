# ExDBN-DNN (CausalGPT runner)

This folder contains a runnable pipeline for:

- **EXDBN (MILP/Gurobi)** adjacency estimation
- **DNN refinement stage** (currently implemented as a tiny **nanoGPT-style** causal transformer in
	`CausalGPT.utils.dnn_constraints_utils.train_dnn_with_constraints`)
- Writing CaMML-style constraint files (`.anc`) and saving adjacency matrices (`.csv/.npy`)
- Optional plotting (heatmap + directed network)

The main runnable entrypoints live under `src/CausalGPT/`.

## Quickstart (macOS)

### 1) Use the Python 3.11 venv (recommended)

We keep a local venv under `src/CausalGPT/.venv`.

```bash
cd /Users/xiaoyuhe/EXDBN-LLM/ExDBN-DNN/src/CausalGPT
source .venv/bin/activate

# editable install (so imports work)
python -m pip install -e .

# verify torch
python -c "import torch; print(torch.__version__, 'mps=', torch.backends.mps.is_available())"
```

Notes:

- Python 3.13 often breaks torch; this repo is set up to work with **Python 3.11/3.12**.
- You still need a working **Gurobi license** to run the EXDBN MILP stage.

### 2) Run CoDiet experiment

```bash
cd /Users/xiaoyuhe/EXDBN-LLM/ExDBN-DNN/src/CausalGPT
bash Test_codiet.sh
```

This script drives `tests/ExDBN_perform.py` against `codiet_302_*.csv` replicates.

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

## LLM priors (variable-name → priors)

If you want an LLM to look at **variable names** and propose causal priors, use:

- `CausalGPT.utils.llm_prior_cli`

Example (Ollama, OpenAI-compatible):

```bash
cd /Users/xiaoyuhe/EXDBN-LLM/ExDBN-DNN/src/CausalGPT
python -m CausalGPT.utils.llm_prior_cli \
	--csv /path/to/your.csv \
	--provider ollama \
	--base_url http://localhost:11434 \
	--model llama3.1
```

This prints `--ancs="[...]"` and `--forb_ancs="[...]"` strings you can pass to the runner.
It also caches the JSON response next to the CSV by default.

## Repo map

- `src/CausalGPT/tests/ExDBN_perform.py`: main end-to-end runner (EXDBN → constraints → DNN → outputs)
- `src/CausalGPT/exdbn_ban_edges.py`: EXDBN call + constraint writer utilities
- `src/CausalGPT/utils/dnn_constraints_utils.py`: DNN stage (nanoGPT-style transformer) + `.anc` parser
- `src/CausalGPT/utils/llm_prior.py`: OpenAI-compatible LLM prior querying + caching
- `src/CausalGPT/utils/llm_prior_cli.py`: small CLI for generating priors from CSV headers
