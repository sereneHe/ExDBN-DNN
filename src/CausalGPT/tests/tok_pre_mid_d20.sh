#!/bin/bash

set -e

# This script is the "Best ChatGPT clone that $100 can buy",
# It is designed to run in ~4 hours on 8XH100 node at $3/GPU/hour.

# 1) Example launch (simplest):
# bash speedrun.sh
# 2) Example launch in a screen session (because the run takes ~4 hours):
# screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh
# 3) Example launch with wandb logging, but see below for setting up wandb first:
# WANDB_RUN=speedrun screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh

# Default intermediate artifacts directory is in ~/.cache/nanochat
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# Resolve this repo root (for running EXDBN -> .anc generation).
THIS_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXDBN_DNN_ROOT="$(cd "$THIS_SCRIPT_DIR/../../.." && pwd)"

ANC_PATCH_FILE="$THIS_SCRIPT_DIR/patches/constr-nanochat-chat_sft_constr-anc.patch"

# -----------------------------------------------------------------------------
# This repository does not vendor the upstream `nanochat` training code.
# We bootstrap a local checkout under $NANOCHAT_BASE_DIR so `python -m nanochat.*`
# and `python -m scripts.*` work reliably.

NANOCHAT_REPO_URL="${NANOCHAT_REPO_URL:-https://github.com/karpathy/nanochat.git}"
NANOCHAT_CODE_DIR="${NANOCHAT_CODE_DIR:-$NANOCHAT_BASE_DIR/nanochat}"

if [ ! -d "$NANOCHAT_CODE_DIR" ]; then
    echo "Cloning nanochat into $NANOCHAT_CODE_DIR ..."
    git clone --depth 1 "$NANOCHAT_REPO_URL" "$NANOCHAT_CODE_DIR"
fi

cd "$NANOCHAT_CODE_DIR"

# -----------------------------------------------------------------------------
# Python venv setup with uv

# install uv (if not already installed)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || uv venv
# install the repo dependencies

# Some repos use uv extras, others don't. Prefer GPU extra when available.
# On macOS, the GPU extra often pins CUDA wheels that don't exist, so skip it.
if [ "$(uname -s)" = "Darwin" ]; then
    uv sync
else
    if ! uv sync --extra gpu; then
        echo "uv sync --extra gpu failed; falling back to uv sync"
        uv sync
    fi
fi
uv pip install --upgrade pip setuptools requests urllib3 certifi

# activate venv so that `python` uses the project's venv instead of system python
source .venv/bin/activate

# If this nanochat checkout provides scripts.chat_sft_constr, optionally patch it
# to accept a --anc flag.
if [ -f "scripts/chat_sft_constr.py" ] && [ -f "$ANC_PATCH_FILE" ]; then
    if git apply --reverse --check "$ANC_PATCH_FILE" >/dev/null 2>&1; then
        echo "[PATCH] constr-nanochat --anc patch already applied"
    elif git apply --check "$ANC_PATCH_FILE" >/dev/null 2>&1; then
        echo "[PATCH] applying constr-nanochat --anc patch"
        git apply "$ANC_PATCH_FILE"
    else
        echo "[PATCH] WARNING: could not apply $ANC_PATCH_FILE (continuing)" >&2
    fi
fi

# -----------------------------------------------------------------------------
# wandb setup
# If you wish to use wandb for logging (it's nice!, recommended).
# 1) Make sure to first log in to wandb, e.g. run:
#    `wandb login`
# 2) Set the WANDB_RUN environment variable when running this script, e.g.:
#    `WANDB_RUN=d26 bash speedrun.sh`
if [ -z "$WANDB_RUN" ]; then
    # by default use "dummy" : it's handled as a special case, skips logging to wandb
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# During the course of the run, we will be writing markdown reports to the report/
# directory in the base dir. This command clears it out and writes a header section
# with a bunch of system info and a timestamp that marks the start of the run.
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Optional: Generate EXDBN-predicted .anc priors for downstream constrained runs.
#
# Usage:
#   RUN_EXDBN_ANC=1 EXDBN_DATA_CSV=/path/to.csv bash tok_pre_mid_d20.sh
#
# Outputs:
#   - Named priors: $EXDBN_ANC_OUT (default: $NANOCHAT_BASE_DIR/exdbn_priors/ExDBN_LLM.anc)

RUN_EXDBN_ANC=${RUN_EXDBN_ANC:-0}
EXDBN_DATA_CSV=${EXDBN_DATA_CSV:-}
EXDBN_ANC_OUT=${EXDBN_ANC_OUT:-$NANOCHAT_BASE_DIR/exdbn_priors/ExDBN_LLM.anc}

DEGREE=${DEGREE:-1}
EXDBN_TIME_LIMIT=${EXDBN_TIME_LIMIT:-120}
EXDBN_TARGET_MIP_GAP=${EXDBN_TARGET_MIP_GAP:-0.10}

if [ "$RUN_EXDBN_ANC" = "1" ]; then
    if [ -z "$EXDBN_DATA_CSV" ] || [ ! -f "$EXDBN_DATA_CSV" ]; then
        echo "Missing EXDBN_DATA_CSV (or file not found): $EXDBN_DATA_CSV" >&2
        exit 2
    fi

    # Prefer activated venv python if present; else local .venv; else system python.
    PY_EXDBN=""
    if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/bin/python" ]; then
        PY_EXDBN="${VIRTUAL_ENV}/bin/python"
    elif [ -x "$EXDBN_DNN_ROOT/.venv/bin/python" ]; then
        PY_EXDBN="$EXDBN_DNN_ROOT/.venv/bin/python"
    else
        PY_EXDBN=python3
    fi

    base="$(basename "$EXDBN_DATA_CSV" .csv)"
    exdbn_run_dir="$NANOCHAT_BASE_DIR/exdbn_runs/${base}_conf0.99999"
    mkdir -p "$exdbn_run_dir"

    echo "[EXDBN] generating named priors .anc from $EXDBN_DATA_CSV"
    DATASET=codiet \
    EXDBN_TIME_LIMIT="$EXDBN_TIME_LIMIT" \
    EXDBN_TARGET_MIP_GAP="$EXDBN_TARGET_MIP_GAP" \
    PYTHONPATH="$EXDBN_DNN_ROOT/src" \
    "$PY_EXDBN" -m CausalGPT.tests.ExDBN_perform \
        --data_path "$EXDBN_DATA_CSV" \
        --anc_path "$exdbn_run_dir/test_ancs.anc" \
        --out_dir "$exdbn_run_dir" \
        --write_anc \
        --skip_dnn \
        --epochs 1 \
        --degree "$DEGREE" \
        --conf 0.99999 \
        --ancs "[]" \
        --forb_ancs "[]"

    if [ ! -f "$exdbn_run_dir/ExDBN_LLM.anc" ]; then
        echo "Missing expected EXDBN output: $exdbn_run_dir/ExDBN_LLM.anc" >&2
        exit 2
    fi

    mkdir -p "$(dirname "$EXDBN_ANC_OUT")"
    cp "$exdbn_run_dir/ExDBN_LLM.anc" "$EXDBN_ANC_OUT"
    echo "[EXDBN] wrote priors anc: $EXDBN_ANC_OUT"
fi

# By default, do a quick smoke-test on macOS (imports + entrypoint presence).
# Set RUN_FULL=1 to actually run dataset download + tokenizer + torchrun training.
if [ "$(uname -s)" = "Darwin" ] && [ "${RUN_FULL:-0}" != "1" ]; then
    echo "[SMOKE] macOS detected; skipping dataset + tokenizer + torchrun (set RUN_FULL=1 to run)."
    python -c "import nanochat; print('nanochat import: OK')"
    python -c "import scripts; print('scripts import: OK')" || true
    python - <<'PY'
import importlib.util
mods = [
    "nanochat.dataset",
    "scripts.tok_train",
    "scripts.tok_eval",
    "scripts.base_train",
    "scripts.mid_train",
    "scripts.chat_eval",
]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit(f"Missing modules: {missing}")
print("entrypoints present: OK")
PY
    echo "[SMOKE] OK"
    exit 0
fi


# -----------------------------------------------------------------------------
# Tokenizer

# Download the first ~2B characters of pretraining dataset
# each data shard is ~250M chars
# so we download 2e9 / 250e6 = 8 data shards at this point
# each shard is ~100MB of text (compressed), so this is about ~800MB of data on disk
# look at dev/repackage_data_reference.py for details on how this data was prepared
python -m nanochat.dataset -n 8
# Immediately also kick off downloading more shards in the background while tokenizer trains
# Approximately 350 shards are needed for 10B tokens of data for pretraining.
# The maximum total number of shards available in the entire dataset is 1822.
python -m nanochat.dataset -n 370 &
DATASET_DOWNLOAD_PID=$!
# train the tokenizer with vocab size 2**15 = 32768 on ~2B characters of data
python -m scripts.tok_train
# evaluate the tokenizer (report compression ratio etc.)
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Base model (pretraining)
echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

# Number of processes/GPUs to use
NPROC_PER_NODE=2
if [ "$(uname -s)" = "Darwin" ]; then
    NPROC_PER_NODE=1
fi

torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train -- --depth=20 --device-batch-size=8 --total-batch-size=262144 --eval-tokens=524288 --eval-every=1000 --sample-every=1000 --save-every=2000 --core-metric-every=-1 --run=$WANDB_RUN

# -----------------------------------------------------------------------------
# Midtraining (teach the model conversation special tokens, tool use, multiple choice)

# run midtraining and eval the model
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.mid_train -- --run=$WANDB_RUN --device_batch_size=8 --model_tag='d20'
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i mid

# ----------------------------------------------------------------------------
# Optional: Constrained SFT stage (requires a nanochat fork that provides
# scripts.chat_sft_constr, e.g. andrewklayk/constr-nanochat).

RUN_CONSTR_SFT=${RUN_CONSTR_SFT:-0}
SFT_MODEL_SUFFIX=${SFT_MODEL_SUFFIX:-_exdbn_anc}

if [ "$RUN_CONSTR_SFT" = "1" ]; then
    if ! python -c "import importlib; importlib.import_module('scripts.chat_sft_constr')" >/dev/null 2>&1; then
        echo "[SFT_CONSTR] scripts.chat_sft_constr not available in this checkout." >&2
        echo "[SFT_CONSTR] Set NANOCHAT_REPO_URL=https://github.com/andrewklayk/constr-nanochat.git and rerun." >&2
        exit 2
    fi

    if [ ! -f "$EXDBN_ANC_OUT" ]; then
        echo "[SFT_CONSTR] Missing EXDBN_ANC_OUT: $EXDBN_ANC_OUT" >&2
        echo "[SFT_CONSTR] Run with RUN_EXDBN_ANC=1 (and set EXDBN_DATA_CSV)." >&2
        exit 2
    fi

    torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_sft_constr -- \
        --run=$WANDB_RUN \
        --device-batch-size=4 \
        --num-epochs=3 \
        --model-name="$SFT_MODEL_SUFFIX" \
        --opt="pbm" \
        --model-tag="d20" \
        --anc "$EXDBN_ANC_OUT"

    torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i sft --model-tag="d20${SFT_MODEL_SUFFIX}"
fi

# chat with the model over CLI! Leave out the -p to chat interactively
# python -m scripts.chat_cli -p "Why is the sky blue?"

# even better, chat with your model over a pretty WebUI ChatGPT style
# python -m scripts.chat_web

# -----------------------------------------------------------------------------
# Reinforcement Learning. Optional, and currently only on GSM8K
# (optional)

# run reinforcement learning
# torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_rl -- --run=$WANDB_RUN
# eval the RL model only on GSM8K
# torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i rl -a GSM8K

# -----------------------------------------------------------------------------
# Generate the full report by putting together all the sections
# report.md is the output and will be copied to current directory for convenience