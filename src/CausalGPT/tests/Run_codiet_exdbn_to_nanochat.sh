#!/bin/bash
set -euo pipefail

# Pipeline: EXDBN -> ExDBN_LLM.anc -> NanoChat (priors=ExDBN_LLM.anc) -> generated hard .anc
#
# Single-file smoke mode:
#   SINGLE_DATA_CSV=/path/to/small.csv bash Run_codiet_exdbn_to_nanochat.sh
# This bypasses codiet_302_r.csv replicate selection and runs EXDBN directly on that CSV.
#
# Required env vars for NanoChat:
#   NANOCHAT_CKPT   : path to ckpt.pt
#   NANOCHAT_VOCAB  : path to vocab.txt
# Optional:
#   NANOCHAT_PROMPT_TOKENS : comma-separated ids (default: 1,2,3 placeholder)
#   NANOCHAT_MAX_GEN       : default 512
#
# EXDBN knobs (conservative defaults):
#   DEGREE (default 1)
#   EXDBN_TIME_LIMIT (default 120)
#   EXDBN_TARGET_MIP_GAP (default 0.10)
#   EXDBN_MAX_FEATURES: set empty to use ALL features (default empty)
#
# Replicates:
#   R_LIST="0 1 2 3 4 5" (default)
#   CODIET_N_ROWS=302 (default)

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXDBN_DNN_ROOT="$(cd "$HERE/../.." && pwd)"

DATA_DIR=${DATA_DIR:-/Users/xiaoyuhe/Datasets/CoDiet}
OUT_BASE=${OUT_BASE:-$EXDBN_DNN_ROOT/reports/causalgpt_runs_codiet}

SINGLE_DATA_CSV=${SINGLE_DATA_CSV:-}

R_LIST=${R_LIST:-"0 1 2 3 4 5"}
CODIET_N_ROWS=${CODIET_N_ROWS:-302}

DEGREE=${DEGREE:-1}
EXDBN_TIME_LIMIT=${EXDBN_TIME_LIMIT:-120}
EXDBN_TARGET_MIP_GAP=${EXDBN_TARGET_MIP_GAP:-0.10}
EXDBN_MAX_FEATURES=${EXDBN_MAX_FEATURES-}

NANOCHAT_CKPT=${NANOCHAT_CKPT:-}
NANOCHAT_VOCAB=${NANOCHAT_VOCAB:-}
NANOCHAT_PROMPT_TOKENS=${NANOCHAT_PROMPT_TOKENS:-1,2,3}
NANOCHAT_MAX_GEN=${NANOCHAT_MAX_GEN:-512}

# Convenience defaults: if you generated NanoChat vocab/ckpt under reports/tmp_sanity,
# use them automatically unless the user explicitly provides paths.
DEFAULT_TMP_SANITY_DIR="$EXDBN_DNN_ROOT/reports/tmp_sanity"
if [[ -z "$NANOCHAT_VOCAB" && -f "$DEFAULT_TMP_SANITY_DIR/vocab.txt" ]]; then
  NANOCHAT_VOCAB="$DEFAULT_TMP_SANITY_DIR/vocab.txt"
  echo "[NANOCHAT] defaulting NANOCHAT_VOCAB=$NANOCHAT_VOCAB"
fi
if [[ -z "$NANOCHAT_CKPT" && -f "$DEFAULT_TMP_SANITY_DIR/dummy_ckpt.pt" ]]; then
  NANOCHAT_CKPT="$DEFAULT_TMP_SANITY_DIR/dummy_ckpt.pt"
  echo "[NANOCHAT] defaulting NANOCHAT_CKPT=$NANOCHAT_CKPT"
fi

# If you don't have real NanoChat artifacts yet, set DUMMY_NANOCHAT=1 to
# auto-generate a random-initialized ckpt + vocab for pipeline smoke tests.
# NOTE: This does not produce meaningful constraints; it only validates the code path.
DUMMY_NANOCHAT=${DUMMY_NANOCHAT:-0}

if [[ -z "$NANOCHAT_CKPT" || -z "$NANOCHAT_VOCAB" ]]; then
  if [[ "$DUMMY_NANOCHAT" != "1" ]]; then
    echo "Missing NANOCHAT_CKPT or NANOCHAT_VOCAB" >&2
    echo "Provide real artifacts, or set DUMMY_NANOCHAT=1 for a smoke-test dummy model." >&2
    exit 2
  fi
fi

cd "$HERE"

# Prefer activated venv python if present; else local .venv; else system python.
PY=""
if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
  PY="${VIRTUAL_ENV}/bin/python"
elif [[ -x "$HERE/.venv/bin/python" ]]; then
  PY="$HERE/.venv/bin/python"
else
  PY=python3
fi

# Sanity check strictness:
#   1 = fail fast if .anc node tokens are missing from vocab
#   0 = warn only (useful for dummy smoke tests)
NANOCHAT_SANITY_STRICT=${NANOCHAT_SANITY_STRICT-}
if [[ -z "$NANOCHAT_SANITY_STRICT" ]]; then
  if [[ "$DUMMY_NANOCHAT" == "1" ]]; then
    NANOCHAT_SANITY_STRICT=0
  else
    NANOCHAT_SANITY_STRICT=1
  fi
fi

if [[ -n "$SINGLE_DATA_CSV" ]]; then
  data_csv="$SINGLE_DATA_CSV"
  if [[ ! -f "$data_csv" ]]; then
    echo "Missing SINGLE_DATA_CSV: $data_csv" >&2
    exit 2
  fi

  base="$(basename "$data_csv" .csv)"
  run_out_dir="$OUT_BASE/codiet/${base}_conf0.99999"
  anc_named="$run_out_dir/ExDBN_LLM.anc"

  echo "[STEP 1/2] EXDBN (single-file) -> $run_out_dir"
  mkdir -p "$run_out_dir"

  # Run EXDBN_perform directly on the provided CSV.
  DATASET=codiet \
  EXDBN_TIME_LIMIT="$EXDBN_TIME_LIMIT" \
  EXDBN_TARGET_MIP_GAP="$EXDBN_TARGET_MIP_GAP" \
  EXDBN_GUROBI_OUTPUTFLAG="$EXDBN_GUROBI_OUTPUTFLAG" \
  EXDBN_GUROBI_DISPLAYINTERVAL="$EXDBN_GUROBI_DISPLAYINTERVAL" \
  "$PY" -m CausalGPT.tests.ExDBN_perform \
    --data_path "$data_csv" \
    --anc_path "$run_out_dir/test_ancs.anc" \
    --out_dir "$run_out_dir" \
    --write_anc \
    --skip_dnn \
    --epochs 1 \
    --degree "$DEGREE" \
    --conf 0.99999 \
    --ancs "[]" \
    --forb_ancs "[]"

  if [[ ! -f "$anc_named" ]]; then
    echo "Missing EXDBN anc: $anc_named" >&2
    exit 2
  fi

  if [[ "$DUMMY_NANOCHAT" == "1" && ( -z "$NANOCHAT_CKPT" || -z "$NANOCHAT_VOCAB" ) ]]; then
    echo "[DUMMY] Generating dummy NanoChat assets (ckpt+vocab) for smoke test"
    NANOCHAT_CKPT="$run_out_dir/dummy_nanochat_ckpt.pt"
    NANOCHAT_VOCAB="$run_out_dir/dummy_nanochat_vocab.txt"
    "$PY" -m CausalGPT.make_dummy_nanochat_assets \
      --out_ckpt "$NANOCHAT_CKPT" \
      --out_vocab "$NANOCHAT_VOCAB" \
      --vocab_size 64 \
      --sequence_len 128 \
      --n_layer 2 \
      --n_head 2 \
      --n_kv_head 2 \
      --n_embd 64
  fi

  echo "[STEP 2/2] NanoChat priors=ExDBN_LLM.anc -> generate hard anc"
  out_anc="$run_out_dir/nanochat_generated_hard_from_exdbnpriors.anc"

  echo "[CHECK] Sanity check ckpt+vocab+anc (strict=$NANOCHAT_SANITY_STRICT)"
  "$PY" -m CausalGPT.tests.Test_sanity_check_vocub \
    --ckpt "$NANOCHAT_CKPT" \
    --vocab "$NANOCHAT_VOCAB" \
    --anc "$anc_named" \
    --strict "$NANOCHAT_SANITY_STRICT"

  "$PY" -m CausalGPT.run_exdbn_to_nanochat \
    --data_csv "$data_csv" \
    --exdbn_anc "$anc_named" \
    --ckpt "$NANOCHAT_CKPT" \
    --vocab "$NANOCHAT_VOCAB" \
    --prompt_tokens "$NANOCHAT_PROMPT_TOKENS" \
    --max_gen "$NANOCHAT_MAX_GEN" \
    --out_anc "$out_anc"

  exit 0
fi

# Run EXDBN one-by-one (skip DNN for stability)
for r in $R_LIST; do
  data_csv="$DATA_DIR/codiet_${CODIET_N_ROWS}_${r}.csv"
  if [[ ! -f "$data_csv" ]]; then
    echo "Missing data: $data_csv" >&2
    exit 2
  fi

  run_out_dir="$OUT_BASE/codiet/codiet_${CODIET_N_ROWS}_${r}_conf0.99999"
  anc_named="$run_out_dir/ExDBN_LLM.anc"

  echo "[STEP 1/2] EXDBN replicate r=${r} -> $run_out_dir"
  DATASET=codiet \
  DATA_DIR="$DATA_DIR" \
  OUT_BASE="$OUT_BASE" \
  CONF_LIST="0.99999" \
  SKIP_DNN=1 \
  FAST_RUN=1 \
  DEGREE="$DEGREE" \
  EXDBN_TIME_LIMIT="$EXDBN_TIME_LIMIT" \
  EXDBN_TARGET_MIP_GAP="$EXDBN_TARGET_MIP_GAP" \
  EXDBN_MAX_FEATURES="$EXDBN_MAX_FEATURES" \
  CODIET_N_ROWS="$CODIET_N_ROWS" \
  CODIET_GLOB="codiet_${CODIET_N_ROWS}_${r}.csv" \
  MAX_CASES=1 \
  bash Test_codiet.sh

  if [[ ! -f "$anc_named" ]]; then
    echo "Missing EXDBN anc: $anc_named" >&2
    exit 2
  fi

  echo "[STEP 2/2] NanoChat priors=ExDBN_LLM.anc -> generate hard anc"
  out_anc="$run_out_dir/nanochat_generated_hard_from_exdbnpriors.anc"

  echo "[CHECK] Sanity check ckpt+vocab+anc (strict=$NANOCHAT_SANITY_STRICT)"
  "$PY" -m CausalGPT.tests.Test_sanity_check_vocub \
    --ckpt "$NANOCHAT_CKPT" \
    --vocab "$NANOCHAT_VOCAB" \
    --anc "$anc_named" \
    --strict "$NANOCHAT_SANITY_STRICT"

  "$PY" -m CausalGPT.run_exdbn_to_nanochat \
    --data_csv "$data_csv" \
    --exdbn_anc "$anc_named" \
    --ckpt "$NANOCHAT_CKPT" \
    --vocab "$NANOCHAT_VOCAB" \
    --prompt_tokens "$NANOCHAT_PROMPT_TOKENS" \
    --max_gen "$NANOCHAT_MAX_GEN" \
    --out_anc "$out_anc"

done
