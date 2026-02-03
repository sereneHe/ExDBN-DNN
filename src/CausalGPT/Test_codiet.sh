#!/bin/bash
set -euo pipefail

# CausalGPT (EXDBN-DNN) CoDiet runner.
# Scans /Users/xiaoyuhe/Datasets/CoDiet for codiet*.csv and runs ExDBN_perform.
#
# Environment overrides:
#   DATASET    : "codiet" (default) or "asia" (kept for compatibility)
#   DATA_DIR   : directory containing input CSV files (pattern depends on DATASET)
#   OUT_BASE   : base output directory (each run gets its own subfolder)
#   S_LIST     : (asia only) sample sizes, e.g. "250 1000" (optional; if unset, auto-discover)
#   R_LIST     : (asia only) replicate indices, e.g. "0 1 2 3 4 5 6" (optional; if unset, auto-discover)
#   CONF_LIST  : confidence list, e.g. "0.9 0.99999" (default: "0.99999")
#   EPOCHS     : DNN epochs (ExDBN_perform runs DNN stage), default 1 (smoke)
#   DEGREE     : max in-degree for EXDBN MILP, default 5
#   FAST_RUN   : if "1", set fast EXDBN MILP env vars (TimeLimit/MIPGap + gurobi progress) (default: 1)
#   MAX_CASES  : if set, stop after running this many files (useful for smoke tests)
#
# NOTE: ExDBN_perform expects numeric CSV values. The LLM_CD Asia CSVs are yes/no strings,
# so by default this script points to /Users/xiaoyuhe/Datasets/Asia.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXDBN_DNN_ROOT="$(cd "$HERE/../.." && pwd)"   # .../ExDBN-DNN

DATASET="${DATASET:-codiet}"

if [[ "$DATASET" == "asia" ]]; then
  DATA_DIR="${DATA_DIR:-/Users/xiaoyuhe/Datasets/Asia}"
elif [[ "$DATASET" == "codiet" ]]; then
  DATA_DIR="${DATA_DIR:-/Users/xiaoyuhe/Datasets/CoDiet}"
else
  echo "Unknown DATASET=$DATASET (expected: asia|codiet)" >&2
  exit 2
fi

OUT_BASE="${OUT_BASE:-$EXDBN_DNN_ROOT/reports/causalgpt_runs_codiet}"

CONF_LIST=${CONF_LIST:-"0.99999"}

EPOCHS=${EPOCHS:-1}
DEGREE=${DEGREE:-5}

# DNN stage requires torch; some environments (e.g. Python 3.13) may not have a compatible build.
# Default to skipping DNN for smoke tests; set SKIP_DNN=0 to enable.
SKIP_DNN=${SKIP_DNN:-1}

# If set to 1, fail fast unless the DNN stage actually runs.
REQUIRE_DNN=${REQUIRE_DNN:-0}

if [[ "$REQUIRE_DNN" == "1" && "$SKIP_DNN" == "1" ]]; then
  echo "REQUIRE_DNN=1 but SKIP_DNN=1; set SKIP_DNN=0 to enable DNN." >&2
  exit 2
fi

SKIP_DNN_ARG=()
if [[ "$SKIP_DNN" == "1" ]]; then
  SKIP_DNN_ARG=(--skip_dnn)
fi

REQUIRE_DNN_ARG=()
if [[ "$REQUIRE_DNN" == "1" ]]; then
  REQUIRE_DNN_ARG=(--require_dnn)
fi

echo "[CFG] DATASET=$DATASET EPOCHS=$EPOCHS SKIP_DNN=$SKIP_DNN REQUIRE_DNN=$REQUIRE_DNN"



# Prefer the local .venv (Python 3.11) to avoid torch issues on Python 3.13.
# Set RUNNER=uv to force the old behavior.
RUNNER="${RUNNER:-venv}"
VENV_PY="$HERE/.venv/bin/python"
USE_UV=0
if [[ "$RUNNER" == "uv" ]]; then
  USE_UV=1
elif [[ -x "$VENV_PY" ]]; then
  USE_UV=0
else
  USE_UV=1
fi

RUN_CMD=()
if [[ "$USE_UV" == "1" ]]; then
  RUN_CMD=(uv run "${UV_GROUPS[@]}")
else
  RUN_CMD=("$VENV_PY")
fi
UV_GROUPS=(--group dev)
if [[ "$SKIP_DNN" != "1" ]]; then
  # torch is in the optional dependency group 'dnn'
  UV_GROUPS+=(--group dnn)
fi

FAST_RUN=${FAST_RUN:-1}
MAX_CASES=${MAX_CASES:-}

# MILP on high-dimensional data can spend a long time in pre-processing (score enumeration)
# before Gurobi starts printing optimization progress. For smoke tests, cap the number of
# features used. Override by setting EXDBN_MAX_FEATURES (or set empty to disable).
EXDBN_MAX_FEATURES=${EXDBN_MAX_FEATURES:-}

if [[ "$FAST_RUN" == "1" ]]; then
  export EXDBN_TIME_LIMIT="${EXDBN_TIME_LIMIT:-30}"
  export EXDBN_TARGET_MIP_GAP="${EXDBN_TARGET_MIP_GAP:-0.10}"
  export EXDBN_GUROBI_OUTPUTFLAG="${EXDBN_GUROBI_OUTPUTFLAG:-1}"
  export EXDBN_GUROBI_DISPLAYINTERVAL="${EXDBN_GUROBI_DISPLAYINTERVAL:-1}"

  if [[ -z "${EXDBN_MAX_FEATURES:-}" ]]; then
    EXDBN_MAX_FEATURES=20
  fi
fi

MAX_FEATURES_ARG=()
if [[ -n "${EXDBN_MAX_FEATURES:-}" ]]; then
  MAX_FEATURES_ARG=(--max_features "$EXDBN_MAX_FEATURES")
fi

# Soft priors (same as perform_CaMML.sh for Asia).
ANCS_ASIA="[(0,1),(1,5),(1,7),(2,3),(2,4),(3,5),(3,7),(4,7),(5,6)]"
FORB_ANCS_ASIA="[]"

# CoDiet: default to no priors.
ANCS_CODIET="[]"
FORB_ANCS_CODIET="[]"

cd "$HERE"

shopt -s nullglob

cases_run=0

if [[ "$DATASET" == "asia" ]]; then
  DATA_FILES=("$DATA_DIR"/asia_*_*.csv)
  if [[ ${#DATA_FILES[@]} -eq 0 ]]; then
    echo "No files matched: $DATA_DIR/asia_*_*.csv" >&2
    exit 2
  fi

  for conf in $CONF_LIST; do
    for data_path in "${DATA_FILES[@]}"; do
      base="$(basename "$data_path" .csv)"
      # Expect: asia_<N>_<r>
      IFS="_" read -r dataset s r <<< "$base"
      if [[ "$dataset" != "asia" || -z "${s:-}" || -z "${r:-}" ]]; then
        echo "[SKIP] unexpected filename: $data_path" >&2
        continue
      fi

      run_out_dir="$OUT_BASE/asia/N${s}_r${r}_conf${conf}"

      echo "[RUN] $base conf=$conf -> $run_out_dir"
      "${RUN_CMD[@]}" -m CausalGPT.tests.ExDBN_perform \
        --data_path "$data_path" \
        --out_dir "$run_out_dir" \
        --align_asia \
        --write_anc \
        "${MAX_FEATURES_ARG[@]+"${MAX_FEATURES_ARG[@]}"}" \
        "${SKIP_DNN_ARG[@]+"${SKIP_DNN_ARG[@]}"}" \
        "${REQUIRE_DNN_ARG[@]+"${REQUIRE_DNN_ARG[@]}"}" \
        --epochs "$EPOCHS" \
        --degree "$DEGREE" \
        --conf "$conf" \
        --ancs "$ANCS_ASIA" \
        --forb_ancs "$FORB_ANCS_ASIA"

      cases_run=$((cases_run + 1))
      if [[ -n "${MAX_CASES:-}" && "$cases_run" -ge "$MAX_CASES" ]]; then
        echo "[STOP] MAX_CASES=$MAX_CASES reached" >&2
        exit 0
      fi
    done
  done

elif [[ "$DATASET" == "codiet" ]]; then
  # CoDiet: prefer running the deterministic replicate files generated by
  # tools/generate_codiet_replicates.py (e.g. codiet_302_0..5.csv).
  # This avoids accidentally picking up other codiet*.csv files that may be
  # semicolon-separated or otherwise incompatible.
  #
  # Override file selection via:
  #   CODIET_GLOB="codiet*.csv" DATASET=codiet ./Test_codiet.sh
  CODIET_GLOB="${CODIET_GLOB:-codiet_302_*.csv}"

  DATA_FILES=("$DATA_DIR"/$CODIET_GLOB)
  if [[ ${#DATA_FILES[@]} -eq 0 && "$CODIET_GLOB" != "codiet*.csv" ]]; then
    CODIET_GLOB="codiet*.csv"
    DATA_FILES=("$DATA_DIR"/$CODIET_GLOB)
  fi

  if [[ ${#DATA_FILES[@]} -eq 0 ]]; then
    echo "No files matched: $DATA_DIR/$CODIET_GLOB" >&2
    exit 2
  fi

  for conf in $CONF_LIST; do
    for data_path in "${DATA_FILES[@]}"; do
      base="$(basename "$data_path" .csv)"
      if [[ "$base" == "codiet_data_code" || "$base" == codiet_data_* ]]; then
        echo "[SKIP] non-compatible CSV format: $base" >&2
        continue
      fi

      run_out_dir="$OUT_BASE/codiet/${base}_conf${conf}"
      anc_path="$run_out_dir/test_ancs.anc"

      echo "[RUN] $base conf=$conf -> $run_out_dir"
      "${RUN_CMD[@]}" -m CausalGPT.tests.ExDBN_perform \
        --data_path "$data_path" \
        --anc_path "$anc_path" \
        --out_dir "$run_out_dir" \
        --write_anc \
        "${MAX_FEATURES_ARG[@]+"${MAX_FEATURES_ARG[@]}"}" \
        "${SKIP_DNN_ARG[@]+"${SKIP_DNN_ARG[@]}"}" \
        "${REQUIRE_DNN_ARG[@]+"${REQUIRE_DNN_ARG[@]}"}" \
        --epochs "$EPOCHS" \
        --degree "$DEGREE" \
        --conf "$conf" \
        --ancs "$ANCS_CODIET" \
        --forb_ancs "$FORB_ANCS_CODIET"

      cases_run=$((cases_run + 1))
      if [[ -n "${MAX_CASES:-}" && "$cases_run" -ge "$MAX_CASES" ]]; then
        echo "[STOP] MAX_CASES=$MAX_CASES reached" >&2
        exit 0
      fi
    done
  done
fi
