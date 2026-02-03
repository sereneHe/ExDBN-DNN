#!/bin/bash
set -euo pipefail

# CausalGPT (EXDBN-DNN) runner, modeled after LLM_CD/perform_CaMML.sh.
# Runs Asia experiments with optional soft ancestor priors.
#
# Environment overrides:
#   DATA_DIR   : directory containing numeric asia_<N>_<r>.csv files
#   OUT_BASE   : base output directory (each run gets its own subfolder)
#   S_LIST     : sample sizes, e.g. "250 1000" (optional; if unset, auto-discover)
#   R_LIST     : replicate indices, e.g. "0 1 2 3 4 5 6" (optional; if unset, auto-discover)
#   CONF_LIST  : confidence list, e.g. "0.9 0.99999"
#   EPOCHS     : DNN epochs (ExDBN_perform runs DNN stage), default 300
#   DEGREE     : max in-degree for EXDBN MILP, default 5
#   FAST_RUN   : if "1", set fast EXDBN MILP env vars (TimeLimit/MIPGap + gurobi progress)
#   MAX_CASES  : if set, stop after running this many files (useful for smoke tests)
#
# NOTE: ExDBN_perform expects numeric CSV values. The LLM_CD Asia CSVs are yes/no strings,
# so by default this script points to /Users/xiaoyuhe/Datasets/Asia.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXDBN_DNN_ROOT="$(cd "$HERE/../.." && pwd)"   # .../ExDBN-DNN

DATA_DIR="${DATA_DIR:-/Users/xiaoyuhe/Datasets/Asia}"
OUT_BASE="${OUT_BASE:-$EXDBN_DNN_ROOT/reports/causalgpt_runs}"

CONF_LIST=${CONF_LIST:-"0.5 0.6 0.7 0.8 0.9 0.99999"}

EPOCHS=${EPOCHS:-300}
DEGREE=${DEGREE:-5}

FAST_RUN=${FAST_RUN:-0}
MAX_CASES=${MAX_CASES:-}

if [[ "$FAST_RUN" == "1" ]]; then
  export EXDBN_TIME_LIMIT="${EXDBN_TIME_LIMIT:-30}"
  export EXDBN_TARGET_MIP_GAP="${EXDBN_TARGET_MIP_GAP:-0.10}"
  export EXDBN_GUROBI_OUTPUTFLAG="${EXDBN_GUROBI_OUTPUTFLAG:-1}"
  export EXDBN_GUROBI_DISPLAYINTERVAL="${EXDBN_GUROBI_DISPLAYINTERVAL:-1}"
fi

# Soft priors (same as perform_CaMML.sh for Asia).
ANCS_ASIA="[(0,1),(1,5),(1,7),(2,3),(2,4),(3,5),(3,7),(4,7),(5,6)]"
FORB_ANCS_ASIA="[]"

cd "$HERE"

shopt -s nullglob
DATA_FILES=("$DATA_DIR"/asia_*_*.csv)
if [[ ${#DATA_FILES[@]} -eq 0 ]]; then
  echo "No files matched: $DATA_DIR/asia_*_*.csv" >&2
  exit 2
fi

cases_run=0
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
    uv run --group dev -m CausalGPT.tests.ExDBN_perform \
      --data_path "$data_path" \
      --out_dir "$run_out_dir" \
      --align_asia \
      --write_anc \
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
