#!/bin/bash
set -euo pipefail

# Run EXDBN on CoDiet replicates one-by-one using ALL features (no feature cap).
# This is intentionally conservative by default to reduce OOM risk.
#
# Usage:
#   cd /Users/xiaoyuhe/EXDBN-LLM/ExDBN-DNN/src/CausalGPT
#   bash Run_codiet_full_features.sh
#
# Override examples:
#   DEGREE=2 EXDBN_TIME_LIMIT=300 EXDBN_TARGET_MIP_GAP=0.05 bash Run_codiet_full_features.sh
#   R_LIST="0 1" bash Run_codiet_full_features.sh

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Replicates to run.
R_LIST=${R_LIST:-"0 1 2 3 4 5"}

# CoDiet replicate shape.
CODIET_N_ROWS=${CODIET_N_ROWS:-302}

# Solver / EXDBN settings (conservative defaults).
DEGREE=${DEGREE:-1}
EXDBN_TIME_LIMIT=${EXDBN_TIME_LIMIT:-120}
EXDBN_TARGET_MIP_GAP=${EXDBN_TARGET_MIP_GAP:-0.10}
EXDBN_GUROBI_OUTPUTFLAG=${EXDBN_GUROBI_OUTPUTFLAG:-1}
EXDBN_GUROBI_DISPLAYINTERVAL=${EXDBN_GUROBI_DISPLAYINTERVAL:-1}

# DNN is optional and can introduce extra memory use.
SKIP_DNN=${SKIP_DNN:-1}

# Keep FAST_RUN=1 so TimeLimit/MIPGap env vars take effect.
FAST_RUN=${FAST_RUN:-1}

# IMPORTANT: empty means "no cap" after the Test_codiet.sh patch.
EXDBN_MAX_FEATURES=${EXDBN_MAX_FEATURES-}

cd "$HERE"

for r in $R_LIST; do
  f="codiet_${CODIET_N_ROWS}_${r}.csv"
  echo "[FULL-FEATURES] Running replicate r=${r} file=${f}"

  DATASET=codiet \
  FAST_RUN="$FAST_RUN" \
  SKIP_DNN="$SKIP_DNN" \
  DEGREE="$DEGREE" \
  EXDBN_MAX_FEATURES="$EXDBN_MAX_FEATURES" \
  EXDBN_TIME_LIMIT="$EXDBN_TIME_LIMIT" \
  EXDBN_TARGET_MIP_GAP="$EXDBN_TARGET_MIP_GAP" \
  EXDBN_GUROBI_OUTPUTFLAG="$EXDBN_GUROBI_OUTPUTFLAG" \
  EXDBN_GUROBI_DISPLAYINTERVAL="$EXDBN_GUROBI_DISPLAYINTERVAL" \
  CODIET_N_ROWS="$CODIET_N_ROWS" \
  CODIET_GLOB="$f" \
  MAX_CASES=1 \
  bash Test_codiet.sh

done
