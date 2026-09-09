#!/usr/bin/env bash
# Formal v9 exact-fit collection. Raw episodes contain the common Pi0.5/ACT/DP
# view plus PAP-MoE force observations and four-expert routing targets.  Legacy
# semantic/progress metadata may remain in raw files for recovery indexing, but
# it is not an input or supervision target of the current PAP-MoE model.
set -euo pipefail

WS_DIR="/home/ubuntu/ur3_ft300_ws"
export PAP_MOE_PILOT_ROOT="${PAP_MOE_PILOT_ROOT:-$WS_DIR/pap_moe_framework/datasets/raw_v9_exactfit_scripted}"

if [[ "$#" -eq 0 ]]; then
  set -- false 1
elif [[ "$#" -eq 1 ]]; then
  set -- "$1" 1
fi

exec "$WS_DIR/scripts/run_pap_moe_v8_exactfit_pilot_collection.sh" "$@"
