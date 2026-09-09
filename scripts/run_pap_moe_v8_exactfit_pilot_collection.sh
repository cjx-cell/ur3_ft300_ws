#!/usr/bin/env bash
# PAP-MoE v8 exact-fit pilot. Keep new sim-time, q/c/h-routed episodes
# physically separate from the legacy raw_v6_pilot corpus.
set -euo pipefail

WS_DIR="/home/ubuntu/ur3_ft300_ws"
export PAP_MOE_PILOT_ROOT="${PAP_MOE_PILOT_ROOT:-$WS_DIR/pap_moe_framework/datasets/raw_v8_exactfit_pilot}"

# The legacy runner defaults to episode 9001.  The isolated v8 corpus starts
# from 0001; preserve an explicitly supplied GUI/episode pair.
if [[ "$#" -eq 0 ]]; then
  set -- true 1
elif [[ "$#" -eq 1 ]]; then
  set -- "$1" 1
fi

exec "$WS_DIR/scripts/run_pap_moe_v6_pilot_collection.sh" "$@"
