#!/usr/bin/env bash
set -uo pipefail
D=$HOME/e34/DHK/distributed-kernels/fused_moe
ADP=$D/moe_mps_adapter.cuh
KRN=$D/k0pf6gm_device_tile_mps.hip
echo "################ ADP 250-300 ################"; sed -n '250,300p' "$ADP"
echo "################ ADP 400-432 ################"; sed -n '400,432p' "$ADP"
echo "################ KRN 405,425 ################"; sed -n '405,425p' "$KRN"
echo "################ KRN 100,112 (rev) ################"; sed -n '100,112p' "$KRN"
echo "===DONE==="
