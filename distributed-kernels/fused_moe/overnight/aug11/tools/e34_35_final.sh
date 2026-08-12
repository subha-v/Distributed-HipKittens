#!/usr/bin/env bash
# exp_34: condition 5 log format, plus final hashes / diffs for the deliverable.
set -uo pipefail
K0=$HOME/amd-master/auto-gpu-kernel/k0_fused_moe
HOST=$K0/prefill_opt/host/e004pf_k0pf_ab.py
D=$HOME/e34/DHK/distributed-kernels/fused_moe

echo "############ 5. the soak / output gate print, verbatim ############"
sed -n '5040,5075p' "$HOST"
echo "--- output_gate assembly ---"
sed -n '5188,5205p' "$HOST"

echo
echo "############ negative-control diff (the ONLY difference vs the arm) ############"
diff -u "$D/k0pf6gm_device_tile_mps.hip" \
        "$HOME/e34/negctl/distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip" \
  | tee "$HOME/e34/out/negctl.patch" | head -40
echo "(negctl.patch written)"

echo
echo "############ arm diff vs pristine HEAD ############"
cd "$HOME/e34/DHK"
git diff --stat
git diff > "$HOME/e34/out/e34_final_v2.diff"
wc -l "$HOME/e34/out/e34_final_v2.diff"

echo
echo "############ hashes of the delivered files ############"
sha256sum "$D/k0pf6gm_device_tile_mps.hip" "$D/moe_mps_adapter.cuh"
echo
echo "############ built artifacts ############"
ls -l "$HOME/e34/out/"*.hsaco
echo "===DONE==="
