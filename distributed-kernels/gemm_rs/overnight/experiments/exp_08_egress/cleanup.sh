#!/usr/bin/env bash
# Remove the two scratch .so files prof_confirm.sh left in harness/build so a
# later `ls build/*.so` is not misleading. The WGM=4 control arm is rebuildable
# in one line -- the command is recorded in result.md.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
docker exec dhk-gemmrs bash -lc \
  "rm -f $ON/harness/build/win_keep.so $ON/harness/build/gemm_rs_wgm4.so; \
   ls $ON/harness/build/*.so | sed 's#.*/##'"
echo "--- node clean ---"
rocm-smi --showpids 2>/dev/null | grep -c '^[0-9]' || true
