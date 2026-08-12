#!/usr/bin/env bash
# exp_28 probe 1 (read-only): G-history, donor helper semantics, container env.
set -uo pipefail
AM="$HOME/amd-master/auto-gpu-kernel/k0_fused_moe"
SOL="$AM/solution/hip"

echo "===GPUJOBS==="
pgrep -af 'run_campaign|torchrun|mpirun' || echo "(none)"

echo "===G_HISTORY_DIRS==="
ls -d "$AM"/experiments/*large_m* "$AM"/experiments/*exp_59* "$AM"/experiments/*exp_63* \
      "$AM"/experiments/*exp_64* 2>/dev/null || echo "(no large_m dirs)"

echo "===G_GREP_RESULTS==="
grep -rln --include=result.md -e 'G=4' -e 'kGM = 4' -e 'N2GM_G=4' -e 'G-stack' "$AM/experiments" 2>/dev/null | head -40

echo "===G4_ANY_MENTION==="
grep -rn --include='*.md' -e 'G=4' -e 'N2GM_G=4' -e 'K0P6GM_G=4' "$AM/experiments" 2>/dev/null | head -60

echo "===EXP59_RESULT_HEAD==="
for d in "$AM"/experiments/exp_59*/ "$AM"/experiments/exp_63*/ "$AM"/experiments/exp_64*/; do
  [ -d "$d" ] || continue
  echo "--- $d"
  ls "$d"
done

echo "===STAGE_AGPR==="
grep -n -B 4 -A 20 'stage_agpr' "$SOL/n2_device_common.cuh" | head -60

echo "===LOAD_BFRAG==="
grep -n -B 4 -A 24 'load_bfrag' "$SOL/n2_device_common.cuh" | head -70

echo "===RFP8_RACC_DEFS==="
grep -n -e 'using rfp8' -e 'using racc' -e 'accv' -e 'accf' -e 'kBlockM' -e 'kWaves' \
     -e 'kThreads' -e 'kAChunks' -e 'kAChunkBytes' -e 'kKGroups2' -e 'kInter' \
     "$SOL/n2_device_common.cuh" "$SOL/n2_fused_moe.hpp" 2>/dev/null | head -60

echo "===CONTAINER_SEES_HOME==="
docker exec subha_k1 bash -lc 'ls -d ~/amd-master ~/Distributed-HipKittens 2>/dev/null; echo "whoami=$(whoami)"; hipcc --version | head -3'

echo "===MORI_JIT==="
docker exec subha_k1 bash -lc 'ls -d /usr/local/lib/python3.12/dist-packages/mori/_jit-sources 2>/dev/null || echo MISSING'

echo "===NODE_HEAD==="
git -C "$HOME/Distributed-HipKittens" log --oneline -3
git -C "$HOME/Distributed-HipKittens" status --porcelain=v1 | head -20
