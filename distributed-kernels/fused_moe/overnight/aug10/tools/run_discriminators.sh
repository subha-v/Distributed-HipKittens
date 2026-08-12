#!/usr/bin/env bash
# exp_01 part 2: three FULL runs (no debug stop) that are all pre-registered
# sweep points AND fault discriminators. Serialized, one 8-GPU job at a time.
#   A: mode=2 + pull_fallback=1 -> streamed flags kept, M8 switched to remote
#      `part` pulls. CLEAN => fault is slot addressing; FAULT => service/stream.
#   B: mode=0, C=8              -> reserved-CTA tax control, no push.
#   C: mode=1, C=0              -> push layout only.
set -uo pipefail
cd "$HOME/amd-master/auto-gpu-kernel/k0_fused_moe" || exit 2

run_one() {
  local tag="$1"; local cfg="$2"
  local out="$HOME/k0-mok-mps-$tag"
  local log="$HOME/exp01_$tag.log"
  rm -rf "$out"; mkdir -p "$out"; rm -f "$log"

  echo "########## $tag : K0_MPS_CFG=$cfg ##########"
  if pgrep -af 'torchrun|mpirun' > /dev/null; then
    echo "REFUSING $tag: a gpu job is already running"; pgrep -af 'torchrun|mpirun'; return 3
  fi

  setsid timeout 900 env \
    K0_MOK_ARMS=production,mps_mega \
    K0_MOK_WARMUP_ITERS=1 K0_MOK_TIMED_ITERS=1 \
    K0_MOK_OUTPUT_ROOT="$out" \
    K0_MOK_RUN_TIMEOUT=700 \
    K0_MPS_CFG="$cfg" \
    K0_MPS_TRACE=1 \
    bash benchmarks/mok_synthetic_prefill/run_campaign.sh "$tag" 1 \
    > "$log" 2>&1 < /dev/null

  echo "--- exit=$? ---"
  echo "--- fault lines ---"
  grep -c 'Memory access fault' "$log"
  grep -m 3 'Memory access fault' "$log"
  echo "--- progress rank0 ---"
  cat "$out"/run1/progress_rank0.log 2>/dev/null || echo "(no progress log)"
  echo "--- correctness lines ---"
  grep -m 6 -E 'rel_L2|pperr|control_fails|rel_L1|max_abs' "$log"
  echo
  sleep 20
}

run_one pullfb  "C=8,g=2,mode=2,flush_rows=16,pull_fallback=1"
run_one mode0c8 "C=8,mode=0"
run_one mode1c0 "C=0,mode=1"

echo "########## ALL DONE ##########"
