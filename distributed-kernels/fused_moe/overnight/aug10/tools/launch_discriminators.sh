#!/usr/bin/env bash
# Stage run_discriminators-equivalent body on the node and nohup it, so the
# ssh call returns immediately and the 8-GPU job keeps running.
set -uo pipefail
mkdir -p "$HOME/ovn"
cat > "$HOME/ovn/discriminators.sh" <<'BODY'
#!/usr/bin/env bash
set -uo pipefail
cd "$HOME/amd-master/auto-gpu-kernel/k0_fused_moe" || exit 2

run_one() {
  local tag="$1"; local cfg="$2"
  local out="$HOME/k0-mok-mps-$tag"
  local log="$HOME/exp01_$tag.log"
  rm -rf "$out"; mkdir -p "$out"; rm -f "$log"

  echo "########## $tag : K0_MPS_CFG=$cfg ##########"
  date -u +%FT%TZ
  if pgrep -af 'torchrun|mpirun' > /dev/null; then
    echo "REFUSING $tag: a gpu job is already running"; pgrep -af 'torchrun|mpirun'; return 3
  fi

  timeout 900 env \
    K0_MOK_ARMS=production,mps_mega \
    K0_MOK_WARMUP_ITERS=1 K0_MOK_TIMED_ITERS=1 \
    K0_MOK_OUTPUT_ROOT="$out" \
    K0_MOK_RUN_TIMEOUT=700 \
    K0_MPS_CFG="$cfg" \
    K0_MPS_TRACE=1 \
    bash benchmarks/mok_synthetic_prefill/run_campaign.sh "$tag" 1 \
    > "$log" 2>&1 < /dev/null
  echo "--- campaign exit=$? ---"

  echo "--- memory-fault count ---"
  grep -c 'Memory access fault' "$log"
  grep -m 2 'Memory access fault' "$log"
  echo "--- progress rank0 ---"
  cat "$out"/run1/progress_rank0.log 2>/dev/null || echo "(no progress log)"
  echo "--- correctness ---"
  grep -m 8 -E 'rel_L2|pperr|control_fails|rel_L1|max_abs' "$log"
  echo
  sleep 20
}

run_one pullfb  "C=8,g=2,mode=2,flush_rows=16,pull_fallback=1"
run_one mode0c8 "C=8,mode=0"
run_one mode1c0 "C=0,mode=1"

echo "########## ALL DISCRIMINATORS DONE ##########"
date -u +%FT%TZ
BODY

chmod +x "$HOME/ovn/discriminators.sh"
rm -f "$HOME/ovn/discriminators.out"
nohup setsid bash "$HOME/ovn/discriminators.sh" > "$HOME/ovn/discriminators.out" 2>&1 < /dev/null &
echo "launched discriminators pid $!"
sleep 3
echo "--- preflight snapshot ---"
head -5 "$HOME/ovn/discriminators.out"
exit 0
