#!/usr/bin/env bash
# Relaunch the mode-0 / mode-1 controls with COMPLETE K0_MPS_CFG strings.
# The previous attempt passed only C and mode; the host requires all four of
# C,g,mode,flush_rows (optional: pull_fallback,timestamps) and rejected it
# before launch, so those runs were void rather than clean.
#
# mode 0 = reserved-CTA capacity tax only, no push. If this is CLEAN with
# pperr=0 it proves the reservation + tail are fine when the push transport is
# off, isolating the fault to the push path (which translates desc[61]).
set -uo pipefail
mkdir -p "$HOME/ovn"
cat > "$HOME/ovn/mode01.sh" <<'BODY'
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
    echo "REFUSING $tag: a gpu job is already running"; return 3
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

  echo "--- config rejected? ---"
  grep -c 'K0_MPS_CFG requires' "$log"
  echo "--- memory-fault count ---"
  grep -c 'Memory access fault' "$log"
  echo "--- progress rank0 ---"
  cat "$out"/run1/progress_rank0.log 2>/dev/null || echo "(no progress log)"
  echo "--- MARK lines (all arms) ---"
  grep -E '\[MARK\]' "$log" | head -10
  echo "--- control ---"
  grep -E 'control_fails' "$log" | head -3
  echo
  sleep 20
}

run_one mode0c8v2 "C=8,g=2,mode=0,flush_rows=16"
run_one mode1c0v2 "C=0,g=2,mode=1,flush_rows=16"

echo "########## MODE 0/1 CONTROLS DONE ##########"
date -u +%FT%TZ
BODY

chmod +x "$HOME/ovn/mode01.sh"
rm -f "$HOME/ovn/mode01.out"
nohup setsid bash "$HOME/ovn/mode01.sh" > "$HOME/ovn/mode01.out" 2>&1 < /dev/null &
echo "launched mode01 controls pid $!"
sleep 3
head -3 "$HOME/ovn/mode01.out"
exit 0
