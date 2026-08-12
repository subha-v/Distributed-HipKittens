#!/usr/bin/env bash
# t23: leave the node in the shipping state and record it.
set -uo pipefail
DHK="$HOME/Distributed-HipKittens"
MPSSRC="$DHK/distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip"
echo "== node idle? =="
pgrep -af 'torchrun|mpirun|screen.sh' || echo "no torchrun/mpirun/screen.sh"
/opt/rocm/bin/rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+[ \t]/ {print $1" "$2}'
ls -ld /tmp/k0_mok_synthetic_gpu_lock 2>/dev/null || echo "(no stale lock dir)"
echo "== sync node checkout to origin =="
git -C "$DHK" fetch --all -q
git -C "$DHK" reset -q --hard origin/codex/distributed-hipkittens-scaffold
echo "node HEAD  : $(git -C "$DHK" rev-parse HEAD)"
echo "node short : $(git -C "$DHK" rev-parse --short HEAD)"
git -C "$DHK" status --porcelain=v1 | head -5
echo "== shipping kernel state =="
grep -nE '^#define K0P6_MPS_SRC_REV [0-9]+|^#define N2GM_P1_SCHED_GSCALE [0-9]+|^#include "n2_phase1_gm(_mps)?\.cpp"' "$MPSSRC"
echo "kernel sha256: $(sha256sum "$MPSSRC" | cut -c1-32)"
echo "== harness state =="
K0="$HOME/amd-master/auto-gpu-kernel/k0_fused_moe"
for f in prefill_opt/host/e004pf_k0pf_ab.py \
         benchmarks/mok_synthetic_prefill/run_campaign.sh \
         benchmarks/mok_synthetic_prefill/MPS_OVERNIGHT_HARNESS_NOTE.md; do
  printf '%s  %s  %s lines\n' "$(sha256sum "$K0/$f" | cut -c1-16)" "$f" "$(wc -l < "$K0/$f")"
done
echo "== poison knobs forwarded? =="
grep -n 'K0_MOK_POISON' "$K0/benchmarks/mok_synthetic_prefill/run_campaign.sh"
echo "== batch inventory =="
for t in e26_m0a e26_m4 e26_m1 e26_m1b e26_m5 e26_m0b e26_m4b e26_m5b e32regate; do
  f="$HOME/overnight-scratch/screen_${t}.csv"
  # the cfg field is quoted and contains commas, so blank quoted fields first
  [ -f "$f" ] && printf '%-12s %d data rows, statuses: %s\n' "$t" "$(( $(wc -l < "$f") - 1 ))" \
    "$(sed 's/"[^"]*"/Q/g' "$f" | awk -F, 'NR>1{print $4}' | sort -u | tr '\n' ' ')"
done
echo "== done =="
exit 0
