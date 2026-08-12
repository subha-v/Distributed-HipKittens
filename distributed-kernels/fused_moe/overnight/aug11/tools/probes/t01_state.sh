#!/usr/bin/env bash
# t01: node state before any write. Idleness, git, harness identity, SRC_REV.
set -uo pipefail
K0="$HOME/amd-master/auto-gpu-kernel/k0_fused_moe"
DHK="$HOME/Distributed-HipKittens"
echo "== date =="; date -u +%FT%TZ
echo "== idle: torchrun/mpirun =="; pgrep -af 'torchrun|mpirun' || echo "(none)"
echo "== idle: rocm-smi showpids =="
/opt/rocm/bin/rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+[ \t]/ {print $1" "$2}' || true
echo "== lock =="; ls -ld /tmp/k0_mok_synthetic_gpu_lock 2>/dev/null || echo "(no lock dir)"
echo "== DHK git =="
git -C "$DHK" rev-parse HEAD
git -C "$DHK" rev-parse --abbrev-ref HEAD
git -C "$DHK" status --porcelain=v1 | head -20
echo "== SRC_REV =="
grep -n 'K0P6_MPS_SRC_REV' "$DHK/distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip" | head -5
echo "== phase1 include site =="
sed -n '388,400p' "$DHK/distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip"
echo "== harness identity =="
for f in prefill_opt/host/e004pf_k0pf_ab.py \
         benchmarks/mok_synthetic_prefill/run_campaign.sh \
         benchmarks/mok_synthetic_prefill/correctness.py \
         benchmarks/mok_synthetic_prefill/summarize.py ; do
  if [ -f "$K0/$f" ]; then
    printf '%s  %s  %s lines  mtime %s\n' "$(sha256sum "$K0/$f" | cut -c1-16)" "$f" \
      "$(wc -l < "$K0/$f")" "$(stat -c %y "$K0/$f" | cut -c1-19)"
  else
    echo "MISSING $f"
  fi
done
echo "== harness git status (amd-master) =="
git -C "$HOME/amd-master" status --porcelain=v1 -- auto-gpu-kernel/k0_fused_moe/prefill_opt/host/e004pf_k0pf_ab.py auto-gpu-kernel/k0_fused_moe/benchmarks/mok_synthetic_prefill/ 2>/dev/null | head -20 || echo "(not a git repo / no changes)"
echo "== patch file present in DHK checkout? =="
ls -l "$DHK/distributed-kernels/fused_moe/overnight/aug11/exp_32_gate_hardening/" 2>/dev/null || echo "(exp_32 dir absent at node HEAD)"
echo "== vendored phase1 body =="
ls -l "$DHK/distributed-kernels/fused_moe/n2_phase1_gm_mps.cpp" 2>/dev/null || echo "(absent)"
echo "== hsaco stamp =="
JIT="$HOME/.cache/k0-mok-synthetic-prefill/mori/jit/gfx950_mlx5"
L="$JIT/latest/k0pf6gm_mps_mega.hsaco"
if [ -e "$L" ]; then R="$(readlink -f "$L")"; echo "$R"; stat -L -c '%y %s' "$R"; else echo "(no latest hsaco)"; fi
echo "== done =="
