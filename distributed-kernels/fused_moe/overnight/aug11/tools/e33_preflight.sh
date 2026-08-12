#!/usr/bin/env bash
# exp_33 pre-flight: node idleness, arm identity, and the production phase-signal survey.
# Read-only. Writes nothing outside $HOME/e33.
set -uo pipefail

DHK="$HOME/Distributed-HipKittens"
K0="$HOME/amd-master/auto-gpu-kernel/k0_fused_moe"
AB="$K0/prefill_opt/host/e004pf_k0pf_ab.py"
MPSSRC="$DHK/distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip"
mkdir -p "$HOME/e33"

echo "=== host/time ==="
hostname; date -u

echo "=== foreign torchrun/mpirun ==="
pgrep -af 'torchrun' 2>/dev/null || echo "(none: torchrun)"
pgrep -af 'mpirun' 2>/dev/null || echo "(none: mpirun)"

echo "=== rocm-smi --showpids ==="
/opt/rocm/bin/rocm-smi --showpids 2>&1 | sed -n '1,40p'

echo "=== gpu lock ==="
ls -ld /tmp/k0_mok_synthetic_gpu_lock 2>&1 || echo "(no lock dir)"

echo "=== node checkout sync ==="
git -C "$DHK" fetch --all -q
git -C "$DHK" reset -q --hard origin/codex/distributed-hipkittens-scaffold
git -C "$DHK" log -1 --format='HEAD=%H%n subj=%s%n date=%cI'
git -C "$DHK" rev-parse --abbrev-ref HEAD

echo "=== ratchet build flags in mps .hip ==="
grep -nE 'define +K0P6_MPS_ASCALE_TM' "$MPSSRC" || echo "ASCALE_TM: NOT FOUND"
grep -nE 'define +K0P6_MPS_SRC_REV' "$MPSSRC" || echo "SRC_REV: NOT FOUND"

echo "=== poison knob default ==="
grep -n 'K0_MOK_POISON_OUT' "$AB" | sed -n '1,10p'
grep -n 'K0_MOK_POISON_OUT' "$K0/benchmarks/mok_synthetic_prefill/run_campaign.sh" | sed -n '1,10p'

echo "=== production phase-signal survey (K0PF PROFILE / torch profiler hooks) ==="
grep -n 'K0PF PROFILE' "$AB" | sed -n '1,20p'
echo "--- profiler-ish knobs in ab.py ---"
grep -nE 'K0_[A-Z0-9_]*PROFIL|torch\.profiler|profile_memory|record_shapes|K0PF_PROFILE' "$AB" | sed -n '1,40p'
echo "--- MPS TS emitters ---"
grep -n 'MPS TS\|MPS SPIN' "$AB" | sed -n '1,20p'

echo "=== disk ==="
df -h / | sed -n '1,2p'
echo "=== screen.sh copies ==="
ls -l "$HOME/.overnight-scripts/screen.sh" 2>&1 || echo "(no node screen.sh)"
ls -l "$DHK/distributed-kernels/fused_moe/overnight/aug11/tools/screen.sh" 2>&1
md5sum "$HOME/.overnight-scripts/screen.sh" "$DHK/distributed-kernels/fused_moe/overnight/aug11/tools/screen.sh" 2>&1
echo "=== PREFLIGHT DONE ==="
exit 0
