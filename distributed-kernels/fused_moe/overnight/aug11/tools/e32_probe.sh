#!/usr/bin/env bash
# exp_32 read-only reconnaissance. NO GPU WORK, NO WRITES to any tracked tree.
#  1. hash the harness files we intend to patch (so the diff applies cleanly)
#  2. locate the exp_26 compile scaffolding + any already-disassembled ISA
set -uo pipefail
K0=$HOME/amd-master/auto-gpu-kernel/k0_fused_moe
SC=$HOME/overnight-scratch/e26

echo "===HARNESS FILE IDENTITY==="
for f in "$K0/prefill_opt/host/e004pf_k0pf_ab.py" \
         "$K0/benchmarks/mok_synthetic_prefill/run_campaign.sh" \
         "$K0/benchmarks/mok_synthetic_prefill/correctness.py" \
         "$K0/benchmarks/mok_synthetic_prefill/MPS_OVERNIGHT_HARNESS_NOTE.md"; do
  if [[ -f "$f" ]]; then
    printf '%s  %s  lines=%s  mtime=%s\n' "$(sha256sum "$f" | cut -c1-16)" "$f" \
      "$(wc -l < "$f")" "$(date -u -r "$f" +%Y-%m-%dT%H:%M:%SZ)"
  else
    echo "MISSING $f"
  fi
done

echo "===AB.PY ANCHOR LINES (the four insertion points)==="
grep -n 'cand_out = torch.zeros\|if name not in _PROD_LIKE: cand_out.zero_()\|hbarrier(); cand_out.zero_()\|for _mps_epoch in range(_mps_soak_iters)\|def _mok_measure_arm\|strict_gate = gate(_obuf(name))' \
  "$K0/prefill_opt/host/e004pf_k0pf_ab.py"

echo "===RUN_CAMPAIGN -e FORWARD LIST==="
grep -n '\-e ' "$K0/benchmarks/mok_synthetic_prefill/run_campaign.sh"

echo "===EXP_26 SCAFFOLDING==="
ls -la "$SC" 2>&1
echo "--- out2 ---"
ls -la "$SC/out2" 2>&1
echo "--- tu ---"
ls -laR "$SC/tu" 2>&1
echo "--- dhk kernel identity in the snapshot ---"
sha256sum "$SC/dhk/distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip" 2>&1
wc -l "$SC/dhk/distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip" 2>&1
echo "--- LIVE node checkout kernel identity (read-only) ---"
sha256sum "$HOME/Distributed-HipKittens/distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip" 2>&1
wc -l "$HOME/Distributed-HipKittens/distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip" 2>&1
cd "$HOME/Distributed-HipKittens" && git log --oneline -1 && git status --porcelain | head -20

echo "===GPU BUSY CHECK (informational only; we launch nothing)==="
rocm-smi --showpids 2>&1 | head -20
exit 0
