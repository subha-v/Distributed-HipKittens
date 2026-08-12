#!/usr/bin/env bash
# exp_33 probe 4: is there prior MEASURED pf6gm_mega per-module data (exp_58 residual
# attribution) that can serve as a cross-run reference for the M7 epilogue surcharge?
# Read-only.
set -uo pipefail
K0="$HOME/amd-master/auto-gpu-kernel/k0_fused_moe"

echo "=== exp_58 folder ==="
ls -la "$K0/experiments/exp_58_residual_attribution/" 2>&1 | sed -n '1,30p'

echo
echo "=== any recorded PF6GM DECOMP numbers anywhere on the node ==="
grep -rl 'PF6GM DECOMP' "$K0/experiments" 2>/dev/null | sed -n '1,20p'
grep -rh 'M7_n2_phase2\|M6_n2_phase1' "$K0/experiments" 2>/dev/null | sed -n '1,30p'

echo
echo "=== result.md of exp_58, module table region ==="
for f in "$K0/experiments/exp_58_residual_attribution/"*.md; do
  echo "--- $f ---"
  grep -nE 'M0_retire|M1_qpush|M2_stream|M3M4M5|M6_n2|M7_n2|M8M9|steady_skew|phase=[0-6]' "$f" | sed -n '1,40p'
done

echo
echo "=== driver progress ==="
tail -n 6 "$HOME/e33/e33a_driver.log"
LG=$(ls -t $HOME/overnight-scratch/e33a_*.log 2>/dev/null | head -1)
echo "--- run markers so far ---"
grep -nE 'starting run=|completed run=|campaign complete' "$LG" | sed -n '1,20p'
echo "--- K0PF PROFILE lines so far ---"
grep -h 'K0PF PROFILE' "$LG" | sed -n '1,10p'
echo "=== PROBE4 DONE ==="
exit 0
