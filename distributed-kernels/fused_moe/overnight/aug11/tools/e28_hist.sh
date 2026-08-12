#!/usr/bin/env bash
# exp_28 probe 2 (read-only): the authoritative historical G-kill record.
set -uo pipefail
AM="$HOME/amd-master/auto-gpu-kernel/k0_fused_moe"
E59="$AM/experiments/exp_59_large_m_tiles"
E63="$AM/experiments/exp_63_large_m_rerun"

echo "===E59_RESULT==="
cat "$E59/result.md"
echo "===E59_REGFP==="
cat "$E59/rescheck/register_fingerprint.md"
echo "===E59_RESCHECK_LS==="
ls -la "$E59/rescheck"
echo "===E59_DESIGN_REG_SECTION==="
sed -n '230,300p' "$E59/design.md"
echo "===E63_RESULT_TAIL==="
sed -n '100,150p' "$E63/result.md"
echo "===SUMMARY_G_PARA==="
sed -n '800,830p' "$AM/experiments/summary.md"
echo "===LEVER_MATRIX_G==="
sed -n '195,235p' "$AM/experiments/lever_matrix.md"
echo "===DONE==="
