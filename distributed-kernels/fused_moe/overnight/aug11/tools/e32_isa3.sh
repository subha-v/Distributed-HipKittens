#!/usr/bin/env bash
# exp_32 Job 2 step 3: opcode histogram straight off D.isa (it holds exactly one
# kernel symbol, k0pf6gm_mps_mega, at file lines 6..end). READ-ONLY.
set -uo pipefail
SC=$HOME/overnight-scratch/e26
OUT=$HOME/overnight-scratch/e32
mkdir -p "$OUT"
ISA=$SC/out2/D.isa

echo "===OPCODE HISTOGRAM (top 70)==="
sed 's|//.*||' "$ISA" | sed 's/^[[:space:]]*//' | awk 'NF{print $1}' \
  | grep -E '^[a-z]' | sort | uniq -c | sort -rn | head -70

echo "===MEMORY OPCODES==="
sed 's|//.*||' "$ISA" | sed 's/^[[:space:]]*//' | awk 'NF{print $1}' \
  | grep -E '^(global|flat|buffer|scratch|ds)_' | sort | uniq -c | sort -rn
exit 0
