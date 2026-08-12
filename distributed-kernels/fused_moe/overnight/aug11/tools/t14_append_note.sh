#!/usr/bin/env bash
# t14: append the exp_32 + exp_26 harness-edit record to MPS_OVERNIGHT_HARNESS_NOTE.md
# (project rule: every harness edit is logged there).
set -uo pipefail
NOTE="$HOME/amd-master/auto-gpu-kernel/k0_fused_moe/benchmarks/mok_synthetic_prefill/MPS_OVERNIGHT_HARNESS_NOTE.md"
SRC="$HOME/overnight-scratch/e26act/incoming/t13_note.md"
[ -f "$SRC" ] || { echo "missing $SRC"; exit 3; }
if grep -q '^## exp_32 — NaN poison' "$NOTE" 2>/dev/null; then
  echo "already appended -- skipping"; exit 0
fi
cp -p "$NOTE" "$HOME/harness-backups/e32/note.pre.$(date -u +%Y%m%dT%H%M%SZ).md"
printf '\n' >> "$NOTE"
cat "$SRC" >> "$NOTE"
echo "appended. note is now $(wc -l < "$NOTE") lines, sha $(sha256sum "$NOTE" | cut -c1-16)"
echo "== new section headings =="
grep -n '^## ' "$NOTE" | tail -8
exit 0
