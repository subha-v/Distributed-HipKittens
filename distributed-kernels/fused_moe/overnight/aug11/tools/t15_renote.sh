#!/usr/bin/env bash
# t15: the first append went through a non-UTF8 pipe and mangled em-dashes to
# '???'. Restore the note from the pre-append backup and re-append the
# ASCII-only text.
set -uo pipefail
NOTE="$HOME/amd-master/auto-gpu-kernel/k0_fused_moe/benchmarks/mok_synthetic_prefill/MPS_OVERNIGHT_HARNESS_NOTE.md"
SRC="$HOME/overnight-scratch/e26act/incoming/t13_note.md"
PRE="$(ls -1t "$HOME/harness-backups/e32/note.pre."*.md 2>/dev/null | tail -1)"
[ -n "$PRE" ] && [ -f "$PRE" ] || { echo "no pre-append backup found"; exit 3; }
echo "restoring from $PRE ($(wc -l < "$PRE") lines)"
cp -p "$PRE" "$NOTE"
grep -c '???' "$NOTE" && echo "WARNING mangling in the restored base" || true
printf '\n' >> "$NOTE"
cat "$SRC" >> "$NOTE"
echo "note now $(wc -l < "$NOTE") lines, sha $(sha256sum "$NOTE" | cut -c1-16)"
echo "non-ASCII bytes remaining: $(LC_ALL=C grep -c '[^ -~\t]' "$NOTE" || true)"
grep -n '^## ' "$NOTE" | tail -4
exit 0
