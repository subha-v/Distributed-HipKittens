#!/usr/bin/env bash
# exp_23: instruction-level corroboration of the parity gate.
#
# G1 (identical resource tuple) says the ring is free in ALLOCATION. This says
# what actually landed in the instruction stream, which is the other half of the
# story. Rather than guess at mnemonics, take the full opcode histogram of each
# TU and diff it against the reference: the tier-A delta must be exactly the ring
# stores and their address arithmetic, with ZERO change to the clock reads
# (s_memrealtime) and ZERO change to the max-atomics that maintain the coarse
# cells. If the atomic count moves, ts_mark is not sharing one read.
set -uo pipefail
SC=${SC:-$HOME/overnight-scratch/e23}
cd "$SC/out"

hist() {   # opcode histogram of one disassembly
  awk '{ for (i = 1; i <= NF; i++) if ($i ~ /^(s_|v_|global_|flat_|scratch_|buffer_|ds_)/) { print $i; break } }' "$1" \
    | sort | uniq -c | awk '{printf "%s %s\n", $2, $1}' | sort
}

echo "=== headline counts ==="
printf '%-5s %-14s %-16s %-11s %-11s %s\n' TU s_memrealtime flat_atomic_umax_x2 s_waitcnt isa_lines text_B
for v in R TA TAON TCON; do
  [ -f "$v.isa" ] || continue
  printf '%-5s %-14s %-16s %-11s %-11s %s\n' "$v" \
    "$(grep -c 's_memrealtime' "$v.isa")" \
    "$(grep -cE '(flat|global)_atomic_[su]?max_x2' "$v.isa")" \
    "$(grep -c 's_waitcnt' "$v.isa")" \
    "$(wc -l < "$v.isa")" \
    "$(stat -c %s "$v.text.bin")"
done

for v in TA TAON TCON; do
  [ -f "$v.isa" ] || continue
  echo
  echo "=== opcode histogram delta: $v vs R (only opcodes whose count moved) ==="
  hist R.isa > /tmp/e23_h_R.txt
  hist "$v.isa" > "/tmp/e23_h_$v.txt"
  join -a1 -a2 -e 0 -o 0,1.2,2.2 /tmp/e23_h_R.txt "/tmp/e23_h_$v.txt" \
    | awk -v tu="$v" 'BEGIN{printf "%-34s %8s %8s %8s\n","opcode","R",tu,"delta"}
        { d = $3 - $2; if (d != 0) printf "%-34s %8d %8d %+8d\n", $1, $2, $3, d }'
done
