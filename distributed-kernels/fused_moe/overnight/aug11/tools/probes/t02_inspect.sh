#!/usr/bin/env bash
# t02: READ-ONLY. Dry-run the exp_32 patch, confirm anchors + ordering facts.
# Writes nothing outside /tmp.
set -uo pipefail
K0="$HOME/amd-master/auto-gpu-kernel/k0_fused_moe"
DHK="$HOME/Distributed-HipKittens"
PATCH="$DHK/distributed-kernels/fused_moe/overnight/aug11/exp_32_gate_hardening/exp_32_poison.patch"
AB="$K0/prefill_opt/host/e004pf_k0pf_ab.py"

echo "== patch sha =="; sha256sum "$PATCH"
echo "== git apply --check (in $K0) =="
cd "$K0" && git apply -p1 --check --verbose "$PATCH" 2>&1 | tail -20
echo "rc=$?"

echo "== is mok_gate defined before the eager loop? =="
grep -n '^def mok_gate\|^    def mok_gate\|mok_gate = \|def mok_gate' "$AB" | head -10
echo "-- eager loop start --"
grep -n 'eager correctness on every arm' "$AB" | head -3
echo "== K0_BENCHMARK_PROTOCOL definition =="
grep -n 'K0_BENCHMARK_PROTOCOL *=' "$AB" | head -5

echo "== anchor A: eager loop 4540..4575 =="
sed -n '4540,4575p' "$AB"
echo "== anchor B: MOK GATE loop 4615,4640 =="
sed -n '4615,4640p' "$AB"
echo "== anchor C: soak 4995,5015 =="
sed -n '4995,5015p' "$AB"
echo "== anchor D: measure arm 5118,5160p =="
sed -n '5118,5160p' "$AB"

echo "== who parses the [MPS SOAK] printed line? =="
grep -rn 'MPS SOAK' "$K0/benchmarks/mok_synthetic_prefill/" 2>/dev/null | head -10
echo "-- summarize.py soak handling --"
grep -n 'soak\|BLOCKED_STATUSES' "$K0/benchmarks/mok_synthetic_prefill/summarize.py" | head -25

echo "== existing pushed copies of screen.sh outside the checkout? =="
ls -l "$HOME/overnight-scratch/"*.sh 2>/dev/null | head -20 || echo "(none)"
ls -l "$HOME/tools/" 2>/dev/null | head -20 || echo "(no ~/tools)"
echo "== done =="
exit 0
