#!/usr/bin/env bash
# exp_03 per-arm build + ISA verification.   arm_isa.sh <arm_name>
#
# Why this exists and must run BEFORE gate_ladder.sh: gate_ladder's M2 step runs
# tools/m2_report.sh, which only *reads* $ON/build/m1a.log and
# $ON/build/isa/*.s. Neither is produced by harness/build.sh (that writes
# harness/build/*.log). So M2 inside the ladder reports whatever the last
# tools/m1_build.sh + tools/m2_isa.sh left behind. This refreshes both, archives
# them per arm, and prints the three things an arm is judged on:
#   1. per-instantiation VGPR/AGPR/SGPR/scratch/spill
#   2. the k-loop's sync-relevant instructions in PROGRAM ORDER
#   3. whether any scratch op landed inside the k-loop
# No GPU work: compile + text analysis only.
set -uo pipefail

ARM=${1:?usage: arm_isa.sh <arm_name>}
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
EXP=$ON/experiments/exp_03_mainloop
A=$EXP/arms/$ARM
S=$ON/build/isa/gemm_rs_mi300x-hip-amdgcn-amd-amdhsa-gfx942.s
mkdir -p "$A"

echo "########## $ARM: M1 compile (resource remarks) ##########"
docker exec dhk-gemmrs bash $ON/tools/m1_build.sh > "$A/m1_build.log" 2>&1
echo "m1_build exit=$?"
grep -E '^M1a exit|^M1b exit|^OK|^FAIL' "$A/m1_build.log"
if grep -qE 'error:' "$A/m1_build.log"; then
  echo "=== COMPILE ERRORS ==="
  grep -E 'error:' "$A/m1_build.log" | head -40
  exit 1
fi

echo
echo "########## $ARM: ISA (--save-temps) ##########"
docker exec dhk-gemmrs bash $ON/tools/m2_isa.sh > "$A/m2_isa.log" 2>&1
echo "m2_isa exit=$?"
grep -E 'save-temps exit|^ISA file:|^ *[0-9]+ /' "$A/m2_isa.log" | head -5
if [ ! -f "$S" ]; then echo "NO ISA at $S"; exit 1; fi
cp "$S" "$A/gemm_rs_mi300x.gfx942.s"

echo
echo "########## $ARM: resource table ##########"
docker exec dhk-gemmrs bash $ON/tools/m2_report.sh > "$A/m2_report.log" 2>&1
sed -n '/resource tuples/,/^$/p;/instantiation/,/CTA\/CU/p' "$A/m2_report.log" | head -40

echo
echo "########## $ARM: k-loop program order ##########"
docker exec dhk-gemmrs bash $EXP/p0_30_kloop.sh > "$A/kloop_run.log" 2>&1
for f in "$EXP"/logs/p0_kloop_inventory.txt; do
  [ -f "$f" ] && cp "$f" "$A/kloop_inventory.txt"
done
for f in "$EXP"/isa/kloop_256_256_32_*.s; do
  [ -f "$f" ] && cp "$f" "$A/$(basename "$f")"
done
if [ -f "$A/kloop_inventory.txt" ]; then
  sed -n '/DEEPEST MFMA LOOP/,/raw dump/p' "$A/kloop_inventory.txt" | head -120
else
  echo "no inventory produced; see $A/kloop_run.log"
  tail -20 "$A/kloop_run.log"
fi

echo
echo "########## $ARM: scratch inside the k-loop? ##########"
python3 - "$A/kloop_inventory.txt" <<'PY'
import re, sys
try: t = open(sys.argv[1], errors='replace').read()
except Exception as e: print("no inventory:", e); raise SystemExit(0)
for blk in t.split('#'*110):
    m = re.search(r'SYMBOL (\S+)', blk)
    if not m: continue
    n = len(re.findall(r'scratch_(store|load)', blk))
    print(f"  {m.group(1):<22} scratch ops inside k-loop: {n}"
          f"   {'OK' if n == 0 else '*** SPILL IN K-LOOP ***'}")
PY

echo
echo "########## $ARM: archived to $A ##########"
ls -la "$A"
echo "################ DONE ################"
