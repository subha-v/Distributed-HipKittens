#!/usr/bin/env bash
# exp_32 Job 1: dump the exact NODE text of every region the poison patch touches,
# with line numbers, so the unified diff applies to the live file. READ-ONLY.
set -uo pipefail
AB=$HOME/amd-master/auto-gpu-kernel/k0_fused_moe/prefill_opt/host/e004pf_k0pf_ab.py
RC=$HOME/amd-master/auto-gpu-kernel/k0_fused_moe/benchmarks/mok_synthetic_prefill/run_campaign.sh

echo "===AB sha256==="; sha256sum "$AB"
echo "===R1 alloc 1790-1806==="; sed -n '1790,1806p' "$AB" | cat -n | awk '{printf "%d\t%s\n", $1+1789, substr($0, index($0,$2))}'
echo "===R2 env-knob home: MOK tolerance block 130-165==="; awk 'NR>=130 && NR<=165 {printf "%d\t%s\n", NR, $0}' "$AB"
echo "===R3 eager loop 4498-4525==="; awk 'NR>=4498 && NR<=4525 {printf "%d\t%s\n", NR, $0}' "$AB"
echo "===R4 mok gate loop 4565-4592==="; awk 'NR>=4565 && NR<=4592 {printf "%d\t%s\n", NR, $0}' "$AB"
echo "===R5 negative control 4810-4835==="; awk 'NR>=4810 && NR<=4835 {printf "%d\t%s\n", NR, $0}' "$AB"
echo "===R6 soak 4930,5005==="; awk 'NR>=4930 && NR<=5005 {printf "%d\t%s\n", NR, $0}' "$AB"
echo "===R7 measure_arm 5045,5105==="; awk 'NR>=5045 && NR<=5105 {printf "%d\t%s\n", NR, $0}' "$AB"
echo "===RC sha256==="; sha256sum "$RC"
echo "===RC 100,160==="; awk 'NR>=100 && NR<=160 {printf "%d\t%s\n", NR, $0}' "$RC"
exit 0
