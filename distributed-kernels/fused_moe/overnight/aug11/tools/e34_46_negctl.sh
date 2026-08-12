#!/usr/bin/env bash
# exp_34 ladder step 4b: THE PROTOCOL NEGATIVE CONTROL. Expected to FAIL.
#
# The pinned checkout is now owned by another agent, so the control lives in a
# PRIVATE COPY of the pinned tree and is selected with DHK_ROOT, which
# run_campaign.sh:8 exposes as an env knob and bind-mounts read-only. Nothing
# under ~/Distributed-HipKittens is touched.
#
# The only source difference is mode 14's rendezvous publish loop: `R < world-1`
# leaves rank 7 untold. REQUIRED outcome: pperr bit 25 (33554432,
# K0P6_MPS_ERR_M7DONE) on rank 7 only, mps_mega gate FAIL. If it passes, the
# rendezvous is not load-bearing and the whole rung is meaningless.
set -uo pipefail
SRC=$HOME/Distributed-HipKittens
NC=$HOME/e34/negctl_dhk

echo "############ build the private control tree from the PIN ############"
rm -rf "$NC"; mkdir -p "$NC"
cp -a "$SRC/." "$NC/"
git -C "$NC" rev-parse --short HEAD

python3 - <<'PY'
import io, re, sys
p = "/home/subvadla/e34/negctl_dhk/distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip"
s = io.open(p, encoding="utf-8").read()
m = re.search(r"( *)for \(int R = 0; R < world; \+\+R\) \{\n(?:.*\n){0,8}?"
              r".*publish_epoch<hk_moe::scope::agent>\(self_slot, epoch32\);", s)
if not m: print("ANCHOR FAIL"); sys.exit(1)
head = m.group(0)
if s.count(head) != 1: print("NOT UNIQUE"); sys.exit(1)
s = s.replace(head, head.replace("for (int R = 0; R < world; ++R) {",
                                 "for (int R = 0; R < world - 1; ++R) {   // exp_34 NEGATIVE CONTROL"))
s = s.replace("#define K0P6_MPS_SRC_REV 28", "#define K0P6_MPS_SRC_REV 1028")
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("control patched")
PY
[ $? -eq 0 ] || { echo "PATCH FAILED"; exit 1; }
echo "--- the whole difference vs the pin ---"
diff -u "$SRC/distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip" \
        "$NC/distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip"
echo "--- pinned tree still clean? ---"
git -C "$SRC" status --porcelain | head

echo
echo "############ run the control (1/1/1) ############"
export DHK_ROOT="$NC"
export SCREEN_ARMS=production,pf6gm_mega,mps_mega
export SCREEN_WARMUP=1 SCREEN_TIMED=1 SCREEN_PROCS=1
export SCREEN_SYNC=0
export SCREEN_TRACE=1
export SCREEN_JOB_TIMEOUT=3600 SCREEN_RUN_TIMEOUT=3300
CFG=/tmp/e34_neg_cfgs.txt
echo 'C=0,g=353,mode=14,flush_rows=16,timestamps=1' > "$CFG"
date -u
setsid -w timeout 4000 bash "$HOME/tools/screen.sh" e34neg "$CFG"
echo "===SCREEN_RC=$?==="; date -u

echo
echo "############ EXPECTED-FAILURE EVIDENCE ############"
L=$(ls -t $HOME/overnight-scratch/e34neg_*.log | head -1)
echo "log: $L"
echo "--- pperr values (33554432 = bit 25 expected) ---"
grep -hoE "pperr=[0-9]+" "$L" | sort | uniq -c | sort -rn | head
echo "--- per-rank pperr / gate lines ---"
grep -hE "\[MOK GATE\]|pperr=|rank=|RANK" "$L" | grep -iE "pperr|gate" | head -25
echo "--- bit decode of every distinct nonzero pperr ---"
python3 - <<PY
import re
t=open("$L",errors="ignore").read()
vals=sorted({int(v) for v in re.findall(r"pperr=(\d+)",t)})
names={2097152:"GRIDBAR(21)",16777216:"A2DONE(24)",33554432:"M7DONE(25)",
       67108864:"SERVICE(26)",134217728:"DUAL(27)",4194304:"PAD(22)",65536:"POS(16)"}
for v in vals:
    bits=[names.get(1<<i,f"bit{i}") for i in range(32) if v>>i&1]
    print(v, bits)
PY
echo "===DONE==="
