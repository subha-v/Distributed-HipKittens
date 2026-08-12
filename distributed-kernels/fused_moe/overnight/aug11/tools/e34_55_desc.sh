#!/usr/bin/env bash
# exp_34 BLOCKING diagnostic: my same-session mode-12 control at the ratchet cfg
# (C=16,g=353) reads 7,222 us where rev 26 read 6,488 -- and rev 26's g=65 read
# 7,111. So either (A) rev 28 decodes g=353 differently for mode 12, or (B) my
# explicit "timestamps=0" perturbs the parse. The harness dumps the packed
# descriptor word the KERNEL actually read; compare mine against rev 26's.
set -uo pipefail
python3 - <<'PY'
import glob, json, os, re
HOME = os.path.expanduser("~")

def descs(root):
    out = {}
    for f in sorted(glob.glob(os.path.join(root, "**", "mps_desc_rank0.txt"), recursive=True)):
        try:
            out[f] = open(f, errors="ignore").read().strip()[:400]
        except Exception:
            pass
    return out

print("################ my rev-28 runs (tag e34c1) ################")
for d in sorted(glob.glob(f"{HOME}/k0-mok-e34c1/*/")):
    cfg = os.path.basename(d.rstrip("/"))
    ds = descs(d)
    print(f"--- {cfg}")
    for f, v in list(ds.items())[:1]:
        print("    desc:", v)
    if not ds:
        print("    (no mps_desc dump)")

print("\n################ historical rev-26 roots ################")
cands = []
for root in glob.glob(f"{HOME}/k0-mok-*/*/"):
    n = os.path.basename(root.rstrip("/"))
    if "mode12" in n and ("g353" in n or "g65" in n):
        cands.append(root)
for root in sorted(cands)[-14:]:
    ds = descs(root)
    print(f"--- {os.path.basename(root.rstrip('/'))}")
    for f, v in list(ds.items())[:1]:
        print("    desc:", v)
    if not ds:
        print("    (no mps_desc dump)")
PY
echo
echo "===what the kernel/adapter says about the g bitfield and the new 0x80 bit==="
R=$HOME/Distributed-HipKittens/distributed-kernels/fused_moe
grep -n "kCoarseKeepDrain\|0x80\b\|0x8000\b" "$R/moe_mps_adapter.cuh" | head -30
echo "---- parse of the cfg string (C,g,mode,flush_rows,timestamps) ----"
grep -rn "timestamps" "$R/moe_mps_adapter.cuh" | head -20
echo "===DONE==="
