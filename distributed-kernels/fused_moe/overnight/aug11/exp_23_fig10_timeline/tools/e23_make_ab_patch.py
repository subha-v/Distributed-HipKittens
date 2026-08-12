#!/usr/bin/env python3
"""exp_23: generate e23_ab.patch for the node harness (outside the repo).

The harness file `prefill_opt/host/e004pf_k0pf_ab.py` lives in
~/amd-master/auto-gpu-kernel/k0_fused_moe, not in Distributed-HipKittens, so the
edit is carried as a committed patch the way exp_32 did. This script edits a
COPY and emits a unified diff; it never touches the harness itself. That matters
tonight: another agent's campaign is reading the live harness, and a mid-flight
edit would silently make some of its rotations a different arm.

Two hunks:
  1. grow `_pf6_mps_state` by 256*16 int64 for the ring's tail space
  2. add an opt-in `[MPS E23]` dump after the existing `[MPS SPIN]` block

Run on the node:  python3 e23_make_ab_patch.py > /tmp/e23_ab.patch
"""
import hashlib
import os
import subprocess
import sys
import tempfile

SRC = os.path.expanduser(
    "~/amd-master/auto-gpu-kernel/k0_fused_moe/prefill_opt/host/e004pf_k0pf_ab.py")
REL = "prefill_opt/host/e004pf_k0pf_ab.py"

OLD_ALLOC = '''        # 8 u32 words followed by 8 u64 timestamps = 96 bytes.
        _pf6_mps_state = torch.zeros((12,), dtype=torch.int64, device="cuda")
'''

NEW_ALLOC = '''        # 8 u32 words followed by 8 u64 timestamps = 96 bytes, then exp_23's
        # per-CTA phase ring: 256 CTAs x 16 slots x 8 B = 32 KiB of ADDITIVE
        # tail space in the SAME descriptor slot (K0P6_D_MPS_STATE). No new
        # slot, no host-bridge signature change, no kernel ABI change -- the
        # kernel finds the ring at u64 offset 12. This zero-init is the ring's
        # ONLY floor: M0 deliberately does not widen its reset loop, because
        # that would put a grid-wide 32 KiB store into the reset path of an arm
        # that ships with the ring off (see the K0P6_MPS_E23_* block in
        # moe_mps_adapter.cuh). Sound because every CTA rewrites the slots it
        # owns on every epoch, so a final-epoch read is current.
        _pf6_e23_slots = 16
        _pf6_e23_ctas = 256
        _pf6_mps_state = torch.zeros(
            (12 + _pf6_e23_ctas * _pf6_e23_slots,), dtype=torch.int64,
            device="cuda",
        )
'''

OLD_PRINT = '''            except Exception as _e2:
                print(f"[MPS SPIN] unavailable: {type(_e2).__name__}: {_e2}", flush=True)
'''

NEW_PRINT = '''            except Exception as _e2:
                print(f"[MPS SPIN] unavailable: {type(_e2).__name__}: {_e2}", flush=True)
            # --- exp_23 (paper Fig 3 / Q4b): per-CTA phase event ring dump.
            # Fixed slots in the tail of the same mps_state buffer, from u64
            # offset 12: cell (b, k) is the last time CTA b crossed boundary k,
            # in raw s_memrealtime ticks (100 MHz HSA domain, 1 tick = 0.01 us).
            # Like the [MPS TS] block above, these are FINAL-EPOCH values after
            # the soak.
            # OPT-IN on K0_E23_DUMP so that every existing arm's stdout stays
            # byte-identical: without the flag this block prints nothing, which
            # is what lets the patch sit in the harness while other campaigns
            # run. Needs K0_MPS_CFG timestamps=1 to have anything to print.
            if os.environ.get("K0_E23_DUMP", "0").strip() not in ("", "0"):
                try:
                    _e23t = None
                    try:
                        _e23t = pf6_state["mps_state"]
                    except Exception:
                        _e23t = _pf6_mps_state
                    _e23 = _e23t.detach().cpu().tolist()[12:]
                    _e23ns = 16
                    _e23nc = len(_e23) // _e23ns
                    print(f"[MPS E23] slots={_e23ns} ctas={_e23nc} tick_us=0.01 "
                          "names=KSTART,M2_DONE,M5_DONE,M6_DONE,M7_DONE,"
                          "SVC_ENTER,SVC_EXIT,M75_ENTER,M75_BAR,M75_EXIT,"
                          "M8_ENTER,REDUCE_DONE,M9_DONE,rsv13,rsv14,META",
                          flush=True)
                    _e23live = 0
                    for _e23b in range(_e23nc):
                        _e23row = [int(x) & 0xFFFFFFFFFFFFFFFF for x in
                                   _e23[_e23b * _e23ns:(_e23b + 1) * _e23ns]]
                        if not any(_e23row):
                            continue
                        _e23live += 1
                        print(f"[MPS E23 CTA] {_e23b} "
                              + " ".join(str(_v) for _v in _e23row), flush=True)
                    print(f"[MPS E23] live_ctas={_e23live}", flush=True)
                except Exception as _e3:
                    print(f"[MPS E23] unavailable: "
                          f"{type(_e3).__name__}: {_e3}", flush=True)
'''


def main():
    src = open(SRC, "r", encoding="utf-8", newline="").read()
    base_sha = hashlib.sha256(src.encode("utf-8")).hexdigest()

    for name, old in (("alloc", OLD_ALLOC), ("print", OLD_PRINT)):
        n = src.count(old)
        if n != 1:
            sys.stderr.write("anchor %s matched %d times, want 1\n" % (name, n))
            return 2

    out = src.replace(OLD_ALLOC, NEW_ALLOC).replace(OLD_PRINT, NEW_PRINT)
    new_sha = hashlib.sha256(out.encode("utf-8")).hexdigest()

    d = tempfile.mkdtemp(prefix="e23ab")
    a = os.path.join(d, "a")
    b = os.path.join(d, "b")
    open(a, "w", encoding="utf-8", newline="").write(src)
    open(b, "w", encoding="utf-8", newline="").write(out)
    diff = subprocess.run(
        ["git", "diff", "--no-index", "--no-color", "-U6", a, b],
        capture_output=True, text=True).stdout
    body = "\n".join(
        l for l in diff.splitlines()
        if not l.startswith(("diff --git", "index ", "--- ", "+++ ")))

    sys.stdout.write(
        "# exp_23 tier A -- node harness edit (file lives OUTSIDE this repo).\n"
        "# target:      ~/amd-master/auto-gpu-kernel/k0_fused_moe/%s\n"
        "# base sha256: %s\n"
        "# post sha256: %s\n"
        "# backup to:   ~/amd-master/auto-gpu-kernel/k0_fused_moe/%s.e23bak\n"
        "# apply:  cd ~/amd-master/auto-gpu-kernel/k0_fused_moe \\\n"
        "#           && cp %s %s.e23bak \\\n"
        "#           && git apply -p1 --check <this> && git apply -p1 <this>\n"
        "# revert: cp %s.e23bak %s\n"
        "#\n"
        "# NOT APPLIED YET AS OF WRITING: another agent's campaign was reading\n"
        "# the live harness, and editing it mid-flight would have made some of\n"
        "# its rotations a different arm. Apply at lease handover.\n"
        "--- a/%s\n+++ b/%s\n%s\n"
        % (REL, base_sha, new_sha, REL, REL, REL, REL, REL, REL, REL, body))
    return 0


if __name__ == "__main__":
    sys.exit(main())
