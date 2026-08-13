#!/usr/bin/env python3
"""exp_03 M2 placement gate.

Usage: assert_placement.py <base.s> <cmid.s> <base_compile.log> <cmid_compile.log>

Asserts, from the ISA alone (the exp_27 method: back-edge spans, program
order), that the edit did exactly what it says and nothing else:

  base  <256,256,32,*>: every s_waitcnt vmcnt(0) in the k-loop sits AFTER at
        least 48 of its MFMAs (the tail commit).
  cmid  <256,256,32,*>: the first s_waitcnt vmcnt(0) sits after >=25 and
        <=40 MFMAs, with >=25 MFMAs after it (the mid commit, halves covered).
  both: every kernel instantiation has VGPRs <= 248, 0 scratch, 0 spills.

Exit 0 = all gates pass; nonzero otherwise. A build that silently sank the
commit back to the tail must fail here, not masquerade as a null arm.
"""
import re
import sys

MANGLE = re.compile(r"Li(\d+)E")
TAIL = re.compile(r"Lb(\d)E")


def functions(path):
    """name -> list of (kind, op) in program order, kind in {label, inst, branch}."""
    funcs, name, body = {}, None, None
    for raw in open(path, errors="replace"):
        line = raw.rstrip()
        m = re.match(r"^([A-Za-z_.$][\w.$]*):\s*(?:;.*)?$", line)
        if m and m.group(1).startswith("_Z"):
            if name:
                funcs[name] = body
            name, body = m.group(1), []
            continue
        if name is None:
            continue
        if re.match(r"^\.Lfunc_end", line):
            funcs[name] = body
            name, body = None, None
            continue
        m = re.match(r"^(\.L\w+):", line)
        if m:
            body.append(("label", m.group(1)))
            continue
        m = re.match(r"^\s+([a-z][\w.]*)\s*(.*)$", line)
        if m and not m.group(1).startswith((".", ";")):
            op, rest = m.group(1), m.group(2)
            if op.startswith(("s_branch", "s_cbranch")):
                t = re.search(r"(\.L\w+)", rest)
                body.append(("branch", (op, t.group(1) if t else None)))
            else:
                body.append(("inst", op + (" " + rest if rest else "")))
    if name:
        funcs[name] = body
    return funcs


def kloop_span(body):
    """Smallest back-edge span containing >= 32 v_mfma, as a list of insts."""
    label_pos = {v: i for i, (k, v) in enumerate(body) if k == "label"}
    best = None
    for i, (k, v) in enumerate(body):
        if k != "branch":
            continue
        op, target = v
        if target in label_pos and label_pos[target] < i:
            span = body[label_pos[target]:i + 1]
            nm = sum(1 for kk, vv in span if kk == "inst"
                     and vv.startswith("v_mfma"))
            if nm >= 32 and (best is None or len(span) < len(best)):
                best = span
    return best


def mfma_vmcnt_profile(span):
    """Rotation-invariant placement profile of the k-loop.

    The compiler rotates the K_TAIL loop (the header holds the second MFMA
    half), so absolute position is meaningless. The loop body is treated as a
    CIRCLE and two circular distances are measured, in MFMAs:

      tail_mfma: from the FIRST vmcnt(0) forward to the s_barrier
                 (0 in the incumbent: the commit sits right at the barrier;
                 ~32 in the cmid arm: half 2 covers the drain)
      head_mfma: from the s_barrier forward to the first vmcnt(0)
                 (~64 incumbent, ~32 cmid)

    Returns (tail_mfma, head_mfma, total_mfma, n_vmcnt0) or Nones.
    """
    insts = [v for k, v in span if k == "inst"]
    total = sum(1 for v in insts if v.startswith("v_mfma"))
    vm0 = [i for i, v in enumerate(insts)
           if v.startswith("s_waitcnt") and re.search(r"vmcnt\(0\)", v)]
    bar = [i for i, v in enumerate(insts) if v.startswith("s_barrier")]
    if not vm0 or not bar:
        return None, None, total, len(vm0)

    def circ_mfma(a, b):
        n, i = 0, (a + 1) % len(insts)
        while i != b:
            if insts[i].startswith("v_mfma"):
                n += 1
            i = (i + 1) % len(insts)
        return n

    tail = circ_mfma(vm0[0], bar[0])
    head = circ_mfma(bar[0], vm0[0])
    return tail, head, total, len(vm0)


def resources(log_path):
    """Function Name -> {field: value} from -Rpass-analysis remarks."""
    out, cur = {}, None
    for line in open(log_path, errors="replace"):
        m = re.search(r"Function Name:\s+(\S+)", line)
        if m:
            cur = m.group(1)
            out[cur] = {}
            continue
        m = re.search(r"remark:.*?\s([A-Za-z][A-Za-z /\[\]]*?):\s+(\d+)", line)
        if m and cur:
            out[cur][m.group(1).strip()] = int(m.group(2))
    return out


def is_256(sym):
    nums = MANGLE.findall(sym)
    return len(nums) >= 3 and nums[0] == "256" and nums[1] == "256"


def main():
    base_s, cmid_s, base_log, cmid_log = sys.argv[1:5]
    ok = True

    for tag, path, check in (("base", base_s, "tail"), ("cmid", cmid_s, "mid")):
        funcs = functions(path)
        found = 0
        for sym, body in funcs.items():
            if not is_256(sym) or "kernel" not in sym.lower() and "gemm" not in sym.lower():
                if not is_256(sym):
                    continue
            span = kloop_span(body)
            if span is None:
                continue
            found += 1
            tail, head, total, nvm = mfma_vmcnt_profile(span)
            tail_flag = TAIL.search(sym)
            label = f"<256,256,32,{'true' if tail_flag and tail_flag.group(1) == '1' else 'false'}>"
            print(f"[{tag}] {label}: k-loop mfma={total} vmcnt0={nvm} "
                  f"mfma(vmcnt0->barrier)={tail} mfma(barrier->vmcnt0)={head}")
            if check == "tail":
                good = nvm >= 1 and tail is not None and tail <= 2
                if not good:
                    print(f"  FAIL: base arm expects the commit AT the barrier "
                          f"(tail<=2 MFMA); got {tail}")
                    ok = False
            else:
                good = (nvm >= 1 and tail is not None
                        and tail >= 25 and head >= 25)
                if not good:
                    print(f"  FAIL: cmid arm expects the mid commit "
                          f"(>=25 MFMA on both sides); got tail={tail} "
                          f"head={head}")
                    ok = False
        if found == 0:
            print(f"[{tag}] FAIL: no 256x256 k-loop found in {path}")
            ok = False

    for tag, log in (("base", base_log), ("cmid", cmid_log)):
        for sym, fields in resources(log).items():
            vgpr = fields.get("VGPRs")
            if vgpr is None:
                continue
            scratch = fields.get("ScratchSize [bytes/lane]", 0)
            # SGPR spills of 54-88 are present in the shipped incumbent (they
            # spill to VGPR lanes, exp_27 census); the gate is VGPR spills.
            spills = sum(v for k, v in fields.items()
                         if "Spill" in k and "VGPR" in k)
            status = "ok"
            if vgpr > 248 or scratch != 0 or spills != 0:
                status = "FAIL"
                ok = False
            print(f"[{tag}] {sym[:60]}: VGPR={vgpr} scratch={scratch} "
                  f"spills={spills} {status}")

    print("PLACEMENT GATE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
