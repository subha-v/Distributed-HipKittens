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
        m = re.match(r"^([A-Za-z_.$][\w.$]*):\s*$", line)
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
    """(mfmas before first vmcnt(0), mfmas after it, total mfma, n vmcnt0)."""
    insts = [v for k, v in span if k == "inst"]
    total = sum(1 for v in insts if v.startswith("v_mfma"))
    vm0 = [i for i, v in enumerate(insts)
           if v.startswith("s_waitcnt") and re.search(r"vmcnt\(0\)", v)]
    if not vm0:
        return None, None, total, 0
    before = sum(1 for v in insts[:vm0[0]] if v.startswith("v_mfma"))
    after = sum(1 for v in insts[vm0[0]:] if v.startswith("v_mfma"))
    return before, after, total, len(vm0)


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
            before, after, total, nvm = mfma_vmcnt_profile(span)
            tail_flag = TAIL.search(sym)
            label = f"<256,256,32,{'true' if tail_flag and tail_flag.group(1) == '1' else 'false'}>"
            print(f"[{tag}] {label}: k-loop mfma={total} "
                  f"vmcnt0_in_loop={nvm} mfma_before_first={before} after={after}")
            if check == "tail":
                good = nvm >= 1 and before is not None and before >= 48
                if not good:
                    print(f"  FAIL: base arm expects the tail commit "
                          f"(>=48 MFMA before vmcnt(0)); got {before}")
                    ok = False
            else:
                good = (nvm >= 1 and before is not None
                        and 25 <= before <= 40 and after >= 25)
                if not good:
                    print(f"  FAIL: cmid arm expects the mid commit "
                          f"(25<=before<=40, after>=25); got before={before} "
                          f"after={after}")
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
            spills = sum(v for k, v in fields.items() if "Spill" in k)
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
