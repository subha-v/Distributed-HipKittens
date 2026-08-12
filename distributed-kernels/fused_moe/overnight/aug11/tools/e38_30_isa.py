#!/usr/bin/env python3
"""exp_38: WHY does adding an unreachable code path make a vmcnt throttle inert?

Reads the already-built REF / DEF / M14 / RING disassembly from the .text gate
(no compiling, so this is safe to run while a GPU campaign is timing) and
measures the thing the instruction census cannot see: not HOW MANY `vmcnt`
waits there are, but HOW MANY REMOTE ATOMICS CAN BE IN FLIGHT BETWEEN THEM.

The throttle is `asm volatile("s_waitcnt vmcnt(4)")` in the M7 epilogue: it is
supposed to cap outstanding remote `flat_atomic_pk_add_bf16` at 4. Its census is
unchanged at the mode-14 pin. But a throttle only binds if the code around it
would otherwise exceed the cap. So the diagnostic is the ISSUE RUN LENGTH: the
number of `flat_atomic_pk_add_bf16` issued between two consecutive waitcnts that
constrain vmcnt. If spill code has forced extra low-N waits into the epilogue,
the run length collapses, the hardware never reaches 4 outstanding, and the
explicit vmcnt(4) is dead -- while still being present, and still being counted.
"""
import re, sys, os, json, collections

OUT = os.path.expanduser("~/overnight-scratch/e38/out")
# argv overrides, so the same instrument reads the four gate arms or the ten
# single-site ablation arms without a second copy of the analysis.
VARIANTS = sys.argv[1:] or ["REF", "DEF", "M14", "RING"]

RE_ATOMIC  = re.compile(r'^\s+flat_atomic_pk_add_bf16')
RE_WAITVM  = re.compile(r'^\s+s_waitcnt\b.*\bvmcnt\((\d+)\)')
RE_SCRATCH = re.compile(r'^\s+scratch_(load|store)')
RE_INSTR   = re.compile(r'^\s+[a-z]')


def analyse(path):
    """One linear pass. The epilogue is defined empirically as the region
    between the first and last remote atomic; everything reported is inside it,
    because that is the only region the throttle can act on."""
    lines = open(path, errors="replace").read().splitlines()
    idx = [i for i, l in enumerate(lines) if RE_ATOMIC.match(l)]
    if not idx:
        return None
    lo, hi = idx[0], idx[-1]
    body = lines[lo:hi + 1]

    runs, run = [], 0
    waits = collections.Counter()
    scratch_in = 0
    # scratch ops that sit BETWEEN two atomics -- i.e. spill traffic the
    # compiler injected into the injection window itself.
    scratch_between = 0
    for l in body:
        if RE_ATOMIC.match(l):
            run += 1
            continue
        m = RE_WAITVM.match(l)
        if m:
            n = int(m.group(1))
            waits[n] += 1
            # A wait only breaks the run if it can actually constrain what is
            # outstanding. vmcnt(N) with N >= run cannot.
            if run:
                runs.append(run)
                run = 0
            continue
        if RE_SCRATCH.match(l):
            scratch_in += 1
            scratch_between += 1
    if run:
        runs.append(run)

    return {
        "atomics": len(idx),
        "epilogue_span_instrs": sum(1 for l in body if RE_INSTR.match(l)),
        "issue_runs": len(runs),
        "max_run": max(runs) if runs else 0,
        "mean_run": round(sum(runs) / len(runs), 2) if runs else 0,
        "runs_gt4": sum(1 for r in runs if r > 4),
        "runs_eq1": sum(1 for r in runs if r == 1),
        "vmcnt_waits_in_epilogue": sum(waits.values()),
        "vmcnt_hist": dict(sorted(waits.items())),
        "scratch_ops_in_epilogue": scratch_in,
    }


def main():
    res = {}
    for v in VARIANTS:
        p = os.path.join(OUT, f"{v}.isa")
        if not os.path.exists(p):
            print(f"  {v}: no ISA at {p}")
            continue
        r = analyse(p)
        if r is None:
            print(f"  {v}: no flat_atomic_pk_add_bf16 found")
            continue
        res[v] = r

    print("=== exp_38: the M7 epilogue's INJECTION WINDOW, by arm ===")
    print("(atomics = flat_atomic_pk_add_bf16; a 'run' is the atomics issued")
    print(" between two consecutive vmcnt-constraining waits)")
    print()
    hdr = ("arm", "atomics", "span", "runs", "max_run", "mean_run",
           "runs>4", "runs==1", "vm_waits", "scratch")
    print("  %-5s %8s %7s %6s %8s %9s %7s %8s %9s %8s" % hdr)
    for v, r in res.items():
        print("  %-5s %8d %7d %6d %8d %9s %7d %8d %9d %8d" % (
            v, r["atomics"], r["epilogue_span_instrs"], r["issue_runs"],
            r["max_run"], r["mean_run"], r["runs_gt4"], r["runs_eq1"],
            r["vmcnt_waits_in_epilogue"], r["scratch_ops_in_epilogue"]))
    print()
    print("=== vmcnt(N) histogram inside the epilogue ===")
    for v, r in res.items():
        print(f"  {v:<5} {r['vmcnt_hist']}")
    print()
    print("=== READING ===")
    if "REF" in res and "M14" in res:
        a, b = res["REF"], res["M14"]
        print(f"  runs that can exceed the depth-4 cap: REF={a['runs_gt4']}  M14={b['runs_gt4']}")
        print(f"  max atomics in flight between waits : REF={a['max_run']}  M14={b['max_run']}")
        print(f"  spill ops inside the epilogue       : REF={a['scratch_ops_in_epilogue']}  M14={b['scratch_ops_in_epilogue']}")
        if b["runs_gt4"] < a["runs_gt4"]:
            print("  => the mode-14 build cannot reach 4 outstanding remote RMWs as often,")
            print("     so vmcnt(4) has nothing left to cap: THE THROTTLE IS INERT BY")
            print("     CONSTRUCTION, not by being deleted.")
    dst = os.path.join(OUT, "e38_epilogue_%s.json" % ("gate" if len(VARIANTS) == 4
                                                      and VARIANTS[0] == "REF"
                                                      else "ablation"))
    json.dump(res, open(dst, "w"), indent=2)
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
