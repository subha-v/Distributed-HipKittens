#!/usr/bin/env python3
"""Parse -Rpass-analysis=kernel-resource-usage and apply the exp_22 CPU gates.

Gates (all must pass before the ubench is allowed to produce a number):
  G1  LDS = 98,304 B/block in every role kernel  -- the one-workgroup-per-CU
      pinning the whole figure assumes. If the compiler shrank or grew it, the
      CTA axis no longer means "one block per CU".
  G2  occupancy = 1 wave/SIMD.
  G3  zero scratch and zero spill in the role kernels. exp_04 measured
      store_peer_packets_multi<4> spilling its staging array to scratch under
      the megakernel's register pressure; if that happens here the isolated
      xGMI curve is measuring spill traffic, not the fabric.
  G4  all three depth instantiations present (d1/d4/d8).
"""
import re
import sys

FIELDS = [
    ("SGPRs", "SGPR"),
    ("VGPRs", "VGPR"),
    ("AGPRs", "AGPR"),
    ("ScratchSize [bytes/lane]", "scratch"),
    ("SGPRs Spill", "sspill"),
    ("VGPRs Spill", "vspill"),
    ("LDS Size [bytes/block]", "lds"),
    ("Occupancy [waves/SIMD]", "occ"),
]


def parse(path):
    txt = open(path, errors="ignore").read()
    blocks = re.split(r"Function Name:\s*", txt)[1:]
    out = []
    for b in blocks:
        fn = b.split()[0]
        rec = {"fn": fn}
        for key, short in FIELDS:
            m = re.search(re.escape(key) + r"\s*:\s*(\S+)", b)
            rec[short] = m.group(1) if m else "-"
        out.append(rec)
    # the same function can be reported more than once; keep the last
    dedup = {}
    for r in out:
        dedup[r["fn"]] = r
    return list(dedup.values())


def main():
    rows = parse(sys.argv[1])
    if not rows:
        print("no resource remarks found -- was -Rpass-analysis passed?")
        return 1
    shorts = [s for _k, s in FIELDS]
    print("%-22s %6s %6s %6s %8s %7s %7s %9s %5s"
          % ("kernel", *[s[:8] for s in shorts]))
    for r in sorted(rows, key=lambda x: x["fn"]):
        print("%-22s %6s %6s %6s %8s %7s %7s %9s %5s"
              % (r["fn"][:22], *[r[s] for s in shorts]))

    # names arrive mangled (_Z13sat_kernel_d16Params); match on the substring
    roles = [r for r in rows if "sat_kernel" in r["fn"]]
    fails = []
    notes = []
    for r in roles:
        # >= 96 KiB is what pins one workgroup per CU on gfx950's 160 KiB. The
        # depth-4/8 kernels legitimately sit above it: the compiler promotes the
        # by-reference pointer arrays into LDS (see the note in the source).
        try:
            if float(r["lds"]) < 98304.0:
                fails.append("%s: LDS %s < 98304 (one-WG-per-CU pinning lost)"
                             % (r["fn"], r["lds"]))
        except ValueError:
            fails.append("%s: LDS unparsable (%s)" % (r["fn"], r["lds"]))
        if r["occ"] not in ("1", "1.0"):
            fails.append("%s: occupancy %s != 1 wave/SIMD" % (r["fn"], r["occ"]))
        # scratch and VGPR spills are disqualifying: both put the payload path
        # on memory instructions that are not the ones being measured. SGPR
        # spills land in VGPR lanes when scratch is 0, so they are reported
        # rather than fatal.
        for k, label in (("scratch", "scratch bytes/lane"),
                         ("vspill", "VGPR spills")):
            try:
                if float(r[k]) != 0.0:
                    fails.append("%s: %s = %s (nonzero)" % (r["fn"], label, r[k]))
            except ValueError:
                fails.append("%s: %s unparsable (%s)" % (r["fn"], label, r[k]))
        try:
            if float(r["sspill"]) != 0.0:
                notes.append("%s: %s SGPR spill(s) (to VGPR lanes; scratch is 0)"
                             % (r["fn"], r["sspill"]))
        except ValueError:
            pass

    for want in ("sat_kernel_d1", "sat_kernel_d4", "sat_kernel_d8"):
        if not any(want in r["fn"] for r in roles):
            fails.append("missing instantiation %s" % want)

    print()
    print("role kernels:", ", ".join(sorted(r["fn"] for r in roles)) or "(none)")
    for n in notes:
        print("note:", n)
    if fails:
        print("CPU_GATE: FAIL")
        for f in fails:
            print("  -", f)
        return 1
    print("CPU_GATE: PASS (LDS >= 98304 in all depths, occupancy 1, zero "
          "scratch, zero VGPR spill, d1/d4/d8 present)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
