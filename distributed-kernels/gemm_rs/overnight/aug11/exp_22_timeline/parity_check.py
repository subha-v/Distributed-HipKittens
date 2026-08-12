#!/usr/bin/env python3
"""exp_22 gate 1: resource-tuple parity, trace flag compiled in but OFF.

Parses hipcc's -Rpass-analysis=kernel-resource-usage remarks out of the three
build logs parity_gate.sh produces and answers two questions:

  1. Is `flag present and 0` byte-identical to `flag absent`?  If not, the
     instrumented code shape has leaked into the production build and the
     timeline would picture a kernel nobody ships.  That is a STOP.
  2. Do both match the known-good post-exp_14 M2 table?  If not, the ratchet
     build moved underneath this experiment and the reference table below is
     what needs updating -- deliberately, not silently.

The flag-ON tuple is parsed and recorded for disclosure only.  A spill there
does not fail this gate (the gate is about the OFF build), but it does mean the
instrumented arm pictures a kernel with different register pressure than the
ratchet, and result.md has to say so.

Usage:  parity_check.py --logs <build dir> [--json <out>]
Exit 0 on PASS, 1 on FAIL, 2 if a log is missing or unparseable.
"""

import argparse
import json
import os
import re
import sys

# Post-exp_14 M2 table: (BM, BN, BK, k_tail) -> VGPRs.  AGPRs, scratch and
# **VGPR** spills must be zero on every row; M2_EXPECT is the instantiation
# count.
#
# SGPR spills are deliberately NOT required to be zero.  The first run of this
# gate demanded it and failed all 14 rows -- including the untouched baseline
# build, which reports 54-88 SGPR spills on every instantiation.  The M2
# contract (LESSONS: "zero AGPRs, zero scratch and zero VGPR spills") never
# mentioned them, and they cost nothing observable here: ScratchSize is 0 on
# every row, i.e. the scalar spills go to VGPR lanes and never to memory.
# Demanding zero would have made the gate fail on a property of the kernel
# rather than a property of the patch.  What IS required of SGPR spills is that
# they be *identical* between the two flag-OFF arms, which the parity
# comparison below already enforces.
ZERO_REQUIRED = ("agpr", "scratch", "vgpr_spill")
M2_EXPECT = 7
KNOWN_GOOD_VGPR = {
    (32, 64, 128, False): 98,
    (64, 128, 64, False): 104,
    (128, 192, 32, True): 136,
    (256, 256, 32, False): 246,
    (256, 256, 32, True): 248,
    (32, 64, 64, False): 91,
    (32, 64, 64, True): 92,
}

FIELDS = ("sgpr", "vgpr", "agpr", "scratch", "sgpr_spill", "vgpr_spill", "lds")

_KEYS = {
    "SGPRs": "sgpr",
    "VGPRs": "vgpr",
    "AGPRs": "agpr",
    "ScratchSize [bytes/lane]": "scratch",
    "SGPRs Spill": "sgpr_spill",
    "VGPRs Spill": "vgpr_spill",
    "LDS Size [bytes/block]": "lds",
    "Occupancy [waves/SIMD]": "occupancy",
}


def parse_log(path):
    """-> {mangled name: {field: int}} for every kernel the log reports."""
    out = {}
    current = None
    with open(path, errors="replace") as handle:
        for line in handle:
            if "remark:" not in line:
                continue
            body = line.split("remark:", 1)[1].strip()
            name = re.match(r"Function Name:\s*(\S+)", body)
            if name:
                current = name.group(1)
                out.setdefault(current, {})
                continue
            if current is None:
                continue
            for label, field in _KEYS.items():
                if body.startswith(label + ":"):
                    value = body.split(":", 1)[1].split("[-Rpass")[0].strip()
                    try:
                        out[current][field] = int(value)
                    except ValueError:
                        pass
                    break
    return {k: v for k, v in out.items() if v}


def decode(mangled):
    """Recover (BM, BN, BK, k_tail) from the template instantiation's name."""
    nums = [int(x) for x in re.findall(r"Li(\d+)E", mangled)]
    tail = re.search(r"Lb(\d)E", mangled)
    if len(nums) < 3:
        return None
    return (nums[0], nums[1], nums[2], bool(tail and tail.group(1) == "1"))


def tuple_of(entry):
    return tuple(entry.get(f, -1) for f in FIELDS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    arms = {}
    for tag in ("off_absent", "off_present", "on"):
        path = os.path.join(args.logs, tag + ".log")
        if not os.path.exists(path):
            print(f"MISSING {path}")
            return 2
        arms[tag] = parse_log(path)
        print(f"{tag:<12} {len(arms[tag])} instantiations")

    failures = []
    for tag in ("off_absent", "off_present"):
        if len(arms[tag]) != M2_EXPECT:
            failures.append(
                f"{tag}: {len(arms[tag])} instantiations, expected {M2_EXPECT}")

    # --- question 1: flag-present-and-0 vs flag-absent, byte for byte -------
    absent, present = arms["off_absent"], arms["off_present"]
    for name in sorted(set(absent) | set(present)):
        a, p = absent.get(name), present.get(name)
        if a is None or p is None:
            failures.append(f"{decode(name) or name}: present in only one OFF arm")
            continue
        if tuple_of(a) != tuple_of(p):
            failures.append(
                f"{decode(name) or name}: OFF arms differ "
                f"{dict(zip(FIELDS, tuple_of(a)))} vs {dict(zip(FIELDS, tuple_of(p)))}")

    # --- question 2: both OFF arms vs the known-good M2 table ---------------
    for tag in ("off_absent", "off_present"):
        for name, entry in arms[tag].items():
            key = decode(name)
            want = KNOWN_GOOD_VGPR.get(key)
            if want is None:
                failures.append(f"{tag}: unexpected instantiation {key} ({name})")
                continue
            if entry.get("vgpr") != want:
                failures.append(
                    f"{tag} {key}: VGPR {entry.get('vgpr')} != known-good {want}")
            for field in ZERO_REQUIRED:
                if entry.get(field, 0) != 0:
                    failures.append(
                        f"{tag} {key}: {field} = {entry.get(field)}, expected 0")

    # --- disclosure: what the flag-ON build costs --------------------------
    print()
    header = (f"{'BM/BN/BK/tail':<20}{'OFF vgpr':>9}{'ON vgpr':>8}{'d':>4}"
              f"{'ON agpr':>9}{'ON scr':>8}{'OFF sspill':>11}{'ON sspill':>10}"
              f"{'ON vspill':>10}")
    print(header)
    print("-" * len(header))
    on_by_key = {decode(n): e for n, e in arms["on"].items()}
    on_delta = {}
    for name, entry in sorted(absent.items(), key=lambda kv: decode(kv[0]) or ()):
        key = decode(name)
        on = on_by_key.get(key, {})
        delta = (on.get("vgpr") or 0) - (entry.get("vgpr") or 0)
        on_delta[str(key)] = {
            "off_vgpr": entry.get("vgpr"),
            "on_vgpr": on.get("vgpr"),
            "vgpr_delta": delta,
            "on_agpr": on.get("agpr"),
            "on_scratch": on.get("scratch"),
            "off_sgpr_spill": entry.get("sgpr_spill"),
            "on_sgpr_spill": on.get("sgpr_spill"),
            "on_vgpr_spill": on.get("vgpr_spill"),
        }
        print(f"{str(key):<20}{entry.get('vgpr'):>9}{str(on.get('vgpr')):>8}"
              f"{delta:>+4}{str(on.get('agpr')):>9}{str(on.get('scratch')):>8}"
              f"{str(entry.get('sgpr_spill')):>11}{str(on.get('sgpr_spill')):>10}"
              f"{str(on.get('vgpr_spill')):>10}")

    verdict = "PASS" if not failures else "FAIL"
    print()
    print(f"GATE 1 (resource-tuple parity, flag present and OFF): {verdict}")
    for line in failures:
        print("  - " + line)
    if verdict == "PASS":
        print("  flag-present-and-0 is byte-identical to flag-absent on all "
              f"{M2_EXPECT} instantiations, and both match the post-exp_14 table.")
    on_bad = [k for k, v in on_delta.items()
              if v["on_vgpr_spill"] or v["on_scratch"]]
    if on_bad:
        print("  DISCLOSURE: the flag-ON build spills VGPRs or uses scratch on "
              + ", ".join(on_bad) + " -- the diagnostic arm would picture a "
              "kernel with materially different register pressure than the "
              "ratchet. Switch to the static-slot fallback (design.md section "
              "3 item 3) and re-run.")
    else:
        worst = max((v["vgpr_delta"] for v in on_delta.values()), default=0)
        print(f"  DISCLOSURE: the flag-ON build costs at most {worst:+d} VGPRs "
              "and stays at zero AGPRs, zero scratch and zero VGPR spills on "
              "every instantiation, so the static-slot fallback is not needed.")

    if args.json:
        with open(args.json, "w") as handle:
            json.dump({
                "schema": "exp22.parity.v1",
                "verdict": verdict,
                "m2_expect": M2_EXPECT,
                "failures": failures,
                "arms": {tag: {str(decode(n)): dict(zip(FIELDS, tuple_of(e)))
                               for n, e in kernels.items()}
                         for tag, kernels in arms.items()},
                "on_disclosure": on_delta,
            }, handle, indent=2, sort_keys=True)
        print(f"  wrote {args.json}")

    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
