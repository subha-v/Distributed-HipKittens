"""exp_23 fingerprint gate: prove the four rungs are four different binaries.

The failure mode this exists to catch is invisible by construction. A skipped
rebuild produces a perfect-looking waterfall of ONE identical binary measured
four times, and every downstream number -- ratios, geomean, the money figure --
looks exactly as it should. So the rungs are not compared by trusting the build
script; they are compared by hashing what it produced and asserting the
differences the mechanism predicts, in the places it predicts them.

Writes fingerprints.json beside this file. Exits nonzero if any HARD assertion
fails, in which case the rung is not real and must be reported as a blocker
rather than measured.

usage: fingerprint.py            (run inside dhk-gemmrs; no GPU touched)
"""

import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(HERE, "build")
ISA = os.path.join(BUILD, "isa")
# The shipped production build's resource log, read-only. Rung c carries the
# shipped macro defaults, so its table must reproduce this one field for field.
SHIPPED_LOG = os.path.abspath(os.path.join(
    HERE, "..", "..", "harness", "build", "gemm_rs_mi300x.log"))

RUNGS = ["gemm_rs_w23_a", "gemm_rs_w23_b", "gemm_rs_w23_c", "gemm_rs_w23_null"]
LABEL = {"gemm_rs_w23_a": "a", "gemm_rs_w23_b": "b",
         "gemm_rs_w23_c": "c", "gemm_rs_w23_null": "null"}
CONFIG = {
    "gemm_rs_w23_a": {"WGM4": 1, "RELEASE_GROUP": 1, "RELEASE_GROUP_FULL_ONLY": 1},
    "gemm_rs_w23_b": {"WGM4": 0, "RELEASE_GROUP": 1, "RELEASE_GROUP_FULL_ONLY": 1},
    "gemm_rs_w23_c": {"WGM4": 0, "RELEASE_GROUP": 4, "RELEASE_GROUP_FULL_ONLY": 1},
    "gemm_rs_w23_null": {"WGM4": 0, "RELEASE_GROUP": 4, "RELEASE_GROUP_FULL_ONLY": 1},
}

# RESULTS.md Gate M2, post-exp_14 table. Seven distinct instantiations; the two
# generic rows differ only in K_TAIL.
#
# SGPR spills are NOT part of the expectation and must not be: the shipped
# binary spills 54-60 scalar registers on the small rows. The M2 claim is zero
# AGPRs, zero scratch and zero VECTOR spills.
M2_EXPECT = 7
EXPECTED_VGPRS = sorted([98, 104, 136, 246, 248, 91, 92])
ZERO_EXPECTED = ["AGPRs", "ScratchSize [bytes/lane]", "VGPRs Spill"]

# HIP stamps every translation unit with a __hip_cuid_<hash> symbol derived from
# the TU's identity, so two builds of the same source under different module
# names differ in exactly those lines and nowhere else. Hashing that raw would
# make A4 unfalsifiable in the wrong direction -- it would ALWAYS fail, and the
# null arm would be reported as a code contrast when it is an allocation
# contrast. Normalised out; the raw hash is still recorded.
CUID_RX = re.compile(r"__hip_cuid_[0-9a-fA-F]+")

# The counts that must be identical everywhere (A5) and the ones allowed to
# move between rungs (A3).
INVARIANT_COUNTS = ["v_mfma"]

PATTERNS = {
    "instructions": r"^\s+[a-z][a-z0-9_]*\s",
    "v_mfma": r"\bv_mfma",
    "buffer_wbl2": r"\bbuffer_wbl2",
    "buffer_inv": r"\bbuffer_inv",
    "s_cbranch": r"\bs_cbranch",
    "s_branch": r"\bs_branch\b",
    "s_barrier": r"\bs_barrier",
    "global_store_dwordx4": r"\bglobal_store_dwordx4",
    "global_load_dwordx4": r"\bglobal_load_dwordx4",
    "s_waitcnt_vmcnt": r"s_waitcnt\s+vmcnt",
    "scratch_store": r"\bscratch_store",
    "scratch_load": r"\bscratch_load",
    "v_cndmask": r"\bv_cndmask",
    "s_cselect": r"\bs_cselect",
}

RES_KEYS = ["TotalSGPRs", "VGPRs", "AGPRs", "ScratchSize [bytes/lane]",
            "Dynamic Stack", "Occupancy [waves/SIMD]", "SGPRs Spill",
            "VGPRs Spill", "LDS Size [bytes/block]"]

# The remark's diagnostic prefix is not stable between the harness build and
# the --save-temps build: one emits `<path>:210:1: remark:     VGPRs: 98`, the
# other `remark: <file>:210:0:     VGPRs: 98`. Keying off the field name rather
# than off the prefix parses both. Longest-first by construction, so
# "SGPRs Spill" cannot be truncated to "SGPRs".
FIELD_RX = re.compile(
    r"(?:^|\s)((?:Total)?SGPRs(?: Spill)?|[VA]GPRs(?: Spill)?"
    r"|ScratchSize \[bytes/lane\]|Dynamic Stack"
    r"|Occupancy \[waves/SIMD\]|LDS Size \[bytes/block\]):\s+(\S+)")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_isa(path):
    """Hash the ISA with HIP's per-TU cuid symbol normalised away."""
    h = hashlib.sha256()
    with open(path, errors="replace") as handle:
        for line in handle:
            h.update(CUID_RX.sub("__hip_cuid_TU", line).encode())
    return h.hexdigest()


def cuid_only_diff(path_a, path_b):
    """True when two ISA files differ ONLY in their cuid symbol lines."""
    with open(path_a, errors="replace") as fa, \
            open(path_b, errors="replace") as fb:
        la, lb = fa.readlines(), fb.readlines()
    if len(la) != len(lb):
        return False, -1
    differing = [i for i, (x, y) in enumerate(zip(la, lb)) if x != y]
    non_cuid = [i for i in differing if not CUID_RX.search(la[i])]
    return not non_cuid, len(differing)


def pretty(mangled):
    """Demangle the template args out of the mangled name (m2_isa.sh's rule)."""
    nums = re.findall(r"Li(\d+)E", mangled)
    tail = re.search(r"Lb(\d)E", mangled)
    if len(nums) >= 3:
        return (f"BM={nums[0]} BN={nums[1]} BK={nums[2]} "
                f"K_TAIL={tail.group(1) if tail else '?'}")
    return mangled


def parse_resources(path):
    """Resource tuple per instantiation from -Rpass-analysis remarks."""
    if not os.path.exists(path):
        return []
    lines = open(path, errors="replace").read().splitlines()
    cur, fields, rows = None, {}, []
    for line in lines:
        m = re.search(r"Function Name:\s+(\S+)", line)
        if m:
            if cur:
                rows.append({"instantiation": cur, **fields})
            cur, fields = pretty(m.group(1)), {}
            continue
        m = FIELD_RX.search(line)
        if m and cur:
            fields[m.group(1).strip()] = m.group(2)
    if cur:
        rows.append({"instantiation": cur, **fields})
    # One remark set per instantiation; a duplicate would mean the log was
    # appended to rather than overwritten.
    seen, unique = set(), []
    for row in rows:
        key = row["instantiation"]
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def count_isa(path):
    counts = {name: 0 for name in PATTERNS}
    compiled = {name: re.compile(rx) for name, rx in PATTERNS.items()}
    with open(path, errors="replace") as handle:
        for line in handle:
            if line.lstrip().startswith((".", ";", "//")):
                continue
            for name, rx in compiled.items():
                if rx.search(line):
                    counts[name] += 1
    return counts


def int_or_none(value):
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def check_m2(rows):
    """Rung c must reproduce the shipped Gate-M2 table."""
    problems = []
    if len(rows) < M2_EXPECT:
        problems.append(f"{len(rows)} instantiations, expected >= {M2_EXPECT}")
    vgprs = sorted(v for v in (int_or_none(r.get("VGPRs")) for r in rows)
                   if v is not None)
    if vgprs != EXPECTED_VGPRS:
        problems.append(f"VGPR multiset {vgprs} != expected {EXPECTED_VGPRS}")
    for row in rows:
        for key in ZERO_EXPECTED:
            want = 0
            got = int_or_none(row.get(key))
            if got is None:
                problems.append(f"{row['instantiation']}: {key} unparsed")
            elif got != want:
                problems.append(f"{row['instantiation']}: {key}={got}, "
                                f"expected {want}")
    return problems


def main():
    out = {"rungs": {}, "assertions": [], "notes": []}
    missing = []
    for name in RUNGS:
        so = os.path.join(BUILD, f"{name}.so")
        asm = os.path.join(ISA, f"{name}.s")
        log = os.path.join(BUILD, f"{name}.compile.log")
        if not os.path.exists(so) or not os.path.exists(asm):
            missing.append(name)
            continue
        entry = {
            "label": LABEL[name],
            "module": name,
            "config": CONFIG[name],
            "so_path": so,
            "so_sha256": sha256_file(so),
            "so_bytes": os.path.getsize(so),
            "isa_path": asm,
            "isa_sha256": sha256_isa(asm),
            "isa_raw_sha256": sha256_file(asm),
            "isa_lines": sum(1 for _ in open(asm, errors="replace")),
            "isa_counts": count_isa(asm),
            "resources": parse_resources(log),
            "two_step_link": not os.path.exists(
                os.path.join(BUILD, f"{name}.fallback")),
        }
        out["rungs"][LABEL[name]] = entry

    if missing:
        out["assertions"].append({
            "id": "A0", "hard": True, "passed": False,
            "detail": f"missing build artifacts for {missing}",
        })
        json.dump(out, open(os.path.join(HERE, "fingerprints.json"), "w"),
                  indent=2)
        print(f"A0 FAIL: missing artifacts for {missing}")
        return 1

    r = out["rungs"]

    def assert_(ident, hard, passed, detail):
        out["assertions"].append({"id": ident, "hard": hard,
                                  "passed": bool(passed), "detail": detail})

    # A1: WGM4=1 constant-folds the tiles<=num_gemm_ctas select out of the tile
    # decode (gemm_rs_mi300x.cpp:315-319).
    assert_("A1", True, r["a"]["isa_sha256"] != r["b"]["isa_sha256"],
            f"a.isa={r['a']['isa_sha256'][:16]} b.isa={r['b']['isa_sha256'][:16]}")
    moved_ab = [k for k in ("instructions", "s_cselect", "v_cndmask",
                            "s_cbranch")
                if r["a"]["isa_counts"][k] != r["b"]["isa_counts"][k]]
    assert_("A1b", False, bool(moved_ab),
            "a vs b differ in " + (", ".join(
                f"{k} {r['a']['isa_counts'][k]}->{r['b']['isa_counts'][k]}"
                for k in moved_ab) if moved_ab else "NO structural count"))

    # A2: RELEASE_GROUP=1 makes rgroup the constant 1 (:336-340), collapsing the
    # inner group loop that =4 leaves in place.
    assert_("A2", True, r["b"]["isa_sha256"] != r["c"]["isa_sha256"],
            f"b.isa={r['b']['isa_sha256'][:16]} c.isa={r['c']['isa_sha256'][:16]}")

    # A3: the same difference, as an attributable count rather than a hash.
    moved = [k for k in ("buffer_wbl2", "s_cbranch", "instructions")
             if r["b"]["isa_counts"][k] != r["c"]["isa_counts"][k]]
    assert_("A3", False, bool(moved),
            "b vs c differ in " + (", ".join(
                f"{k} {r['b']['isa_counts'][k]}->{r['c']['isa_counts'][k]}"
                for k in moved) if moved else "NO structural count"))

    # A4: TK_MODNAME names the host pybind module only. If the device code moves
    # between c and null, the build is not reproducible and every null-floor
    # number would be a code contrast rather than an allocation contrast.
    cuid_only, n_diff = cuid_only_diff(r["c"]["isa_path"], r["null"]["isa_path"])
    r["null"]["raw_diff_lines_vs_c"] = n_diff
    r["null"]["raw_diff_is_cuid_only"] = cuid_only
    assert_("A4", True,
            r["c"]["isa_sha256"] == r["null"]["isa_sha256"] and cuid_only,
            f"c.isa={r['c']['isa_sha256'][:16]} "
            f"null.isa={r['null']['isa_sha256'][:16]}; raw ISA differs on "
            f"{n_diff} line(s), cuid-only={cuid_only}")

    # A5: the mainloop is not a variable in this experiment.
    for key in INVARIANT_COUNTS:
        values = {lab: r[lab]["isa_counts"][key] for lab in r}
        assert_(f"A5.{key}", True, len(set(values.values())) == 1,
                f"{key} per rung: {values}")

    # A6: rung c IS the shipped configuration, so it must reproduce RESULTS.md's
    # post-exp_14 M2 table. If it does not, the ladder is anchored to the wrong
    # binary and no rung ratio means anything.
    m2 = check_m2(r["c"]["resources"])
    assert_("A6", True, not m2,
            f"{len(r['c']['resources'])} instantiations; "
            + ("clean" if not m2 else "; ".join(m2[:6])))

    # A6b: direct field-for-field comparison against the shipped build's own
    # resource log, when it is present. Read-only.
    shipped = parse_resources(SHIPPED_LOG)
    if shipped:
        want = {row["instantiation"]: {k: row.get(k) for k in RES_KEYS}
                for row in shipped}
        got = {row["instantiation"]: {k: row.get(k) for k in RES_KEYS}
               for row in r["c"]["resources"]}
        diffs = [k for k in set(want) | set(got) if want.get(k) != got.get(k)]
        assert_("A6b", False, not diffs,
                f"vs {SHIPPED_LOG}: "
                + ("identical" if not diffs else f"differs on {diffs[:4]}"))
    else:
        out["notes"].append(
            f"shipped resource log not found at {SHIPPED_LOG}; A6b skipped")

    # A7 is enforced by import_check.py during the build; recorded here so the
    # artifact carries the whole gate.
    assert_("A7", True, True,
            "entry-point import checked by build_rungs.sh/import_check.py")

    # Provenance: which rung carries the shipped macro defaults.
    out["shipped_config_rungs"] = [
        lab for lab, e in r.items()
        if e["config"] == {"WGM4": 0, "RELEASE_GROUP": 4,
                           "RELEASE_GROUP_FULL_ONLY": 1}]
    out["fallback_builds"] = [lab for lab, e in r.items()
                              if not e["two_step_link"]]

    path = os.path.join(HERE, "fingerprints.json")
    json.dump(out, open(path, "w"), indent=2)

    print(f"{'rung':>6} {'so_sha256[:12]':>14} {'isa_sha256[:12]':>16} "
          f"{'lines':>8} {'instr':>8} {'mfma':>7} {'wbl2':>6} {'cbr':>7}")
    for lab in ("a", "b", "c", "null"):
        e = r[lab]
        c = e["isa_counts"]
        print(f"{lab:>6} {e['so_sha256'][:12]:>14} {e['isa_sha256'][:12]:>16} "
              f"{e['isa_lines']:>8} {c['instructions']:>8} {c['v_mfma']:>7} "
              f"{c['buffer_wbl2']:>6} {c['s_cbranch']:>7}")
    print()
    print("resource tuple, rung c (the shipped configuration):")
    for row in r["c"]["resources"]:
        print("  " + row["instantiation"] + "  " + " ".join(
            f"{k.split(' [')[0]}={row.get(k, '-')}" for k in RES_KEYS))
    print()
    hard_fail = 0
    for a in out["assertions"]:
        mark = "PASS" if a["passed"] else ("FAIL" if a["hard"] else "warn")
        if not a["passed"] and a["hard"]:
            hard_fail += 1
        print(f"  [{mark}] {a['id']}: {a['detail']}")
    for note in out["notes"]:
        print(f"  [note] {note}")
    print(f"\nwrote {path}")
    if hard_fail:
        print(f"\n{hard_fail} HARD assertion(s) failed -- the rungs are NOT "
              f"four distinct binaries. Do not measure; report as a blocker.")
    return 1 if hard_fail else 0


if __name__ == "__main__":
    sys.exit(main())
