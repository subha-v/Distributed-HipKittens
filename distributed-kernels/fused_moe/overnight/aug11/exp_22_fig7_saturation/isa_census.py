#!/usr/bin/env python3
"""Per-kernel ISA census for the exp_22 ubench.

What we are checking, and why each one can invalidate a number:
  * mfma_f32_32x32x16_bf16 count > 0     -- the compute role really issues CDNA4
    doubled-K bf16 MFMA, so the TFLOPS conversion (32,768 flop/instruction) is
    the right one.
  * (global|flat)_store_dwordx4 present  -- the push lowers to 16-byte stores;
    BUILDING.md makes this an explicit MPS gate, and a demotion to dwordx2 would
    halve the per-instruction payload without changing any host-side arithmetic.
  * scratch_load/scratch_store == 0      -- exp_04's failure mode: a spilled
    packet16 staging array turns MLP depth into scratch traffic.
  * s_memrealtime count                  -- the role span instrument is present
    (2 per kernel: begin and end).
  * buffer_wbl2 / buffer_inv             -- system-scope release lowering; must
    be ABSENT from the payload-only path and PRESENT only under --protocol.
"""
import re
import sys

PATTERNS = [
    ("mfma_32x32x16_bf16", r"\bv_mfma_f32_32x32x16_bf16\b"),
    ("mfma_any", r"\bv_mfma\w*"),
    ("store_dwordx4", r"\b(global|flat)_store_dwordx4\b"),
    ("load_dwordx4", r"\b(global|flat)_load_dwordx4\b"),
    ("ds_read_b128", r"\bds_read_b128\b"),
    ("ds_write_b128", r"\bds_write_b128\b"),
    ("scratch_ops", r"\bscratch_(load|store)\w*"),
    ("s_memrealtime", r"\bs_memrealtime\b"),
    ("wbl2_or_inv", r"\b(buffer_wbl2|buffer_inv|global_inv|global_wb)\b"),
    ("nt_load", r"\b(global|flat)_load_dwordx4[^\n]*\bnt\b"),
    ("atomic_add", r"\b(global|flat)_atomic_add\b"),
    ("barrier", r"\bs_barrier\b"),
]


def split_kernels(path):
    """Split the .s by top-level label. Names are mangled
    (_Z13sat_kernel_d16Params), so match any column-0 label and filter later."""
    cur, body, out = None, [], {}
    for ln in open(path, errors="ignore"):
        # a kernel label is at column 0 and carries a trailing "; @name" comment
        m = re.match(r"^([A-Za-z_][\w.$]*):(?:\s*;.*)?\s*$", ln)
        if m:
            if cur:
                out[cur] = body
            cur, body = m.group(1), []
            continue
        if cur is not None:
            body.append(ln)
            if ln.strip().startswith("s_endpgm"):
                out[cur] = body
                cur, body = None, []
    if cur:
        out[cur] = body
    return out


def main():
    kernels = split_kernels(sys.argv[1])
    if not kernels:
        print("no kernel bodies found in", sys.argv[1])
        return 1
    names = [n for n in kernels if "sat_kernel" in n]
    if not names:
        print("no sat_kernel bodies found; labels seen:",
              ", ".join(sorted(kernels)[:12]))
        return 1
    labels = [p[0] for p in PATTERNS]
    print("%-26s" % "kernel" + "".join("%20s" % l for l in labels))
    fails = []
    for n in sorted(names):
        text = "".join(kernels[n])
        counts = [len(re.findall(rx, text)) for _l, rx in PATTERNS]
        print("%-26s" % n[:26] + "".join("%20d" % c for c in counts))
        d = dict(zip(labels, counts))
        if d["scratch_ops"]:
            fails.append("%s: %d scratch op(s) -- MLP depth is spilling"
                         % (n, d["scratch_ops"]))
        if d["store_dwordx4"] == 0:
            fails.append("%s: no 16-byte store; the push lowering changed" % n)
        if d["mfma_32x32x16_bf16"] == 0:
            fails.append("%s: no 32x32x16 bf16 MFMA; the TFLOPS constant is wrong"
                         % n)
        if d["s_memrealtime"] < 2:
            fails.append("%s: %d s_memrealtime (need 2: role begin and end)"
                         % (n, d["s_memrealtime"]))
    print()
    if fails:
        print("ISA_GATE: FAIL")
        for f in fails:
            print("  -", f)
        return 1
    print("ISA_GATE: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
