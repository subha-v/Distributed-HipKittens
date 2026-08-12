"""Attribute the kernel's runtime to its three phases by ablation.

Generates a SCRATCH copy of the kernel with three macro-gated cuts and builds
one module per arm. The deliverable source is not touched: MI300X_DESIGN.md
section 9 states no ablation code is present in it, and that stays true.

  full        everything
  no_mainloop skip the GEMM k-loop (accumulators stay zero) -> isolates the
              mainloop's math and A/B global->LDS traffic
  emit_local  emit the same bytes through the same path but to the local rank's
              own slot -> the delta from full is the XGMI cost of egress,
              holding the staging and store instruction count fixed
  no_reduce   reducers still wait, acquire, drain and credit, but do not pull
              or sum -> isolates the reduction's read/sum/store
  no_protocol reducers exit immediately and producers skip the credit wait ->
              the delta from full prices all cross-rank synchronization

Only `full` is numerically correct; the others exist to price the phases.

An earlier arm that skipped emission entirely faulted on three of four shapes,
so it was replaced by emit_local, which keeps every address in bounds while
preserving the store volume.
"""

import os
import subprocess
import sys

import harness_lib as H
from harness_lib import rt, WORLD, GemmRS

REPO = "/home/subvadla/dhk"
SRC = f"{REPO}/distributed-kernels/gemm_rs/gemm_rs_mi300x.cpp"
HARNESS = f"{REPO}/distributed-kernels/gemm_rs/overnight/harness"
SCRATCH = f"{HARNESS}/ablate"
BUILD = f"{HARNESS}/build"

# (anchor, replacement) pairs. Each anchor must appear exactly once.
#
# ANCHOR HYGIENE, learned by paying for it (aug11/exp_20_attribution): three of
# these nine anchors were dead against the kernel they were meant to cut, and
# `generate()` raises on the first one, so the whole table was unobtainable.
# Two causes, both avoidable:
#   - Anchors that quoted COMMENT text. E3's group loop rewrote the egress
#     comment from "per-band credit wait, emit, one release, publish" to
#     "per-band credit wait, then emit", killing the mainloop's #endif anchor.
#     Every anchor below now quotes only executable code.
#   - Anchors that carried the WRONG INDENTATION. E3 wrapped the tile body in an
#     outer group loop, moving the mainloop from 12 to 16 spaces and the emit
#     body from 16 to 20. A leading-whitespace mismatch on the FIRST line of a
#     multi-line anchor still matches (str.count sees the tail of the real
#     indent), which is why anchors 1 and 8 kept working by luck while 2, 3 and
#     9 -- whose mismatch is on a CONTINUATION line, where the newline pins the
#     column -- silently went to zero. Indentation below is exact.
PATCHES = [
    # 1. mainloop
    #
    # Re-anchored after exp_03 (E1b) replaced the fused `G::load` with the
    # issue/commit split, and again after E3 re-indented the body. Anchor on the
    # k=0 prologue issue pair: the in-loop issues carry `kn`, so `{0, 0, tm, 0}`
    # is unique, and no comment wording is involved.
    ("""                m3::load_issue<ST_A, NT>(abuf, g.a, {0, 0, tm, 0});
                m3::load_issue<ST_B, NT>(bbuf, g.b, {0, 0, tn, 0});""",
     """#if !ABL_SKIP_MAINLOOP
                m3::load_issue<ST_A, NT>(abuf, g.a, {0, 0, tm, 0});
                m3::load_issue<ST_B, NT>(bbuf, g.b, {0, 0, tn, 0});"""),
    # Closes the mainloop cut on the k-loop's own last statements plus its brace
    # -- structural, and `As[(k + 1) & 1]` appears exactly once in the file.
    ("""                    m3::load_commit<NT>(As[(k + 1) & 1], abuf);
                    m3::load_commit<NT>(Bs[(k + 1) & 1], bbuf);
                    __syncthreads();
                }""",
     """                    m3::load_commit<NT>(As[(k + 1) & 1], abuf);
                    m3::load_commit<NT>(Bs[(k + 1) & 1], bbuf);
                    __syncthreads();
                }
#endif"""),
    # 2. emit destination: same bytes, same path, local slot.
    # The declaration alone is unique and is the whole anchor; the previous
    # version dragged in the `lrow` line above it, which also occurs in the
    # credit-wait block and at a different indent.
    ("""                    int dest_eff = dest;""",
     """                    int dest_eff = dest;
#if ABL_EMIT_LOCAL
                    dest_eff = me;   // same volume, no XGMI
#endif"""),
    # 3. reduce
    ("""        m3::pull_sum_bf16_strip_mlp8(""",
     """#if !ABL_SKIP_REDUCE
        m3::pull_sum_bf16_strip_mlp8("""),
    ("""            threadIdx.x, m3::CTA_THREADS);

        // All consumer reads must drain before any retirement credit is""",
     """            threadIdx.x, m3::CTA_THREADS);
#endif

        // All consumer reads must drain before any retirement credit is"""),
    # 4. the per-tile directional release (an L2 writeback, buffer_wbl2 sc0 sc1)
    ("""            m3::release_payload_system();""",
     """#if !ABL_NO_RELEASE
            m3::release_payload_system();
#endif"""),
    # 5. all cross-rank synchronization
    ("""    if (m3::error_bit_set(errp, m3::ERR_PRODUCER_CREDIT |
                                m3::ERR_REDUCER_READY)) return;

    const int red_tiles = lrows * cols;""",
     """    if (m3::error_bit_set(errp, m3::ERR_PRODUCER_CREDIT |
                                m3::ERR_REDUCER_READY)) return;
#if ABL_NO_PROTOCOL
    return;
#endif

    const int red_tiles = lrows * cols;"""),
    ("""                if (threadIdx.x < (unsigned)bands) {""",
     """#if !ABL_NO_PROTOCOL
                if (threadIdx.x < (unsigned)bands) {"""),
    ("""                __syncthreads();
                if (m3::error_bit_set(errp, m3::ERR_PRODUCER_CREDIT)) return;""",
     """                __syncthreads();
                if (m3::error_bit_set(errp, m3::ERR_PRODUCER_CREDIT)) return;
#endif"""),
]

PROLOGUE = """
// ---- SCRATCH ABLATION BUILD, generated by exp_ablation.py ----
#ifndef ABL_SKIP_MAINLOOP
#define ABL_SKIP_MAINLOOP 0
#endif
#ifndef ABL_EMIT_LOCAL
#define ABL_EMIT_LOCAL 0
#endif
#ifndef ABL_SKIP_REDUCE
#define ABL_SKIP_REDUCE 0
#endif
#ifndef ABL_NO_PROTOCOL
#define ABL_NO_PROTOCOL 0
#endif
#ifndef ABL_NO_RELEASE
#define ABL_NO_RELEASE 0
#endif
"""

ARMS = [
    ("gemm_rs_abl_full", []),
    ("gemm_rs_abl_nomain", ["-DABL_SKIP_MAINLOOP=1"]),
    ("gemm_rs_abl_emitlocal", ["-DABL_EMIT_LOCAL=1"]),
    ("gemm_rs_abl_nored", ["-DABL_SKIP_REDUCE=1"]),
    ("gemm_rs_abl_noproto", ["-DABL_NO_PROTOCOL=1"]),
    ("gemm_rs_abl_norelease", ["-DABL_NO_RELEASE=1"]),
]

SHAPES = [
    (64, 7168, 18432, False, 1234),
    (512, 4096, 12288, True, 663),
    (2048, 2880, 2880, True, 166),
    (4096, 4096, 4096, False, 1371),
    (8192, 4096, 14336, True, 7168),
    (8192, 8192, 29568, False, 42),
]


def generate():
    os.makedirs(SCRATCH, exist_ok=True)
    text = open(SRC).read()
    for anchor, replacement in PATCHES:
        count = text.count(anchor)
        if count != 1:
            raise SystemExit(f"anchor appears {count} times, expected 1:\n"
                             f"{anchor[:120]}")
        text = text.replace(anchor, replacement)
    marker = '#include "gemm_rs_mi300x_hk_adapter.cuh"'
    text = text.replace(marker, marker + PROLOGUE, 1)
    path = f"{SCRATCH}/gemm_rs_ablate.cpp"
    with open(path, "w") as handle:
        handle.write(text)
    print(f"generated {path} ({len(text)} bytes, {len(PATCHES)} cuts)")
    return path


def build(path):
    rocm = os.environ.get("ROCM_PATH", "/opt/rocm")
    pyinc = subprocess.check_output(
        ["python3", "-c",
         "import sysconfig;print(sysconfig.get_paths()['include'])"],
        text=True).strip()
    pbinc = subprocess.check_output(
        ["python3", "-c", "import pybind11;print(pybind11.get_include())"],
        text=True).strip()
    common = [
        "hipcc", "-std=c++20", "-O3", "-DKITTENS_CDNA3",
        "-DHIP_ENABLE_WARP_SYNC_BUILTINS", "-ffast-math",
        "--offload-arch=gfx942", "-shared", "-fPIC",
        f"-I{REPO}/include", f"-I{REPO}/include/pyutils",
        f"-I{rocm}/include/hip", f"-I{pbinc}", f"-I{pyinc}",
        f"-I{REPO}/distributed-kernels/gemm_rs",
        "-Wno-nan-infinity-disabled",
    ]
    for name, flags in ARMS:
        target = f"{BUILD}/{name}.so"
        if os.path.exists(target):
            print(f"  {name}: already built")
            continue
        cmd = common + flags + [f"-DTK_MODNAME={name}", path, "-o", target]
        print(f"  building {name} ...", flush=True)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0 or not os.path.exists(target):
            print(result.stdout[-3000:])
            print(result.stderr[-3000:])
            raise SystemExit(f"build failed for {name}")
    print("  all ablation arms built")


def run_cell(module_name, shape, iters):
    """Each cell runs in its own process: a GPU fault aborts the process."""
    m, n, k, bias, seed = shape
    cmd = [sys.executable, "-u", f"{HARNESS}/exp_ablation_one.py",
           module_name, str(m), str(n), str(k), str(int(bias)), str(iters),
           str(seed)]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=HARNESS,
                            timeout=900)
    for line in result.stdout.splitlines():
        if line.startswith("RESULT"):
            fields = dict(part.split("=", 1)
                          for part in line.split()[1:] if "=" in part)
            return {"us": float(fields["wall_us"]),
                    "device_us": float(fields["device_us"]),
                    "correct": fields.get("correct") == "1",
                    "errors": fields.get("errors", "none")}
    tail = (result.stdout + result.stderr).strip().splitlines()
    reason = "FAULT" if any("Memory access fault" in x for x in tail) else "FAIL"
    return {"us": float("nan"), "device_us": float("nan"), "correct": False,
            "errors": reason, "tail": tail[-6:], "rc": result.returncode}


def main():
    path = generate()
    build(path)
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 30

    print("\n" + "=" * 104)
    print("phase attribution (pipelined us per world-8 operation)")
    print("=" * 104)
    header = (f"{'shape':<20}{'full':>9}{'noMain':>9}{'emitLoc':>9}"
              f"{'noRed':>9}{'noProto':>9}{'noRelse':>9}"
              f"{'GEMM':>8}{'XGMI':>8}{'reduce':>8}{'sync':>8}{'release':>8}")
    print(header)
    print("-" * len(header))
    for shape in SHAPES:
        m, n, k, bias, seed = shape
        times, notes = {}, []
        for name, _ in ARMS:
            cell = run_cell(name, shape, iters)
            times[name] = cell["us"]
            if cell["errors"] != "none":
                notes.append(f"{name}: {cell['errors']}")
                if "tail" in cell:
                    notes.append(f"    {cell['tail'][-3:]}")

        def value(key):
            return times.get(f"gemm_rs_abl_{key}", float("nan"))

        full = value("full")
        print(f"{f'{m}x{n}x{k}':<20}{full:>9.1f}{value('nomain'):>9.1f}"
              f"{value('emitlocal'):>9.1f}{value('nored'):>9.1f}"
              f"{value('noproto'):>9.1f}{value('norelease'):>9.1f}"
              f"{full - value('nomain'):>8.1f}"
              f"{full - value('emitlocal'):>8.1f}"
              f"{full - value('nored'):>8.1f}"
              f"{full - value('noproto'):>8.1f}"
              f"{full - value('norelease'):>8.1f}", flush=True)
        for note in notes:
            print(f"    {note}")
    print("-" * len(header))
    print("GEMM   = full - noMain    (mainloop math + A/B global->LDS traffic)")
    print("XGMI   = full - emitLoc   (peer cost of egress, store volume fixed)")
    print("reduce = full - noRed     (8-source pull, fp32 sum, bf16 store)")
    print("sync   = full - noProto   (cross-rank waits, credits, publishes)")
    print("release= full - noRelse   (the per-tile buffer_wbl2 L2 writeback)")
    print("Single-cut deltas overlap where the phases overlap; they need not "
          "sum to `full`. noRelse is NOT a correct kernel -- it removes the "
          "publication ordering -- it exists only to price the writeback.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
