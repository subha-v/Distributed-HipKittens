"""Apply the minimum compatibility repair to the frozen rank-1 submission.

The submission replaces Triton's HIPLauncher with its own, whose generated C
launcher parses Triton's `packed_metadata` as six ints:

    int num_warps, num_ctas, shared_memory, clusterDimX, clusterDimY, clusterDimZ;
    PyArg_ParseTuple(kernel_metadata, "iiiiii", ...)

Triton 3.6.0 (the version in every ROCm image on this node) packs three:
`(num_warps, num_ctas, shared_memory)`. Its own launcher parses "iii". So the
submission raises `TypeError: function takes exactly 6 arguments (3 given)` on
the first launch.

This repair is behaviour-preserving, not merely convenient: `clusterDimX/Y/Z`
are forwarded to `_launch` and then *never read* -- `_launch` passes only
`num_warps` and `shared_memory` to `hipModuleLaunchKernel`. So parsing three
ints instead of six and pinning the cluster dims to 1 changes nothing that
reaches the GPU, and rank-1 keeps its own lean launcher rather than falling back
to Triton's (which would have understated its performance on the small shapes,
where launch overhead is a large fraction of a 6-8 us budget).

Nothing else in the submission is touched. The patch fails loudly if the anchor
is not found exactly once.
"""

import hashlib
import sys

EXPECTED_SHA = "7940fcb81df06c1d8b1e1a77051f23c934149a688441ef48b2751b3f336f0dc5"

ANCHOR = '''  int num_warps, num_ctas, shared_memory, clusterDimX, clusterDimY, clusterDimZ;
  if (!PyArg_ParseTuple(kernel_metadata, \\"iiiiii\\", &num_warps, &num_ctas, &shared_memory, &clusterDimX, &clusterDimY, &clusterDimZ)) {{'''

REPLACEMENT = '''  int num_warps, num_ctas, shared_memory;
  // COMPAT(triton 3.6.0): packed_metadata is (num_warps, num_ctas,
  // shared_memory); the cluster dims below are forwarded to _launch but never
  // read by it, so pinning them to 1 changes nothing that reaches the GPU.
  int clusterDimX = 1, clusterDimY = 1, clusterDimZ = 1;
  if (!PyArg_ParseTuple(kernel_metadata, \\"iii\\", &num_warps, &num_ctas, &shared_memory)) {{'''


def main():
    source, destination = sys.argv[1], sys.argv[2]
    original = open(source).read()

    digest = hashlib.sha256(original.encode()).hexdigest()
    print(f"source sha256 : {digest}")
    print(f"expected      : {EXPECTED_SHA}")
    if digest != EXPECTED_SHA:
        print("REFUSING: source is not the frozen rank-1 submission")
        return 1

    count = original.count(ANCHOR)
    print(f"anchor occurrences: {count}")
    if count != 1:
        print("REFUSING: anchor not found exactly once. Anchor was:")
        print(ANCHOR)
        return 1

    patched = original.replace(ANCHOR, REPLACEMENT)
    with open(destination, "w") as handle:
        handle.write(patched)

    print(f"wrote {destination}")
    print(f"patched sha256: {hashlib.sha256(patched.encode()).hexdigest()}")
    print(f"line delta    : {len(patched.splitlines()) - len(original.splitlines())}")
    print("\n--- the only change ---")
    for line in ANCHOR.splitlines():
        print(f"  - {line}")
    for line in REPLACEMENT.splitlines():
        print(f"  + {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
