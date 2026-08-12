#!/usr/bin/env bash
# torchrun launches this once per rank; it wraps rocprofv3 around RANK 0 only.
#
# Profiling all eight ranks would multiply the profiler's own overhead across
# the collective and interleave eight dispatch streams in one output tree, and
# the figure plots rank 0 anyway (rank symmetry is stated, not averaged --
# FIGURE_SPECS.md section 6).  The other seven ranks run unprofiled so the
# collective still has real peers.
#
#   b0_rank0_wrap.sh <profile out dir> <python args...>
set -uo pipefail
OUTDIR=$1; shift

if [ "${RANK:-1}" = "0" ]; then
  exec /opt/rocm/bin/rocprofv3 --kernel-trace \
       -d "$OUTDIR" -o k --output-format csv -- python3 -u "$@"
else
  exec python3 -u "$@"
fi
