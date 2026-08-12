#!/usr/bin/env bash
# Phase B: the official-evaluator cross-check, with the repaired run_rank1_bench3.sh.
#
# ONE rotation, not the designed two. Reason: five other experiments are cycling
# this node tonight and this run was already preempted once at 06:40; two rotations
# is a ~3 h lease hold and invites a second preemption, whereas instrument A -- the
# headline, which already carries the full arm-order rotation -- is finished and
# safe on disk. Instrument B exists to cross-check ORDERING under the competition's
# real one-process-per-rank topology, and its arms are separate process pools with
# no shared warm state, so first-arm bias is far weaker here than in A. The reduced
# rotation is disclosed in result.md.
set -uo pipefail
D=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/aug11/exp_24_ladders
export LAD_KEEP=1 LAD_EXPECT_A=6 LAD_PHASE=B LAD_ROTATIONS=1 LAD_CAP=9000
exec bash "$D/go_full.sh"
