#!/usr/bin/env bash
# exp_22 build -- host-side wrapper (CPU ONLY; compiles and inspects, never
# launches a GPU job).
#
#   bash build.sh          # link the ubench + resource remark + CPU gates
#   bash build.sh isa      # ... and disassemble the three role kernels
#
# All real work happens in build_in.sh inside the CPU container `subha_k1`,
# against the PRIVATE scratch clone at $HOME/e22/DHK. Nothing here touches
# $HOME/Distributed-HipKittens (a pinned GPU campaign owns that checkout) or the
# megakernel sources.
set -uo pipefail
docker exec subha_k1 bash /home/subvadla/e22/src/build_in.sh "${1:-}"
